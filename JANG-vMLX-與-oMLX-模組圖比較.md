# JANG / vLLM-metal 與 oMLX 模組圖技術比較

## 1. 比較前提

`oMLX-系統架構模組圖.md` 的主線很清楚：

- 核心目標是 Apple Silicon / MLX 上的大模型、長 context、多用戶 serving
- 核心瓶頸不是只有模型能不能載入，而是：
- `KV cache` 的容量
- decode 階段的記憶體頻寬
- 多租戶調度
- LPDDR / SSD 分層存放
- page 級精準讀取與預取

從 `jangq` 這條線裡，我認為真正值得借的技術可以拆成 3 類：

- `JANG mixed-precision quantization`
- `JANG runtime` 的 `mmap + fused dequant + Metal kernels`
- `vLLM + vllm-metal` 類的 serving 骨架：`OpenAI API + continuous batching + paged cache`

所以這次不是在問「哪一邊整體比較強」，而是要問：

**在 oMLX 的每一層裡，JANG / vLLM-metal 這條線有沒有更適合直接借來落地的部分；如果有，是取代主線，還是補強主線。**

---

## 2. 快速結論

如果只先講結論，我的判斷是：

1. **Serving 骨架層：`vLLM + vllm-metal` 比模組圖裡的抽象模組更適合先拿來落地**
   - 因為它已經有 OpenAI-compatible API、continuous batching、paged KV 這種可跑骨架。

2. **模型 / 權重量化層：`JANG` 比模組圖內既有技術更適合直接補進來**
   - 因為它對 Apple Silicon 上的大型 MoE 權重混合精度量化很有實績。

3. **KV lifecycle 主線：模組圖內技術明顯比 `JANG / vLLM-metal` 更適合**
   - 包含 `SnapKV / MiniCache / xKV / Quest / AdaptCache / EvicPress / SSD Prefetch`。
   - 這些技術直接對準你專案最核心的長 context + 多用戶 + decode memory-bound 問題。

4. **Metal kernel 層：兩邊不是二選一，而是分工**
   - `JANG runtime` 更適合做權重側 `dequant + GEMV/GEMM` 參考。
   - 模組圖裡的 `PolarQuant + QJL + fused attention` 更適合做 KV 側主線。

一句話總結：

**`vLLM + vllm-metal` 適合當 serving 骨架，`JANG` 適合當權重量化插件，而 oMLX 模組圖內的 KV policy / storage / retrieval 技術才應該是主線。**

---

## 3. 分層比較

### 3.1 Serving 層：`vLLM + vllm-metal` vs 模組圖的 Request Gateway / Scheduler / Continuous Batching

模組圖內對應區塊：

- Request Gateway
- Scheduler
- Continuous Batching
- Session / Sequence Manager

`vLLM + vllm-metal` 已經相對接近可落地骨架，優點是：

- 已有 `OpenAI-compatible API`
- 已有 `continuous batching`
- 已有 `paged KV cache`
- 已有多模型 / 多模態 / reasoning parser 等工程能力

### 哪個比較適合？

- **短中期落地：`vLLM + vllm-metal` 更適合**
- **長期架構主線：模組圖仍更適合**

原因是：

- `vLLM + vllm-metal` 解的是「Apple Silicon 上怎麼先把 serving 跑起來」
- 模組圖解的是「跑起來之後，怎麼把 KV cache 變成真正的資料系統」

也就是說：

- `vLLM + vllm-metal` 更像骨架
- 模組圖更像差異化能力設計

### 結論

**Serving 層先借 `vLLM + vllm-metal`，比完全從模組圖白手起家更適合。**

---

### 3.2 Runtime / Cache 層：`vLLM + vllm-metal` 的 paged cache vs 模組圖的 PagedAttention / Block Manager

模組圖內對應區塊：

- PagedAttention Engine
- Block Manager
- LPDDR Hot Cache
- Compressed Swap
- SSD Cold Cache

`vLLM + vllm-metal` 的強項在於：

- 已經有 page/block 概念
- 已經有 prefix cache / paged cache 的工程骨架
- 已經服務多請求場景

但它的弱點是：

- 目前重點仍是記憶體內快取與 serving 效率
- 沒有明確把 KV cache 做成你模組圖那種 `LPDDR -> warm swap -> SSD -> metadata -> prefetch` 完整生命週期

### 哪個比較適合？

- **作為 baseline runtime：`vLLM + vllm-metal` 更適合**
- **作為最終多層 KV 系統：模組圖更適合**

### 結論

**PagedAttention / Block Manager 的第一版可以直接借 `vLLM + vllm-metal` 思路，但真正的 SSD tiering、placement、eviction，還是模組圖方向比較適合。**

---

### 3.3 模型 / 權重量化層：`JANG` vs 模組圖裡的 Quantization Policy / PolarQuant / QJL

這一層最容易混淆，因為兩邊都在講「量化」，但其實打的目標不一樣。

#### `JANG` 主要解什麼

- 權重 mixed precision
- Attention、router、embedding 保較高 bit
- MLP / experts 壓低 bit
- 目標是讓大型 MoE 模型在 Apple Silicon 上低 bit 仍可用

#### 模組圖裡的量化主線主要解什麼

- `PolarQuant + QJL`
- `Hadamard / FWHT`
- `Quantization Policy`
- 目標更偏向 `KV cache` 的容量與 decode 搬運頻寬

### 哪個比較適合？

- **如果你問的是模型權重：`JANG` 更適合**
- **如果你問的是 KV cache：模組圖技術更適合**

### 原因

- `JANG` 對 Apple Silicon 的大型 MoE 權重低 bit 非常有價值，尤其能改善 MLX 對某些模型在 2-bit / 3-bit 下崩掉、NaN、失真的問題。
- 但 `JANG` 並沒有直接處理你專案最痛的 `KV lifecycle` 問題。
- `PolarQuant / QJL` 則直接對準 KV 壓縮與 decode bandwidth。

### 結論

**`JANG` 不應取代模組圖內的 `PolarQuant / QJL`，而應該作為「權重量化層」的補強。**

換句話說：

- `JANG`：權重側主力候選
- `PolarQuant / QJL`：KV 側主力候選

兩者可並存，而且這樣分工最合理。

---

### 3.4 Kernel / Compute 層：`JANG runtime` vs 模組圖的 Metal Kernel Layer

模組圖內對應區塊：

- Hadamard / FWHT
- PolarQuant + QJL Encode
- QJL Decode + Fused Attention/GEMV

`JANG runtime` 現成可借的點包括：

- `zero-copy mmap loading`
- `fused dequant + GEMV`
- `fused dequant + GEMM`
- Apple GPU threadgroup memory 的 kernel 實作經驗

但它的主力仍是：

- 權重載入
- 權重 dequant
- 一般 attention / GEMV / GEMM 路徑

而模組圖內的 kernel 層要處理的是：

- KV 壓縮前後的 encode / decode
- QJL / PolarQuant 的資料表示
- 將 `decode` 與 `attention` 融合
- 用算力換掉 KV bandwidth

### 哪個比較適合？

- **權重側 kernel 參考：`JANG runtime` 更適合**
- **KV 側 kernel 主線：模組圖更適合**

### 結論

**`JANG runtime` 很值得拿來借 kernel engineering 風格，但它不會取代你模組圖裡的 KV 專用 kernel 主線。**

---

### 3.5 Storage / Retrieval / Policy 層：`JANG / vLLM-metal` vs 模組圖的 SnapKV / Quest / AdaptCache / EvicPress

模組圖內對應區塊：

- SnapKV Filter
- Layer-wise KV Cache
- Quantization Policy
- Eviction / Placement Policy
- Quest Retrieval Policy
- Metadata Index
- Async Double Buffer Prefetch

這一層其實就是兩邊差距最大的地方。

`JANG / vLLM-metal` 這條線目前比較明確的強項不是：

- token 級過濾
- cross-layer KV 壓縮
- page 級 metadata retrieval
- utility-aware eviction
- SSD 預取排程

而模組圖內這些技術正好就是專案最核心的差異化能力。

### 哪個比較適合？

- **毫無疑問是模組圖內技術更適合**

### 原因

- 這一層直接解的是：
- 超長 context
- decode memory-bound
- 多用戶 sharing
- SSD 分層存放
- page 級精準取回

而這正是本專案真正要贏的地方。

### 結論

**在 policy / storage / retrieval 這層，不應該讓 `JANG / vLLM-metal` 搶主線；模組圖內技術才是更適合的方向。**

---

## 4. 對照總表

| 可借技術 | 對應模組圖區塊 | 哪個更適合 | 判斷 |
| --- | --- | --- | --- |
| `vLLM + vllm-metal` OpenAI API + continuous batching | Request Gateway / Scheduler / Continuous Batching | `vLLM + vllm-metal` | 先落地 serving 骨架時更適合，因為已經有成熟 serving 語意與 Apple 後端方向。 |
| `vLLM + vllm-metal` paged KV cache | PagedAttention / Block Manager | `vLLM + vllm-metal` 作 baseline，模組圖作長期目標 | 第一版 runtime 借它最快；但要做到 SSD tiering，模組圖更完整。 |
| `JANG` mixed-precision weight quantization | Quantization Policy（模型側） | `JANG` | 對 Apple Silicon 大型 MoE 的權重量化更成熟、更直接。 |
| `PolarQuant + QJL` | Quantization Policy（KV 側） / QJL Decode + Fused Attention | 模組圖 | 直接解 KV 容量與 decode bandwidth，與 `JANG` 不同層次。 |
| `JANG runtime` fused dequant GEMV/GEMM | Metal Kernel Layer | 分工，各自較適合不同路徑 | `JANG runtime` 借權重側 kernel；模組圖保留 KV 專用 fused attention 路線。 |
| `JANG runtime` zero-copy mmap | 權重載入 / SSD 映射思路 | `JANG runtime` 作工程參考 | 很值得借，但它本身不是完整的 KV 分層存放方案。 |
| `SnapKV / MiniCache / xKV / Quest / AdaptCache / EvicPress / MegaTrain prefetch` | KV Policy / Storage / Retrieval | 模組圖 | 這些才直接命中本專案主問題。 |

---

## 5. 最適合的組合方式

如果目標是最符合 oMLX 的方向，我建議這樣組：

### A. 骨架層

- 借 `vLLM + vllm-metal`
- 用它來提供：
- OpenAI-compatible API
- scheduler
- continuous batching
- paged KV baseline

### B. 模型層

- 借 `JANG`
- 用它來提供：
- Apple Silicon 友善的大型 MoE 權重量化
- mixed-precision bit allocation
- 某些模型在低 bit 下的穩定性

### C. 主線差異化能力

- 保留模組圖主線：
- SnapKV
- MiniCache / xKV
- PolarQuant / QJL
- Quest
- AdaptCache / EvicPress
- SSD cold cache + metadata index + async prefetch

### D. Kernel 分工

- 權重側：參考 `JANG runtime`
- KV 側：走模組圖的 `Hadamard / PolarQuant / QJL / fused attention`

---

## 6. 最後判斷

如果只問一句：

**把可借的 `JANG / vLLM-metal` 技術，和 `oMLX-系統架構模組圖.md` 裡的技術相比，誰比較適合？**

答案不是單選，而是分層：

1. **Serving 骨架層：`vLLM + vllm-metal` 比模組圖更適合先用**
2. **模型權重量化層：`JANG` 比模組圖現有內容更適合直接補進來**
3. **KV lifecycle 主線：模組圖內技術比 `JANG / vLLM-metal` 更適合，這不該讓位**
4. **最合理策略不是替代，而是拼接**

所以最終建議是：

**以 `vLLM + vllm-metal` 當骨架、以 `JANG` 當權重側插件、以模組圖內的 KV policy / storage / retrieval 技術當真正主線。**

這樣最符合你專案的方向，也最不容易被外部專案的現成能力帶偏重心。
