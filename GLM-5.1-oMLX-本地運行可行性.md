# GLM 5.1 在 oMLX 本地運行可行性備忘錄

日期：2026-04-27

這份文件不是技術評分表，而是針對 **GLM 5.1 這個具體模型**，評估它是否可能依據 [oMLX-系統架構模組圖.md](./oMLX-系統架構模組圖.md) 在本地 Apple Silicon 上運行。必要時參考 [LLM-推理-KV-壓縮與效能瓶頸.md](./LLM-推理-KV-壓縮與效能瓶頸.md) 內對 KV 壓縮、長 context 與 memory-bound decode 的推估。

結論先講：

> **GLM 5.1 很適合驗證 oMLX，但它比 Kimi K2.6 多一個更硬的條件：除了 MoE expert paging，還必須支援 `glm_moe_dsa` 的 DSA sparse attention / index cache 路徑。**

更務實地說：

- **256GB Mac：可做低併發 PoC，但不適合作為「完整可用」目標。**
- **512GB UMA 或多 GPU / 多節點：比較像穩定平台。**
- **192GB Mac：只適合做 loader、adapter、DSA/MLA/KV simulator、單層 expert path。**
- **第一階段不應追 200K context 或多用戶；應先追 text-only、短 context、batch 1 的穩定 decode。**

---

## 1. GLM 5.1 公開規格

| 項目 | 數值 |
| --- | --- |
| 架構 | MoE + DSA |
| HF architecture | `GlmMoeDsaForCausalLM` |
| HF model type | `glm_moe_dsa` |
| 總參數 | 約 744B / 754B |
| Active parameters | 約 40B per token |
| Layers | 78 |
| Dense layers | 前 3 層 dense |
| Hidden size | 6144 |
| Attention heads | 64 |
| Routed experts | 256 |
| Selected experts per token | 8 |
| Shared experts | 1 |
| MoE intermediate size | 2048 |
| Context length | 200K / 202,752 tokens |
| Attention | MLA + DSA |
| KV latent rank | `kv_lora_rank = 512` |
| RoPE head dim | `qk_rope_head_dim = 64` |
| DSA index heads | `index_n_heads = 32` |
| DSA index head dim | `index_head_dim = 128` |
| DSA top-k | `index_topk = 2048` |
| 官方 BF16 權重檔 | 約 1.51TB |
| 官方 FP8 權重檔 | 約 756GB |
| 模態 | Text-only |

來源：

- [Z.AI GLM-5.1 official docs](https://docs.z.ai/guides/llm/glm-5.1)
- [zai-org/GLM-5.1 model card](https://huggingface.co/zai-org/GLM-5.1)
- [zai-org/GLM-5.1 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)
- [zai-org/GLM-5.1 files](https://huggingface.co/zai-org/GLM-5.1/tree/main)
- [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8)
- [Transformers GlmMoeDsa docs](https://huggingface.co/docs/transformers/main/model_doc/glm_moe_dsa)
- [mlx-community/GLM-5.1](https://huggingface.co/mlx-community/GLM-5.1)

重要現實：

- Z.AI 文件標示 GLM 5.1 context length 為 200K、max output 128K。
- Hugging Face 官方 model card 提到可用 `SGLang`、`vLLM`、`xLLM`、`Transformers`、`KTransformers` 本地部署。
- `mlx-community/GLM-5.1` 已有 MLX 格式，但檔案仍約 1.49TB，這代表「MLX 格式存在」不等於「256GB Mac 可完整運行」。

---

## 2. 對 oMLX 架構的意義

GLM 5.1 同時壓到 oMLX 三條主線：

1. **MoE Weight Memory System**
   - BF16 1.51TB、FP8 756GB，完整常駐 256GB UMA 不可能。
   - 必須有 SSD cold expert store。
   - 必須有 active expert cache。
   - 必須有 route-aware prefetch。

2. **KV / DSA Memory System**
   - MLA 讓 KV latent cache 不至於像 full KV 那麼大。
   - DSA 透過 indexer 做 sparse attention，理論上降低 attention read bandwidth。
   - 但 DSA 需要 index cache / sparse selector，這會新增一條 metadata / index working set。

3. **Metal-native Execution Runtime**
   - 需要 MoE expert GEMV/GEMM。
   - 需要 MLA attention path。
   - 需要 DSA top-k index selection / sparse gather path。
   - 需要 fused dequant + attention，否則 Apple LPDDR 頻寬會被打爆。

因此，GLM 5.1 對 oMLX 的定位應該是：

> **不是單純 MoE 權重分層測試，而是 MoE 權重分層 + MLA KV + DSA sparse attention 的整合壓力測試。**

---

## 3. 權重工作集估算

官方 BF16 權重約 1.51TB，FP8 權重約 756GB。

對 256GB UMA 來說，兩者都不能完整 resident。必須採用：

```text
resident trunk/router/shared path
+ active expert cache
+ expert staging buffer
+ SSD cold expert store
```

粗估單個 routed expert：

```text
single expert params ~= 3 * hidden_size * moe_intermediate_size
                     ~= 3 * 6144 * 2048
                     ~= 37.7M params
```

以 FP8 約 `1 byte / param`：

```text
single expert ~= 37.7 MB
8 selected experts / layer ~= 302 MB
75 MoE layers selected route ~= 21.1 GB
```

若未來有可用 Q4 / INT4 權重量化，粗抓 `0.59 byte / param`：

```text
single expert ~= 22 MB
75 MoE layers selected route ~= 12.5 GB
```

這裡有兩個關鍵點：

- GLM 5.1 active params 約 40B，比 Kimi K2.6 的 32B active 更重。
- GLM 5.1 官方 FP8 權重檔 756GB，比 Kimi K2.6 native INT4 檔案更大，所以對 256GB Mac 來說，GLM 5.1 不比 K2.6 輕。

---

## 4. KV 與 DSA index 工作集估算

GLM 5.1 config 顯示：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
max_position_embeddings = 202752
num_hidden_layers = 78
```

MLA latent KV 可用下列近似估算：

```text
per-token per-layer MLA KV ~= 512 + 64 = 576 BF16 values
```

單 session 200K：

```text
576 values * 2 bytes * 78 layers * 202752 tokens
~= 17.0 GB
```

不同 KV 精度下：

| KV 形態 | 單 session 200K 粗估 |
| --- | --- |
| BF16 MLA KV | 約 17GB |
| 8-bit KV | 約 8.5GB |
| 4-bit KV | 約 4.2GB |
| 2-bit KV | 約 2.1GB |

但 GLM 5.1 不能只看 MLA KV，還要看 DSA indexer。

DSA config：

```text
index_n_heads = 32
index_head_dim = 128
index_topk = 2048
```

如果天真地把 `32 * 128 = 4096` 維 index feature 以 BF16 對每 token / layer 保存，單 session 200K 的 index cache 會達到：

```text
4096 values * 2 bytes * 78 layers * 202752 tokens
~= 120GB
```

這不代表實作一定真的要這樣存，但它說明一件事：

> **GLM 5.1 的長 context 壓力可能從「KV cache」轉移到「DSA index / sparse selector / gather locality」。**

因此 oMLX 的 Quest / Metadata Index 模組對 GLM 5.1 不是外掛，而是很接近 DSA runtime 的核心。  
Quest-style page selector 和 DSA indexer 不能各做一套互不相干的 metadata，否則會重複吃 UMA。

---

## 5. 與 Kimi K2.6 的差異

| 項目 | GLM 5.1 | Kimi K2.6 | 對 oMLX 的含義 |
| --- | --- | --- | --- |
| 總參數 | 約 744B / 754B | 1T | GLM 總參數較小 |
| Active params | 約 40B | 約 32B | GLM 每 token active 更重 |
| Layers | 78 | 61 | GLM 層數更多，KV / scheduler 更複雜 |
| Experts | 256 | 384 | GLM experts 較少，但每層仍有 route entropy |
| Active experts | 8 | 8 | 類似 |
| Context | 200K | 256K | Kimi 原生 context 更長 |
| Attention | MLA + DSA | MLA | GLM 多了 DSA index path |
| 官方權重 | BF16 1.51TB / FP8 756GB | INT4 約 595GB | GLM FP8 對 256GB 仍更重 |
| 模態 | Text-only | Multimodal | GLM Phase 1 較乾淨 |

簡短判斷：

- **Kimi K2.6 的難點更偏權重總量與 multimodal。**
- **GLM 5.1 的難點更偏 active compute、DSA runtime、index cache、sparse gather locality。**

---

## 6. 256GB UMA 分帳套用

依據架構圖，256GB Mac 保守抓約 192GB 可調度工作集。

對 GLM 5.1，建議初期分帳如下：

| 區塊 | 256GB PoC 建議 |
| --- | --- |
| Resident trunk / router / dense layers / shared path | 35GB - 70GB，需實測 |
| Active expert cache | 60GB - 90GB |
| Expert staging / prefetch buffer | 24GB - 32GB |
| MLA KV hot / warm cache | 16GB - 40GB |
| DSA index / sparse metadata working set | 8GB - 32GB，取決於實作 |
| Runtime / scheduler / Metal temp | 16GB - 24GB |
| Safety headroom | 12GB - 24GB |

這份分帳的含義：

- GLM 5.1 不能照架構圖原本把 80GB - 95GB 給 KV。
- MLA KV 單 session 不大，但 DSA index / metadata 可能吃掉一大塊。
- active expert cache 必須比一般 1T MoE PoC 更慎重，因為 GLM active params 約 40B。

---

## 7. 可運行性分層

| 目標 | 可行性 | 說明 |
| --- | --- | --- |
| 解析 config / tokenizer / weight index | 高 | 可立即做。 |
| 使用 MLX 格式載入 metadata | 高 | `mlx-community/GLM-5.1` 已存在，但檔案巨大。 |
| 建立 BF16 / FP8 expert shard index | 高 | 需要 safetensors / MLX weight map。 |
| 單層 routed expert execution | 中 | 需要 FP8/BF16 或 Q4 dequant + Metal expert kernel。 |
| MLA paged KV baseline | 中 | 與 DeepSeek / Kimi 路線相近。 |
| DSA indexer 正確實作 | 中偏低 | 這是 GLM 5.1 的新硬點。 |
| Text-only、batch 1、短 context decode | 中偏低 | 需 MoE + MLA + DSA 同時跑通。 |
| 32K context | 低到中 | DSA index path 開始重要。 |
| 200K context | 低 | 需要 DSA index cache / sparse gather / KV policy 全部成熟。 |
| 多用戶 serving | 很低 | expert route entropy + DSA sparse access 會讓 prefetch 很難。 |
| 256GB Mac 完整實用運行 | 低 | 可做 PoC，不建議視為穩定目標。 |
| 512GB Mac / 多 GPU | 中 | 比較合理。 |

---

## 8. 必要模組清單

若目標是讓 GLM 5.1 真的在 oMLX 本地跑起來，必要模組如下。

### 8.1 `glm_moe_dsa` model adapter

需要支援：

- `GlmMoeDsaForCausalLM`
- GLM MoE block
- first 3 dense layers
- grouped expert selection
- sigmoid scoring
- shared expert
- MTP layer 排除或後補

### 8.2 MLA attention path

需要支援：

- `kv_lora_rank = 512`
- `q_lora_rank = 2048`
- `qk_nope_head_dim = 192`
- `qk_rope_head_dim = 64`
- `v_head_dim = 256`
- `rope_interleave`

第一版先做 correctness，不急著壓 KV。

### 8.3 DSA sparse attention / indexer path

這是 GLM 5.1 和 Kimi K2.6 最大不同。

需要支援：

- indexer projection
- index cache / metadata cache
- top-k KV position selection
- sparse gather
- interleaved RoPE for indexer
- DSA selection 與 paged KV 的映射

如果 DSA 沒有正確實作，就算 MoE 權重能載入，生成品質和長 context 都不可信。

### 8.4 Active expert cache

最低需求：

- `(layer_id, expert_id)` key
- FP8/BF16/Q4 active buffer
- LRU / LFU / route-aware policy
- cache hit / miss metrics
- batch 內 unique expert 統計

### 8.5 Expert staging / prefetch

最低需求：

- SSD -> staging
- staging -> active expert cache
- double buffer
- next-layer prefetch
- I/O queue depth control

### 8.6 Metal routed expert path

最低需求：

- FP8/BF16 baseline
- Q4 / INT4 path 可作第二階段
- fused dequant + GEMV for decode
- fused dequant + GEMM for chunked prefill
- expert gather / scatter
- SwiGLU

### 8.7 DSA-aware scheduler

一般 scheduler 看：

- KV blocks
- prefill/decode
- batch size

GLM 5.1 還要看：

- active expert cache pressure
- DSA top-k sparse pages
- DSA index cache size
- sparse gather locality
- SSD expert queue
- SSD KV / index queue

也就是說：

> **GLM 5.1 的 scheduler 必須同時是 expert-aware、KV-aware、DSA-index-aware。**

---

## 9. 建議 PoC 路線

### Phase 0：離線 config / memory simulator

不載模型，先建估算器。

輸出：

- resident / expert / staging budget
- MLA KV bytes / context
- DSA index cache bytes / context
- active expert cache hit rate 模擬
- DSA top-k sparse page 分布模擬

### Phase 1：GLM 5.1 loader skeleton

目標：

- 解析 config
- 解析 BF16 / FP8 / MLX weight layout
- 建立 expert shard index
- 分類 dense / router / shared / routed experts / DSA indexer weights

成功標準：

- 可以按 `(layer_id, expert_id)` 定位 expert tensor。
- 可以估算 resident trunk 大小。
- 可以單獨從 SSD 載入 expert shard。

### Phase 2：單層 MoE expert forward

目標：

- 先不跑完整模型
- 只跑單層 routed expert
- 測 FP8/BF16 baseline
- 後續再加 Q4

成功標準：

- 單層輸出與參考實作接近。
- cache hit / miss latency 可量測。

### Phase 3：MLA paged KV baseline

目標：

- 完成 MLA latent KV block lifecycle
- 測 4K / 16K / 32K / 128K / 200K memory accounting
- 不急著上 PolarQuant / QJL

### Phase 4：DSA indexer / sparse selector baseline

目標：

- indexer cache 正確
- top-k selection 正確
- sparse gather 正確
- DSA selected KV positions 能映射到 paged KV

這一步是 GLM 5.1 的必過關卡。

### Phase 5：End-to-end 短 context decode

目標：

- text-only
- batch 1
- 4K -> 16K context
- decode 128 tokens

必量測：

- TTFT
- tok/s
- expert miss / token
- DSA selected pages / token
- DSA index cache bytes
- SSD read GB/s
- UMA resident bytes
- Metal kernel time breakdown

### Phase 6：擴到 32K / 128K / 200K

只有 Phase 5 成立後，再逐步拉長 context。

200K 成功標準不應只看「不 OOM」，還要看：

- sparse attention 品質
- DSA index latency
- DSA gather locality
- KV / index cache hit rate
- tail latency

### Phase 7：再接 KV 壓縮與 SSD KV

此時才導入：

- PolarQuant / QJL
- TurboQuant-style baseline
- Quest metadata
- AdaptCache / EvicPress
- SSD cold KV
- IndexCache / HiSparse 類概念可作參考

原因：

> **GLM 5.1 已經自帶 DSA sparse attention。若沒有先理解 DSA index path，再疊 Quest / SSD KV，很容易做出兩套互相打架的 sparse retrieval。**

---

## 10. 目前不該列為成功標準

### 10.1 不把 200K context 當 Phase 1

GLM 5.1 官方 context 是 200K，但本地 oMLX Phase 1 應先證明：

```text
MoE expert paging + MLA + DSA correctness
```

不是一開始就跑滿 200K。

### 10.2 不把多用戶 serving 當 Phase 1

多用戶會同時放大：

- active expert cache miss
- DSA sparse gather fragmentation
- KV page pressure
- SSD queue contention

應等 batch 1 / low-concurrency 成立後再做。

### 10.3 不先做 FP8 以外的激進量化

官方 FP8 已經有 756GB 檔案，可作第一個權重側 baseline。  
Q4 / INT4 能大幅改善 256GB 可行性，但它應該是第二階段，因為 DSA / MLA / MoE correctness 更優先。

---

## 11. 對 oMLX 架構圖的具體調整建議

GLM 5.1 會讓 [oMLX-系統架構模組圖.md](./oMLX-系統架構模組圖.md) 需要新增或細化幾個模組。

### 11.1 KV Memory System 改成 `MLA + DSA-aware KV System`

新增：

- MLA latent KV block
- DSA index cache
- DSA sparse page map
- index metadata placement
- top-k selected page prefetch

### 11.2 Quest-style Page Selector 與 DSA indexer 要合併設計

GLM 5.1 已有 learned sparse selector。  
Quest-style metadata 不應重複掃一遍，而應作為：

- DSA selected page 的 storage-level accelerator
- SSD cold KV / index prefetch 的 page planner
- DSA top-k 結果到 paged KV block 的映射層

### 11.3 Scheduler 加入 `DSA I/O Admission`

Admission control 應新增：

- index cache budget
- selected sparse pages / token
- DSA gather locality
- KV hot page pressure
- expert miss + DSA miss 的聯合預估

### 11.4 Phase 1 成功標準改成 GLM-specific

對 GLM 5.1，Phase 1 成功標準應該是：

> **不完整載入 756GB FP8 權重，仍能透過 SSD cold expert + active expert cache + MLA + DSA 正確完成 text-only 短 context decode。**

---

## 12. 最終判斷

GLM 5.1 在 oMLX 本地運行的可能性可以這樣判斷：

- **模型規格上：高度符合 oMLX north-star。**
- **權重容量上：FP8 756GB，256GB UMA 不能完整常駐，必須 expert paging。**
- **KV 上：MLA 讓單 session 200K KV 約 17GB，不是最可怕的部分。**
- **DSA 上：index cache / sparse gather / page mapping 是 GLM 5.1 的真正新增硬點。**
- **工程上：GLM 5.1 比 Kimi K2.6 更需要 DSA-aware runtime。**

一句話：

> **GLM 5.1 在 256GB Mac 上可以作為 oMLX 研究 PoC，但完整可用的關鍵不是單純把 KV 壓小，而是把 756GB FP8 MoE 權重與 DSA sparse attention 都變成可快取、可預取、可調度的 working set。**

建議近期目標：

1. `glm_moe_dsa` adapter / loader skeleton
2. FP8 expert shard index
3. 單層 MoE expert Metal path
4. MLA paged KV baseline
5. DSA indexer / top-k sparse selector correctness
6. batch 1 短 context end-to-end decode

在這六步之前，不建議把 200K context、多用戶 serving、QJL/PolarQuant、SSD cold KV 放進成功標準。
