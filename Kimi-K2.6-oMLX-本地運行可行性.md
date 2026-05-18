# Kimi K2.6 在 oMLX 本地運行可行性備忘錄

日期：2026-04-27

這份文件不是技術評分表，而是針對 **Kimi K2.6 這個具體模型**，評估它是否可能依據 [oMLX-系統架構模組圖.md](./oMLX-系統架構模組圖.md) 在本地 Apple Silicon 上運行。

結論先講：

> **Kimi K2.6 很適合作為 oMLX 的 north-star 模型，但不能當成一般 MLX 本地模型直接載入。它在 256GB Apple Silicon 上的可行性，取決於 MoE 權重分層、SSD cold expert、active expert cache、route-aware prefetch、MLA-native KV path 是否先成立。**

更務實地說：

- **256GB Mac：可作為研究 PoC 目標。**
- **192GB Mac：可能可做 loader、routing、expert cache simulator、短鏈路 kernel PoC；完整 K2.6 低併發運行風險很高。**
- **128GB Mac：不適合作為完整 K2.6 目標，只適合做分模組驗證。**
- **第一階段不應追大量用戶，也不應追 1M context；應先追 text-only、低併發、短到中 context 的穩定 decode。**

---

## 1. Kimi K2.6 公開規格

| 項目 | 數值 |
| --- | --- |
| 架構 | MoE |
| 總參數 | 1T |
| Active parameters | 32B |
| Layers | 61 |
| Dense layers | 1 |
| Attention hidden dimension | 7168 |
| Experts | 384 |
| Selected experts per token | 8 |
| Shared experts | 1 |
| Context length | 256K / 262,144 tokens |
| Attention | MLA |
| Vision encoder | MoonViT, 400M |
| 權重量化 | native INT4 / compressed-tensors |
| HF 權重檔案總量 | 約 595GB |

來源：

- [Kimi K2.6 model card](https://huggingface.co/moonshotai/Kimi-K2.6)
- [Kimi K2.6 config.json](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json)
- [Kimi K2.6 files](https://huggingface.co/moonshotai/Kimi-K2.6/tree/main)
- [Kimi K2.6 deployment guide](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md)

重要現實：

K2.6 官方建議的 inference engines 是 `vLLM`、`SGLang`、`KTransformers`，部署範例以 H200 TP8 或 CPU+GPU heterogeneous inference 為主。這表示 oMLX 若要在單機 Mac 上跑 K2.6，本質上是在做一條新的 memory hierarchy runtime，而不是套現成官方部署路線。

---

## 2. 對 oMLX 架構的意義

K2.6 同時壓到 oMLX 的兩條主線：

1. **MoE Weight Memory System**
   - 595GB 權重檔無法完整常駐 256GB UMA。
   - 必須有 SSD cold expert store。
   - 必須有 active expert cache。
   - 必須根據 router 結果做 route-aware prefetch。

2. **KV Memory System**
   - K2.6 使用 MLA，單 session 256K KV 壓力比標準 full KV 小很多。
   - KV 壓縮仍有價值，但對 K2.6 的第一價值更偏多 session，而不是讓單一 256K session 放得下。

因此，對 K2.6 來說：

> **第一瓶頸是 MoE 權重與 expert I/O，不是單 session KV。**

這點會改變 oMLX 的實作優先順序：應先把權重側生命週期跑通，再把 KV 壓縮和 SSD KV 分層推進來。

---

## 3. 權重工作集估算

K2.6 HF 檔案約 595GB，所以完整 resident 不成立。

必須採用：

```text
resident trunk/router/shared path
+ active expert cache
+ expert staging buffer
+ SSD cold expert store
```

粗估 routed expert 大小：

```text
single expert params ~= 3 * hidden_size * moe_intermediate_size
                     ~= 3 * 7168 * 2048
                     ~= 42M params
```

以 INT4 pack 加 scale 粗抓約 `0.59 byte / param`：

```text
single expert ~= 24.8 MB
8 selected experts / layer ~= 198 MB
60 MoE layers selected route ~= 11.6 GB
```

這代表：

- 單 token 的 route 工作集不誇張。
- 真正可怕的是 batch 內 route 分散，導致每層 unique experts 快速膨脹。
- 如果每層命中接近 384 experts，active expert cache 和 SSD I/O 會直接失控。

所以 K2.6 的核心問題不是「每 token active 32B 算不算得動」而是：

> **在連續 decode 中，能不能讓 routed experts 的 cache hit rate 高到足以遮蔽 SSD I/O。**

---

## 4. KV 工作集估算

K2.6 config 顯示：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
```

所以 MLA KV 可用下列近似估算：

```text
per-token per-layer MLA KV ~= 512 + 64 = 576 BF16 values
```

單 session 256K：

```text
576 values * 2 bytes * 61 layers * 262144 tokens
~= 17.2 GB
```

不同 KV 精度下：

| KV 形態 | 單 session 256K 粗估 |
| --- | --- |
| BF16 MLA KV | 約 17.2GB |
| 8-bit KV | 約 8.6GB |
| 4-bit KV | 約 4.3GB |
| 2-bit KV | 約 2.1GB |

這和標準 full KV 差很多。若用 full attention KV 估算，256K 會落到數百 GB 級；但 K2.6 的 MLA 已先把 KV footprint 大幅壓低。

因此：

- 單 session 256K：KV 不是最大阻塞。
- 多 session 256K：KV 壓縮開始重要。
- 1M context：KV 壓縮、Quest、SnapKV、MiniCache / xKV、SSD KV 才會變成必要條件。

---

## 5. 256GB UMA 分帳套用

依據架構圖，目前 256GB Mac 保守抓約 192GB 可調度工作集：

| 區塊 | 原建議預算 |
| --- | --- |
| Active Model Working Set | 72GB - 80GB |
| Expert Staging / Prefetch Buffer | 16GB - 24GB |
| KV Working Set | 80GB - 95GB |
| Runtime Headroom | 10GB - 16GB |

對 K2.6 建議調整成：

| 區塊 | K2.6 初期建議 |
| --- | --- |
| Resident trunk / router / shared path | 30GB - 50GB，需實測 |
| Active Expert Cache | 50GB - 80GB |
| Expert Staging / Prefetch Buffer | 24GB - 32GB |
| MLA KV Working Set | 20GB - 60GB，視 session 數決定 |
| Runtime Headroom | 16GB+ |

原因：

K2.6 的 MLA 讓單 session KV 變小，但 595GB expert weights 讓權重側變成最大壓力。因此初期應把更多 UMA 預算給 active expert cache，而不是一開始就保留 95GB 給 KV。

---

## 6. 可運行性分層

| 目標 | 可行性 | 說明 |
| --- | --- | --- |
| 只解析 config / tokenizer / weight index | 高 | 可立即做。 |
| 建立 expert shard metadata | 高 | 需要 safetensors / compressed-tensors index。 |
| 單層 routed expert 從 SSD 載入並執行 | 中 | 需要 INT4 unpack/dequant + Metal GEMV/GEMM。 |
| Text-only、batch 1、短 context decode | 中 | 第一個真正 end-to-end 目標。 |
| Text-only、單 session 128K-256K | 中偏低 | KV 放得下，但 prefill 與 expert streaming 壓力大。 |
| 256K 多 session 低併發 | 低到中 | 需要 paged MLA KV、prefix sharing、expert-aware scheduler。 |
| 10+ unique users 長 context | 低 | route entropy 與 SSD expert I/O 會是主要風險。 |
| 48-way concurrency 類官方 KTransformers 範例 | 很低 | 官方範例是多 GPU + CPU heterogeneous，不能對標單機 Mac。 |
| 1M context | 很低 | 超出官方 256K context，屬於 oMLX 自研長 context extension。 |
| Vision / video | 很低 | Phase 1 不應納入。 |

---

## 7. 必要模組清單

若目標是「真的讓 K2.6 在本地跑起來」，必要模組不是 KV 壓縮起手，而是以下順序。

### 7.1 Kimi / DeepSeekV3-style model adapter

需要支援：

- `KimiK25ForConditionalGeneration`
- DeepSeekV3-style MoE block
- MLA attention
- YaRN RoPE
- SwiGLU expert
- native INT4 compressed-tensors

先做 text-only，不做 vision。

### 7.2 Weight index / expert metadata

需要建立：

```text
layer_id
expert_id
tensor_name
file_path
byte_offset / span
quantization metadata
resident_or_cold
```

這是 SSD cold expert store 的基礎。

### 7.3 Active expert cache

最低需求：

- 以 `(layer_id, expert_id)` 為 key
- active resident buffer
- LRU / LFU / route-aware policy
- async load state
- cache hit / miss metrics

### 7.4 Expert staging / prefetch

最低需求：

- SSD -> staging buffer
- staging buffer -> active expert cache
- double buffer
- queue depth control
- next-layer prefetch

### 7.5 Metal routed expert path

最低需求：

- INT4 unpack / dequant
- fused dequant + GEMV for decode
- fused dequant + GEMM for prefill chunk
- expert gather / scatter
- SwiGLU

### 7.6 MLA paged KV baseline

第一版先做 BF16 / native precision：

- MLA latent KV block
- page/block allocator
- prefix sharing
- per-session KV accounting

PolarQuant / QJL 之後再加，避免一開始同時 debug model adapter、expert paging、KV quantization。

### 7.7 Expert-aware scheduler

K2.6 的 scheduler 不能只看 token/KV budget，還要看：

- batch 內每層 unique experts
- active expert cache hit rate
- SSD read queue
- staging buffer occupancy
- prefetch miss rate

這會是 oMLX 和一般只看 token / KV block 的 serving scheduler 最大的差異之一。

---

## 8. 建議 PoC 路線

### Phase 0：離線 memory / route simulator

不載模型，先用 config 做估算。

輸出：

- KV bytes / context length
- expert bytes / layer
- active cache capacity 模擬
- route entropy 對 SSD read 的影響
- 不同 batch size 下 unique experts / layer

### Phase 1：K2.6 loader skeleton

目標：

- 解析 model config
- 解析 safetensors / compressed-tensors 檔案列表
- 建立 expert shard index
- 分類 resident weights 與 cold experts

成功標準：

- 可以按 `(layer_id, expert_id)` 定位 expert tensor。
- 可以估算 resident trunk 大小。
- 可以單獨從 SSD 讀取一個 expert shard。

### Phase 2：單層 expert execution

目標：

- 從 SSD 載入 selected experts
- INT4 unpack/dequant
- 跑單層 routed expert forward

成功標準：

- 單層輸出與參考實作可對齊。
- 可量測 cache hit/miss latency。

### Phase 3：MLA paged KV baseline

目標：

- 不做 KV 壓縮
- 先完成 MLA KV block lifecycle
- 測 4K / 16K / 32K / 128K / 256K 的 KV memory accounting

成功標準：

- KV allocation/free/reuse 正確。
- prefix sharing 可工作。
- 單 session 256K KV 不爆預算。

### Phase 4：End-to-end text-only decode

目標：

- batch 1
- 4K -> 32K context
- decode 128 tokens
- 不做 vision
- 不做大量用戶

必量測：

- TTFT
- tok/s
- SSD read GB/s
- expert miss / token
- active expert cache hit rate
- UMA resident bytes
- Metal kernel time breakdown

### Phase 5：擴到 128K / 256K 與低併發

目標：

- 128K / 256K single session
- batch 2-4
- 加入 admission control
- 測 route overlap 對效能的影響

### Phase 6：再接 KV 壓縮與 SSD KV

只有在權重側成立後，再接：

- TurboQuant-style baseline
- PolarQuant / QJL 對照
- Quest metadata
- AdaptCache / EvicPress placement
- SSD cold KV

理由：

> **如果 expert paging 不成立，KV 壓縮再漂亮也無法讓 K2.6 跑完整模型。**

---

## 9. 目前不該列為成功標準的東西

### 9.1 不把 1M context 當 Phase 1

K2.6 官方 context 是 256K。1M 是 oMLX 自己的 extension，需要額外驗證：

- RoPE / YaRN extrapolation 品質
- long-context perplexity
- retrieval accuracy
- KV 壓縮損失
- Quest / SnapKV / MiniCache / xKV 是否能疊加

### 9.2 不把大量用戶當 Phase 1

大量用戶需要：

- continuous batching
- paged MLA KV
- prefix cache
- admission control
- expert-aware scheduling
- QoS

這些應在單機低併發 decode 成立後再做。

### 9.3 不先做 vision

K2.6 是 multimodal，但 MoonViT / image-video preprocessing 會引入另一條複雜路徑。Phase 1 應先 text-only。

---

## 10. 對架構圖的具體調整建議

K2.6 會讓 [oMLX-系統架構模組圖.md](./oMLX-系統架構模組圖.md) 裡幾個模組更明確。

### 10.1 KV Memory System 加入 MLA-native path

新增或細化：

- MLA latent KV block
- rope key component cache
- MLA-specific KV metadata
- MLA compression policy

### 10.2 Scheduler 加入 Expert I/O Admission

Admission control 不只看：

- tokens
- KV blocks
- prefill/decode

還要看：

- unique experts / layer
- active expert cache pressure
- SSD queue depth
- staging buffer pressure

### 10.3 Phase 1 改成先證明 MoE weight lifecycle

對 K2.6，Phase 1 成功標準應該是：

> **不完整載入 595GB 權重，仍能透過 SSD cold expert + active expert cache 完成 text-only 短 context decode。**

長 context 和多用戶是下一階段。

---

## 11. 最終判斷

Kimi K2.6 在 oMLX 本地運行的可能性可以這樣判斷：

- **模型規格上：非常適合 oMLX。**
- **記憶體容量上：完整常駐不可能，分層才可能。**
- **KV 上：MLA 讓單 session 256K 沒有想像中可怕。**
- **權重上：595GB native INT4 MoE 是真正硬點。**
- **工程上：要先做 expert paging / route-aware prefetch，再談 KV 壓縮。**

一句話：

> **K2.6 在 256GB Mac 上不是完全不可能，但可行性的核心不在「把 KV 再壓小」，而在「把 595GB MoE 權重變成可快取、可預取、可調度的 SSD expert working set」。**
