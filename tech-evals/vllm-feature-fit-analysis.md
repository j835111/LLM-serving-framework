# 技術盤點：vLLM / vllm-metal 還有哪些特性適合本專案

日期：2026-04-26

這份文件的目的，不是再重複一次「vLLM 很強」，而是把 **vLLM / vllm-metal 目前已公開的特性**，放回本專案自己的目標來看：

- Apple Silicon
- memory-first
- 超大 MoE
- 長 context
- 後續多用戶 serving

結論先講：

- **最值得現在就吸收的，不是產品層功能，而是記憶體生命週期與 scheduler 語意。**
- 對本專案最有價值的新增吸收點，是：
  - `V1 process architecture`
  - `paged KV + automatic prefix caching`
  - `chunked prefill + admission control`
  - `vllm-metal TurboQuant`
  - `selective offload + async prefetch`
- 真正該先觀察、不要搶主線的，是：
  - `EPLB`
  - `disaggregated prefill`
  - upstream 通用 `FP8 KV cache`
  - 各種偏 CUDA 的 executor 優化

---

## 1. 快速總表

| 特性 | 適合度 | 為什麼適合 | 現階段建議 |
| --- | --- | --- | --- |
| `V1 process architecture` + `AsyncLLMEngine` | 高 | 很適合當 control plane 骨架，且不會綁死底層 Metal kernel | 直接吸收 |
| `paged KV` + `automatic prefix caching` | 高 | 直接對應本專案的 block/page abstraction 與 session memory system | 直接吸收 |
| `chunked prefill` + `scheduler-reserve-full-isl` | 高 | 直接改善長 prompt / 多請求下的 decode 干擾與 KV thrashing | 直接吸收 |
| `vllm-metal TurboQuant` | 很高 | Apple-native KV 壓縮，直接命中容量與長 context | 優先 PoC，甚至可升主線候選 |
| `selective weight/KV offload` + `async prefetch` | 高 | 非常符合 SSD cold tier / UMA staging / expert paging 方向 | 先吸收控制面設計 |
| `Hybrid KV Cache Manager` | 中高 | 對 hybrid attention / sliding-window / KV sharing 很有參考價值 | 做 PoC，別直接照抄 |
| `enable-ep-weight-filter` | 中高 | 對大 MoE 的 expert shard 載入 / storage I/O 很有啟發 | 吸收概念 |
| `EPLB` | 中 | 能處理 expert skew，但會吃記憶體 | 先觀察 |
| `disaggregated prefill` | 中 | 有 SLA 價值，但不提高 throughput | 後期再做 |
| upstream `FP8 KV cache` | 中低 | 概念對，但目前官方文件重點仍偏 CUDA / ROCm / FlashAttention | 參考介面，不拿來當 Apple 主線 |
| `Sleep Mode` | 低 | 官方明講只支援 CUDA | 不納入 |

---

## 2. 最值得直接吸收的幾個點

### 2.1 `V1 process architecture` 很適合當 control plane 骨架

vLLM 的 stable 架構文件已經把責任切得很清楚：

- API server：收 HTTP、做 tokenization / input processing、stream 回 client
- Engine core：跑 scheduler、管理 KV cache、協調 worker
- GPU worker：載入權重、執行 forward、管理裝置記憶體
- DP coordinator：多副本時做 load balancing 與同步

這跟本專案 README 的方向幾乎一致：

- 上層借 `vLLM-style control plane`
- 下層保留 `Metal-first runtime`

所以真正值得借的不是「把 vLLM 全搬進來」，而是把責任邊界借來：

- API server 不碰熱路徑
- scheduler / admission / KV lifecycle 由 engine core 擁有
- worker 只專注在 execution path
- 讓後續自研 Metal kernel、SSD prefetch、expert staging 都能掛在 worker / runtime 層

結論：  
**這一層非常適合直接採用。**

來源：

- [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/)

---

### 2.2 `paged KV` + `automatic prefix caching` 是最該借的基線

這部分對本專案很重要，因為它不只是「prefix cache」而已，而是一整套 block/page 級管理思路。

vLLM 官方設計文件裡幾個很值得借的點：

- prefix cache 不是靠 request-level cache，而是 **hash 到 KV block**
- hash key 同時看：
  - block 內 token
  - block 前面的 prefix token
- KV block 會先建成固定大小的 **block pool**
- free list 用 linked-list queue 管理
- cache block evict 採 **LRU**
- cache hit block 會先被 touch，避免在同一輪分配裡被誤逐出

這非常符合本專案想做的：

- `page/block 級映射`
- `metadata-guided retrieval`
- `session memory system`

換句話說，這不是最終答案，但它是很好的 **第一版 memory object model**。

更重要的是，`vllm-metal` 現在已經把 paged KV 與 prefix cache 納進 Apple 路徑配置：

- `VLLM_METAL_USE_PAGED_ATTENTION=1`
- `VLLM_METAL_PREFIX_CACHE`
- `VLLM_METAL_PREFIX_CACHE_FRACTION`

這代表：

- 這條路線不再只是 CUDA 世界的概念
- Apple 路徑已經開始把它變成正式配置面的一部分

但有一個很重要的限制：

- `vllm-metal` 最新 supported models 頁面明寫，對 hybrid / Mamba 模型，upstream 目前仍維持 automatic prefix caching 預設關閉
- 目前像 `Qwen3.6`、`Qwen3-Next` 這些 hybrid 路線，不能假設 APC 已經成熟可直接吃

所以這層的正確結論不是「Prefix cache 已經 solved」，而是：

- **block pool / block hash / LRU / prefix reuse 這套骨架非常值得借**
- **但 hybrid model 上的 prefix reuse 還需要本專案自己往前推**

來源：

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vllm-metal Configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/)
- [vllm-metal Supported Models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/)

---

### 2.3 `chunked prefill` + `scheduler-reserve-full-isl` 很適合本專案

這一塊很容易被低估，但其實非常對本專案的多用戶 serving 主線。

vLLM 在 V1 的優化文件中明講：

- chunked prefill 在可用時預設開啟
- scheduler 會優先 decode request
- 先把 pending decode 排進 batch，再塞 prefill
- 若 prefill 放不下，才切 chunk

這個 policy 的價值非常直接：

- 改善 `ITL`
- 讓 compute-bound prefill 與 memory-bound decode 更好地共存

另外，CLI 還有一個本專案很該注意的選項：

- `--scheduler-reserve-full-isl`

官方描述很明確：  
它會在 admission 時檢查 **整個 input sequence length** 是否真的放得進 KV cache，而不是只看第一個 chunk，目的是避免 chunked prefill 造成 over-admission 與 KV thrashing。

這幾乎就是本專案未來一定會碰到的問題：

- 長 prompt 可以先切進來
- 但如果只看第一塊，就可能把 system 擠到後面才爆

所以這裡最值得借的不只是 chunking 本身，而是：

- decode-first scheduling
- long / short prefill lane 的概念
- full-input admission 檢查
- async scheduling 避免 executor gap

這些都非常適合 Apple memory-bound serving。

來源：

- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM `serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/)

---

### 2.4 `vllm-metal TurboQuant` 是目前最值得認真追的 Apple 特性

如果只問「還有哪個 vLLM 生態特性最直接適合本專案」，我的答案其實不是 upstream `FP8 KV cache`，而是 **`vllm-metal TurboQuant`**。

根據 `vllm-metal` 最新文件：

- 它是 Apple Silicon 上原生跑的 KV cache compression
- 路徑是 `MLX + Metal kernel`
- 核心作法是：
  - Walsh-Hadamard rotation
  - per-block quantization
- 文件給的數字是：
  - 約 `2.5x - 5x` KV 壓縮
  - `q8_0/q3_0` 預設組合大約可多出 `2.5x` context 預算

這跟本專案的方向高度重合，因為它不是只提升 benchmark，而是直接換取：

- 更多 KV token
- 更長 context
- 更大 multi-session 空間

更關鍵的是，它對 Apple 路徑的限制條件寫得很清楚：

- 需要 paged attention
- 支援 hybrid `SDPA + GDN linear`
- `MLA` 不支援
- head dim 目前要落在 `64 / 128 / 256`

這種限制資訊對本專案很有價值，因為它能幫我們更早知道：

- 哪些 model family 能先吃
- 哪些 attention family 要分流

#### 對本專案更重要的額外訊號

在最新的 developer-preview `TurboQuant` API 文件裡，vLLM 把方法描述得更直接：

- K：Hadamard rotation + per-coordinate Lloyd-Max scalar quantization
- V：uniform quantization

而且它特別寫到：

- **QJL 沒有被納入**
- 原因是多個獨立實作觀察到，它會在 softmax 前放大量化誤差，傷 attention 品質

這點很值得本專案注意，因為你們目前文件裡把 `PolarQuant + QJL` 放得相對前面。  
這不代表 QJL 在你們的目標模型上一定不行，但它代表：

- **不能再把 QJL 視為預設正確方向**
- 比較合理的做法，是把它降成「比較組」
- 讓 `Hadamard + Lloyd-Max / TurboQuant-style` 成為更強的 baseline

我對這點的建議很明確：

1. 先用 `TurboQuant-style KV compression` 做 Apple baseline
2. `QJL` 只保留成對照實驗
3. 只有在你們自己的 long-context / MoE / perplexity 測試真的贏，才把 QJL 拉回主線

來源：

- [vllm-metal TurboQuant](https://docs.vllm.ai/projects/vllm-metal/en/latest/turboquant/)
- [vLLM TurboQuant API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/)

---

### 2.5 `selective offload` + `async prefetch` 的控制面很值得借

這塊我認為是最接近你們 `MoE weight memory system` 的 vLLM 現成資產。

`vllm serve` 的最新 CLI 已經把 offload 做成很明確的控制面：

- `--offload-backend`
  - `auto`
  - `prefetch`
  - `uva`
- `--cpu-offload-gb`
- `--cpu-offload-params`
- `--offload-group-size`
- `--offload-num-in-group`
- `--offload-prefetch-step`
- `--offload-params`

它背後隱含的設計很適合本專案：

- offload 不該是全有或全無
- 要能 **選參數子集合**
- 要能以 **group / layer** 為單位排 prefetch
- 要把「容量」與「latency hiding」拆開控制

對本專案來說，這可直接映射到：

- shared trunk 常駐
- expert shard hot/cold 分級
- SSD -> UMA -> GPU staging
- route-aware prefetch

同樣地，KV 這邊也有對應的控制面：

- `--kv-offloading-size`
- `--kv-offloading-backend`

再加上 disaggregated prefill 頁面裡的 `OffloadingConnector`，可以看到 vLLM 已經把：

- KV transfer
- KV offload
- connector abstraction

開始做成獨立模組，而不只是 scheduler 內部硬寫。

當然，現成實作目前主要還是：

- CPU offload
- UVA
- CUDA 路徑

所以我不建議直接搬 code path。  
真正該借的是：

- **offload 的控制面**
- **prefetch 的配置面**
- **targeted offload 的 metadata 模型**

這對你們做 SSD cold weight tier 會很有幫助。

來源：

- [vLLM `serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

---

## 3. 值得做 PoC，但不要急著當主線的點

### 3.1 `Hybrid KV Cache Manager`

這份設計文件比一般人想像的更有參考價值，因為它不是只處理「full attention」。

它處理的是：

- full attention
- sliding window attention
- Mamba 類 state
- KV sharing layer

最值得借的點有三個：

1. **按 attention type 分組配置 KV block**
2. 對 hybrid model 的 prefix hit 要做 **group intersection**
3. `KV sharing` layer 不一定需要自己獨立配置 KV

這對本專案的價值在於：

- 未來你們不只會碰傳統 GQA
- 很可能還會碰 hybrid attention family
- 而 Apple 記憶體預算又逼你不能「所有層都一視同仁地配一樣多」

但它也有清楚的限制：

- 目前 prefix-cache coordinator 只處理 **兩類 KV group**
- 必須包含一組 full attention + 一組其他 efficient attention
- 更多類型還沒實作完成

所以這份文件最適合的角色是：

- **第一版 hybrid allocator 的參考**
- 不是最終形態

來源：

- [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/v0.18.2/design/hybrid_kv_cache_manager/)

---

### 3.2 `enable-ep-weight-filter` 值得借概念，但 `EPLB` 先不要

這塊要拆開看。

#### 值得借的是 `enable-ep-weight-filter`

最新 CLI 文件明寫：

- 開啟 expert parallel 時，可以跳過 non-local expert weights
- 每個 rank 只讀自己的 expert shard
- 這能大幅降低 DeepSeek / Mixtral / Kimi-K2.5 這類模型的 storage I/O

這個點即使你現在不做真正的 EP，也非常值得借概念，因為它本質上在說：

- **expert weights 不該一股腦全讀進來**
- 應該按 ownership / locality / route 去決定載入集合

這跟本專案的：

- active expert cache
- SSD cold expert tier
- route-aware prefetch

高度一致。

#### 不該急著借的是 `EPLB`

`EPLB` 能處理 expert skew 沒錯，但官方文件也直接提醒：

- 它靠 redundant experts 來換 balancedness
- 這會吃額外記憶體
- 文件甚至舉 `DeepSeekV3` 為例，說一個 redundant expert per EP rank 大約就要多 `2.4 GB`

對本專案現階段來說，這個 tradeoff 不漂亮：

- 你們現在最缺的是容量
- 不是先追跨 rank 的 expert load 均衡

所以我建議：

- `ep-weight-filter`：吸收概念
- `EPLB`：先觀察，不列入 Phase 1 / 2

來源：

- [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM `serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/)

---

### 3.3 `disaggregated prefill` 有價值，但比較像後期能力

這項特性官方定位很清楚：

- 把 prefill 與 decode 放到不同 vLLM instance
- 分開調 `TTFT` 與 `ITL`
- 更穩地控制 tail latency

但官方也明寫：

- **它不會提升 throughput**

這對本專案的意思很簡單：

- 如果未來你們要做嚴格 SLA、多 instance、多節點，這很有價值
- 但現在主線還在證明 Apple Silicon 上的大模型 + 長 context + memory hierarchy 是否成立
- 那它就不是第一優先

不過，它裡面的 connector 思維很值得先記住：

- `KV transfer`
- `OffloadingConnector`
- `connector extra config`

因為這跟你們未來想做的：

- SSD / host / worker 間 KV movement
- prefill / decode 分離
- metadata-guided 搬運

其實是同一類問題。

來源：

- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

---

## 4. 目前不建議優先投入的點

### 4.1 upstream 通用 `FP8 KV cache`

這功能本身當然有價值，官方也明講它能：

- 降低 KV 記憶體
- 放更多 token
- 拉長 context

但以本專案來看，現階段它不如 `vllm-metal TurboQuant` 值得優先，原因有三個：

1. 官方文件的主路徑還是偏 CUDA / ROCm
2. 更進一步的 per-head quantization 依賴 `Flash Attention`
3. 你們的專案是 Apple-first，不是 FA3-first

所以這項的正確使用方式是：

- 借它的 API / config 形狀
- 不把它當 Apple 主線實作

來源：

- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

---

### 4.2 `Sleep Mode`

官方文件直接寫：

- 只支援 `CUDA`

所以這項對本專案現階段幾乎沒有價值。

來源：

- [vLLM Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/)

---

### 4.3 產品層功能不該搶主線

像這些能力：

- structured outputs
- tool calling
- reasoning outputs
- 一般型 API 相容功能

都很有產品價值，但不是目前本專案最缺的差異化核心。

本專案現在真正要贏的是：

- 更大的模型
- 更長的 context
- 更好的 memory hierarchy

所以這些功能可以沿用 vLLM 骨架，但不應該搶資源。

`MTP` 也一樣。  
它已經有獨立評估文件，而且它更像 decode accelerator，不是容量主線。

---

## 5. 最合理的吸收順序

### Phase 1：現在就該吸收

- `V1 process architecture`
- `AsyncLLMEngine` / API server / engine core / worker 責任分離
- `paged KV` block pool / free queue / LRU / prefix block hash
- `chunked prefill` 的 decode-first scheduling
- `scheduler-reserve-full-isl` 的 admission 概念
- `vllm-metal TurboQuant` 當 Apple KV 壓縮 baseline

### Phase 2：做出專案差異化

- `Hybrid KV Cache Manager` 的分組配置思路
- selective weight offload / KV offload metadata
- `offload_group_size` / `offload_prefetch_step` 這類 async prefetch 控制面
- `enable-ep-weight-filter` 啟發的 route-aware expert loading
- hybrid model 上可工作的 prefix caching

### Phase 3：當模型 / context / 單機路徑已成立後

- `disaggregated prefill`
- DP / external LB / hybrid LB
- 真正的 multi-instance serving
- 若容量足夠，再考慮 `EPLB`

---

## 6. 最終判斷

如果把這次盤點濃縮成一句話：

> **vLLM 現在最值得本專案吸收的，不是更多產品 API，也不是更多 CUDA-only 加速，而是它對 KV lifecycle、scheduler admission、prefix reuse、hybrid cache grouping、以及 selective offloading 的系統化做法。**

再更務實一點講：

1. **最該借的是控制面與記憶體物件模型**
2. **最該追的 Apple 特性是 `vllm-metal TurboQuant`**
3. **最該保留自研空間的是 SSD tiering、MoE expert lifecycle、以及 hybrid / long-context 下的 KV retrieval policy**
4. **`QJL` 不該再被視為預設主線，而應該退回比較組**

這樣最符合本專案目前的 north star，也最不會被 upstream 現成 feature 帶離 memory-first 主線。
