# 技術評估：oMLX 語言改寫與極限效能路線

日期：2026-05-12

## 1. 快速結論

不建議把 `oMLX` 全量改寫成另一種語言。

比較合理、也最接近極限效能的路線是：

> **保留 Python / FastAPI / oMLX 現有 control plane，將 decode / KV / expert / kernel 熱路徑逐段 native 化：先用 MLX custom Metal kernel，之後把穩定熱點沉到 C++ extension + Metal。**

原因很直接：

- `oMLX` 的現有價值主要在 serving substrate：
  - OpenAI / Anthropic API
  - EnginePool / 多模型生命週期
  - continuous batching
  - block-based paged KV
  - hot RAM + cold SSD tier
  - prefix sharing / Copy-on-Write
  - process-level memory enforcement
- 這些多半是控制面與記憶體生命週期問題，不是每 token 的核心算子熱點。
- 真正影響極限效能的路徑通常是：
  - attention / decode kernel
  - KV encode / decode / compression
  - fused dequant + GEMV / GEMM
  - expert gather / scatter
  - route-aware prefetch
  - SSD -> UMA staging
  - MLX lazy evaluation 的 `eval` / sync 邊界

換句話說，`oMLX` 本身不該被視為「要換掉的 Python server」，而應被視為 **control plane substrate**。真正要改寫的是它下面那些會在每個 decode step、每個 batch、每個 expert 或每個 KV block 上重複發生的熱路徑。

## 2. 基本資訊

- 技術名稱：oMLX 語言改寫 / native hot path acceleration
- 類型：system design / runtime optimization / kernel optimization
- 來源：
  - 本 repo：`README.md`、`oMLX-系統架構模組圖.md`
  - oMLX public repo / docs
  - MLX 官方文件
- 日期：2026-05-12

## 3. 它主要解哪個問題

- 主要解決維度：讀取方式 / 存放位置 / kernel execution path / control-plane overhead
- 目標瓶頸：頻寬 / 併發 / scheduler overhead / kernel launch overhead / metadata 管理
- 主要作用階段：decode 為主，prefill 次之
- 適用場景：large MoE / long context / 多用戶 serving / SSD tiering

需要先切清楚：

- **語言改寫不能讓 LPDDR 頻寬變大。**
- **語言改寫也不能自動讓 MLX kernel 更快。**
- 它真正能改善的是：
  - 減少 Python 在高頻 loop 裡的 dispatch / object / GIL 開銷
  - 讓 page table / block map / expert index 更接近 cache-friendly native data structure
  - 把多個 MLX op / memory movement fuse 成一個 Metal kernel
  - 把 async I/O、prefetch、staging 和 kernel dispatch 排得更穩

因此評估重點不是「哪個語言最快」，而是：

> **哪一層換語言之後，可以少搬資料、少同步、少 launch、少走 Python hot loop。**

## 4. 現有 oMLX 的語言與架構邊界

公開 oMLX 目前仍是以 Python server 為主，README 中的架構包含：

- `FastAPI Server`
- `EnginePool`
- `BatchedEngine`
- `Scheduler`
- `mlx-lm BatchGenerator`
- `PagedCacheManager`
- `PagedSSDCacheManager`
- macOS menu bar app 使用 PyObjC

這代表現有 oMLX 的「應保留價值」在：

- API compatibility
- model lifecycle
- dashboard / settings / app packaging
- request scheduling
- cache object model
- MLX-format 模型整合
- agentic coding workload 的 SSD KV reuse

而不是它已經把所有 compute hot path 都做到極限。

## 5. MLX 生態給的語言選項

MLX 官方文件顯示：

- Python API 接近 NumPy，適合快速組裝模型與實驗。
- MLX 有 C++ API，並且 closely follows Python API。
- MLX 採 lazy computation，array 只有在需要時 materialize。
- MLX 的 unified memory 讓 CPU / GPU 操作同一個 shared memory pool，不需要傳統 device copy。
- MLX 支援 custom Metal kernels，且可從 Python 和 C++ API 寫。
- MLX C 可作為其他語言 binding 的橋；MLX Swift 也是透過 MLX C 形成 Swift API。
- MLX export/import function 可讓一個 front-end 寫出的 computation 在另一個 front-end 執行，例如 Python -> C++。

這些事實導出一個重要判斷：

> **MLX 的最佳改寫路線不是離開 MLX，而是在 MLX 內部把穩定熱點從 Python orchestration 移到 C++ / Metal。**

## 6. 語言候選評估

### 6.1 Python

適合保留的層：

- FastAPI / OpenAI / Anthropic API
- admin dashboard backend
- model discovery / config / CLI
- 實驗性 scheduler policy
- 初版 block policy / eviction policy
- MLX / mlx-lm model adapter

不適合繼續留在 Python 的層：

- 每 token decode inner loop
- per-block KV encode / decode
- bitpack / unpack
- Hadamard / FWHT
- QJL / TurboQuant-style transform
- expert dispatch inner loop
- 大量 page table 掃描與 compact
- 高頻 SSD prefetch decision hot loop

判斷：

- Python 很適合維持產品速度與研究迭代。
- 但只要某段邏輯進入每 token / 每 layer / 每 block 熱路徑，就應該抽出 native backend。

### 6.2 C++ + Metal

最適合的層：

- MLX extension
- custom op
- fused attention / dequant / GEMV
- KV compression kernel
- expert gather / scatter
- page/block table compact
- runtime profiler hooks
- stable decode backend

優點：

- 和 MLX core / C++ API 最貼近。
- custom Metal kernel 可在 Python 原型成熟後沉下來。
- 適合把多個 MLX op fuse 成少數 kernel。
- 能避開 Python hot loop 和頻繁 object dispatch。

缺點：

- 開發與除錯成本最高。
- ABI / packaging / macOS wheel / code signing 會變成實際維護問題。
- Metal kernel 需要針對 Apple GPU threadgroup、memory layout、SIMD group 做真實 profiling。

判斷：

> **如果目標是極限效能，C++ + Metal 是最該投資的 execution path，不是 Swift 或 Rust。**

### 6.3 Swift

適合的層：

- macOS app
- native service wrapper
- long-running daemon
- dashboard / settings / app lifecycle
- structured concurrency 的 request orchestration
- 需要深度 macOS 整合時的外層 runtime

可能適合的中層：

- EnginePool / scheduler 的 production rewrite
- memory policy engine
- model manager

不適合作為第一個熱路徑改寫目標：

- KV compression kernel
- attention kernel
- expert GEMV / GEMM
- bit-level packing

原因：

- Swift 可以降低 Python server / app 的工程摩擦，但真正的 compute 仍會落到 MLX / Metal。
- 若只是把 Python API server 改成 Swift，通常不會改善每 token TPS。
- Swift 的價值更像「產品化與穩定 daemon」，不是第一優先的 kernel acceleration。

判斷：

> **Swift 是 oMLX 長期產品化 control plane 的好候選，但不是極限效能的第一刀。**

### 6.4 Rust

適合的層：

- SSD cache metadata
- page table / block index
- async I/O pipeline
- prefetch queue
- admission control
- memory accounting
- CLI / daemon

優點：

- memory safety 與 concurrency 很適合 cache / scheduler / I/O。
- 可用 FFI 接 MLX C。
- 很適合寫長時間運行、低 bug density 的服務中層。

缺點：

- MLX 沒有官方 Rust first-class API。
- 需要經過 C API / FFI，熱路徑整合摩擦比 C++ 高。
- 如果最後仍要呼叫 Metal / MLX C++，Rust 可能變成中間層負擔。

判斷：

> **Rust 適合 cache metadata / async I/O / scheduler 的高可靠實作，但不適合作為 MLX compute path 的第一語言。**

### 6.5 Go / Java / Zig / Mojo

判斷：

- Go：server concurrency 很好，但 MLX / Metal 整合弱，不適合本專案主線。
- Java / Kotlin：不符合 MLX / Apple native kernel 路線。
- Zig：可以做系統層，但 MLX 生態缺口太大。
- Mojo：長期可觀察，但現在不適合承擔 Apple Silicon / MLX serving 主線。

## 7. 建議改寫邊界

### 保留 Python / oMLX 的部分

| 模組 | 建議 |
| --- | --- |
| API server | 保留 |
| OpenAI / Anthropic compatibility | 保留 |
| dashboard / settings | 保留 |
| EnginePool | 短中期保留 |
| model lifecycle / TTL / pinning | 保留 |
| request object / streaming protocol | 保留 |
| 初版 scheduler policy | 保留 |
| 實驗性 cache policy | 保留 |

### 優先 native 化的部分

| 熱路徑 | 建議語言 | 原因 |
| --- | --- | --- |
| KV block pack / unpack | C++ + Metal | 每 block 高頻，直接影響頻寬與 latency |
| KV compression encode / decode | C++ + Metal | 需要 fuse transform + quant / dequant |
| Hadamard / FWHT | Metal | 不能靠 Python / MLX op chain 堆出極限效能 |
| attention read compressed KV | Metal | 應避免先解壓成大張量再 attention |
| fused dequant + GEMV / GEMM | Metal | MoE 權重側最重要 hot path |
| expert gather / scatter | C++ + Metal | sparse MoE working set 會大量打到 |
| route-aware expert prefetch | C++ 或 Rust | 需要低 overhead queue / metadata scan |
| page table compact / lookup | C++ 或 Rust | 高併發下 Python object 會放大成本 |
| SSD staging / double buffer | C++ 或 Rust | 需要穩定 async I/O 與 memory accounting |

### 可晚點 native 化的部分

| 模組 | 判斷 |
| --- | --- |
| API server | 高併發 HTTP 不是第一瓶頸 |
| dashboard | 不影響 TPS |
| model downloader | 不影響推理熱路徑 |
| tool calling parser | 只有特定 workload 才可能成為 CPU 熱點 |
| config / settings | 不值得先改 |

## 8. 分階段路線

### Phase 0：先 profile，不先改語言

目標：

- 確定 Python overhead 是否真的進入熱路徑。
- 建立每 token / 每 layer / 每 block 的時間帳。
- 把 MLX lazy eval / synchronize 邊界明確化。

需要量：

- `TTFT`
- `ITL`
- aggregate token TPS
- per request p50 / p95 / p99 latency
- prefill / decode 分段時間
- scheduler wait time
- Python CPU time
- GIL contention
- MLX active / peak / cache memory
- SSD read/write bytes
- KV hot / cold hit rate
- page fault / cache restore latency

進入下一階段門檻：

- Python / scheduler / metadata time 在 decode loop 中超過 5%。
- 或 KV compression / page lookup / expert routing 需要每 token 高頻呼叫 Python。
- 或 MLX op chain 產生大量中間張量與 sync。

### Phase 1：Python 外殼 + MLX custom Metal kernel

目標：

- 最快驗證 kernel ROI。
- 不重寫 oMLX。
- 在 Python 內先使用 `mx.fast.metal_kernel()` 實作 hot kernels。

優先 PoC：

1. FWHT / Hadamard Metal kernel
2. KV block quantize / dequantize kernel
3. compressed KV read + attention microkernel
4. bitpack / unpack
5. expert gather / scatter microkernel

成功標準：

- 長 context decode TPS 明顯提升。
- p95 ITL 不因壓縮而惡化。
- KV 容量下降能換成更多 concurrent sessions。
- 短 context 可動態關閉壓縮，避免 overhead。

### Phase 2：C++ extension + stable Metal backend

目標：

- 將 Phase 1 證明有效的 kernels 和資料結構移到 C++ extension。
- Python 只保留 orchestration。

適合沉下去的內容：

- kernel registry
- persistent compiled Metal kernels
- page table native representation
- compressed block descriptor
- native KV block allocator
- C++ profiling hooks
- batch decode execution plan

成功標準：

- 每 token Python call 數量下降。
- kernel launch 數量下降。
- 中間 tensor materialization 下降。
- high concurrency 下 p99 更穩。

### Phase 3：native scheduler / cache metadata

目標：

- 當多用戶 serving 變成主要瓶頸，再把 scheduler / cache metadata 下沉。

候選語言：

- C++：如果要貼近 MLX runtime / execution plan。
- Rust：如果主要是 cache metadata / async I/O / prefetch queue。
- Swift：如果要做 macOS-native daemon / app lifecycle。

進入條件：

- `max-concurrent-requests >= 32` 以上時，Python scheduler / queue / metadata 成為 p99 主因。
- SSD tier hit / restore 的 metadata path 比實際 I/O 還慢。
- EnginePool / model lifecycle 在多模型高 churn 下造成明顯 stall。

### Phase 4：Swift / native daemon 產品化

這不是性能第一階段，而是產品穩定性階段。

適合在以下情況做：

- 要把 oMLX 變成長駐 macOS service。
- 要更好地接 system memory pressure、app lifecycle、menu bar、auto-update。
- Python packaging / venv / PyObjC 成為維護瓶頸。

但即使做到這一步，compute hot path 仍應是 C++ / Metal。

## 9. Gate Check

1. 它是否面向 decode memory-bound 問題？
   - 是，但只有 native hot path 改寫直接面向 decode memory-bound。
   - 全量 server rewrite 不直接解 decode memory-bound。

2. 它是否適用於 MLX / Metal / UMA / SSD？
   - 是。
   - 最適配的語言組合是 Python control plane + C++ / Metal execution path。

3. 它是否能接入 oMLX 現有 API / EnginePool / BatchGenerator / paged SSD KV / ProcessMemoryEnforcer？
   - 是。
   - 最佳方式是 extension / backend，而不是 fork 後重寫整個 oMLX。

4. 它是在補 oMLX 缺口，還是在重做 oMLX 已有 baseline？
   - full rewrite 是重做 baseline，不建議。
   - hot path rewrite 是補缺口，建議。

5. 它是否對 122B+、長 context、多用戶場景有實益？
   - 有，但集中在 KV / expert / scheduler metadata 熱路徑。
   - API server 換語言對 122B+ 的收益有限。

6. 它的額外 compute / latency / complexity 是否值得？
   - C++ / Metal hot kernel 值得。
   - Swift / Rust 全量改寫暫時不值得。

## 10. Scoring

### 10.1 全量改寫 oMLX

| 維度 | 分數 |
| --- | --- |
| 容量收益 | 1/5 |
| 頻寬收益 | 1/5 |
| Serving 整合度 | 2/5 |
| oMLX substrate fit | 1/5 |
| Apple 適配度 | 3/5 |
| 品質風險 | 2/5 |
| 工程可行性 | 1/5 |

- 總分：11/35
- 建議：放棄

### 10.2 Python control plane + C++ / Metal hot path

| 維度 | 分數 |
| --- | --- |
| 容量收益 | 4/5 |
| 頻寬收益 | 5/5 |
| Serving 整合度 | 4/5 |
| oMLX substrate fit | 5/5 |
| Apple 適配度 | 5/5 |
| 品質風險 | 3/5 |
| 工程可行性 | 4/5 |

- 總分：30/35
- 建議：優先整合

### 10.3 Swift control plane rewrite

| 維度 | 分數 |
| --- | --- |
| 容量收益 | 1/5 |
| 頻寬收益 | 1/5 |
| Serving 整合度 | 3/5 |
| oMLX substrate fit | 3/5 |
| Apple 適配度 | 5/5 |
| 品質風險 | 3/5 |
| 工程可行性 | 3/5 |

- 總分：19/35
- 建議：觀察，等產品化需求或 Python daemon 成本明確化後再做

### 10.4 Rust cache / scheduler subsystem

| 維度 | 分數 |
| --- | --- |
| 容量收益 | 2/5 |
| 頻寬收益 | 2/5 |
| Serving 整合度 | 3/5 |
| oMLX substrate fit | 3/5 |
| Apple 適配度 | 3/5 |
| 品質風險 | 4/5 |
| 工程可行性 | 3/5 |

- 總分：20/35
- 建議：觀察 / 局部 PoC

## 11. 建議的實作切入點

第一個真正值得做的 PoC 不是重寫 server，而是：

> **Compressed KV decode microbenchmark：同一批 KV block，測 MLX baseline、Python custom Metal kernel、C++/Metal extension 三條路徑。**

PoC 內容：

1. 固定模型與 head dim：
   - 先用 `head_dim = 128` 或 `256`
   - context 長度至少測 `32k / 128k / 1M-equivalent synthetic KV`

2. 固定 workload：
   - single request decode
   - batch 4 / 8 / 16 continuous decode
   - prefix cache hit / partial hit / miss

3. 測三個版本：
   - uncompressed BF16 / FP16 KV
   - MLX op-chain compression
   - fused Metal compressed KV read

4. 指標：
   - token TPS
   - ITL p50 / p95
   - active memory
   - cache memory
   - SSD restore latency
   - kernel launch count
   - intermediate tensor bytes
   - CPU time per token

成功才進 C++ extension。

如果這個 PoC 沒贏，換語言也不會神奇地贏。

## 12. 最終建議

本專案應採用以下分工：

| 層級 | 建議語言 |
| --- | --- |
| API / dashboard / config | Python |
| EnginePool / scheduler policy 第一版 | Python |
| scheduler execution plan 熱點 | C++ 或 Rust |
| cache metadata 第一版 | Python |
| cache metadata 高併發版 | C++ 或 Rust |
| KV compression / decode | C++ + Metal |
| expert paging / gather-scatter | C++ + Metal |
| weight dequant + GEMV / GEMM | C++ + Metal |
| macOS app / daemon 長期產品化 | Swift |

總結：

> **不要重寫 oMLX。把 oMLX 當控制面，把 MLX / Metal 當執行面，讓每一段 native rewrite 都必須通過 profile 與 microbenchmark 門檻。**

更尖銳一點說：

> **若目標是極限效能，第一語言不是 Swift、Rust 或 Go，而是 Metal；C++ 是把 Metal 接進 MLX runtime 的主通道。**

## 13. 參考資料

- `README.md`
- `oMLX-系統架構模組圖.md`
- `TECH_EVAL_TEMPLATE.md`
- `tech-evals/JANG-vMLX-與-oMLX-模組圖比較.md`
- oMLX GitHub:
  - https://github.com/jundot/omlx
- oMLX site:
  - https://omlx.ai/
- MLX documentation:
  - https://ml-explore.github.io/mlx/build/html/
- MLX Unified Memory:
  - https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX Custom Metal Kernels:
  - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX C:
  - https://ml-explore.github.io/mlx-c/build/html/
- MLX Swift:
  - https://github.com/ml-explore/mlx-swift
