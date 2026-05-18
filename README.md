# LLM Serving Framework

## 專案目的

這個專案的目標，是在 **Apple Silicon 的有限統一記憶體** 下，打造一個以 **記憶體效率優先（memory-first）** 為核心的 LLM serving framework。

核心問題不是單純「把模型跑起來」，而是：

- 在固定 RAM 預算下，盡可能放入 **更大的模型**
- 在放得下之後，盡可能保留 **實用的長 context**
- 最後再進一步支援 **多用戶 serving 與更高吞吐**

一句話總結：

> **以計算量換空間，在 Mac 256GB 這類固定 UMA 預算下，聯合優化模型權重與 KV cache，最大化可服務的 MoE 模型規模、上下文長度與後續多租戶能力。**

---

## 問題定義

對超大模型推理來說，真正的限制不是單一指標，而是以下三者的平衡：

1. 模型參數量
2. 上下文長度
3. serving 吞吐與多用戶能力

在 Apple Silicon 上，這個問題更直接，因為權重、KV cache、runtime buffer、scheduler 開銷都在同一個統一記憶體池中競爭。

本專案把它視為一個 **固定記憶體預算下的聯合記憶體管理問題**：

`Total Memory = Weights + Active Experts + KV Cache + Runtime Headroom`

因此，這個專案不只處理 KV 壓縮，也不只處理權重量化，而是同時處理：

- **靜態模型記憶體系統**
  - 權重量化
  - MoE expert hot/cold cache
  - layer / expert paging
  - SSD cold weight tier
- **動態 session 記憶體系統**
  - KV 壓縮
  - KV 分層快取
  - page/block 級映射
  - metadata-guided retrieval
  - async prefetch

---

## 專案定位

這不是單純的「MLX 版 vLLM」，也不是只做「KV cache 量化實驗」。

本專案的定位更接近：

- **控制平面（control plane）** 承接現有 `oMLX` 的 API、EnginePool、continuous batching、paged SSD KV cache
- **執行路徑（execution path）** 先以 `mlx-lm / MLX` 為 baseline，再針對熱點下探 **Metal**
- 在必要時以 extension 方式實作 expert paging、MLA/DSA-aware KV/index 與 custom Metal kernels
- 以 **oMLX serving substrate**、**MoE 權重分層**、**MLA / KV / Index 分層快取** 為三條並列主線

換句話說，這個專案希望建立的是：

> **一個承接 oMLX 既有 serving 能力、再面向超大 MoE 與長 context 擴充 memory hierarchy 的 Apple Silicon serving 系統。**

---

## 優先順序

目前專案的優先順序如下：

1. **模型規模優先**
   - 第一目標是讓更大的 MoE 模型能在單機 Apple Silicon 上可用化
2. **長 context 第二**
   - 在不犧牲模型規模的前提下，盡可能保留實用級長 context
3. **多用戶 serving 第三**
   - 在模型與 context 成立後，再擴展到更成熟的多租戶與高吞吐 serving

這代表設計取捨上，會優先站在 **權重側預算** 與 **MoE 工作集管理**，而不是先把所有空間都留給 KV。

---

## 目標模型

本專案的主目標模型不是小型 dense model，而是 **大規模 sparse MoE**。

目前的 north star 類型包括：

- `Kimi K2.6` / `1T class MoE`
- `GLM 5.1` / `MLA + DSA MoE`
- `DeepSeek-V3` 類大型 expert model

`Mixtral` 類模型仍然有價值，但主要定位會是：

- bring-up
- correctness 驗證
- kernel / cache path 除錯

而不是專案最終價值的代表。

---

## 設計原則

### 1. Memory-first，而不是 benchmark-first

如果一項技術不能實際改善以下至少一項，就不應該優先進主線：

- 容量
- 頻寬
- context 可行性
- expert working set 管理
- 多用戶整合度
- Apple 平台適配

### 2. 用算力換空間

在這個專案裡，額外計算量是可交易資源。

我們願意用：

- 量化 / 解量化
- fused kernel
- metadata scan
- prefetch / staging
- route-aware scheduling

去交換：

- 更大的模型規模
- 更長的 context
- 更少的記憶體搬運壓力

### 3. 權重與 KV 必須聯合優化

如果只優化權重，不優化 KV，長 context 會失敗。  
如果只優化 KV，不優化權重，超大 MoE 模型放不下。

因此兩邊都必須進主線：

- **權重側**
  - mixed-precision quantization
  - expert hot/cold cache
  - SSD offloading
  - prefetch
- **KV 側**
  - KV quantization
  - tiered KV storage
  - sparse retrieval
  - async prefetch

### 4. 先承接 oMLX baseline，再保留自研自由

本專案會先承接 `oMLX` 已經具備的 server / engine / cache 能力，包括：

- OpenAI / Anthropic-compatible API
- EnginePool / model lifecycle
- `mlx-lm BatchGenerator` continuous batching
- paged SSD KV cache
- prefix sharing / Copy-on-Write
- process-level memory enforcement

但不會把核心熱路徑完全綁死在現成 runtime 上。

只要某段是專案差異化核心，就應保留直接下探 Metal 的能力，例如：

- Hadamard / FWHT
- PolarQuant + QJL encode/decode
- fused attention / GEMV
- expert gather/scatter
- SSD -> GPU-visible UMA 預取與 staging

---

## 技術主線

### A. oMLX Serving / Control Plane

- OpenAI / Anthropic-compatible API
- EnginePool / model lifecycle
- scheduler
- continuous batching
- sequence / session manager
- paged block abstraction
- process memory enforcement

### B. MoE Weight Memory System

- mixed-precision weight quantization
- shared trunk / router 常駐
- active expert cache
- SSD cold expert tier
- route-aware prefetch

### C. MLA / KV / Index Memory System

- oMLX paged SSD KV cache
- MLA latent KV block lifecycle
- DSA index cache / sparse page map
- KV quantization
- KV tiered cache
- page/block mapping
- metadata-guided retrieval
- async double-buffer prefetch

### D. MLX + Metal Compute Path

- MLX / mlx-lm baseline
- Metal kernel optimization
- fused dequant + GEMV / attention
- custom memory movement path
- 必要時直接使用 Metal API

---

## MVP 目標

第一階段 MVP 不先追求「最高吞吐」，也不直接挑戰完整 1T MoE end-to-end，而是先證明以下事情成立：

> **在現有 oMLX substrate 上，paged SSD KV / continuous batching / process memory accounting 可被穩定量測，並能逐步接上超大 MoE 的 weight index、active expert cache 與短鏈路 expert execution。**

這裡的重點是：

- 不是只做靜態載入成功
- 不是只跑短 context demo
- 不是先追多用戶 benchmark

而是先證明：

1. oMLX baseline 的 KV / batching / model lifecycle 先可觀測、可調度
2. 大型 MoE 模型可以透過 weight index 定位 resident / cold experts
3. 單層或短鏈路 routed expert 能透過 active expert cache / staging buffer 運行
4. 後續再把 MLA / DSA-aware KV correctness、長 context、KV compression 與多用戶 serving 疊上去

---

## 非目標

現階段本專案的非目標包括：

- 只針對小模型 / 短上下文 benchmark 做優化
- 只做單一量化方法的 paper reproduction
- 只做本地單用戶 demo，而不考慮 serving 架構
- 把專案綁死在單一高層 runtime，而失去對底層 Metal 路徑的控制

---

## 相關文件

目前 repo 中與專案方向直接相關的文件：

- [LLM-推理-KV-壓縮與效能瓶頸.md](./LLM-推理-KV-壓縮與效能瓶頸.md)
- [oMLX-系統架構模組圖.md](./oMLX-系統架構模組圖.md)
- [JANG-vMLX-與-oMLX-模組圖比較.md](./tech-evals/JANG-vMLX-與-oMLX-模組圖比較.md)
- [TECH_EVAL_TEMPLATE.md](./TECH_EVAL_TEMPLATE.md)
- [vllm-mtp-tech-eval.md](./tech-evals/vllm-mtp-tech-eval.md)（歷史 / 對照參考）
- [vllm-feature-fit-analysis.md](./tech-evals/vllm-feature-fit-analysis.md)（歷史 / 對照參考）

這份 README 的角色，是提供後續所有技術評估、架構設計與實作文件的總綱。
