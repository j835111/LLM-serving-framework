# TECH EVAL TEMPLATE

這份文件用來快速評估新技術是否值得納入本專案。

## Project Goal

在 Apple Silicon / MLX / oMLX 上，打造一個以「記憶體頻寬與容量效率」為核心，支援超大模型、超長 context、可多用戶高併發 serving 的推理框架。

## One-line Rule

如果一個技術不能明顯改善「容量、頻寬、併發、分層快取、或 Apple 平台適配」其中至少一項，而且整合成本過高，就不應優先納入主線。

## Gate Check

在進入詳細評估前，先回答這 6 題：

1. 它是為 decode memory-bound 問題服務，還是只優化小模型/短上下文 benchmark？
2. 它是否適用於 MLX / Metal / UMA / SSD 分層快取，而不只是 CUDA/HBM？
3. 它能否接進 oMLX 現有 API / EnginePool / BatchGenerator / paged SSD KV / ProcessMemoryEnforcer 其中一段？
4. 它是在補 oMLX 缺口，還是在重做 oMLX 已有 baseline？
5. 它是否對 122B+、長 context、多用戶場景仍有收益？
6. 它帶來的額外 compute / latency / complexity，是否低於它省下的 bandwidth / memory？

快速判定：

- 前 3 題有 2 題答「否」：先放入觀察名單
- 前 3 題有 3 題答「是」：可進入詳細評估

## Scoring Rubric

每項 0-5 分，總分 35 分。

| 維度 | 評估問題 | 分數 |
| --- | --- | --- |
| 容量收益 | 是否讓單機能放更大模型、更多 KV、更多併發？ | /5 |
| 頻寬收益 | 是否減少 decode 時搬運資料量或掃描量？ | /5 |
| Serving 整合度 | 是否能接到 batching / scheduler / block manager / prefetch？ | /5 |
| oMLX substrate fit | 是否承接 oMLX 現有 API / EnginePool / paged SSD KV / memory enforcement，而不是重造？ | /5 |
| Apple 適配度 | 是否善用 Metal、UMA、SSD、zero-copy、CPU/GPU 分工？ | /5 |
| 品質風險 | 精度或生成品質損失是否可控？ | /5 |
| 工程可行性 | 實作、維護、除錯成本是否合理？ | /5 |

## Score Guide

- 28-35：優先整合
- 21-27：值得做 PoC
- 14-20：先追蹤，不急著投入
- 0-13：暫不納入主線

## Required Questions

每次評估新技術，固定回答以下 9 題：

1. 它主要壓的是長度、深度、精度、存放位置、還是讀取方式？
2. 它解的是容量瓶頸、頻寬瓶頸，還是兩者都有？
3. 它改善的是 prefill、decode，還是兩者？
4. 它對長 context 是否比對短 context 更有價值？
5. 它對多用戶 serving 是否真的有幫助？
6. 它是否需要大改模型，還是可直接套在現有模型上？
7. 它接到 oMLX 現有哪個模組？如果接不上，是不是應該作為 extension 而非主線？
8. 它是否適合 Apple Silicon，還是其實比較偏 NVIDIA 生態？
9. 它最終提高的是可服務能力，還是只提高單點 benchmark？

## Five Technical Buckets

這個專案目前可把候選技術分成 5 類：

- 減量：先少存一點，例如 SnapKV、H2O
- 壓縮：同樣資料更小，例如 PolarQuant、QJL、KIVI
- 分層：跨 layer 減冗餘，例如 MiniCache、xKV
- 分層存放：LPDDR / SSD 分級管理，例如 AdaptCache、EvicPress
- 精準讀取：只讀需要的 page，例如 Quest、prefetch

## Decision Priority

優先順序不是「壓縮比最高」，而是：

能否在 Apple Silicon 上，讓大模型 + 長 context + 多用戶 serving 真的更可行。

## Evaluation Template

複製以下區塊開始評估：

```md
# 技術評估：<技術名稱>

## 1. 基本資訊

- 技術名稱：
- 類型：paper / repo / algorithm / system design / kernel optimization
- 來源：
- 日期：

## 2. 它主要解哪個問題

- 主要解決維度：長度 / 深度 / 精度 / 存放位置 / 讀取方式
- 目標瓶頸：容量 / 頻寬 / 併發 / 快取命中 / 其他
- 主要作用階段：prefill / decode / both
- 適用場景：小模型 / 大模型 / 長 context / 多用戶

## 3. 與本專案的關聯

- 是否適合 Apple Silicon / MLX：
- 是否能接入現有 serving 架構：
- 可能接入的層：kernel / cache / scheduler / storage / model
- 與現有技術的關係：互補 / 替代 / 可疊加

## 4. 主要收益

- 容量收益：
- 頻寬收益：
- 併發收益：
- TTFT / TPS 影響：
- 對超長 context 的價值：

## 5. 主要成本與風險

- 額外 compute 開銷：
- 精度或品質風險：
- 工程複雜度：
- 維護成本：
- 平台限制：

## 6. Gate Check

1. 它是否面向 decode memory-bound 問題？
2. 它是否適用於 MLX / Metal / UMA / SSD？
3. 它是否能接入 oMLX 現有 API / EnginePool / BatchGenerator / paged SSD KV / ProcessMemoryEnforcer？
4. 它是在補 oMLX 缺口，還是在重做 oMLX 已有 baseline？
5. 它是否對 122B+、長 context、多用戶場景有實益？
6. 它的額外 compute / latency / complexity 是否值得？

## 7. Scoring

- 容量收益：_/5
- 頻寬收益：_/5
- Serving 整合度：_/5
- oMLX substrate fit：_/5
- Apple 適配度：_/5
- 品質風險：_/5
- 工程可行性：_/5

- 總分：_/35

## 8. 結論

- 建議：整合 / PoC / 觀察 / 放棄
- 原因：
- 下一步：
```

## Short Version

如果只是快速判斷，可先回答這 5 句：

1. 它幫我們省的是容量、頻寬，還是兩者？
2. 它是否對 Apple Silicon 友善？
3. 它能否接進 oMLX 現有 serving substrate？
4. 它是在補缺口，還是在重做 oMLX 已有 baseline？
5. 它是否真的提升大模型 + 長 context + 多用戶能力？

如果這 5 題裡有 4 題以上答「是」，就值得至少做一個 PoC。
