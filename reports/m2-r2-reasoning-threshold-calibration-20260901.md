# M2-R2 Reasoning Train-only threshold calibration

状态：`CALIBRATED_REASONING_DIAGNOSTIC_REJECTED`；`selected_candidate=false`。

R2 只复用已接受的 `RBT3_REASONING_CORRECTIVE_V1` 三个 checkpoint，未重新训练
Encoder 或 head。每个 seed 使用自己的 checkpoint，`weights_only=true`，并验证了
`hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`、最后一个 Transformer block、
`reasoning_tags` head 和 seed identity。阈值选择只使用 Train 1,822 的预测/weak labels；
阈值冻结后才计算 Dev 448。候选网格为 `0.05..0.95`，步长 `0.01`，含 `0.50`；并列按
距离 `0.50` 最近、再按较高阈值处理。没有保存逐条概率文件。

## Artifact

- 路径：[m2-r2-reasoning-threshold-calibration-20260901-retry](/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-r2-reasoning-threshold-calibration-20260901-retry)
- content address：`d7c09a4933c7c57f0b0174a0da831ddc651a255fd4c710561f89a4da42967ae9`
- manifest：[content-addressed-manifest.json](/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-r2-reasoning-threshold-calibration-20260901-retry/content-addressed-manifest.json:1)
- 文件总大小：`73,369` bytes
- parent corrective artifact（只读）：`0654d2ad1e537ef0637f71e8604ea02943fd08d2ffdae8900ed9f1c70eaf4238`
- 设备：三个 seed 均为 MPS；每个 seed 均完成 CPU reload finite-output smoke（`[1,15]`）。

## 每 seed 阈值

| Label | Seed 35 | Seed 71 | Seed 107 |
| --- | ---: | ---: | ---: |
| CROSS_STOCK_REFERENCE | 0.39 | 0.39 | 0.38 |
| FLOW_POSITIONING | 0.47 | 0.36 | 0.47 |
| FUNDAMENTAL | 0.31 | 0.46 | 0.30 |
| MACRO_POLICY | 0.28 | 0.31 | 0.33 |
| NEWS_EVENT | 0.38 | 0.32 | 0.19 |
| NO_REASON_GIVEN | 0.33 | 0.49 | 0.37 |
| RELATIVE_PERFORMANCE | 0.31 | 0.43 | 0.43 |
| RUMOR | 0.26 | 0.13 | 0.21 |
| SARCASM_IRONY | 0.13 | 0.14 | 0.17 |
| SOCIAL_PROOF | 0.25 | 0.27 | 0.40 |
| TECHNICAL_PRICE | 0.33 | 0.44 | 0.33 |
| THEME_NARRATIVE | 0.44 | 0.47 | 0.28 |
| UNKNOWN | 0.19 | 0.22 | 0.25 |
| VALUATION | 0.16 | 0.10 | 0.18 |
| WORDPLAY | 0.23 | 0.20 | 0.23 |

## Raw 0.50 与校准后的 Dev 指标

| Seed | Raw Macro / Micro / Exact | Calibrated Macro / Micro / Exact | CPU reload |
| ---: | --- | --- | --- |
| 35 | 0.352299 / 0.501319 / 0.263393 | 0.436283 / 0.543909 / 0.263393 | PASS, finite |
| 71 | 0.347568 / 0.506849 / 0.256696 | 0.447048 / 0.540779 / 0.270089 | PASS, finite |
| 107 | 0.369610 / 0.514486 / 0.263393 | 0.453477 / 0.560056 / 0.270089 | PASS, finite |
| **mean** | **0.356492 / 0.507551 / 0.261161** | **0.445603 / 0.548248 / 0.267857** | — |
| **sample std** | **0.011604 / 0.006612 / 0.003867** | **0.008688 / 0.010345 / 0.003866** | — |

校准后的 15-label F1、Train 校准摘要以及完整 manifest 位于 artifact 的每个
`seed-{35,71,107}` 目录；没有 Test、Anchor、Gold、OOD 或 reference predictions
输入。

## Classical feasibility gate

Aggregate 数值门槛均通过：

- Macro mean `0.445603 >= 0.400256050141`；worst seed `0.436283 >= 0.380256050141`；sample std `0.008688 <= 0.05`。
- Micro mean `0.548248 >= 0.476338797814`；worst seed `0.540779 >= 0.456338797814`。
- Exact mean `0.267857 >= 0.117232142857`；worst seed `0.263393 >= 0.097232142857`。

Critical labels（均 support >=20）逐 seed 结果如下，delta 为 calibrated F1 减 Classical F1：

| Label (support) | Classical F1 | Seed 35 delta | Seed 71 delta | Seed 107 delta | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| NO_REASON_GIVEN (86) | 0.481481 | +0.012637 | +0.015413 | +0.002179 | PASS |
| TECHNICAL_PRICE (180) | 0.615385 | +0.060077 | +0.055169 | +0.075524 | PASS |
| FUNDAMENTAL (24) | 0.666667 | -0.086022 | -0.037037 | -0.043716 | **FAIL** |
| SARCASM_IRONY (33) | 0.222222 | -0.081377 | -0.093190 | -0.045007 | **FAIL** |

`FUNDAMENTAL` seed 35 and `SARCASM_IRONY` seeds 35/71 exceed the permitted `-0.05`
drop. 因此最终状态为 `CALIBRATED_REASONING_DIAGNOSTIC_REJECTED`，不能称为 Classical
feasible、不能选择为七头候选，也不自动增加 epoch、下载 LERT 或启动新的 fit。

R2 没有训练、没有下载模型、没有改变 checkpoint、没有访问 Test/Anchor/Gold/OOD/
reference predictions，没有调用 LLM/云服务，也没有生产推理。
