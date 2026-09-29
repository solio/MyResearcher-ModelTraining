# 市场监督帖子权重：首轮固定实验（2026-09-29）

结论先行：这是已执行、可复跑的**回顾性方法原型**，不是已验证的交易模型。梯度学出的帖子权重在训练集降低损失，却未改善后段日期的十交易日收益误差；在个股高分日排序上有一个小而不稳的正向迹象。不得据此宣称“找到了悲观极点”或部署买入规则。

## 研究问题与预先固定的操作

给定股票 (s) 的入场交易日 (d)，在 (d) 开盘前只使用映射到最近五个实际交易日的论坛帖和截至前一交易日的价量。假设在 (d) 的前复权开盘价买入，在第十个实际交易日的前复权收盘价卖出，目标为 (r_{s,d,10}=P^{qfq}_{s,d+9,close}/P^{qfq}_{s,d,open}-1)。这只是固定的研究结算假设，不含交易成本、可成交性或投资建议。

逐帖冻结已存在的 `SEM7-20260915-A0` 七头概率，不重新训练语义模型。逐帖极性锚为 (v_i=P_i(BEAR)-P_i(BULL))。帖子内容产生的概率、文本长度和相对日期组成 (z_i)，训练 (a_i=softmax(w^\top z_i))、群体分数 (S_{s,d}=\sum_i a_i v_i)，再与已知价量、发帖量共同预测 (r)。(w) 和收益映射参数由同一个固定十日收益损失反向传播；七头 Encoder 与逐帖概率不参与反向传播。因此本轮确实用梯度学习“哪些帖子权重大”，但**尚未**端到端调整字词/Encoder 权重，也不能把市场相关分数自动命名为情绪真值。注意力权重本身不是因果解释。

三个同时运行的固定对照是：价量＋发帖量；同样的价量＋发帖量再加等权 `BEAR−BULL`；同样输入再加学习的帖子权重。三者均用单 seed、200 epoch、AdamW、训练收益标准化后的 SmoothL1，未在诊断段上调学习率、窗口、期限或模型结构。事件 `N1` 是另一个**事后条件诊断**：用训练期绝对日涨跌幅第 90 百分位定义冲击，查看其前五个交易日的预测前三与事后真实收益前三的差距。事件本身不知道于候选入场时，不能拿“事后前三”当唯一训练标签，也不能把这个条件分析当全日期可交易预测。

## 输入身份和时间角色

- 冻结历史语料：[25,275 条抽样帖及 `sample_id`/时间/文本](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-extreme-unenriched-20260928/source/sampled-post-metadata.jsonl)，每股日最多 16 帖；其中约 94.9% 只使用标题。逐帖 A0 概率按 `sample_id` 一对一连接。
- 行情：[2026-05-06 至 2026-09-24 的 16 股前复权日线](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-data-backfill-20260928/daily-bars-qfq.csv)及原价、沪深 300 交易日历。目标和可用价量由项目现有 `market_signal_multiday.build_panel` 统一计算。
- 有完整五日帖子窗和已成熟十日目标的面板共 1,392 股日。训练：768 股日、48 个入场日期，最终入场为 7 月 20 日，所有训练目标至 7 月 31 日已到期。后段诊断：8 月 3 日至 9 月 11 日的 480 股日、30 个入场日期，目标至 9 月 24 日已到期。两段的目标到期/入场日期严格分离。
- 证据等级固定为 `RETROSPECTIVE_DIAGNOSTIC_ONLY`。大量 5—8 月帖子是 9 月回补入库，当前库的标题/正文仍可原地更新；A0 概率也晚于这些历史入场日，且项目之前已多次研究相同行情日期。所以后段虽然在本脚本中晚于训练段，**不是研究层面的未见样本，更不是完整 point-in-time 前向测试**。

## 实际结果（不筛掉失败指标）

下表的收益和误差均为小数单位，例如 RMSE `0.08` 约为 8 个收益百分点。十日窗口重叠，480 股日不是 480 次独立机会。

| 后段诊断 480 股日 | MAE ↓ | RMSE ↓ | 预测/实际 Pearson ↑ |
| --- | ---: | ---: | ---: |
| 价量＋帖量 | 0.055662 | 0.077967 | 0.1002 |
| 加等权语义 | **0.055513** | **0.077821** | **0.1142** |
| 加学习的帖子权重 | 0.057634 | 0.080813 | 0.0916 |

学习权重模型训练损失 `0.36455`，价量＋帖量为 `0.37875`，但后段误差反而更大：现阶段更像训练期拟合，而非已验证的增量信息。

按每股后段 30 个日期的**离线**预测分数选前四分之一，学习权重版有 10/16 股的选中日期平均真实收益高于价量＋帖量；16 股等权平均差为 `+0.86` 个收益百分点，中位差为 `+0.60` 个百分点。这个离线分位数依赖整段未来日期的分数分布，不是一个实时买入阈值，也未扣交易成本。它与更差的 MAE/RMSE 并存，属于下一批前向日期应检验的候选现象，不能挑它单独宣布成功。

冲击阈值为训练段日绝对收盘涨跌幅 `7.0243%`。训练期后截至 9 月 24 日有 30 个过阈值股日，其中仅 13 个有完整的前五日候选和已成熟十日结果；这 13 个只覆盖 7 个冲击日期、7 股，包含 6 个上涨冲击和 7 个下跌冲击。以“预测前三相对事后最佳前三的平均收益差”为 regret：上涨冲击价量基线为 `1.814` 个百分点，学习权重为 `1.534` 个百分点；下跌冲击分别为 `1.819` 与 `1.950` 个百分点。上涨、下跌方向不能混成“悲观极点有效”的单一结论；某些事件连事后最佳前三的十日买入收益都为负。

## 可还原材料

- 实现：[market_text_gradient.py](/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/tools/market_text_gradient.py)。本报告记录时源码 SHA-256：`ed4724df4cc84dfcde893fc42b6b45bbd56b3647e836f69b1698fb99245c7a3c`。它只读输入，要求新输出目录为空。
- 输入六文件路径和 SHA-256、全部超参数、分组行数、指标见 [summary.json](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-text-gradient-pilot-20260929/summary.json)；已训练的系数、标准化参数和帖子注意力参数见 [checkpoint.json](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-text-gradient-pilot-20260929/checkpoint.json)。
- [逐股日三种模型预测和真实结果](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-text-gradient-pilot-20260929/diagnostic-predictions.jsonl)、[13 个事件逐例结果](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-text-gradient-pilot-20260929/event-cases.jsonl)、[16 股离线选日结果](/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-text-gradient-pilot-20260929/per-stock.jsonl)可直接重算表格；不需要看本报告得出结论。
- 独立复核用行情 CSV 和预测文件重新得到 1,392/768/480 行、阈值、13 个事件身份及 MAE/RMSE；保存的 checkpoint 用纯 NumPy 路径重算逐行预测，与原结果最大差在 `2.3e-8` 内。[单元测试](/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/tests/test_market_text_gradient.py)覆盖语义极性、训练标签到期边界、参数重放和事件候选。

```bash
cd /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining
./.encoder-venv/bin/python tools/market_text_gradient.py \
  --output-dir ../pipeline-runs/market-text-gradient-pilot-REPLAY-UNIQUE
./.venv/bin/python -m pytest -q tests/test_market_text_gradient.py
```

本结果不晋级、不替换 A0/A5，不改变现有“舆论极值 / 悲观冷却”报告。下一阶段的价值不是在这 30 个已看过的日期继续调窗口或损失，而是固定上述输入/输出语义，先保存新日期逐日预测，再等待十日结果到期。若前向阶段仍没有价量＋帖量之外的稳定增量，就应承认当前文本分数无实用证据；若有，再扩股、扩时间并开展独立复核。顶刊级论文需要这种跨时间、跨样本的独立验证；本轮完成的是其可复核的方法起点，不是论文级正结果。
