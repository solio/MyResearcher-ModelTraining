# research/anchors/ — RS-001/003 锚点（规则与词表的原件/重建状态）

## 原件（从 dashboard-k3-experiments 原样复制）

| 文件 | 内容 | 来源 |
|---|---|---|
| `bearw.v1.re` | 看跌词表 v1（`跌|空|砸|崩|跑|卖|割|退市|完|破`） | worktree wordlists/，2026-10-10 从 gen_k3_experiment.py 提取成文件 |
| `declare_sell.v1.re` | 卖出宣言词表 v1 | 同上 |
| `excite.v1.re` | 激动词表 v1（仅 C1 实验用） | 同上 |
| `gen_k3_experiment.py` | 股×日规则分诊聚合脚本（含 normalize、DECLARE_BUY/SELL、EXCITE、QIANG、BEARW、个股黑称 PER_STOCK_NICK、入场日映射） | dashboard-k3-experiments 根目录 |

**口径注意**：RS-003（2026-10-10）时代的看跌密度用的是 `bearw.v1` 单字词表。该词表后来的基准评测显示误伤率约 55%（「空仓」式卖出宣言会被算、单字「完/破」误伤多）——但 P1 复现 (a) 必须按原样用 v1，不许顺手换成新词表。

## 重建（非原件）

| 内容 | 状态 |
|---|---|
| RS-003 的实际计算脚本（产生 A/B 段 +1.43/+1.33 等数字的那份） | **未找到原件**，relation-study-003 文档自己也记了「计算代码和数据面板当时没有提交」。P1 将按文档描述在 research/RS-003-replication/ 重建，所有与原文档可能不一致的点会写进 run-log。 |
| 「历史前 10%」的确切实现（全期分位还是别的） | 原文档未写明，(a) 按「全期分位数」理解复刻，(b) 改为「扩张窗口分位数」；两者并列对照。 |

## 词表之后的版本史（备查，不用作本研究信号）

dashboard-k3-experiments 里 bearw 已演化到 v4（v2 修单字误伤、v4 用户去单字「跌」版）、declare_sell 到 v2。**本研究（RS-003 复现与 RS-004）一律使用 v1 锚定件**，词表升级是另一条研究线的事，混用即口径污染。
