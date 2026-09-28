# Action 语义修复候选：一次训练结果（2026-09-29）

结论：`CANDIDATE_REJECTED__CURRENT_MODEL_UNCHANGED`。这是从现用 selected Action 权重出发的**一次**按头训练，不涉及其他六头；模型产物在 `model-artifacts/semantic-action-repair-candidate-20260929/`，没有替换 M12 hybrid 或覆盖 9 月 28 日预测快照。

Owner 已完成 `semfix-20260928-train17` / `semfix-20260928-check8`，原始答案分别在 `pipeline-runs/semantic-repair-label-20260928/{train,check}-final.jsonl`。其中 Action Train 14、Check 6；不是自然随机样本，而是专门挑选已知边界的 challenge。所有答案原样保留。依据冻结的“动作必须属于作者本人”定义，两个 Train 及一个 Check 答案存在直接冲突，本候选仅从拟合/判分中隔离，未代替 owner 改标：

| 角色 | 原文 | 原答案 | 不用于本次的原因 |
| --- | --- | --- | --- |
| Train | “前几天跟朋友一起买他206.6卖了……我要格局捏着不卖……” | `SELL` | 卖的是朋友，作者明确继续持有。 |
| Train | “明天股权登记，散户都一股不卖……” | `HOLD` | 陈述其他散户，并未表达作者自己的操作。 |
| Check | “大家都别卖，那个的成本就不会降……” | `HOLD` | 对他人的建议，不是作者继续持有。 |

训练集为既有 147 条 Human Train 加本次 12 条有效 Action Train，共 159 条；`REDUCE` 仅 7 条，`BUY` 仅 6 条。仅 Train 内五折 OOF 选 epoch（12 轮内选 11），不使用新 Check、行情、Teacher Dev、Test/Anchor/Gold 选参数。冻结权重后才加载 5 条有效 Action Check。设备 MPS，耗时约 34 秒；脚本见 `src/semantic_model/encoder_action_repair.py`。
候选 checkpoint 已在离线 CPU 上重载并对 5 条 Check 完成形状、有限值及概率和为 1 的推理 smoke；这只证明产物可加载，不改变效果否决。

| 检查 | 现用 Action | 新候选 |
| --- | ---: | ---: |
| 新 Check 5 条正确数 | 0/5 | 0/5 |
| 新 Check macro-F1 | 0 | 0 |
| 三档动作方向一致数（`BUY/ADD`、`SELL/REDUCE`、其余） | 3/5 | 3/5 |
| 候选 Train 内 OOF macro-F1 | — | 0.1931 |

新 Check 逐条：`一股不卖` 人工 `HOLD`，现用 `NO_ACTION_SIGNAL`，候选 `SELL`；`下午早清仓` 人工 `SELL`，两者均 `REDUCE`；`准备清仓` 人工 `SELL`，现用 `BUY`、候选 `REDUCE`；`减仓一半` 人工 `REDUCE`，两者均 `ADD`；`不放量坚决不买` 人工 `WATCH`，两者均 `NO_ACTION_SIGNAL`。按三档方向计分，候选修正了“准备清仓→买入”，却把“一股不卖”从无卖出方向推成卖出，净数不变；这保留了 owner 所说相近类别与反向错误的差别。跨 16 股最小对照中，候选对“我没有买”“别人买了，我没买”仍 16/16 `BUY`，对“我不卖，一直持有”从 14/16 `SELL` 变为 16/16 `SELL`。最小对照不代表自然准确率，但足以否定“修好否定/动作主体”的声称。

因此**不晋级、不重算今日市场报告**：没有理由把失败候选的 Action 概率写进新快照。现有完整 Human147 + 12 条新 Action 的直接九类微调未修复机制，且已有监督中明确“不买”类仅约 3 条、“不卖”类约 5 条，文本有不同主体/动作混合。下一步不继续无目标地改学习率、加 seed 或让 owner 大批量补标；应先决定怎样把“作者本人/他人、否定/肯定、已执行/条件、全卖/部分卖”变成可学习的结构，再做一个隔离候选，并为晋级保留新的时间外人审。Stance 的 3 条新 Train 全为 `BULL`，与既有“过去懊悔不自动等于未来看多”边界有解释冲突，本轮没有盲目微调 Stance。
