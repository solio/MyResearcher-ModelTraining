# Action 动作主体与否定：直接训练实验（2026-09-29）

结论：不是项目失败，也不是现用 Action 已修好。第五个隔离候选在自然帖子对照中出现明确改进，但仍有方向性错读；Action-only 历史影子回放又发现大幅输出迁移，现有极值表现没有改善。**不替换现用 M12/selected 权重**。

## 为什么原来的训练会错

现用 `action_tendency` 是 LERT 编码器上的九类分类头。梯度下降优化的是训练标签的交叉熵，不会自动发现“谁在交易”“否定的是哪个动词”“减半还是清仓”这些关系。原头主要从弱标签学到词面相关性：看到“卖”就可能偏 SELL，看到“买”就可能偏 BUY。现有 159 条 Human Action Train 里 BUY 仅 6、REDUCE 仅 7；逐条检查发现 REDUCE 有 4/7 并非作者减仓，ADD 也有无作者加仓证据的帖子。历史 Human Check 里同样有与冻结的“作者本人动作”定义不一致的答案。人审答案可以有个人语境，不能擅自改，但在上述定义下等权拟合会给模型矛盾梯度。仅多训几个 epoch、换 seed 或增大学习率不能解决这一点。

## 实际改了什么

新增独立训练入口 [`src/semantic_model/encoder_action_contrastive.py`](../src/semantic_model/encoder_action_contrastive.py)：沿用本地官方 LERT 与现用 Action 初始权重，只训练 Action 的 specialist block/分类头；成对训练句显式改变作者/他人、肯定/否定、买/加、减/清、观望/持有。损失仍是带权分类交叉熵；“对照”指成对改动语义，不是声称用了 InfoNCE 等专门对比损失。96 条合成训练句权重 0.65，不冒充 Human。后续增加从既有冻结自然帖快照按文本选的 65 条明确动作和 18 条他人动作/建议/拒绝动作，权重 0.75，均标为专家临时 silver。30 条与作者动作定义缺乏对应文本证据的旧 Human Train 保留原答案，只在此候选将 Action 损失权重降至 0.2；原始标注文件不变。最终变体训练两层 LERT specialist，并以 Human Train 五折 OOF 选择 epoch。没有把旧或新 Check、自然诊断、行情收益放入梯度或 epoch 选择。所有候选保存为独立 checkpoint，CPU 重载推理通过。

五次已完成的训练均使用同一数据源与本地 MPS，未下载模型、未训练其余六头：

| 隔离候选 | 新 Human Check 5 九类正确 | 历史 Check 20 | 历史 Recent Check 19 | 未参与训练的合成 12 | 另选自然帖 14* |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现用 Action | 0 | 4 | 7 | 4 | 6 |
| 一层 + 语义对照 | 3 | 7 | 10 | 6 | 5 |
| 两层 + 语义对照 | 3 | 7 | 10 | 8 | 6 |
| 再加 65 自然 silver | 4 | 6 | 9 | 6 | 7 |
| 再降低 30 条低证据旧标签权重 | 4 | 6 | 10 | 7 | 6 |
| 再加 18 条真实主体/否定反例 | **4** | **6** | **10** | **7** | **11** |

\*14 条是第五次训练前只看原文挑出的、未入训练的真实帖；前四个已保存 checkpoint 在这批帖子上是事后回放。标签是专家临时判断，不是 owner Gold，更不是市场结果。Check 5 已被前几次候选反复看到，表内数值是诊断而非独立泛化证明。历史 Check 的局部答案与“作者本人动作”定义不齐，不能只以九类严格准确率做终判。

第五候选仍将“主要是我没有卖出，一旦卖出后便翻红！”判 SELL（应 HOLD），将“走个面，我抄底了，可以跌停了”判 SELL（应 BUY），把一条讨论散户买入、未讲作者操作的帖子判 WATCH（应 NO_ACTION_SIGNAL）。新 Check 的“不放量坚决不买”也仍判 NO_ACTION_SIGNAL 而非 owner 的 WATCH。因此并非“词典补丁后全对”。

另在冻结自然帖中用固定随机种子抽 1,000 条无标签帖子，只看预测分布：现用/候选 `NO_ACTION_SIGNAL` 为 817/616，`WATCH` 为 13/175。此项不能判谁对谁错，但说明候选改变了整体输出分布，不能用 14 条例子直接替换市场链路。

候选目录在项目外层忽略目录 `model-artifacts/semantic-action-contrast-*-20260929/`，第五次最终诊断为 [`evaluation.json`](../../model-artifacts/semantic-action-contrast-role-negative-v1-20260929/evaluation.json) 和 [`summary.json`](../../model-artifacts/semantic-action-contrast-role-negative-v1-20260929/summary.json)。复跑需新输出目录：

```sh
cd /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_action_contrastive \
  --two-block --natural-silver --human-evidence-weights --role-negative-silver \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/semantic-action-contrast-NEW
```

## 已执行的市场层影子回放

新增 [`tools/market_action_candidate_shadow.py`](../tools/market_action_candidate_shadow.py)，实际对冻结的 25,275 条抽样帖子重算第五候选 Action 九类概率，只在副本里替换此 head；其余六头、原始预测、1,630 个股×日的帖子集合和行情均不改。候选相对现用的九类 argmax 改了 7,368/25,275 条；`action_index` 每日正负号在 755/1,630 个股×日上改变；每股历史最低三日共 48 日中，新旧只重合 18 日。这证明候选不是“几条边界例子的小修补”，它会大幅改变下游信号。

用同一套现有回顾性“每股最低三次操作倾向”与本股相对常态口径对照，10 日期末差值现用 +0.75 个百分点、候选 +0.008 个百分点；20 日现用 -0.64 个百分点、候选 -2.49 个百分点。不同模型选出的极端日期和各自常态成员不同，不能将差值当作同一笔交易的因果增益；且全历史最低三次是事后选点、候选训练见过此时期帖子。这只能说明**目前没有可支持上线的市场侧改善迹象**，不能单凭它否定语义训练或整个项目。逐股日期、原始/候选概率和报告在 [`shadow-summary.json`](../../pipeline-runs/action-contrast-shadow-20260929/shadow-summary.json) 及同目录的 `action-extreme-descriptive/`；原 source 和旧报告没有覆盖。

重跑命令（新输出目录）：

```sh
cd /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining
PYTHONPATH=src .encoder-venv/bin/python tools/market_action_candidate_shadow.py \
  --source-run /Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-extreme-unenriched-20260928/source \
  --candidate-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/semantic-action-contrast-role-negative-v1-20260929 \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/action-contrast-shadow-NEW
```

## 后续怎样判断模型，而不是绕道遮错

这次是**正面改变模型参数**，不是在推理时写“看到某词就改标签”的规则，也没有用市场涨跌去伪造帖子标签。下一步不应继续对已经暴露的 5/14 条反复微调；应针对“作者/他人、否定、部分/全部、已执行/打算/问句”建立另一批不参与训练的真实帖对照，尤其检验现在仍错的两种关系。现有 Human 里同类矛盾应区分“个人标注语境”和冻结标签定义，不能暗中改 owner 答案。若仍不过关，应转向显式关系/多任务监督，或增加有明确证据的自然帖，而不是再堆 epoch。

真正的市场回测不能继续拿事后最低三次冒充入场规则。现有每日操作状态是该股当日抽样帖的 `P(BUY)+P(ADD)-P(SELL)-P(REDUCE)` 均值；要事先固定可实时算出的日状态/信号与入场假设（状态可获得后的下一交易日开盘）、10/20 日持有结果，逐股比较绝对收益、回撤、覆盖数量和本股对照；交易成本和重叠持有期单列。当前候选利用了截至 9 月 28 日的帖子训练，所以 5～9 月历史行情对它至多是诊断性 shadow；要证明时间外效果，需从冻结模型之后的前向帖子预测与随后成熟的行情观察，或真正按时间切分重训/回放。市场结果不能倒流到语义训练集、阈值或个股选择。
