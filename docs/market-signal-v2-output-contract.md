# Market Signal V2 输出合同

状态：`TRACKING_STAGE_CONTRACT_FROZEN__FORWARD_OUTCOMES_ACCUMULATING__PRICE_PROBABILITY_NOT_PROMOTED`

本合同定义 Collector/语义模型之后、Dashboard 和市场评测之前的稳定业务含义。它不把一次回顾性实验的模型结构固化成产品结构。

## 1. 当前目标

当前 16 只股票是跟踪集合，不是买入组合，也不是完整投资范围。

- 跟踪阶段：分别观察每只股票在给定期限内的绝对收益是否为正。
- 当前主期限：3 个交易日；1 日、5 日是辅助观察。
- 当前成功标签：复权后从入场日开盘到期限收盘的 `stock_return > 0`。
- 沪深300：当前只提供已知市场环境；不要求跟踪集里的每只股票都跑赢它。
- 组合阶段：只有 owner 形成真实买入集合和权重后，才评价组合相对沪深300的收益。

## 2. 四层输出必须分开

### 2.1 群体状态层（可持续展示）

这些值来自同一只股票、同一可交易入场日之前的帖子集合。它们描述“散户群体现在是什么状态”，不是价格预测。

- 注意力：帖子总量、去重文本量、抽样覆盖率。
- 多空分布：`stance` 各类别平均概率，以及 `BULL - BEAR` 方向指数。
- 操作分布：`action_tendency` 各类别平均概率，以及
  `BUY + ADD - SELL - REDUCE` 方向指数。
- 情绪分布：`emotion_primary` 的完整类别概率分布；正向、负向、FOMO 质量只作便捷聚合，不能替代完整分布。
- 单帖模型不确定度：每条帖子分类概率熵的均值；高值表示模型对单帖本身拿不准。
- 跨帖群体分歧度：广义 Jensen-Shannon divergence
  `H(mean(p)) - mean(H(p))`；高值表示各帖子给出不同且较明确的方向。
- 跨帖一致度：`1 - 群体分歧度`。
- 群体集中度：`1 - H(mean(p)) / log(K)`；高值表示总体概率质量集中到较少类别。

“一致度”和“集中度”不是同一概念。所有帖子都给出相同的模糊概率时，一致度高，但集中度低；不能把这种情况展示成强共识。

### 2.2 语义解释层（按需下钻）

`target_mode`、`reasoning_tags`、`context_dependency` 等 head 用于解释帖子在谈什么、为什么这么判断、是否依赖外部上下文。它们是生产资料和条件化字段，不是最终收益或交易信号。

情绪跨头输出分为两层：`predictions.emotion_primary` 与
`predictions.emotion_target` 保留模型的 **raw** 预测和完整概率，任何审计或回放都以
这两个字段为准。下游 feed/群体聚合同时可读取明确命名的
`predictions.emotion_target_effective`，其中当 raw `emotion_primary.prediction` 为
`NONE_EXPLICIT` 时，`effective_prediction` 固定为 `NOT_APPLICABLE`；否则它等于
raw `emotion_target.prediction`。该条件化视图不改权重、概率或 raw 字段，也不表示准确率提升。

### 2.3 市场结果层（事实回填）

每个 `stock_code × entry_date × horizon` 记录：

- 入场日和到期日；
- 复权个股绝对收益；
- 个股是否获得正收益；
- 同期沪深300收益，仅作为市场背景；
- 成交量、成交额、换手率等已知行情状态。

实现收益是观察事实，不是模型输出；Dashboard 必须允许它在期限到达后回填。

### 2.4 价格预测层（当前仅实验）

V2 的 `p(stock_return > 0)` 尚未取得晋级证据，不得显示为可用交易信号、买卖建议或已验证“情绪值”。如需展示，只能放在明确标注“回顾性实验/未验证”的诊断区域，并同时显示基线、校准和后续实现结果。

## 3. 唯一时间与实体键

- 群体状态主键：`stock_code × next_CSI300_entry_date`。
- 市场结果主键：`stock_code × entry_date × horizon_trading_days`。
- 多个自然日映射到同一下一交易日时，共用一个市场结果，不能当成多个独立样本。
- 周末和节假日帖子折叠到下一个实际沪深300交易日。
- 任何价格模型训练只能使用 `outcome_date < evaluation entry_date` 的已到期结果。

## 4. 当前证据边界

V2 复用 12,119 条既有 M12 预测，形成 16 只股票、49 个入场日、784 个市场单元。3 日主评测中，市场加群体状态的概率模型没有跑赢同股历史胜率，且明确差于纯市场模型，因此当前决策是：

- 保留群体状态层作为真实产品输出合同；
- 保留市场结果层用于持续检验；
- 不晋级当前价格概率模型；
- 不在相同历史面板上继续调参数；
- 不因市场映射失败而重训 M12 语义模型；
- 等真正新增、互相独立的日期后，按同一合同做一次前瞻验证。

权威产物：
`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-absolute-20260918/`。

## 5. 首个真实前向批次（截至 2026-09-21）

- 2026-09-18 冻结的 1,321 条帖子预测形成 16 个 `FORWARD` 群体状态快照，统一从
  2026-09-21 开盘观察。
- 1 日结果已经全部到期：11/16 个股绝对收益为正，等权平均绝对收益 `+1.0713%`。
- 3 日和 5 日结果尚未到期；当前不做价格概率模型的晋级、否决或再调参判断。
- 实际结果文件为
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-forward-20260918-noon/forward-outcomes-asof-20260921.jsonl`。
- 该批只证明前向冻结到结果回填的闭环已经运行。一个入场日不是独立日期序列，不能据此宣称市场 edge。
