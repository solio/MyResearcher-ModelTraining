# MyResearcher-ModelTraining Expert 入职与实时交接

状态：`ACTION_CONTRAST_CANDIDATE_TRAINED_NOT_PROMOTED__CURRENT_MODEL_UNCHANGED`

最后核对：2026-09-29 CST（Action 对照训练已完成五次隔离候选；现用 M12 和已有前向快照未更改）

适用对象：临时或长期接替当前技术负责、任务编排和 review 职责的 Expert / Agent。

这是一份可执行的入职文档。它回答四件事：当前角色负责什么、项目真实做到哪一步、接手后如何判断产出，以及下一步应推进什么。架构历史仍以
[`architecture-handoff-and-model-roadmap.md`](architecture-handoff-and-model-roadmap.md)
为准，里程碑和优先级仍以
[`milestone-priority-and-agent-routing.md`](milestone-priority-and-agent-routing.md)
为准；Dashboard 与市场层字段含义以
[`market-signal-v2-output-contract.md`](market-signal-v2-output-contract.md)
为准。但这些文档中的时间点快照仍可能落后于本文件和实时 Git/artifact 状态。

## 1. 一分钟接手结论

**2026-09-29 最新：**用户要求继续直接训练，不以单次失败宣告项目失败。已针对 Action 的作者/他人、否定、部分/全部问题实际训练五个隔离 LERT 候选，见 [`../reports/semantic-action-contrast-20260929.md`](../reports/semantic-action-contrast-20260929.md)。第五候选在人审新 Check 由现用 0/5 到 4/5，在另选 14 条真实帖专家临时对照由 6/14 到 11/14；但仍把“我没有卖出”判 SELL、“我抄底了”判 SELL。已对冻结 25,275 帖做 Action-only 市场影子回放：7,368 条标签变化、755/1,630 个股×日操作状态正负翻转，现有事后极值的 10/20 日相对本股常态表现均未改善；不能当作交易回测或证明模型路线失败。第五候选不替换现用模型。下一步用新的真实帖关系对照确认语义泛化，避免在已暴露的少数 Check 上调参，再设计不看未来的时间外市场测试。现用模型、其他六头、旧报告均保持原样。本段优先于下面的 2026-09-28/29 历史快照。

### 终局目标与验收层级（owner 于 2026-09-16 重申）

- 项目的终局目标不是复现 owner 的人工标签，而是形成一个能服务 owner 个人趋势选择、并能在
  **时间外市场数据中回测和持续调整**的研究系统。最终有效性由市场结果检验。
- owner 的人工标注是市场结果尚未到来时的语义监督、方向选择和问题诊断，属于当前阶段的
  **短期初稿**；它不是普适真理、市场 Gold，也不是项目成功或失败的终局判据。
- 验收顺序固定为：`语义诊断 → 冻结带时间戳的预测 → 无未来信息泄漏的市场回测 → 调整模型/标签/决策规则`。
  语义 accuracy、macro-F1 和人工一致率只属于第一层工程指标。
- 某次模型不符合人工 check，只能说明该候选或当前训练方法没有获得晋级证据；不得据此宣布
  “语义模型路线失败”或“项目失败”。同样，迎合一次人工标签也不等于具有市场价值。
- 后续市场回测才是主要工作量。必须把“文本理解质量”“预测信号质量”“交易/趋势决策质量”
  分层评估，禁止用未来收益反向污染文本、split、模型选择或已冻结的预测。
- 只有标签含义、趋势目标、预测期限、行动映射等真实业务方向不确定时，才请 owner 裁决；
  常规实现、重复校验和非阻断优化不得消耗 owner 注意力。

### 当前事实（2026-09-28，优先于本文后续历史快照）

#### 2026-09-28 极端行情读帖压力测试

- 详细方法与证据见 [`price-shock-semantic-review-20260928.md`](price-shock-semantic-review-20260928.md)。6～7 月 16 股、44 个交易日的只读对照发现：大涨/大跌日的帖量及当前方向/情绪状态与普通日有描述性区分；同时有两种事件的 10 股，10/10 大涨日平均方向指数高于本股大跌日。它检验的是事后舆论反应，不是预测能力。
- 一个明确的 P1 语义边界错误已复现：现有抽样中 12 条“卖飞了”被 M12 `stance` 12/12 判 `BEAR`，而既有 Teacher Train/Dev 同类短句 4/4 标 `NEUTRAL + REGRET`；上一版 selected 对相同 13 条短句有 9 条 `BEAR`，M12 有 13 条，不能说错误仅由 M12 引入。M12 在更广自然短语和合成最小对照中也显示负面经历污染未来立场，但全量错率尚未知。
- 本稿“舆论极值”最低方向的 48 个案例共抽样 714 帖，“卖飞了／后悔没买”精确短句命中 0、宽短语命中 2；**没有证据因此撤销当前极值分析**。但操作极值关键案例已有“一股不卖”→`SELL`、“减仓减仓”→`ADD`、“清仓!”→`ADD` 的直接污染。`action_tendency` 来自旧 selected 权重，并非 M12 新训练。更多否定／动作主体最小对照和真实帖反例已确认，不能只修“卖飞”或“清仓”一个词。
- Owner 要求并行确定修改范围后，25,275 条自然预测的并行审计又发现 `target_mode` 将纯大盘讨论判为 `ON_TARGET`，`emotion_primary` 将“有点害怕”判为 `NONE_EXPLICIT`、“爽死了”判为 `FRUSTRATION`，M12 `emotion_target` 将明确公司／管理层指向判为 `MARKET/UNKNOWN`；`context_dependency` 的指代缺失边界也有反例。`reasoning_tags` 2,264 条空集合违反非空输出契约，应先分离输出逻辑问题再谈重训。详细原文、来源和限制见上述审计文档；这些是定向反例，不代表全量错率。旧权重／报告／前向快照不改，下一轮按 head 逐项训练与比较，而非七头整体清零或用行情涨跌伪造帖子标签。
- 第一层自然帖稀疏人审 `semfix-20260928-train17` / `semfix-20260928-check8` 共 25 项，2026-09-29 已标完并拉取，每帖只填 `action_tendency` 或 `stance` 之一；原始导入与 final 在 `pipeline-runs/semantic-repair-label-20260928/`。Action 14 Train/6 Check 中，2 Train/1 Check 与“作者本人动作”冻结定义直接冲突，原答案保留但本候选不拟合/判分。现用 147 Human Train 加新 12 Action 训练一次 selected 初始化的隔离候选；5 条有效新 Check，现用与候选严格九类均 **0/5**、合并三档方向均 **3/5**，候选跨股否定/主体对照仍读反，因此**否决候选、不切换模型、不重算市场**。详见 [`../reports/semantic-action-repair-20260929.md`](../reports/semantic-action-repair-20260929.md)。Stance 新 Train 三条均标 `BULL`，未盲目重训。

统一称谓：`舆论状态极值` = 原线，研究某日的多空/操作/情绪各自的极端**水平**；
`悲观后冷却` = 独立影子线，研究先相对悲观、后帖量/原始成交量下降且看空缓和的**变化过程**。
不要称后者为“综合极值”或“真正的市场底部”；两条线的事件、入场时间、对照组和期限不同，详见
[`market-extreme-events-runner.md`](market-extreme-events-runner.md) 的“统一术语”。

#### 2026-09-28 未 enrich 初稿已交付，enrich 不再阻塞初步研究

- Owner 明确先看未 enrich 版本；个人趋势印象只供后续讨论，不能据此改参数、选样本或结果。
  后续可以先出非 enrich 稿，再单独出 enrich 稿，不因正文补全慢或验证码一直等待。
  两稿沿用同一模型、输入规则、抽样与研究口径；保留原稿，不把语料变化误报为模型提升。
- 原始帖子回溯在本轮运行期间完成。权威最终快照为 2026-09-28 02:53:20 CST：
  212,817条源帖子，16股、25,275条确定性抽样、1,630股×入场日状态；
  1,299条有正文、23,976条仅标题（94.9%）。所有股票最早现有帖子在5月4～6日。
  本轮使用已有正文，否则标题；不是强行剥离已有正文，也没有删掉仅标题帖。
- 权威本稿：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-extreme-unenriched-20260928/`，
  `source/` 保存输入、title/content、语料来源、七头概率、日状态和配置；`analysis/` 为可读图表/报告。
  前面的 `market-extreme-unenriched-20260928-draft/` 是采集结束前的中间快照，不是最终交付。
- Owner 随后要求添加期间最高收益，最新可读报告为本稿的 `analysis-with-window-highs/REPORT.html`。
  原稿保留，只补汇总/展示，原事件案例与期末收益全部一致，没有重新推理或训练。
  指标为期内日线最高价/入场开盘价-1（含入场日，共H个交易日），先事件内取最高、再逐股平均、股票等权。
  10/20/30/60日低值后该均值为10.8053%/17.6319%/21.1207%/27.7222%，普通日为
  9.3128%/13.7876%/17.8968%/21.8703%；期末收益不变。不是可实现最高卖出价，未到期不混入均值。
  原脚本及后续两稿默认包含新列；10项相关测试通过。
- Owner 指出“普通日”不够普通并要求立即定义，最新可读报告为 `analysis-central-reference/REPORT.html`。
  主对照定义为每股、每指标完整冻结状态的Q25～Q75（含界、线性插值），剔除原入选低/高日期；
  只读状态、不看未来收益，各期限不重分组，仅排除尚未到期结果。相对常态不等于股价平稳或绝对中性。
  旧 `ordinary_*` 组更名“其余入场日”并保留兼容字段，不能再把剩余日期称为常态。
  逐股边界、成员在新报告的 `reference-groups.jsonl`；多空常态共802股×入场日。
  20日有647个完整常态日、16股；37个原看空极端案例不变。期末极端/常态+2.7689%/-1.2041%，
  平均最大浮盈17.6319%/14.0887%。旧数值和案例全部不变；13项测试通过，无重新推理/训练。
  方法已写入 `docs/market-extreme-events-runner.md`。没有启动行情匹配、新阈值优化或策略拟合。
- Owner 强调跟踪和买入只针对少数个股，聚合均值不能当主结论。最新阅读入口改为
  `analysis-stock-first/REPORT.html`：每只股票自身10/20/30/60日极端对常态表放到逐事件日期与帖子之前，
  16股汇总移到文末；Markdown首屏为16股20日速览（不是买入排名）。
  仅展示顺序与逐股汇总变化，同一冻结输入和全部旧案例/数值未改；13项测试通过。
- Owner 另提出“舆论极值之后帖子与成交减少、情绪缓和”可能更重要，担心试验损伤现有进展。
  已在独立旁路运行 `tools/market_cooling_shadow.py`，只读上述冻结状态与行情，原模型/脚本/报告不变。
  最新权威探索报告 `pipeline-runs/market-cooling-shadow-20260928-final/REPORT.md`，方法见上述runner。
  20日37个到期冷却日期只形成19段相邻期；10股有本股对照，其中5股较好、5股较差。
  这是逐股分化的回顾性现象，不是统一新策略，不据结果调条件；13项相关测试通过。
- 复用已有M12概率；先在MPS计算12,409个新增输入，约39秒，采集尾部完成后仅新增97个输入，约1.5秒。
  七头、权重、阈值与输入规则未改变；两处batch设备跟随模型设备的最小改动使既有runtime可在MPS推理，
  默认CPU仍可用。4条同输入CPU/MPS全部标签相同，概率最大差1.20e-6；28项相关测试通过。
- 保持每股每入场日最多16帖、三个原指标、最低/最高各3次、10/20/30/60日观察。
  全历史看空低值后10/20/30日均值 +3.6538%/+2.7689%/+1.9343%，普通日 -0.5066%/-1.1177%/-1.7446%；
  已成熟44/37/35个案例，20日9/16股高于普通日。60日18个案例均值 -5.1071%，不是所有期限都会赚钱。
  情绪平衡低值20日均值 -4.3184%（普通日 -0.9129%），30日 +1.2379%且单股影响明显；
  不把“最看空”“最负面情绪”“最想卖”当成同一个反向信号。
- 重跑脚本 `tools/market_extreme_refresh.py` 复用既有采样、M12、聚合和极端分析，不训练市场概率模型。
  旧报告不覆盖。原等待采集后的自动跟进已关闭，避免重复；每日行情自动化不变。
  enrich 后可按新请求出第二版；目前不自动催采集、不调用LLM、不加标注，不派生优化实验。

#### 2026-09-28 行情回填，配合 owner 的帖子回溯

- Owner 已另行启动16只现有已采集个股的帖子回溯至2026-05-04；本任务仅授权补行情。
- 实际日线已完整补到2026-05-06～2026-09-24，共101交易日。5月4日休市，执行时9月28日尚未开盘。
  16只股票各101条 raw/qfq，分别1,616条；沪深300101条。全33请求成功，无缺失交易日。
- 新目录：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-data-backfill-20260928/`。
  包含原始/前复权OHLC、成交量、成交额、涨跌幅和个股换手率；旧行情和旧报告不覆盖。
- `tools/market_data_bootstrap_v0.py` 新增 `--provider`、`--stock-codes`，可独立回填，不依赖旧13股预测文件。
  本轮依据只读 Collector 库的16股列表，未扩到配置里尚未采集的新增股票。
  市场依赖放仓库外；主分支执行，没有新增分支/worktree。
- 运行说明与可复制命令：`docs/market-data-backfill-runner.md`；15项相关测试通过。
  本轮没有重训、没有补语义预测、没有重跑极端事件或其他市场实验。
- Owner 后续明确授权：帖子回溯完成后，保持原口径重跑一次极端群体状态分析。
  已创建本聊天每小时自动跟进“帖子回溯完成后重跑极端状态分析”（`automation-2`）；
  未完成时安静检查，不推理、不分析、不干扰采集，完成后运行一次并关闭跟进。
- 新研究显式使用这份扩展行情，并先用相同M12生成扩展时期的概率/日状态；
  不能仅换行情路径就声称研究扩展到五月。沿用每股每入场日最多16帖、三个原指标、
  低/高各3次与10/20/30/60日期限；不重训、不调参、不追加标注，不自动派生其他实验。
  上述等待安排现已由本轮未 enrich 初稿执行取代，实际报告见前节。

#### 2026-09-27 最极端状态之后的市场表现

- Owner 澄清：最低3次是数据少时寻找最强现象的机会窗口，不是永久筛选限制或交易策略；
  本轮优先检查信号是否存在，不继续添加实时阈值、概率模型或训练。此前“下一步条件作用模型”建议没有启动，
  已由本轮明确请求取代。不能用3个案例限制未来海量数据研究。
- `tools/market_extreme_events.py` 已真实执行，无训练/调参/下载/新标注。复用848个日状态、16只股票和56个行情交易日；
  按多空、操作、情绪平衡分别取每股全历史最低3次和最高3次，不读收益选点。低值144个、高值144个。
- 全历史多空低值的10/20/30日已观察38/25/22个案例；股票等权绝对收益
  +3.9235%/+5.6433%/+5.3873%，同股普通日 +0.2505%/+0.9375%/+1.0144%。
  20日10/16只股票低值表现高于普通日，各股均值中位数 +2.6386%；每次剔除一股后相对普通日差值
  仍 +2.7367～+5.7280个百分点。因此不能继续描述为“最极端状态也无关系”。
- 同一批有10/20/30日完整结果的22个原始多空低值、14只股票，收益依次为
  +5.7352%/+7.7843%/+5.3873%，普通日 +3.0601%/+4.0864%/+1.0144%。
  负面情绪低值主结果相对普通日差值 -1.3168/+1.0896/-0.2212个百分点，结果混合；
  不把各种情绪都解释成同一个反向信号。
- 这是 PROVISIONAL 回顾性极端关联，不是可执行策略、显著性、因果或独立情绪增量证明。
  普通日对照未匹配此前行情，状态仅由每天最多16帖抽样概率估计，日期/收益重叠；
  个股差异和期间亏损明显。60日全部未到期，不能填成零或据此判失败。
- 报告保留全历史未到期极端；可观察历史内重新选的极端单列为补充，不冒充主结果。
  实际事件日期、完整概率、例帖、10/20/30/60收益、相对入场价最深跌幅和价格路径均已输出。
- 权威目录：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-extreme-events-20260927-final/`。
  独立从原行情表重算554条完整主案例收益/跌幅一致，排序与48张图结构通过；相关测试46项通过。
  独立执行说明见 `docs/market-extreme-events-runner.md`；3是可修改参数，不是长期硬限制。
- 本轮完成即停止，不因为发现正均值自动晋级或重训。下一研究应围绕已展示的群体状态和逐股实际表现，
  区分其与此前行情反转的关系；不再拿技术评分替代 owner 能看懂的市场结果。

#### 2026-09-27 同股常态与行情残差限定对照

- Owner 在相关性诊断后授权“继续”：用相同 N=5/20、H=10/20/30、时间切分和股票日，
  固定检验 `relative_combined`（减同股训练均值）与 `residual_combined`（再减已知行情的
  Ridge 线性解释）两个表达方式；C=0.1、Ridge alpha=1，不搜索参数、不改目标、不重训 M12。
- `tools/market_signal_multiday.py --state-contrast` 已实现并真实执行。参考均值、填补、
  缩放和 Ridge 都只在每折已成熟训练行拟合。旧模型已经有股票 one-hot 截距；这里没有增加
  新信息，在线性模型中主要改变正则化几何，不能宣称因果识别或已排除所有行情影响。
- 原五模型的全部 770 条概率、训练/测试身份和原共同样本对照的 414 条概率完全保留。
  仍只有 5→10、5→20、20→10 可做 CV。对应原组合/同股相对/行情残差 Brier 为
  `0.351639/0.376534/0.365655`、`0.355598/0.322235/0.331842`、
  `0.223786/0.384513/0.378538`。两个新增表达只在 5→20 上改善，该组仅 8 个测试日期，
  误差仍高于固定预测 0.5 的 0.25 参照；不能据此晋级或宣称有效交易信号。
- 20→10 的相同 13 个测试日期中，情绪平衡与过去 N 日收益的逐股 Spearman 中位数为
  `+0.538462`，与未来收益为 `-0.203297`；行情残差与未来收益为 `+0.008242`。
  测试行情解释 R² 为 `0.306085`。这支持当前状态较多反映已发生价格变化的解释，但不证明
  因果、所有语义无用或任何其他映射路线失败。
- 新表达在短历史中外推不稳定：20→10 的三项核心状态，原/相对/残差超过训练 ±3σ 的比例为
  `0%/29.15%/31.56%`。因此也不能将新增候选退化直接解释为“去偏后语义没有信息”。
  本轮不围绕缩放、C、alpha 再派生修补实验。
- 权威目录：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-state-contrast-20260927-final/`。
  REPORT.md/html、run-config.json、逐条七候选市场概率、逐股时序相关、配对误差和外推诊断均已落盘。
  这里的七候选不是语义模型的七头。仓库外 `/tmp` 独立复跑的全部候选概率与汇总一致，
  相关测试 40 项通过；原不可变输入与旧结果未修改。
- 当前不采用这两个新增表达，不改变 M12 和已有状态输出。业务目标仍为 5/20 日状态与
  10/20 日收益、30 日辅助。下一方法如继续，应先固定小范围的行情情境作用差异假设，
  而不是继续修线性去均值/残差的 C 或缩放；本轮没有启动该下一实验。无需追加人工标注。

#### 2026-09-27 多日状态与中期收益时间交叉验证

- Owner 已授权过去 5/20 个交易日状态 → 未来 10/20 个交易日正收益，30 日为辅助；同时要求
  参数化独立脚本、尽量复用现有数据和输出报告。当前业务研究目标以此为准，旧 3 日主期限
  只代表早期短期实验。已有 1/3/5 日前向快照仍按原合同回填，不改写。
- `tools/market_signal_multiday.py` 已交付并实际运行，独立命令和参数说明见
  `docs/market-signal-multiday-runner.md`。不需要 Codex、Torch、网络或新的语义推理。
  已复用 15,919 条概率，形成 848 个股票×日状态；行情截至 9/24，共 56 个实际交易日、16 只股票。
- 六个 N/H 组合中，5→10、5→20、20→10 的时间 CV 分别有 28/8/13 个测试日期、444/119/207
  个股票日，共 770 条样本外五模型概率。训练标签的 outcome_date 均严格早于测试日；没有随机分折、
  参数搜索或重训七头模型。5→30、20→20、20→30 当前时间历史不足，保留为缺数据，不判模型失败。
- 组合相对纯行情的 macro-stock Brier 增益分别为 -0.000266、+0.018727、+0.003516。
  20→10 的组合前四分位胜率 47.06%、平均绝对收益 +0.48%，纯行情对应 27.45%、-1.62%；
  这是最大可用样本结果，不能直接用不同 N 的成绩差异判定窗口效果。
- 已额外运行同训练/测试样本、同行情特征的窗口对照：10 日期限均为 13 个测试日期、207 个股票日。
  纯行情 Brier 为 0.229165；5 日/20 日组合分别为 0.239246/0.224569，对行情增益分别
  -0.010081/+0.004596。20 日组合前四分位胜率 45.10%、平均收益仅 +0.0424%（未扣费用）。
  全体历史先验 Brier 为 0.222425，仍优于两个窗口的组合。因此目前只有相对纯行情的
  PROVISIONAL 方向证据，不能宣称已证明稳定预测价值、交易优势、晋级或选型。
- 历史语义模型冻结晚于部分市场日期，本轮为市场层回顾性时间 CV，不能称为全系统真实前向成绩。
  区块长度默认 max(N,H)，不足两个区块时不输出置信区间。历史输入保持原样。
- 权威结果目录：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-multiday-20260927-final/`。
  包括 REPORT.md/html、report.json、run-config.json、CV 概率、逐折切分和 SVG 增益图。
  从 /tmp 以保存配置复跑，770 条主实验概率、414 条窗口对照概率、切分和汇总均一致。
  相关测试 35 项通过。
- 下一步是持续补入真实交易日行情和每日状态，使用同一配置重跑；当前不在此面板追加调参。
  最少新增行情交易日的乐观下限：20→20 需 8 天，5→30 需 13 天，20→30 需 28 天；信号缺日会继续推迟。
  无需 owner 新的决策或人工标注。

#### 2026-09-26 第三批前向冻结（9/28 入场）

- 固定 cutoff 为 `2026-09-24T23:59:59+08:00`。Collector 中 9 月 24 日共有 593 条合格帖子、覆盖 16 只股票；没有晚于 cutoff 的记录进入该批次。
- 使用既有 M12 artifact `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m12-human-first-lert-oof-20260915` 和 selected artifact `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904`。预测于 `2026-09-26T11:19:56+08:00` 冻结，593 条均保留恰好七个 raw heads，并将 `conditioned_outputs` 作为同级字段保留。
- 按仅含交易日期的日历聚合为 16 个股票单元，统一 `next_trading_entry_date=2026-09-28`；9 月 25～27 日休市。快照于 `2026-09-26T11:20:47+08:00` 生成，16/16 均为 `FORWARD`，`market_prices_read=false`、`market_outcomes_read=false`、`outcomes_included=false`。
- 权威目录为 `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-forward-20260924/`；包含 `forward-shadow-predictions.jsonl`、`forward-outcomes-pending.jsonl`、`trading-calendar-dates.json`、`community-state-snapshots.jsonl` 和 `snapshot-summary.json`。本轮没有训练、调参、模型切换或创建标注任务。
- 当前已冻结的三个入场日期为 2026-09-21、2026-09-24 和 2026-09-28。下一步是在各自 1/3/5 个实际交易日期限成熟后，用已取得的本地行情分别写出独立 outcome 文件；新 Collector 批次仍应在下一实际交易日开盘前冻结。

#### 2026-09-26 第二批前向冻结 review

- Main 提交 `332c33175fb47ccd99f01e5397c79615d44b07c4` 已通过独立 review；相关测试复跑 16 项通过。
  9 月 23 日的 1,024 条预测严格保留七头，1,024 条均另有同级 `conditioned_outputs`；
  去重后 1,013 帖聚合为 16 个 `FORWARD` 单元，统一入场日为 9 月 24 日。
  预测于 9 月 24 日 00:43:45 CST、快照于 00:46:46 CST 生成，均早于开盘。
  目录：`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-forward-20260923/`。
- 行情已覆盖至 9 月 24 日。只读重算第二批 1 日结果：16/16 完整，2 只正收益、14 只非正收益，
  个股等权平均收益 `-1.7903%`；本次重算尚未生成独立 outcome 文件，交由 Side 落盘。
  首批 3 日结果已落盘于 `forward-outcomes-asof-20260923.jsonl`，9/16 正收益、等权平均 `+0.4159%`。
  这些是全跟踪集的行情描述，不能当作模型选股收益或预测准确率。
- 中秋休市为 9 月 25～27 日，9 月 28 日恢复；国庆休市为 10 月 1～7 日，10 月 8 日恢复。
  [上交所公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml)。
  首批 5 日结果应在 9 月 28 日到期；第二批 3/5 日结果分别在 9 月 29 日、10 月 8 日到期。
  旧聊天与 bootstrap 中仅按工作日计算的估计不能代替实际交易日历。
- 第三个入场日期已在单独目录冻结；后续 outcome 回填应使用各批冻结时记录的 entry/horizon 合同，不能改写 snapshots。
  维持现有 M12，积累多个不同入场日期的成熟 3 日结果后再统一做初步趋势分析；日期之间可能存在相关性及收益窗口重叠，日期数不是正式显著性或生产验收门槛。

#### 2026-09-22 首个真实前向结果

- 2026-09-18 的 1,321 条 M12 七头预测已在任何未来行情发生前冻结，经规范化文本去重后形成
  16 只股票的 `FORWARD` 群体状态快照；统一入场日为 2026-09-21。该批没有重训语义模型，
  也没有读取未来价格来选帖子、特征或规则。
- 截至 2026-09-21 收盘，16 个 1 日 outcome 已全部回填：11 只正收益、5 只非正收益，
  个股等权平均绝对收益为 `+1.0713%`。3 日和 5 日各 16 个 outcome 仍未到期，状态保持
  `PENDING_HORIZON`，不得提前填值或据此宣布模型成功/失败。
- 增量行情会导致入场日与到期日的 qfq 行来自不同抓取批次。回填逻辑现同时读取 raw 与 qfq：
  若 qfq 抓取批次不同，但 raw/qfq 区间收益绝对差不超过 `0.001`，则接受结果；超过该阈值才返回
  `ADJUSTMENT_VINTAGE_MISMATCH`。首批 16 个 1 日结果的最大 raw/qfq 收益差为 `0`。
- 本批权威目录为
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-forward-20260918-noon/`；
  冻结快照是 `community-state-snapshots.jsonl`，截至 2026-09-21 的独立结果是
  `forward-outcomes-asof-20260921.jsonl`。目录名中的 `noon` 是历史命名，实际输入覆盖
  2026-09-18 全日，不要据目录名误报时间范围。
- collector 主库当前最新帖子时间为 2026-09-19 23:33 CST。该批未在 2026-09-21 开盘前冻结
  预测，因此不能事后冒充第二个前向入场日。下一批必须使用新采集帖子，并在其下一实际交易日
  开盘前冻结；在此之前不重训、不调当前价格概率模型，也不要求 owner 追加人工标注。

#### 2026-09-18 Market Signal V2 目标纠正（优先于 V1）

- owner 的当前跟踪阶段目标是：分别观察跟踪集内每只个股未来绝对收益是否为正，
  不要求每只跑赢沪深300。跟踪集不是实际买入组合；只有未来存在真实买入集合/组合时，
  才用相对沪深300的组合收益检验投研能力。V1 的“单股必须跑赢指数”目标不再是当前主目标。
- V2 没有重训语义模型，也没有重复推理；它直接复用 12,119 条已冻结 M12 七头概率，重新聚合为
  `stock_code × next_CSI300_entry_date` 的独立市场单元。价格标签是复权后个股开盘至 1/3/5 交易日收盘的
  `stock_return > 0`，沪深300只作为当时已知的市场状态特征。
- 群体状态输出已按业务意图拆分：`stance/action/emotion_primary` 概率分布、多空与操作方向指数、
  正/负情绪质量、注意力，以及“单帖模型不确定”和“跨帖群体分歧”两个不可混淆的量。
  `target_mode/reasoning_tags/context_dependency` 仍是解释和条件化的生产资料，不是当前价格目标的主输出。
- 主验收预先固定为 3 日 `combined` 对同股 Beta(1,1) 扩展历史胜率，按 16 只股票等权的
  macro-stock Brier 评分，并按连续 5 个入场日的区块自助法计算 95% 区间。评测只允许
  `outcome_date < evaluation entry_date` 的已到期历史结果进入拟合。
- 唯一 V2 运行的实际主结果是：3 日 `combined` 相对同股历史胜率的 macro-stock Brier gain
  `-0.004126`，5 日区块 bootstrap 95% 区间 `[-0.022423, 0.003683]`；不具备晋级证据。
  相对纯市场模型的 gain 为 `-0.020664`，区间 `[-0.027913, -0.014076]`，明确更差。
  纯群体状态相对同股历史胜率有 `+0.006577` 的小幅点估计，但区间 `[-0.012170, 0.016573]`
  跨零，只能作为未来新日期待验假设。
- 操作结论是 `DO_NOT_PROMOTE_CURRENT_PRICE_PROBABILITY_MODEL`：当前价格涨跌概率不进入生产、不在同一面板继续调参。
  这不是 M12 语义模型失败证据，因为 V2 检验的是市场映射而不是文本理解真值；因此当前也不重训 M12。
- Dashboard 后续应把稳定的“群体状态层”和“市场结果层”分开展示：群体共识/分歧、多空/操作方向、
  情绪分布和注意力是可持续输出；实现收益是回填结果；当前未通过验证的价格概率必须显式标为实验项，
  不得伪装成“情绪值”或交易信号。
- 权威 V2 产物目录为
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v2-absolute-20260918/`，包含绝对收益面板、
  1,440 条时间外 walk-forward 概率、机器可读报告和 `REPORT.md`。

下面 2026-09-17 Market Signal V1 保留为目标误配的历史证据；若与上述 V2 目标和结论冲突，以 V2 为准。

- Market Signal V1 已完成第一次有效的跨日期历史评测：从 collector 的 82,529 条帖子中，
  按上海时区解析发布时间，将周末/节假日折叠到下一个沪深300实际交易日，以
  `stock_code × next_CSI300_entry_date` 作为独立市场单元。最终得到 16 只股票、49 个入场日、
  784 个单元，不再把单帖当成独立市场样本。
- 同一市场单元内先按规范化文本去重，再不读取行情结果地做确定性时间覆盖抽样，每单元最多 16 帖；
  共生成 12,119 条 M12 七头概率预测。本轮没有训练或修改语义模型。
- 市场目标是入场后 1/3/5 个交易日相对沪深300的超额收益是否为正。入场价为下一个实际交易日开盘价；
  expanding walk-forward 只使用 `outcome_date < evaluation entry_date` 的已知结果，避免把未来收益泄漏给拟合。
- 主比较只把有直接方向含义的 `stance` 和 `action_tendency` 概率加入行情基线；七头全特征版只作诊断对照。
  固定逻辑回归不做参数搜索，不在这 49 个日期上反复“炼丹”。
- 结果不支持可用市场信号：1/3/5 日 `combined directional` 相对纯行情模型的 Brier gain 分别为
  `-0.0079/-0.0091/-0.0111`，且日期 block bootstrap 95% 区间均小于 0。换言之，当前 M12 语义概率
  加入后，胜率概率预测反而稳定变差。
- 唯一值得保留但尚未显著的假设是 5 日 `action_score_mean` 横截面排名：平均日 Spearman `0.0725`，
  top-minus-bottom 超额收益 `+1.05%`，但两者 95% 区间都跨 0。它只是一个留待新日期检验的弱假设，
  不是交易信号，也不允许现在围绕它调阈值。
- 权威产物目录是
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-v1-final-20260917/`；其中 `REPORT.md`、
  `market-signal-report.json`、`market-panel.jsonl` 和 `walk-forward-predictions.jsonl` 分别保留可读结论、
  机器可读指标、市场面板和逐日预测。
- 本地 `collector.legacy.db` 不能补早期独立日期：8,251 条中 8,185 条是 `601012`，时间仍从 2026-07-09
  附近开始。其他 smoke/backfill DB 也没有更早可用历史。不得将这些重复/偏科数据冒充新日期。
- 当前阶段不再训练市场层，也不追加同类人工标注。下一次有意义的评测是：获得更早的真实历史日期，
  或累积足够多的以后新日期，然后不改合同地重跑一次。日常不再设置定时器，也不用单一入场日的价格反应做结论。

下面 2026-09-16 的 forward shadow 与早期 36 个单元诊断保留为历史；若与本节冲突，以 Market Signal V1 为准。

### 历史快照（2026-09-16）

- 唯一开发/训练主线仍是本地 `main` 与一个 primary worktree；未创建新分支或 worktree。
- 当前运行基线仍是 `encoder_m12_hybrid_runtime.M12HybridInferenceRuntime`。M16b 的两头
  hybrid runtime 已实现并留下实验证据，但经最新时间外人工检查后**不晋级**，不能替换 M12。

- 情绪跨头输出现在明确分为 raw 与 effective 两层：模型产生的
  `emotion_primary`、`emotion_target` 及完整 probabilities 原样保留；公共推理入口另外
  附加同级 `conditioned_outputs.emotion_target`。当 raw `emotion_primary=NONE_EXPLICIT` 时，effective
  target 仅供下游 feed/群体聚合使用并固定为 `NOT_APPLICABLE`，其他情况与 raw target
  相同。这是确定性语义条件化，不是模型重训、调阈值或效果提升。
- M16b 先在去除 1 条与 Human147 精确文本冲突的 weak sample 后，用 Train1821 做两 epoch
  七头共享预训练；再用 Human147 做五折、最多四 epoch 的人工纠偏。实际使用 MPS，完整运行
  58.175 秒；Weak Dev448、Recent Check19、HumanCheck20、Human48 均未参与拟合或选 epoch。
- M16b artifact：
  `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m16b-weak1821-deduped-multitask-human147-20260916`。
  它曾依据 Recent Check19 / HumanCheck20 / Human48 的旧证据被临时选中：`stance` 相对 M12
  的正确数变化为 `+4/+3/-5`，`emotion_target` 为 `+3/+3/-5`。该结论现已被更新鲜的时间外
  检查否决，保留这里只是为了说明实验历史，不再代表当前选择。
- 已对 292 条 feed 完成 runtime smoke：输出仍是完整七头；相对 M12，只有 `stance`
  与 `emotion_target` 变化。候选快照仍保留为实验记录：
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/recent-chain-20260915/feed/prediction-snapshot-m16b-two-head.jsonl`。
- 最新时间外池来自 2026-09-15 collector 数据：1,332 条 raw，经现有清洗得到 1,000 条；其中
  76 条有正文、924 条仅标题。M12 与 M16b 在 `stance` 上分歧 352 条、在
  `emotion_target` 上分歧 349 条。owner 只对分层抽出的 Train12 / Check8 两头任务完成了
  稀疏标注，共 40 个 assignment；这是一组 challenge-enriched 诊断样本，不代表自然总体。
- 独立 Check8 上，`stance` 的 M12 为 `5/8`（accuracy `0.625`、macro-F1 `0.514286`），
  M16b 为 `2/8`（accuracy `0.25`、macro-F1 `0.123810`）；在 6 个模型分歧中，M12 正确 3 个，
  M16b 正确 0 个。`emotion_target` 两者均为 `2/8`，没有 M16b 的晋级证据。Check8 的 16 个
  head assignment 合计 M12 正确 7 个、M16b 正确 4 个。因此 M16b 两头晋级被否决，M12
  继续作为当前运行基线；Train12 可保留作未来证据，但不触发立即重训。
- 时间外人工检查 artifact：
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/m16b-temporal-20260915/human-evaluation.json`。
  这些结果只说明 M16b 当前更不符合 owner 的语义方向，不是市场验证，也不能推出项目失败。
- M13 frozen embedding、M14b identity-stripped char TF-IDF、M15 Human147-only shared encoder 均已
  有限运行并关闭；它们证明单靠 Human147 会退化为类别先验。不要继续围绕这些方案调 C、阈值、
  epoch 或追加同类小批标注。
- 市场行情层已经真实落地：当前 forward 涉及的 16 只股票及沪深300覆盖 2026-07-09～09-15 共
  49 个交易日，股票 raw/qfq 各 784 行、指数 49 行，OHLCV、成交额和换手率完整。AKShare
  因网络代理失败后由免费 BaoStock 完成；成交量单位是股、成交额单位是人民币。
- 292 条 2026-09-01～09-03 的 M12 历史预测已完成 1/3/5 日结果连接，逐帖共 876 个 outcome；
  主统计先按 `stock_code × entry_date` 聚合，只有 36 个市场单元、13 只股票、3 个入场日。
  `stance` 分数对超额收益的 Spearman 为 `+0.101/+0.035/+0.075`，分档不单调；`action`
  为 `-0.353/-0.111/-0.062`，也没有与预期方向一致的稳定梯度。结论只能是
  `RETROSPECTIVE_PIPELINE_DIAGNOSTIC_ONLY`：链路已跑通，但没有市场效果结论。
- 历史诊断产物：
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-validation-v0-20260916/`。
  该批预测晚于行情结果生成且可能与训练数据重叠，不得用于宣称时间外效果，也不得据此调阈值。
- 第一批真正时间外证据已启动：M12 在 2026-09-16 14:54 CST、未来行情发生前冻结 862 条
  七头预测，覆盖 16 只股票，统一从 2026-09-17 开盘作为入场候选，等待 1/3/5 个实际交易日
  结果。冻结文件是
  `/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-data-v0-20260916/forward-shadow-predictions.jsonl`。
- 初始历史行情清单曾遗漏 `300487`、`603806`、`603997` 三只 forward 股票，现已用冻结的
  forward snapshot 补抓完成，16 只股票各有 49 个交易日，未重做或覆盖预测。正式 forward
  evaluator 已运行并输出 `FORWARD_OUTCOMES_PENDING`：862 条预测对应 16 个股票×入场日单元，
  1/3/5 日结果均为 pending，没有拿旧行情冒充未来结果。
- 下一步不是重训或继续人工标注，而是在真实交易日到达后增量获取行情、按完全相同的 entry/
  horizon 合同回填 forward outcome，并持续冻结多个新日期批次。只有跨多个入场日累积后，才
  判断哪些 head/概率具有市场增量价值；owner 只在趋势期限或行动映射出现真实业务歧义时裁决。

下面的 Feedback Loop V1、RBT3 和早期 LERT 内容是历史证据；若与本节冲突，以本节和实时
artifact/runtime 为准。

### Feedback Loop V1 Phase A（2026-09-09）

M4 Human100 treatment 已关闭为 `HUMAN_SUPERVISION_NO_GAIN_DIAGNOSTIC`，其标签与既有模型均只是可修订的历史证据，不是本轮人工方向的答案。当前唯一主线仍为本地 `main` 与一个 primary worktree；本阶段已把 DataClean pilot 接通为 Collector → Feed → Prediction（训练前快照）→ Human Feed → Training 的循环入口。14 条输入已冻结为 10 条 Human Train 和 4 条 Human Check，并派发到 `fl1-train-20260909`（70 tasks）与 `fl1-check-20260909`（28 tasks）；模型预测未预填进标注应用。完成手机标注后先执行报告中的唯一 `feedback_loop_v1.py pull` 命令，得到可修订的人工快照；Phase B/C 尚未开始。

- 当前本地统一开发/训练分支是 owner 指定的 `main`，不是由 GitHub 默认分支决定的。
- 训练前收口已将 Side `5f06dab543943843a25cfe7c08e4e00814c5a010` 以保留双方历史的
  merge `ee5fd2a999f639c71ce45d06af82b2e4c7cd7b8b` 纳入 `main`；`main` 是新的
  统一 HEAD 线，最终本地盘点只保留 primary worktree 与本地 `main`。
- S3 runner 的实现提交是 `94295b1ab69d2501f9775c3e0cbecb17be67dfb8`；随后
  evidence 文档提交继续在本地 `main`、`origin/main` 上推进，实时 HEAD 以 Git 为准。
- 当前执行只使用一个 primary worktree 和 owner-declared `main`；Side worktree 与
  本地 Side branch 已在其提交由 `main` 可达后删除。远端历史 tracking ref 不阻塞开发、
  训练或里程碑。
- M0 不可变数据和 Classical baseline reference 已冻结并验收。
- M1 `hfl/rbt3` frozen Encoder 七头首跑已经完成并关闭。
- M2-S1 三 seed frozen Encoder control 已完成并独立复核，artifact 已冻结。
- M2-S2 “只解冻最后一个 Transformer block”三 seed 训练已独立复核，平均 Macro-F1
  七个 head 全部优于 S1，但因为两个预先声明的关键标签回归而没有通过 progression
  gate；结论是 `ACCEPT_EVIDENCE_BUT_DO_NOT_PROMOTE`，未晋级、未选型。
- M2-S3 两个触发 head 的 frozen single-task negative-transfer diagnosis 已完成；六个
  run（每个 head 的 seeds 35/71/107）均在 MPS 完成，CPU reload 通过，结果仍不产生
  selected candidate。S3 content address 为
  `7928614bdda834d0de6e3cc6b8d26bc02a10c821c4564dfa61e0ad419ac8899c`。
- `RBT3_REASONING_CORRECTIVE_V1` 与 R2 校准诊断均已完成；R2 的 Classical critical
  label gate 未通过（`FUNDAMENTAL` 与 `SARCASM_IRONY`）。最终七头 specialist
  runner 在 `bb6f131` 从原始 RBT3 权重完成 reasoning 三 seed fail-fast 训练，
  阈值仅由 Train 选择，Dev Macro/Micro/Exact 和 CPU reload 均完成；
  `SARCASM_IRONY` 三个 seed 相对 Classical drop 为 `-0.183760/-0.146750/-0.088889`，
  因而输出 `RBT3_M2_CANDIDATE_REJECTED`，content address
  `73df684f65ec237a0d47da696ea4ca1ee07a7c809892869294275279a7b7b76b`。
- Final specialist fail-fast 在其他六头启动前停止；没有七头 aggregate、统一 bundle、
  M3 candidate 或生产批准。RBT3 M2 正式候选路线到此关闭，不再派生新的 corrective。
- Owner 随后只授权一次新的 `hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460`
  LERT lineage。实现/合同提交为 `f2af729`，但固定 revision retrieval 在真实运行前置阶段
  被网络阻塞：`model_info`/`snapshot_download` 通过 `socks5h://127.0.0.1:10795`
  返回 `ConnectionError/MaxRetryError` 或 `LocalEntryNotFoundError`，取消代理的官方
  `curl` 返回 `SSL_ERROR_SYSCALL`。没有 LERT 文件、模型加载、checkpoint、训练或 artifact；
  也没有更换 revision、升级依赖或尝试第三个模型。
- 当前没有需要 owner 重复决定的事项。不同模型/新下载、新数据、Test 开封、LLM、
  云服务和生产推理仍需要新的明确决定；现有 RBT3 Train/Dev 技术复核不需要。

## 2. 当前角色说明

当前角色是本项目的技术主理人、reviewer 和执行编排者，目标是让项目持续产生真实、
可解释的模型进展，并让 owner 只在确实需要业务或范围选择时介入。

### 2.1 必须承担的职责

1. 把 owner 的目标转换为边界清楚、能完成、能验收的里程碑和任务。
2. 根据当前里程碑给任务定 P0–P3，而不是把所有问题都叫 P0。
3. 为 Main/Side 提供短而完整的提示词；没有真实 owner 决策时，在 review 后直接给
   下一步提示词，不让 owner 反复追问。
4. 对 Main/Side 的产出做证据化 review：检查 live Git、源码、测试和真实 artifact，
   不只复述执行者自己的报告。
5. 区分“代码已经写好”“测试通过”“真实训练完成”“artifact 可复核”“候选模型已选中”
   和“生产验收通过”，禁止跨级表述。
6. 保持主线能训练、artifact 不被 Git 污染、训练使用的代码和数据身份清楚。
7. 在每个阶段及时停止：满足当前 exit criteria 就推进；非阻断优化进入 backlog，
   不让项目变成永远不收敛的治理工程。
8. 维护本文件的实时状态。每次完成一个真实阶段，更新当前 commit、artifact、结论和
   next action。

### 2.2 这个角色不是什么

- 不是 owner 身份认证员，也不是签名、receipt、allowlist 或 remote trust 系统的设计者。
- 不是为了“可信”而无限添加 hash/gate 的流程管理员。Hash 只用于定位和复核实际
  artifact，不代表人的身份或授权。
- 不是用大量文档替代训练的人。文档只在防止重复踩坑或冻结实验变量时有价值。
- 不是生产批准者。当前所有指标来自 weak-label Dev diagnostic，不能写成 Gold、Test
  或 production evidence。
- 不是把 frontier LLM 放入逐条生产分类链路的人。生成式 LLM 只允许未来有限、离线的
  review/验真/证伪/标注辅助，而且输出不能自动成为 Gold。

## 3. 与 owner 协作的固定约定

这些是 owner 最近多轮纠正后已经确定的工作方式，接手人不得重新发明相反流程。

### 3.1 决策和授权

- owner 的直接明确指令足够；本单人项目不增加 owner receipt、签名、身份认证、
  trusted branch、remote allowlist、remote-sync receipt 等机制。
- 技术检查只服务于实际风险：错误数据、split/Test 泄漏、错误 tensor/模型执行、NaN/
  Inf、资源越界、覆盖未合并工作等。
- 不要求 owner 重复已经给过的决定。只有出现真正的新范围时才提问，例如：换一个
  Encoder/revision 并下载、创建 Gold/OOD、打开 Test、使用云服务或 LLM、生产推理。
- 如果不需要 owner 决定，review 结尾直接给下一步任务提示词。

### 3.2 优先级和模型路由

| 等级 | 本项目含义 | 默认执行模型 |
| --- | --- | --- |
| P0 | 阻断当前里程碑，或与当前工作不可并行、必须先完成 | Terra Max |
| P1 | 能明显改善数据/模型效果或决定质量路线 | Luna Max |
| P2 | 流程、工具、开发体验、非阻断审计 | Luna Max |
| P3 | 其他低风险、低收益改进 | Luna Max |

当前 M2 的模型质量实验和分析通常是 P1。不要因为任务“重要”就自动升级成 P0。
Sol Max 只在 Terra Max 无法处理的高代价架构冲突中建议，而且必须直接写在提示词标题。

### 3.3 输出格式

- 提示词放在普通 Markdown fenced code block 中，便于复制。
- 不使用 `<details>` 等伪折叠标签；当前客户端不能可靠折叠，标签只会制造噪声。
- 提示词尽量短，把长期约束引用到仓库文档，不在每次提示词里重抄几十行框架。
- review 先写结论，再写 finding 和证据。owner 若只需要一个决定，就只列那个决定。

### 3.4 事实与权限优先级

发生冲突时按以下顺序处理：

1. owner 最新的明确指令；
2. `AGENTS.md` 和冻结的 machine-readable contract；
3. 当前里程碑/优先级政策与架构合同；
4. decision log 中仍然 active 的决定；
5. stage-specific contract、acceptance 和 runner handoff；
6. 已核验的 live Git 与不可变 content-addressed artifact；
7. dated report、本文快照、commit message、聊天和记忆。

本文是实时交接入口，但仍是 dated snapshot。报告与 artifact JSON 冲突时必须检查两边，
不得挑选更有利的数字。owner 的新决定如果具有长期影响，应在本轮工作后写回合同或
decision log。

### 3.5 派发 Developer / Reviewer 的最小提示词合同

提示词必须在没有聊天上下文时仍可执行。使用下面的骨架，删除无关段落，不增加设计发散：

```text
【<P等级>｜<执行模型>】Milestone <里程碑> — <唯一且有边界的结果>

Role
你是本轮 <Developer / Reviewer>，只处理一个 M2 任务。

Live facts
- repo / branch / commit: <精确值>
- target artifact / content address: <精确值>
- 已接受 evidence: <精确 decision / artifact>

Read first
- AGENTS.md
- docs/expert-onboarding-and-live-handoff.md
- docs/milestone-priority-and-agent-routing.md
- docs/architecture-handoff-and-model-roadmap.md
- <本轮唯一相关的 contract / source / tests>

Objective
只产出：<code / report / review verdict>。

Allowed reads and writes
- read: <精确路径>
- write: <精确路径>
- external action: <none 或 owner 已授权的精确动作>

Hard boundaries
- Train-only fit；Dev-only 的预声明 diagnostic/selection。
- 不打开 Test/Anchor/Gold/OOD/production。
- 不换 model/revision，不下载，不改 Schema/split/gate/threshold。
- 不修改 immutable artifact 或上游仓库。
- 保留并说明不属于本轮的 dirty changes。

Exit criteria
- <有限、可验证的 pass/fail 条件>
- 失败时输出 <blocker / rejection state> 并停止，不设计下一模型。

Handoff
- files changed
- commands and exit codes
- evidence and allowed claim
- known limitations
- exact next owner decision, if any
```

一个 prompt 只给一个 decision surface。Reviewer 默认不实施修复，Developer 默认不自行
接受自己的 evidence；如确需合并角色，必须在 objective 中明说。不要使用“latest”模型或
浮动 revision，也不要通过签名/receipt 代替可复现的技术证据。

## 4. 技术方向和不可混淆的模型角色

### 4.1 当前和未来模型

| 模型/工具 | 角色 | 当前地位 |
| --- | --- | --- |
| TF-IDF + Logistic Regression v0.3.5 | 冻结的诊断、回归和 disagreement baseline | 永久保留，但不是生产模型 |
| `hfl/rbt3` Chinese Encoder | 当前主要的上下文表征和七头实验 lineage | M1 已跑通；M2 正在做稳定性和微调阶段选择 |
| 其他 BERT/RoBERTa/MacBERT 类 Encoder | 未来候选 | 未授权下载或训练 |
| Frontier generative LLM | 有限离线 review、验真、证伪、标注辅助 | 不在常规推理中，不能自动产生 Gold |

owner 所说的“基础模型”是预训练判别式 NLP Encoder，例如 BERT/RoBERTa/MacBERT，
不是生成式 LLM。项目不是永远拒绝基础模型；RBT3 Encoder 已经实际进入训练。

### 4.2 当前七头结构

- 六个 single-label head：`target_mode`、`stance`、`emotion_primary`、
  `emotion_target`、`action_tendency`、`context_dependency`。
- 一个 15-label multi-label head：`reasoning_tags`。
- M1/M2 共用固定 input builder、class order、Train/Dev split 和 sample-by-head weights。
- 当前 S2 使用一个共享 RBT3 Encoder，仅最后一个 Transformer block 与七个 heads 可训练。

### 4.3 OOD、UNKNOWN 和 abstention

三者必须分开：

- `UNKNOWN` 是 taxonomy 内“文本不足以判断该字段”的类别。
- abstention 是模型对某个 head 暂不作可信决定的策略。
- OOD 是输入分布不在已验证范围内，未来需要独立 challenge set 和检测/拒答策略。

固定词表 classical 模型遇到新词主要依靠 char n-gram、已有片段和其他上下文退化处理，
不能可靠理解真正的新概念；Encoder 的 subword/contextual representation 改善这一点，但
仍不能自动解决 OOD。任何高置信度都不能替代未来 OOD 验收。

## 5. 数据和语义底线

接手人开始任何模型工作前应保留这些约束：

1. 只按 `sample_id` 连接 canonical input 和标签；不做文本模糊匹配。
2. Train 1,822 / Dev 448 / Test 467 / Embargo 242 的冻结角色不可随机重建。
3. 当前 M2 只允许 Train 拟合、Dev early stopping 和 diagnostic；不得读取 Test、Anchor、
   Gold、OOD 或 reference predictions 做选择。
4. canonical input、weak label、reviewed label、Human Gold、Anchor、prediction、quarantine
   是不同数据角色；模型输出或模型一致不能自动变成 Gold。
5. Schema 固定为 `schema/semantic-schema-calibrated-v0.2.1.json`，class order 不得静默变更。
6. 每个 head 的 sample weight 独立；不得退化成一个全局 sample weight。
7. `UNKNOWN != NEUTRAL`、`NONE_EXPLICIT != CALM`、
   `NO_ACTION_SIGNAL != WATCH`。
8. 看空不自动等于恐惧，看多不自动等于买入；愿望、条件动作、他人动作不等于作者已
   执行动作；“不割肉”不是 `SELL`，“不追高”不是 `BUY`。

## 6. 当前仓库和运行环境

### 6.1 路径

| 内容 | 位置 |
| --- | --- |
| 仓库 | `/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining` |
| CPython Encoder runtime | `.encoder-venv/`，约 834 MiB |
| 固定 Hugging Face cache | `.encoder-artifacts/hf-cache/` |
| M1 accepted artifact | `.encoder-artifacts/m1-rbt3-0aa0527f-provenance-88f90b1/` |
| M2-S1 artifact | `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s1-first-three-seed-20260831/` |
| M2-S2 artifact | `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s2-partial-last-one-three-seed-20260831/` |
| Side Dev disagreement outputs | `runs/m2-dev-disagreement/` |
| immutable data | `data/local/MyResearcher_Semantic_Immutable_Data_v0.3.5/` |
| baseline reference | `data/local/MyResearcher_Semantic_Baseline_Reference_v0.3.5/` |

`.encoder-venv/`、`.encoder-artifacts/`、`runs/` 和外部 `model-artifacts/` 不进入普通 Git
历史。不要把 checkpoint、CSV/DB/XLSX 或大模型权重提交进仓库。

### 6.2 当前 Git 事实

```text
owner-declared unified branch: main
final specialist implementation/result commits: bb633f5, 8161cae, bb6f131
local worktrees: 1 (primary `main` working directory)
execution branch: main (other local refs, if present, are not an execution gate)
remote tracking refs: origin/main plus one historical origin/feat/... ref
```

S2 manifest 不把 Git commit 当作 owner 身份或 runtime gate；上面的 `fd50471` 是运行时
观察记录，不应伪装成 manifest 内字段。远端旧 ref 或 GitHub 默认分支不是 runtime
前置条件。只有本地实际执行的 source、
未合并工作是否会丢失、以及 owner 指定的统一分支需要关心。不得因为远端清理权限、
默认分支设置或同步状态阻塞本地训练和开发。

## 7. 已完成里程碑

### 7.1 M0 — 不可变数据与 Classical reference

状态：`CLOSED_DIAGNOSTIC_FOUNDATION`

- Data content address：
  `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`。
- Reference content address：
  `828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`。
- Original model SHA-256：
  `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`。
- 原历史环境：Python 3.12.13、Linux x86_64、NumPy 2.3.5、SciPy 1.17.0、
  scikit-learn 1.8.0、joblib 1.5.3、OpenBLAS 0.3.30/pthreads、AMD EPYC。
- 六个 SAGA scalar head 均在 `n_iter_=max_iter=2000` 时未收敛；15 个 liblinear
  Reasoning estimator 均正常收敛。这是原 baseline 本身的事实，不是本地开发训练错。
- frozen reference predictions 重新算指标与历史 metrics 的最大绝对差为 `0.0`。
- macOS/不同 sklearn 版本只能称 `COMPARABLE_DIAGNOSTIC_RUN_ONLY`；exact reproduction
  是隔离的历史 side lane，不阻塞 Encoder 主线。

### 7.2 M1 — 第一个可运行 Encoder 七头闭环

状态：`CLOSED_DIAGNOSTIC_ONLY`

- Model：`hfl/rbt3`。
- Revision：`0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`。
- License：Apache-2.0。
- Accepted artifact content address：
  `b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58`。
- frozen Encoder + 七个 trainable heads；seed 35；Train 1,822；Dev diagnostic；MPS
  训练；CPU reload/inference finite output 已通过。
- M1 证明“训练闭环可以跑”，不证明模型达到生产质量。

### 7.3 M2-S1 — 三 seed frozen shared control

状态：`TECHNICAL_EXIT_PASSED_CONTROL_ONLY`

- Seeds：35、71、107。
- Artifact content address：
  `04a23d76413049e57ff083655f80ad8c3dfc7ed90702a3c9ed66bcfd79f377f6`。
- 三次均在 MPS 完成；CPU checkpoint reload 均通过；所有 head 的 sample standard
  deviation 均不超过 0.05。
- 输出固定为 `S1_CONTROL_EVIDENCE_ONLY`，`selected_candidate=false`。

| Head | S1 Dev Macro-F1 mean | Std |
| --- | ---: | ---: |
| `target_mode` | 0.328733 | 0.008003 |
| `stance` | 0.374859 | 0.007566 |
| `emotion_primary` | 0.198078 | 0.035309 |
| `emotion_target` | 0.286061 | 0.023519 |
| `action_tendency` | 0.139849 | 0.009758 |
| `context_dependency` | 0.326668 | 0.004465 |
| `reasoning_tags` | 0.153885 | 0.003345 |

## 8. 当前实时阶段：M2 LERT FINAL（下载前外部阻塞）

状态：`LERT_SMALL_M2_ATTEMPT_BLOCKED_BEFORE_DOWNLOAD_OR_TRAINING`

实时覆盖（2026-09-02）：最终 RBT3 specialist runner 已从 `bb6f131` 执行 reasoning
fail-fast 三 seed。`SARCASM_IRONY` critical gate 失败，故其余六头未启动；以下
S2/S3 表格是历史 evidence，不是当前待执行阶段。

S2 runner 已在 commit `fd50471` 合入 `main`，三 seed 真实训练已完成并完成独立复核。它只解冻
RBT3 的最后一个 block `encoder.encoder.layer.2` 和七个 heads；heads LR `3e-4`、
Encoder LR `1e-5`、AdamW weight decay `0.01`、max 12 epochs、patience 3。Train/Dev
角色与 S1 相同，Test 等未进入该 runner。

Artifact manifest 当前声明 content address：
`3e452b6105df731abc869adebe73910bfa60d6abf196540765e72f8145932446`。

下面是已独立重算 manifest、metrics 和 checkpoint reload 后确认的 artifact 报告值：

| Head | S1 mean | S2 mean | S2−S1 | S2 Std |
| --- | ---: | ---: | ---: | ---: |
| `target_mode` | 0.328733 | 0.372252 | +0.043519 | 0.019337 |
| `stance` | 0.374859 | 0.398787 | +0.023928 | 0.009429 |
| `emotion_primary` | 0.198078 | 0.261110 | +0.063031 | 0.007852 |
| `emotion_target` | 0.286061 | 0.370535 | +0.084474 | 0.029154 |
| `action_tendency` | 0.139849 | 0.247598 | +0.107750 | 0.012538 |
| `context_dependency` | 0.326668 | 0.378214 | +0.051546 | 0.017575 |
| `reasoning_tags` | 0.153885 | 0.217980 | +0.064096 | 0.013064 |

阶段没有晋级的原因不是 aggregate 变差，而是预先冻结的 critical-label gate 失败：

- `emotion_primary:CALM` 在 seed 107 相对 S1 的 F1 delta 为 `-0.083117`，低于
  允许的 `-0.05`。
- `reasoning_tags:NO_REASON_GIVEN` 在 seed 107 的 delta 为 `-0.067081`，也低于
  `-0.05`。
- 七个 head 的 mean 和 worst-seed primary metric gate 均通过，所有 head 平均提升，
  seed stability 也通过。
- Artifact 当前正确标记 `selected_candidate=false`，结果等价于
  `M2_S2_COMPLETED_NOT_PROMOTED`；不得为了让它通过而事后放宽 gate。

另有一个必须保留 lineage 的报告语义：冻结合同说明“某 head 的 critical-boundary
gate 失败”可触发该 head 的 S3 frozen single-task diagnosis；但 immutable S2 artifact
中的 `s3_triggered_heads` 为空，因为产出它的 `fd50471` 实现只把 head-level primary
metric failures 放入该字段。后续 commit
`58a95c04ef9cd1f29aa7bd42ed9c045b5e3b57ba` 已修复源码并增加单元测试，使同样的失败触发
 `emotion_primary` 和 `reasoning_tags`。不得就地修改旧 artifact，也不能把旧空数组理解为
“没有 S3 诊断对象”。Reviewer 应明确区分 run source、post-run fix 和 immutable bytes。

S3 已按预声明触发条件完成，且只训练上述两个 head、每个 head 三个 matching-seed
run。`emotion_primary` Dev Macro-F1 为 0.216395 / 0.201228 / 0.241920（mean
0.219848，std 0.020565，worst 0.201228），相对 S1 的 seed delta 为 +0.023340 /
+0.035678 / +0.006290。`reasoning_tags` Dev Macro-F1 为 0.152158 / 0.152405 /
0.144848（mean 0.149804，std 0.004294，worst 0.144848），相对 S1 的 delta 为
−0.004690 / −0.002143 / −0.005410；Micro-F1 和 exact-set accuracy 亦已写入
S3 matching-seed 报告。`emotion_primary:CALM`（support 48）在两个 seed 恢复，
`reasoning_tags:NO_REASON_GIVEN`（support 86）在 seed 107 仍回归。S3 stability
diagnostic passed，但 promotion 为 `NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC`，
`selected_candidate=false`。

S3 immutable evidence：
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s3-frozen-single-task-triggered-heads-20260831`
（content address `7928614bdda834d0de6e3cc6b8d26bc02a10c821c4564dfa61e0ad419ac8899c`）。

## 9. 当前下一步

最终 RBT3 specialist 运行已按预声明顺序完成 reasoning 三 seed，并因
`SARCASM_IRONY` critical-label Classical gate 失败而停止；其他六个 specialist 未启动。
因此 RBT3 的 M2 正式候选路线关闭，当前没有 selected candidate，也不启动新的 RBT3
corrective、增加 epoch/LR/seed 或下载 LERT。

保留的最终 rejected evidence：
`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-final-specialist-seven-head-20260901-final`
（`73df684f65ec237a0d47da696ea4ca1ee07a7c809892869294275279a7b7b76b`）。三 seed
checkpoint、Train-only 阈值、Dev 指标、resource-log 和 CPU reload 均可复核。

当前不做：打开 Test、创建 Gold/OOD、读取 reference predictions、下载新模型、
full unfreeze、调用 LLM、生产推理、49,054 条数据推理、云训练或新的 RBT3 阶段。
若未来要比较新模型，必须由 owner 另行给出新的模型/revision 与范围决定。

当前 LERT 尝试的最小恢复入口是恢复到可访问官方 Hub 的网络路径，然后重新执行同一
固定 revision 命令；在此之前不得改 revision、下载其他模型、升级依赖或启动训练。

## 10. Review 工作法

### 10.1 先回答结果层级

每次 review 必须先明确被 review 的东西到底是什么：

| 证据 | 最多能证明什么 |
| --- | --- |
| 单元/合成测试 | 代码路径 `TESTED` |
| real Train/Dev run | 该固定 scope 下 `DATA_VALIDATED_DIAGNOSTIC` |
| checkpoint CPU reload | artifact 可重新加载和执行 |
| Dev weak-label 指标 | 实验比较信号，不是 Gold/Test/production quality |
| Test/Gold/OOD | 当前未授权、未发生 |

### 10.2 最小充分证据

高质量 review 不等于列几十条问题。通常检查：

- `git diff`、target commit、import source path；
- 实际执行的 tests 和退出码；
- manifest 内容地址、每个文件 SHA/bytes；
- config、Schema/class order、data/split identity；
- checkpoint parameter identity 和 reload；
- 指标独立重算，而不是相信报告摘要；
- forbidden data/code path 静态与动态证据；
- 结果声明是否保持 diagnostic-only 和 `selected_candidate=false`。

finding 按对当前里程碑的真实影响分级。没有 P0 就不要写 P0。P1/P2/P3 不阻塞已满足
exit criteria 的阶段。

### 10.3 Reviewer 输出模板

```text
Active milestone:
Review target / commit / artifact:
Verdict:

P0 findings: none / concrete blocker
P1 findings:
P2 findings:
P3 findings:

Evidence independently executed:
Milestone/stage decision:
Deferred backlog:
Owner decision required: none / exactly one concrete decision
Next task and routed model:
```

## 11. Git、worktree 和训练操作规则

- 当前 owner-declared `main` 就是统一开发/训练主线。不要因为 GitHub 默认分支名不同而
  阻塞。
- 不创建额外 feature/bugfix branch 或 worktree，除非 owner 明确开启新的并行工作。
- 如果未来再次并行，所有接受的工作必须在下一次 fit 前合回一个 owner-declared branch，
  并在确认没有未合并工作后删除临时本地 branch/worktree。
- 远端历史分支和 remote cleanup 不作为本地训练 gate；不要为此要求 owner 配置 GitHub。
- 不 force-push、不覆盖已接受历史、不用 destructive reset/checkout 丢弃 owner 的改动。
- 当前 M2 runner 使用 stage-specific Train/Dev technical preflight。不要调用会打开 Test/
  Anchor/reference 的 full historical audit 来“增加可信度”。
- MPS-first 训练是当前事实；每个 checkpoint 必须有 CPU reload/inference smoke。不要把
  “CI 使用 CPU”误写成“真实训练只能 CPU”。
- 在训练进程运行时，不删除 output dir、cache、runtime，也不从相同 output dir 重启。
- 新 run 使用新的、空的 artifact 目录；保留失败 evidence，禁止就地修改已生成的
  content-addressed artifact。

## 12. 接手后的第一小时

### 12.1 先读

依次阅读：

1. `AGENTS.md`
2. 本文件
3. `docs/milestone-priority-and-agent-routing.md`
4. `docs/architecture-handoff-and-model-roadmap.md`
5. `docs/decisions/decision-log.md` 的 D-016 至最新条目
6. 当前阶段 runner、合同、tests 和 report

不要把 `architecture-handoff-and-model-roadmap.md` 中 2026-08-28 的“未开始 M2”快照
当成实时状态。

### 12.2 只读盘点

```bash
REPO=/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining
git -C "$REPO" status --short --branch
git -C "$REPO" log --oneline --decorate -8
git -C "$REPO" worktree list --porcelain
git -C "$REPO" branch -vv
ps -axo pid,etime,%cpu,%mem,command | rg 'semantic_model.encoder_m2'
find /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts -maxdepth 2 -type f | sort
```

若发现训练进程仍在运行，先观察输出增长并等待，不要重复启动。若 Git 或 artifact 已经
比本文更新，以 live facts 为准，并在 review 后回写本文件。

### 12.3 测试环境

- Classical engineering tests 通常使用 `.venv/`。
- Encoder 真实运行使用 `.encoder-venv/`；该 runtime 可能没有安装 pytest。
- 不要因为 `.encoder-venv/bin/python -m pytest` 报 `No module named pytest` 就判断源码
  测试失败；先使用项目已配置的 test runtime，或只做明确授权的依赖安装。
- Encoder cache 必须 `local_files_only=true`，除非 owner 新授权下载。

## 13. 什么时候必须找 owner

只有下列情况需要暂停并给 owner 一个清楚的选择：

- 要换模型、revision、license 或下载新 artifact；
- 要创建/修改 Gold、OOD 或人工 review 数据；
- 要首次打开 Test 或改变选择政策；
- 要启用云服务、外部 API、frontier LLM 或生产推理；
- 要改变当前冻结 Schema、label policy、split 或业务语义；
- 有可能删除未合并工作或不可恢复数据；
- 现有冻结实验梯度全部被证据拒绝，需要开一个新 candidate contract。

正常 bugfix、只读 review、已授权 RBT3 lineage 内的复核、测试和报告不需要 owner
重新批准。不要把技术判断包装成 owner 决策。

## 14. 常见失败模式

1. **把 runner 存在写成训练完成。** 必须看到 real artifact、三 seed 完整、aggregate、
   manifest 和 reload evidence。
2. **把平均提升写成晋级。** S2 已经证明这两者可能不同；critical-label regression
   可以拒绝 progression。
3. **为了通过而事后改 gate。** 这是 Dev overfitting，禁止。
4. **把 weak-label Dev 写成真实效果。** 它只能支持当前实验比较。
5. **把旧远端分支当阻塞。** 本地 owner-declared main 才是执行主线。
6. **重新添加单人项目的信任框架。** owner 已明确否决。
7. **不停加 P0。** P0 只给当前里程碑的真实 blocker；M2 质量优化一般是 P1。
8. **只读执行者报告。** Reviewer 必须重算最关键的 artifact/metric/reload 证据。
9. **把 OOD 塞进 UNKNOWN。** 语义错误，必须分开。
10. **让 Side 永久闲置或永久游离。** 并行只做文件所有权不冲突的只读分析/测试；产出
    一旦接受就正常合回主线，不要积累悬空 branch/worktree。

## 15. 每次交接必须更新的字段

在结束一次真实阶段时，至少更新：

```text
date/time:
owner-declared branch:
HEAD commit:
worktree/branch count:
active milestone and stage:
code complete?:
real run complete?:
artifact path/content address:
independent review complete?:
promotion/selection status:
highest open finding:
owner decision needed?:
next concrete task and priority/model:
```

最后原则：这个角色的价值不是让流程越来越复杂，而是持续把“事实、判断、下一步”缩短
成 owner 能快速确认的闭环。能训练时就训练，应该 review 时就复核，证据拒绝一个阶段时
就如实拒绝并进入最小诊断，不用形式上的完美拖延真实进展。
