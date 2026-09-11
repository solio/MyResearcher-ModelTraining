# M5 Human Train100 taxonomy evidence audit（v0.3 议案）

## 范围与方法

本审查只读取 finalized Human Train100：

`runs/feedback-loop-v3-finalized-20260910/human-train-100-seven-head.jsonl`

以及对应的三批 `selection-train.jsonl` 中的原始 `model_text`，按
`sample_id` 连接。输入为 100 个唯一样本、700 条 final head label，Schema
为 `semantic-schema-calibrated-v0.2.1`。本报告中的“证据”是对可修订人工标签
和原文的只读模式审查，不把现有标签当绝对真值，也不自动改标签。

本轮没有打开 Human Check20、Test、Gold 或 OOD，也没有运行模型或训练。

为避免把主观印象写成新标签，下面的子类型计数是一次可复核的保守编码：只对
文本中有直接线索的行计数；同一行可同时说明多个因素，但在每个小节内只计入
一个主模式。没有足够证据支持的细分类明确标为“不增加”。

## 五项观察的证据

### 1. HOPE 与 EXCITEMENT 可能是连续强度/时态维度

- 实际出现：`emotion_primary=HOPE` 16/100；`EXCITEMENT` 10/100，共 26 行。
- HOPE 的 16 个 sample_id：

  `fl1-1df0323667e82861d82d35123765950f542202019cfc925d948ddacf30fe4762`,
  `fl1-2043c961d262dc78a7e4042080caa8d383dbc7f26b6f2c3f1c2b8735326101e0`,
  `fl1-257df71258e0f3d3592d5e5e5a656a9f252c7d2479cbc4222f928509e8405242`,
  `fl1-468ef7b356a89d2898a01ad3f17f3740d890710df0f0fb0775ae53438bb4ced9`,
  `fl1-49379584ae0018a00e5d2249df6dfeaf5cf6520a1490493947ea41bd26b412b2`,
  `fl1-4c20fd33f44807cf3a909cb26eef5d48f5bb6d4d40f12490b46a6cf879303090`,
  `fl1-61ea4410413d3164fa80d18861e502f812f0309c29be7283df9956ff90ec48da`,
  `fl1-7d724ee939269f4588e667ebd200ca30d5360a2bf949ebf64dcac1a13b19da0e`,
  `fl1-80eded8088d595fbb02627662bca05194ff1a70d3619134a19acaffcedcc5c06`,
  `fl1-98820d49e89768d704b8b52c2bbf5c458820de11c82ffc357ac89eb3d73a7ee1`,
  `fl1-992398ce65c13d9df0d3f2ea4d7d5776de7d410cfdb70a25fa8800e71e213c7e`,
  `fl1-add21489439bf5ad38ac2f912cdf24ec0d35ef78e23c04a21438180d0fe1923b`,
  `fl1-bbf2d130c7fb3d70645f0e6435fe983326cb9cb3a3ed3179162f94ce2d299e9c`,
  `fl1-bc7ea932be33fd2190f648dbde518124abd985476f055ad6b998c8a2b14a582b`,
  `fl1-c00ec59f89abaf37f537db47215072854a135f0d6b6df73e8d98cb8b6ce02354`,
  `fl1-ed6dc2c5418a8d30188ded4066be6f9271b1c891604436134199f8e75cfe2385`。

- EXCITEMENT 的 10 个 sample_id：

  `fl1-09b3ce8c5004f454acd216cb251022843088107fc4ed400f3725e03a97ffd4a6`,
  `fl1-2084b95a10d447f5803789a699abacb66e8de064a6e7110aee85354370e2c68b`,
  `fl1-3b79ad7e81beaaa17d5e3a67d8eed1a4c890397809c3de59f0dc916781b41689`,
  `fl1-43e3eb044f431a4d8e07acc815742f78aaf3c358ccd7f2c7ed33b5e7e2f8488c`,
  `fl1-5fb36012861f099b450547057c305c0d535d3a38154dbb2864280faec62e79cc`,
  `fl1-6e6cb42336ca0a41ce2fb4106998affa6e446b78994a083773ee565341e4abaf`,
  `fl1-867de3cd97f1b7f875e1fad6d3a02e9f9218f882aa476accd34d3d162ba1c16b`,
  `fl1-c4634d39117cea79fdbcd5c0d3756fdd6b9d41d1974114d034fc854f4c733d1e`,
  `fl1-e3256e216ccc550a5c0d36799f323c6dbb52f0647fc5d700295b0f757622fadf`,
  `fl1-e377a2da5fb4022ae2459717b04e040ca025733e1236cc0aba5ddfdfe0a29828`。

- 说明性短文本证据：

  - `fl1-468ef7...` HOPE：`明天涨停板`（未来结果的轻量期待）。
  - `fl1-493795...` HOPE：`这一波可以震荡上行到24块...`（带条件和幅度的期待）。
  - `fl1-5fb360...` EXCITEMENT：`今天涨停了`（已发生强异动）。
  - `fl1-e377a2...` EXCITEMENT：`起飞前的最后一跌罢了，坐等起飞`（高唤醒、强叙事）。
  - `fl1-43e3eb...` EXCITEMENT：`今天这么硬？我要不要追呀28.8亿...`（异动和追入压力）。

- 判断：这是稳定重复的**连续维度/时态差异**，不是一次歧义。现有 HOPE 与
  EXCITEMENT 既有未来期待与已发生异动的差别，也有明显的唤醒强弱梯度；不宜
  直接合并为一个类别。
- v0.3：**值得进入**，但只增加一个可选因子 `expectation_profile`（例如
  `LOW_EXPECTANCY`、`HIGH_AROUSAL`、`REALIZED_MOVE`、`UNKNOWN`），保留
  `emotion_primary` 七头中的原有父层和 11 个细类，不新增一串平面情绪类。

### 2. “希望上涨”与“下跌后买入”被压在同一个 HOPE

- 实际出现：在 16 个 HOPE 行中，5 行有明确的向上预期，2 行有明确的
  “筑底/下跌后买入”语义；两组不重叠。这是方向因子证据，不是对标签的改写。
- 向上预期 5 个 sample_id：

  `fl1-468ef7b356a89d2898a01ad3f17f3740d890710df0f0fb0775ae53438bb4ced9`,
  `fl1-49379584ae0018a00e5d2249df6dfeaf5cf6520a1490493947ea41bd26b412b2`,
  `fl1-98820d49e89768d704b8b52c2bbf5c458820de11c82ffc357ac89eb3d73a7ee1`,
  `fl1-add21489439bf5ad38ac2f912cdf24ec0d35ef78e23c04a21438180d0fe1923b`,
  `fl1-bbf2d130c7fb3d70645f0e6435fe983326cb9cb3a3ed3179162f94ce2d299e9c`。
- 下跌后/底部入场 2 个 sample_id：

  `fl1-992398ce65c13d9df0d3f2ea4d7d5776de7d410cfdb70a25fa8800e71e213c7e`,
  `fl1-ed6dc2c5418a8d30188ded4066be6f9271b1c891604436134199f8e75cfe2385`。

- 说明性短文本证据：

  - `fl1-468ef7...`：`明天涨停板`（直接向上结果）。
  - `fl1-992398...`：`筑底中，12.5左右可以慢慢买入`（先筑底、再入场）。
  - `fl1-ed6dc2...`：`12块抄底，十三块钱卖就行了...十二块钱是铁底`（入场价与目标价同时出现）。
  - `fl1-493795...`：`这一波可以震荡上行到24块...`（上涨路径而非入场触发）。

- 判断：是稳定重复的**因子化方向/条件差异**；样本数足以证明现有单一
  `HOPE` 无法承载“结果方向”和“入场条件”，但还不足以建立更多细粒度情绪类。
- v0.3：**值得进入**，增加 masked `expectation_direction` 因子：
  `UPWARD_OUTCOME`、`DOWNTURN_THEN_ENTRY`、`CONDITIONAL_OR_UNCLEAR`、
  `UNKNOWN`。它与 `action_tendency` 正交：前者表达希望的结果/触发条件，后者
  表达作者交易倾向。没有明确方向的旧行对该因子使用 loss mask，不写成 UNKNOWN。

### 3. emotion_target=OTHER 混合外部对象、争论和叙事

- 实际出现：`emotion_target=OTHER` 共 7/100。
- 7 个 sample_id：

  `fl1-0a5ce13c511ab10c0b991de8b3873474ab7cb61760ef59e7b2e180e5041deee4`,
  `fl1-3a1a4ef001fae404968eee7835d82df3ae6e667662f55cb6e2f2e66a9da4e045`,
  `fl1-8426d932991fa98901a75724bbcc0b24028e5db908e9f524eaf19baa6f17314a`,
  `fl1-88e237564c3993eeed17cfef1264259e409655b9955260f3cf2bca21d955ab57`,
  `fl1-bc7ea932be33fd2190f648dbde518124abd985476f055ad6b998c8a2b14a582b`,
  `fl1-fc9497af634f86af097a1a22eedec7de5d697520e6d83978f4cfd383da93b304`,
  `fl1-fe022a79d0a253ec47af7339755fdb09378c26bf796924dc809bbfc7bdccab63`。
- 保守主模式计数：6 行是其他公司/同业比较或外部实体（中环、晶澳、通威、无锡尚德），
  1 行是事件/叙事对象（台风导致补货）。没有足够清晰的“其他散户”“他人操作”
  独立重复样本来支持五路子类。
- 说明性短文本证据：

  - `fl1-0a5ce1...`：`快了快了，快被中环撵上了`（同业竞争者）。
  - `fl1-8426d9...`：`今天又跑不过通威`（同业比较）。
  - `fl1-3a1a4e...`：`污蔑隆基的小作文到底是不是晶澳写的？...把敌人变成小弟`（争论/叙事对象）。
  - `fl1-88e237...`：`2013年无锡尚德太阳能就是这么破产的吧...`（历史外部公司）。
  - `fl1-bc7ea9...`：`台风把板子吹光光，补货补货`（事件叙事，不是单一公司/持仓）。

- 判断：这是**稳定的粗粒度外部指向模式，但内部子类型证据不足**；不能把
  `OTHER` 的 7 行直接拆成五个新类别。
- v0.3：**不增加 OTHER 子类**。保留 `OTHER` 父层；把 `target_role` 作为后续
  可选、masked 的候选因子，仅在未来出现明确人工证据时再启用。这样不会把本批
  7 行的同业样本伪装成完整的人/行为/论坛叙事分布。

### 4. emotion_target=MARKET 混合交易流、行业市场和大盘环境

- 实际出现：`emotion_target=MARKET` 共 16/100。
- 16 个 sample_id：

  `fl1-1df0323667e82861d82d35123765950f542202019cfc925d948ddacf30fe4762`,
  `fl1-2043c961d262dc78a7e4042080caa8d383dbc7f26b6f2c3f1c2b8735326101e0`,
  `fl1-221cc753018f40157cd29ad522508b6b662f19bf3ed0cd0fb34502358b996e26`,
  `fl1-257df71258e0f3d3592d5e5e5a656a9f252c7d2479cbc4222f928509e8405242`,
  `fl1-431a283b070743b880b1013221500b85fb7d6c1cc0efab94296cea2687cddf90`,
  `fl1-43e3eb044f431a4d8e07acc815742f78aaf3c358ccd7f2c7ed33b5e7e2f8488c`,
  `fl1-5953f840c293f99fa889e130269231aba53f439c527768eb41f26655b8bf2d5e`,
  `fl1-66b8aacc4d05e0011df82e30fa1386987fa19485e6708e2fc01dd0e94851c31b`,
  `fl1-7df03c69f415f9b02cc681d5999caa637b8d524a0a89fedd9f047609ee9430f1`,
  `fl1-8e519506224f2783f49eaa1dfd0a0d98899510b9db53605272ac46da507dfa90`,
  `fl1-98820d49e89768d704b8b52c2bbf5c458820de11c82ffc357ac89eb3d73a7ee1`,
  `fl1-99fcc69452566da871ddd22b5ce0812e939243dc5fb7fcee14935d36750ed096`,
  `fl1-a1376dacfc030bb27b5f82fccde3709aefd9318654928615be24745a134d2698`,
  `fl1-c43fea46b4f9ddcadbf0d6210ab4b8114e5a8ca7ac1d4f35c1e615b5a8d3514d`,
  `fl1-f2bb9c41addb1b396723b28ecd002ee625b6c7f291706ddb393ad272143b66c7`,
  `fl1-fecd37a32d0867de260ba9b87c4afab214b5d45119c411b83bf4c44e4e9b7c4e`。
- 保守主模式计数：6 行交易流/参与者（做 T、主力、散户、大资金、筹码），6 行行业/产品/政策，2 行大盘/市场状态，2 行叙事或跨公司语境。
  计数合计 16，说明 `MARKET` 不是单一粒度。
- 说明性短文本证据：

  - `fl1-2043c9...`：`做T的机会来了`（交易行为/盘面）。
  - `fl1-431a28...`：`主力终于准备拉一次，里面的散户还在那狂T...`（参与者和资金流）。
  - `fl1-221cc7...`：`光伏行业产能永远不会出清...`（行业供给/政策环境）。
  - `fl1-5953f8...`：`市场监管总局会议后多晶硅价格涨停，而组件销售价格不涨？`（政策、产品市场、价格并存）。
  - `fl1-99fcc6...`：`走了大盘要跳水`（大盘状态）。

- 判断：这是稳定重复的**父子/因子化粒度混合**，证据明显足以进入 v0.3。
- v0.3：保留 `emotion_target=MARKET` 父层，增加 masked `market_scope`：
  `TRADING_FLOW`、`SECTOR_INDUSTRY_PRODUCT`、`BROAD_MARKET_REGIME`、
  `NARRATIVE_OR_UNCLEAR`、`UNKNOWN`。老 `MARKET` 概率等于这些子类概率之和；
  没有新 scope 标注的历史样本对该因子使用 loss mask。

### 5. target_mode 只有范围，无法表达话语焦点

- 实际出现：`target_mode=ON_TARGET` 共 77/100，另有 `MARKET_GENERAL` 12、
  `CROSS_TARGET` 7、`UNKNOWN` 4。对 ON_TARGET 行按明确文本焦点保守编码，找到
  14 个不能只用“讨论该股票”解释的样本：6 个他人/受众/争论焦点，8 个作者
  交易行为或持仓行为焦点。
- 14 个 sample_id：

  `fl1-1c883707c8180802e0663babf8ca1a0ef5d8d1552e0bfb9d3c392ded480a5978`,
  `fl1-24f3bb6ff725be2ab18ae078430b76836700da21a57bc6ededbb3c14876c9187`,
  `fl1-3a1a4ef001fae404968eee7835d82df3ae6e667662f55cb6e2f2e66a9da4e045`,
  `fl1-8e519506224f2783f49eaa1dfd0a0d98899510b9db53605272ac46da507dfa90`,
  `fl1-9b2faf3703eb1ffc5a284d4b61f61d18de5fc7d594fc797020ab405309b9fe37`,
  `fl1-9fe4c215065d0b845802dedb122cf59ab97b1592ace7b9c77a807c42f919e65d`,
  `fl1-a00e3349a491d1784806cc76713f74ae68ffe13f9dc4f8e27c681acc89ac2170`,
  `fl1-a8a28625fa8cdb9a1da47559906fe5301cc93e0db9eb383bbf05ef7e8c5be5e5`,
  `fl1-c5441a122a1a19e3cdd09a4aeac4820334ba97387bfaf725296973b9b640346d`,
  `fl1-d385cbb9282f8b2f5d3e611896a7a174623694d6e441869aba91c3605fa851d6`,
  `fl1-e9c92a0c3329fce01fa2d62c5a0ee36e9bc1d183d0ace6c185a1bf53c3079148`,
  `fl1-ed6dc2c5418a8d30188ded4066be6f9271b1c891604436134199f8e75cfe2385`,
  `fl1-f2bb9c41addb1b396723b28ecd002ee625b6c7f291706ddb393ad272143b66c7`,
  `fl1-f3f67770720e10b0839262abacedd5b3f76d9b155bf78cac4fcef3de8622361c`。

- 说明性短文本证据：

  - `fl1-1c8837...`：`看别人吃肉，我重仓绿毛龟...`（他人表现 + 作者持仓）。
  - `fl1-9fe4c2...`：`兄弟们！让我看看你们亏了多少？90万股东...`（受众/其他股东）。
  - `fl1-3a1a4e...`：`污蔑隆基的小作文到底是不是晶澳写的？...`（争论对象/论坛叙事）。
  - `fl1-d385cb...`：`冲高卖`（作者交易行为，不是对象范围）。
  - `fl1-a8a286...`：`47.5清了。还剩100股留个纪念。出来的钱来加隆基了。`（持仓处置与再配置）。

- 判断：这是稳定重复的**正交话语焦点维度**，不是偶发歧义。`target_mode`
  的四个范围类不能表达作者观点、作者动作、他人/受众或论坛叙事之间的差别。
- v0.3：**值得进入**，保留七头中的 `target_mode` 父层，增加 masked
  `discourse_focus`：`ASSET_SCOPE`、`AUTHOR_VIEWPOINT`、`AUTHOR_ACTION_OR_POSITION`、
  `OTHER_PERSON_OR_AUDIENCE`、`FORUM_NARRATIVE_OR_DEBATE`、`UNKNOWN`。旧
  `target_mode` 概率由其子类求和；没有新焦点标注的历史样本只训练旧父层，
  新因子使用 loss mask。

## 最小 Schema v0.3 候选（不在本轮实施）

### 保留七个顶层业务 head

`target_mode`、`stance`、`emotion_primary`、`emotion_target`、
`action_tendency`、`context_dependency`、`reasoning_tags` 仍是七个顶层输出。
v0.3 只在其中增加可选的内部因子/父子报告维度，不删除旧 head，不改变旧
`schema/semantic-schema-calibrated-v0.2.1.json`。

### 建议的新增因子

| 顶层 head | v0.3 最小因子 | 目的 | 当前证据结论 |
|---|---|---|---|
| `emotion_primary` | `expectation_profile` + `expectation_direction` | 区分期待的唤醒/时态，以及向上结果 vs 下跌后入场 | GO，26 行 HOPE/EXCITEMENT；7 行明确方向/入场 |
| `emotion_target` | `market_scope` | 拆开交易流、行业/产品市场、大盘状态 | GO，16 行中出现 6/6/2/2 主模式 |
| `emotion_target` | `target_role`（仅候选、masked） | 未来区分 OTHER 的人/同业/叙事 | **不增加当前子类**，7 行不足 |
| `target_mode` | `discourse_focus` | 把资产范围与作者观点、动作、他人/论坛焦点正交化 | GO，14 个清晰例子 |

`expectation_profile` 与 `expectation_direction` 可以在首个 v0.3 实现中合并为
一个小因子，以控制标注成本；不应扩展为更多平面情绪类别。

### 旧 v0.2.1 到 v0.3 的父子映射与损失

1. v0.2.1 的 11 个 `emotion_primary`、7 个 `emotion_target`、4 个
   `target_mode`、其余 head 的原类别都保留为父层/兼容层。
2. 对真正存在子类的父层，父层概率定义为子类概率和。例如：
   `MARKET = sum(market_scope children)`；`ON_TARGET = sum(discourse_focus
   children that are asset-scoped)`。训练旧粗标签时使用该求和概率的 NLL/BCE，
   因而旧标签仍可贡献监督，不需要猜一个子类。
3. 新因子没有人工标注的行使用 `loss_mask=0`。不把未标注写成 `UNKNOWN`，也不
   把 `UNKNOWN` 当成缺失值。
4. 评估同时输出严格细类指标（每个新子类的 F1/support）和父层方向指标
   （按旧 v0.2.1 类别合并后的 Macro-F1/Accuracy），避免因细类稀疏而掩盖父层
   回归。
5. 当前 Encoder 的共享 tokenizer、底部 embeddings/前 11 个 block、各 head
   的已训练 specialist block 和原七头输出参数均可继续使用。新增因子/子类
   输出层需要新初始化；旧父层 logits 可通过概率求和作为兼容路径，不能把旧
   checkpoint 冒充已训练 v0.3。

### 不进入本次迁移的内容

- 不修改 v0.2.1 Schema、标签文件或已有 artifact。
- 不把本报告的证据编码反写为人工标签。
- 不新增 `OTHER` 的五路平面子类；先保留父层并积累针对性样本。
- 不改 `action_tendency`、`context_dependency`、`reasoning_tags` 的现有类别
  和训练边界。

## 结论与 M5 决策

**NO-GO：应先做 Schema v0.3 最小迁移。**

理由不是“已有七头完全失效”，而是 Human Train100 已经重复显示三类结构性
信息缺口：

1. 情绪需要把结果方向/入场条件与强度/时态分开；
2. `emotion_target=MARKET` 需要交易流 vs 行业/产品 vs 大盘的层次；
3. `target_mode` 需要与资产范围正交的话语焦点。

在这些维度未至少完成最小 masked 因子迁移前，直接按 v0.2.1 跑 M5 会把
“训练指标变高”与“吸收真实语义”混在一起，尤其会继续放大 OTHER/MARKET
和作者动作的混淆。建议先做一次显式、可回滚的 v0.3 schema 迁移设计审查，
再决定是否允许新的 M5 fit；本轮没有训练，也没有替换现有候选。
