# MyResearcher 语义学生模型训练与验收规格 v0.1

> 交付对象：项目经理、模型开发、后端开发、测试  
> 文档状态：实施规格草案，可进入代码库评审  
> 目标：把当前一次性的教师标注与诊断训练，沉淀为可重复执行、可测试、可发布的本地模型工程

## 1. 先纠正当前状态

当前已经训练出的 `TF-IDF + Logistic Regression` 多任务模型只是 **diagnostic baseline**，用途是验证标签能否被普通模型稳定学习、暴露标签漂移和字段混淆。

它不是生产模型，不能直接用于 49,054 条全量推理，也不能作为股票买卖模型。

当前已经确认：

- 3,000 条教师标签已完成，但只能定义为 **weak labels**，不能称为 3,000 条 Gold；
- 其中 21 条存在冻结 Evidence 协议违规，已经隔离；
- 可进入弱监督实验的候选数据为 2,979 条；
- 已按时间切分为 Train 1,822 / Dev 448 / Test 467，并在两个边界设置共 242 条 embargo；
- 部分生产批次存在 `CALM`、`WATCH` 等标签风格漂移；
- 诊断基线在 50 条独立锚点上的 Macro-F1：
  - `target_mode = 0.349`
  - `stance = 0.415`
  - `emotion_primary = 0.256`
  - `emotion_target = 0.209`
  - `action_tendency = 0.111`
  - `context_dependency = 0.378`
  - `reasoning_tags micro-F1 = 0.507`

因此，开发任务不是“把现有 joblib 文件接入生产”，而是实现下面定义的可复现训练、评估和发布流水线。

---

## 2. 我们最终要什么效果

### 2.1 模型的业务职责

模型输入一条散户发言及其最小板块上下文，输出这条发言中可观察到的：

1. 讨论对象；
2. 对未来方向的立场；
3. 显式情绪及其对象；
4. 作者自己的交易行动倾向；
5. 推理依据；
6. 对外部上下文的依赖程度。

它测量的是：

> **belief–emotion–action state**

它不负责：

- 预测明天涨跌；
- 给出买卖建议；
- 直接输出股票级情绪总分；
- 直接判断七种群体状态；
- 根据外部事实纠正作者；
- 把没有表达的信息补全出来。

### 2.2 系统边界

```text
帖子文本 + board context
        ↓
帖子级语义学生模型
        ↓
字段标签 + 概率 + abstain 信息
        ↓
作者去重与权重处理
        ↓
股票 × 日 / 5日聚合
        ↓
分布、分歧、迁移、尾部状态
        ↓
七种群体状态与事件解释
```

七种群体状态属于下游聚合层，不能在这一版中直接蒸馏成帖子级类别。

---

## 3. 输入与输出契约

### 3.1 输入

最小输入：

```json
{
  "sample_id": "...",
  "stock_code": "601012",
  "stock_name": "隆基绿能",
  "model_text": "..."
}
```

推理时允许使用 `stock_code`、`stock_name` 作为 board context，但不得联网补充新闻、价格或公司事实。

训练、评估和推理必须共用同一个文本构造函数，禁止三处各写一套 preprocessing。

### 3.2 V1 必须预测的字段

| 字段 | 任务类型 | 类别 |
|---|---|---|
| `target_mode` | 单标签多分类 | ON_TARGET / CROSS_TARGET / MARKET_GENERAL / UNKNOWN |
| `stance` | 单标签多分类 | BULL / BEAR / NEUTRAL / MIXED / UNKNOWN |
| `emotion_primary` | 单标签多分类 | FEAR / ANXIETY / ANGER / FRUSTRATION / REGRET / HOPE / EXCITEMENT / FOMO / CALM / NONE_EXPLICIT / UNKNOWN |
| `emotion_target` | 单标签多分类 | PRICE / POSITION / COMPANY / MARKET / OTHER / NOT_APPLICABLE / UNKNOWN |
| `action_tendency` | 单标签多分类 | BUY / ADD / HOLD / DO_T / REDUCE / SELL / WATCH / NO_ACTION_SIGNAL / UNKNOWN |
| `reasoning_tags` | 多标签分类 | 冻结 Schema 中的 15 个标签 |
| `context_dependency` | 单标签多分类 | SELF_CONTAINED / PARTIAL_CONTEXT / EXTERNAL_CONTEXT_REQUIRED / UNKNOWN |

### 3.3 V1 不训练的字段

- `label_confidence`：作为训练权重、样本筛选和 QA 信息，不作为预测目标；
- `evidence_spans`：保留用于教师标签审计，V1 学生模型不生成；
- `ambiguity_note`：不做文本生成任务；
- 七种群体状态：由下游聚合计算。

### 3.4 推理返回

除最终类别外，每个 head 必须返回类别概率。服务层必须支持低置信度拒答：

```json
{
  "schema_version": "semantic-schema-calibrated-v0.2.1",
  "model_version": "semantic-student-v0.1.0",
  "predictions": {
    "stance": {
      "label": "UNKNOWN",
      "confidence": 0.43,
      "abstained": true,
      "probabilities": {}
    }
  }
}
```

拒答不能偷偷映射成 `NEUTRAL`、`NONE_EXPLICIT` 或 `NO_ACTION_SIGNAL`。具体阈值从 Dev 集校准并随模型版本保存，不能硬编码在业务调用方。

---

## 4. 必须学对的语义边界

以下规则比总体 Accuracy 更重要，应建立独立挑战集和回归测试：

1. **UNKNOWN 不等于 NEUTRAL**：文本不足、疑问、黑话和残句不得被强行判成中性。
2. **无显式情绪不等于 CALM**：只有明确表达平静、从容时才允许 CALM。
3. **动作必须属于作者本人**：描述别人买卖、建议别人买卖、引用他人动作，不能标成作者 BUY/SELL。
4. **否定不能反转**：`不割肉` 不是 SELL，`不追高` 不是 BUY。
5. **条件动作不等于已经行动**：`如果跌到X再买` 与明确 BUY/ADD 必须区分。
6. **愿望不等于行动**：`快涨吧` 不能推出 HOLD/BUY。
7. **WATCH 不能泛化为一切不确定句**：只有等待、观察或犹豫且确实涉及作者决策时才使用。
8. **Stance、Emotion、Action 相互独立**：看空不自动等于恐惧或卖出；买入不自动等于兴奋或看多。
9. **Evidence 约束由数据校验器执行**：UNKNOWN / NONE_EXPLICIT / NO_ACTION_SIGNAL / NO_REASON_GIVEN 不得附相应 Evidence；现有生产校验器必须补上该规则。

---

## 5. 数据角色与版本管理

### 5.1 数据分层

| 数据 | 用途 | 是否允许训练 | 是否允许最终评估 |
|---|---|---:|---:|
| 原始帖子 | 可复现输入 | 否 | 否 |
| 3,000 条冻结教师标签 | 溯源、审计 | 否，原始版只读 | 否 |
| 2,979 条 weak-label 候选 | 弱监督训练 | 是，需权重 | 否，不得冒充 Gold |
| 21 条隔离集 | 协议与语义修复 | 修复确认前否 | 可作回归样例 |
| 50 条现有锚点 | 固定回归与早期诊断 | 否 | 是，但样本太少，不能单独批准生产 |
| 新增独立 adjudicated Gold | 正式验收 | 否 | 是 |
| Challenge Sets | 关键边界验收 | 否 | 是 |

现有 50 条锚点中只有 11 条是人工确认，39 条属于 Expert weak Gold。报告必须如实披露，不能写成“50 条纯人工 Gold”。

### 5.2 切分规则

禁止逐条随机切分。至少按以下信息分组：

```text
股票 × 时间区间 × 市场事件
```

并尽量隔离作者。时间边界保留 embargo，避免同一事件或近重复发言跨 Train/Dev/Test。

当前 v0.3.5 切分可以作为第一版固定基线：

- Train：1,822
- Dev：448
- Test：467
- Embargo：242

任何重新切分必须生成 manifest，记录样本 ID、分组键、随机种子、输入文件 SHA-256 和切分理由。

### 5.3 弱标签权重

训练代码必须支持按“字段”设置样本权重，而不是一条样本只有一个总权重：

- `HIGH / MEDIUM / LOW` 提供基础权重；
- 已确认发生漂移的批次，对受影响字段单独降权；
- Gate120 可以比普通 weak labels 略高权重，但仍不是人工 Gold；
- 隔离样本在修复确认前权重为 0；
- 权重策略必须写入版本化配置，不能散落在训练脚本里。

---

## 6. 模型与训练方案

### 6.1 实施顺序

开发团队应实现同一套接口下的两个模型：

1. **Classical baseline**：保留字符/词 TF-IDF + 线性分类器，用于快速回归和识别数据问题；
2. **Encoder student**：中文预训练 Encoder + 多任务分类 heads，作为候选生产模型。

不要求项目经理预先锁死具体 Encoder。首轮可在 `MacBERT-base`、`Chinese RoBERTa-wwm-ext` 或团队已有等价中文 Encoder 中选择一个，模型选择必须通过同一数据切分和同一评估脚本比较，而不是凭模型名决定。

### 6.2 推荐网络结构

```text
Chinese Encoder
      ├── target_mode head
      ├── stance head
      ├── emotion_primary head
      ├── emotion_target head
      ├── action_tendency head
      ├── context_dependency head
      └── reasoning_tags multi-label head
```

- 六个单标签 head 使用 cross-entropy；
- `reasoning_tags` 使用 BCEWithLogitsLoss；
- 支持 class weights 或 focal loss，但必须通过 Dev 消融确定，不得只因类别不平衡就默认开启；
- 每个字段分别应用 sample weight；
- Dev 用于 early stopping、阈值选择和概率校准；
- Test 与独立 Gold 只在候选版本冻结后评估；
- 固定 random seed，并至少运行 3 个 seed，报告均值与波动，避免单次偶然结果。

### 6.3 训练阶段

建议按下面顺序推进，而不是一上来追求单个大模型分数：

#### Stage A：数据与基线复现

- 一条命令重建数据 manifest；
- 一条命令复现 TF-IDF baseline；
- 指标与 v0.3.5 报告在容差内一致；
- 补齐 21 条 Evidence 违规校验和语义隔离逻辑。

#### Stage B：Encoder 初训

- 使用 2,979 条 weak labels；
- 按字段权重降低漂移批次影响；
- 输出 Dev/Test/Anchor 的字段级与类别级指标；
- 输出混淆矩阵、置信度分布和错误样本。

#### Stage C：定向 Gold 与再训练

- 从 Emotion、Action、Stance 的高不确定/高分歧样本中抽取 100–200 条；
- 人工或双 Expert adjudication，形成独立 Gold；
- 优先覆盖作者动作归属、CALM/NONE_EXPLICIT、WATCH/NO_ACTION、UNKNOWN/NEUTRAL 和否定/条件句；
- Gold 只用于验收和阈值选择，不反复查看后再改训练数据；需要用于训练时必须另建下一版本，并保留新的未见测试集。

#### Stage D：候选生产模型

- 通过帖子级门槛后导出模型；
- 在 16 只股票上生成每日/5日状态卡；
- 用 10 个真实市场事件验证群体状态是否可解释；
- 通过后才允许对 49,054 条做全量推理。

---

## 7. 验收标准

### 7.1 工程门槛

必须全部通过：

- 从冻结输入到训练、评估、导出可由命令行完整复现；
- 训练集、Dev、Test、Gold 无 sample_id 重叠；
- 无事件边界泄漏；
- Schema、类别顺序、预处理、阈值、权重和模型版本均被保存；
- 相同 seed、依赖和输入哈希下，指标在声明容差内复现；
- 推理输出 100% 通过 JSON Schema；
- CPU 推理路径可用；MPS/CUDA 只是加速路径，不能成为唯一可运行路径；
- 模型与代码均不得执行外部搜索。

### 7.2 帖子级质量门槛

下列数字是 v0.1 的**候选门槛**，应在独立 Gold 达到至少 100–200 条并覆盖关键类别后冻结：

| 字段 | 候选门槛 |
|---|---:|
| target_mode Macro-F1 | ≥ 0.80 |
| stance Macro-F1 | ≥ 0.70 |
| emotion_primary Macro-F1 | ≥ 0.60 |
| emotion_target Macro-F1 | ≥ 0.65 |
| action_tendency Macro-F1 | ≥ 0.65 |
| context_dependency Macro-F1 | ≥ 0.70 |
| reasoning_tags micro-F1 | ≥ 0.70 |

Macro-F1 不能单独决定上线。还必须满足关键错误门槛：

- 把“别人/被建议者的动作”误判为作者 BUY/SELL 的比例 ≤ 5%；
- `NO_ACTION_SIGNAL` 被误报为明确交易动作的比例 ≤ 5%；
- 无显式平静证据却预测 `CALM` 的比例 ≤ 2%；
- 不得把低置信度拒答自动改成 `NEUTRAL`；
- BUY / ADD / REDUCE / SELL 等稀有类若 Gold support < 20，只报告结果，不得宣称该类已通过；
- FOMO 等极稀有类若样本不足，必须保持实验状态或合并到人工复核队列，不能用总体指标掩盖。

### 7.3 群体级门槛

帖子级通过后，还要完成：

- 16 只股票的日/5日状态卡；
- 作者去重后与未去重结果对比；
- 至少 10 个真实市场事件的盲态解释审查；
- 检查状态是否主要由发帖量、单个高频作者或某一批模型错误驱动；
- 不以“预测次日涨跌准确率”作为首要验收标准。

---

## 8. 代码库必须沉淀的内容

建议目录，允许开发团队按现有仓库结构调整，但能力不能缺失：

```text
semantic_model/
├── README.md
├── pyproject.toml / requirements.lock
├── configs/
│   ├── baseline_v0.3.5.yaml
│   └── encoder_v0.1.yaml
├── schema/
│   └── semantic-schema-calibrated-v0.2.1.json
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── weighting.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── export.py
│   └── infer.py
├── tests/
│   ├── test_schema.py
│   ├── test_split_leakage.py
│   ├── test_evidence_dependencies.py
│   ├── test_preprocessing_parity.py
│   └── challenge_cases.jsonl
├── manifests/
├── reports/
└── MODEL_CARD.md
```

最低命令接口：

```bash
python -m semantic_model.prepare --config configs/encoder_v0.1.yaml
python -m semantic_model.train --config configs/encoder_v0.1.yaml
python -m semantic_model.evaluate --run <run_dir>
python -m semantic_model.export --run <run_dir>
python -m semantic_model.infer --model <model_dir> --input <jsonl> --output <jsonl>
```

每次训练必须输出：

- 数据与配置 manifest；
- 输入文件哈希；
- git commit；
- 依赖版本；
- seed；
- 每字段 loss 与指标；
- 每类别 support、precision、recall、F1；
- 混淆矩阵；
- 阈值和校准参数；
- 错误样本清单；
- 模型卡；
- 可部署模型产物。

大体积原始数据和模型权重不应直接进入普通 Git history。代码库保存 manifest、配置、Schema、评估报告和产物索引；权重使用团队已有 artifact storage、release 或 Git LFS/DVC 方案。

---

## 9. 项目经理的拆分建议

### Milestone 1：训练工程可复现

验收物：数据校验器、固定切分、TF-IDF baseline、统一评估器、CLI、测试。  
完成定义：在新环境按 README 可复现 v0.3.5 诊断结果，不依赖 ChatGPT 临时执行。

### Milestone 2：Encoder 候选模型

验收物：多任务 Encoder、3-seed 报告、概率校准、abstention、模型导出。  
完成定义：在固定 Test 与现有 Anchor 上显著超过 baseline，并且关键错误没有恶化。

### Milestone 3：独立 Gold 与语义门槛

验收物：100–200 条独立 adjudicated Gold、Challenge Sets、最终门槛报告。  
完成定义：达到字段指标与关键错误门槛；不足的稀有类明确保持实验状态。

### Milestone 4：下游状态卡验证

验收物：16 股日/5日状态卡、10 个事件审查、作者去重敏感性报告。  
完成定义：证明模型输出能支撑群体状态测量，再批准 49,054 条全量推理。

---

## 10. 后续分工

### 项目经理 / 开发团队负责

- 把训练、评估、导出、推理和回归测试沉淀进代码库；
- 管理运行环境、依赖、模型产物和版本；
- 每次按同一命令重训，而不是依赖某次对话；
- 通过 CI 做 Schema、泄漏、挑战集和推理契约测试；
- 根据验收报告决定是否发布。

### 语义研究 / Expert 负责

- 维护 taxonomy 和冻结 Schema；
- 审查高风险错误与难例；
- 对新增 Gold 做 adjudication；
- 解释字段混淆和群体状态失真；
- 提议标签或聚合规则变更，但不绕过代码库直接替换生产模型。

### ChatGPT 后续正确角色

- 可以帮助评审训练方案、错误样本和模型报告；
- 可以提出下一轮 active learning 样本；
- 可以在开发前做一次性诊断实验；
- 不应成为日常训练执行器、模型注册表或唯一可复现环境。

最终目标是：

> 新标签或新数据到达后，开发团队运行标准命令，自动得到训练结果、验收报告和版本化模型；只有遇到语义边界问题，才把少量高价值样本交给 Expert 复核。

