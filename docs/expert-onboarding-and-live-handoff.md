# MyResearcher-ModelTraining Expert 入职与实时交接

状态：`ACTIVE_OPERATIONAL_HANDOFF`

最后核对：2026-09-01 CST（M2 训练前仓库收口）

适用对象：临时或长期接替当前技术负责、任务编排和 review 职责的 Expert / Agent。

这是一份可执行的入职文档。它回答四件事：当前角色负责什么、项目真实做到哪一步、接手后如何判断产出，以及下一步应推进什么。架构历史仍以
[`architecture-handoff-and-model-roadmap.md`](architecture-handoff-and-model-roadmap.md)
为准，里程碑和优先级仍以
[`milestone-priority-and-agent-routing.md`](milestone-priority-and-agent-routing.md)
为准；但两份旧文档中的时间点快照可能落后于本文件和实时 Git/artifact 状态。

## 1. 一分钟接手结论

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
- 本轮诊断完成后停止，不设计或训练新模型；如需进入新的 M2 实验，必须另行明确范围。
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
S3 fit implementation HEAD: 94295b1ab69d2501f9775c3e0cbecb17be67dfb8
source HEAD operationally observed when S2 ran: fd504719dcd8eb05ea319c0dc1863f7aa4c794eb
local worktrees: 2 (primary `main` working directory plus one external clean side reference; only primary was used for S3)
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

## 8. 当前实时阶段：M2-S3

状态：`M2_S3_DIAGNOSTIC_COMPLETED; SELECTED_CANDIDATE_FALSE`

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

S2 独立 review 已完成，结论为 `ACCEPT_EVIDENCE_BUT_DO_NOT_PROMOTE`；随后预声明的
S3 diagnostic 也已完成，结论为 `M2_S3_DIAGNOSTIC_COMPLETED` 且
`selected_candidate=false`。本轮停止，不启动新的模型或阶段。

S3 使用固定 RBT3 cache、Train 1,822/Dev 448、冻结 Encoder、六个 run units、每个单头
primary Macro-F1 early stopping（reasoning 同报 Micro-F1/exact-set accuracy），并与
matching-seed S1 对比 critical labels；这些证据已固化在上述 immutable artifact。

当前不做：打开 Test、创建 Gold/OOD、下载新模型、full unfreeze、调用 LLM、生产推理、
49,054 条数据推理、云训练、GUI/dashboard。

下一步是 owner 对 `RBT3_REASONING_CORRECTIVE_V1` 是否授权的单一决定。未收到该决定前，
不得启动 corrective fit；本次 Side trigger-data 诊断只提供 Train/Dev weak-label
背景，并不构成训练授权或 selected-candidate 结论。

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
