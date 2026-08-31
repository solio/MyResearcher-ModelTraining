# 里程碑、优先级与 Agent 路由

状态：`ACTIVE_OWNER_EXECUTION_POLICY`

Owner 对齐日期：2026-08-28

适用对象：本仓库的需求方、开发者、reviewer，以及所有后续 AI agent。

## 1. 这份文档解决什么问题

本项目中的任务不能只按“技术上是否重要”排序。每项工作必须先回答：

1. 当前正在交付哪个里程碑；
2. 它是否阻断当前里程碑；
3. 它属于重要/紧急的哪个象限；
4. 它必须串行，还是已经可以与主线并行；
5. 应使用哪个 agent 模型与 reasoning 档位；
6. 本轮做到什么证据就停止，哪些优化明确留到下一阶段。

`docs/architecture-handoff-and-model-roadmap.md` 继续定义模型角色和长期架构；
本文是执行顺序、review 分级、并行门禁和 agent 路由的权威来源。两者冲突时：

- 数据、Schema、模型角色和安全边界以架构/数据合同为准；
- 当前先做什么、是否阻断、何时并行、使用什么模型，以本文为准；
- owner 的最新明确指令优先，并应随后回写本文或 decision log。

## 2. 核心交付原则

### 2.1 当前第一目标是“训练闭环真实跑通”

当前主线不是继续堆规划文档，也不是先把所有质量问题、Gold、OOD、历史精确复现和生产工具一次性做完。当前优先目标是：

> 在通过不可变数据审计后，让一个经 owner 授权的中文预训练 Encoder 完成可重复的最小七头训练、保存、评估和重新加载推理闭环。

第一轮允许存在已知、已记录、不会破坏数据/标签/评估身份的瑕疵。第一轮不要求：

- 同时比较所有 Encoder 候选；
- 三个以上随机种子；
- 完整 Gold/OOD 生产验收；
- 最优 F1；
- 完整性能优化；
- 历史 v0.3.5 在 Linux reference environment 中的精确复现；
- 49,054 条生产推理；
- LLM review 自动化；
- GUI、dashboard 或服务化。

但“先跑起来”不能突破以下底线：

- 不能使用错误、伪造或角色混淆的数据；
- 不能绕过 canonical audit；
- 不能静默改变 Schema/class order；
- 不能重新随机切分或污染 Test/Anchor；
- 不能让 train/inference 使用不同 input builder；
- 不能下载未授权或身份不明确的模型；
- 不能把 weak-label 指标写成 Gold/production 指标；
- 不能让一个 P0 根因被 P1/P2 优化掩盖。

### 2.2 先完成里程碑，再展开优化

每个里程碑只保留最少的 exit gate。只要 P0 全部关闭且 exit evidence 完整，就进入下一里程碑；P1/P2/P3 不自动阻止阶段性交付。

当关键里程碑完成后，互不修改同一冻结合同、同一数据或同一代码所有权的优化任务可以并行。并行的目的，是缩短反馈时间，不是同时改变所有变量。

### 2.3 优先级是相对当前里程碑的，不是永久标签

同一问题会随里程碑变化而升级或降级。例如：

- 没有独立 Gold 不阻止第一次 Encoder 训练，因此在“首次跑通”阶段不是 P0；
- 到生产候选验收阶段，没有独立 Gold 会直接阻断验收，因此升级为 P0；
- CPU latency 优化在首次训练阶段通常是 P2；到生产上线门禁时可能成为 P0；
- exact historical reproduction 对历史治理重要，但不阻止新 Encoder 首跑，所以当前不是全局 P0。

任何 reviewer 都必须先写出 active milestone，再给 P0–P3；不得脱离阶段给一个永久等级。

## 3. P0–P3 的项目定义

### P0 — 阻断目标或必须串行

满足任一条件即可判为 P0：

- 当前里程碑无法开始或无法结束；
- 会造成数据、Schema、split、label、model artifact 或 provenance 错误；
- 会使训练/评估结果失去身份、不可复现或不可解释；
- 两个任务不可安全并行，必须先冻结共同合同；
- 缺失运行所需源码、依赖授权、模型身份、硬件资源或 owner 决策；
- 会造成 Test/Gold 泄漏、生产误授权、破坏性写入或不可恢复结果；
- 当前代码在干净 checkout 中不能执行关键路径；
- review 发现的事实足以推翻当前里程碑结论。

P0 的处理规则：

- 使用 `Terra Max`；
- 默认进入重要且紧急象限；
- 必须先关闭或由 owner 明确接受 blocker；
- 不得一边保留未解决 P0，一边用 P1/P2 结果宣布里程碑完成；
- prompt 必须包含可验证的 exit criteria 和禁止扩大范围的 hard boundaries。

项目示例：

- `src/semantic_model/models/` 被 `.gitignore` 吞掉，干净 checkout 无法 import；
- 未冻结 tokenizer special-token/token-type 策略却开始 Encoder 训练；
- 模型 revision 未确定或下载未授权；
- canonical audit 不通过仍进入 fit；
- Train/Dev/Test/Anchor 身份泄漏；
- 同一共享数据合同被两个并行 agent 同时修改。

### P1 — 对数据或模型效果有明显体感收益

P1 不一定阻止当前最小闭环，但修复后通常能明显改善模型质量、覆盖率、稳定性或用户可感知结果，例如：

- 更合适的 Encoder 候选或微调阶段；
- rare-class、class imbalance、negative transfer 的处理；
- 更高质量的独立 adjudicated Gold；
- tokenizer length/truncation 的实测优化；
- 三种以上 seeds 和稳定性选择；
- single-task 与 shared multi-task 对比；
- OOD/abstention 使高置信错误明显下降；
- 对关键语义边界的定向数据与误差修复。

P1 的处理规则：

- 使用 `Luna Max`；
- 通常属于重要但不紧急；
- 只有当 active milestone 的 exit gate 明确要求该质量项时，才升级为 P0；
- 首次跑通阶段允许记录进 backlog，不应无限扩张首跑范围。

### P2 — 影响体验、流程或工具

P2 改善开发、review、审计、部署、可观察性或操作体验，但通常不直接带来明显模型质量收益，例如：

- exact-environment preflight 和历史复现执行工具；
- CI、报告自动化、命令封装、错误信息和诊断输出；
- artifact 浏览、dashboard、运行索引；
- 非关键性能优化、缓存、批处理便利性；
- 更完整的 tamper tests，但现有关键路径已有充分 gate；
- 文档导航、交接工具和自动清单。

P2 的处理规则：

- 使用 `Luna Max`；
- 一般不阻止当前里程碑；
- 在 P0 完成、文件所有权不冲突时可以并行；
- 如果某个工具缺失导致关键路径根本不能运行，它按实际影响升级为 P0。

### P3 — 其他低风险改进

P3 包括：

- 不改变语义的措辞、排版、命名和注释；
- 不影响执行的轻微 whitespace 或文档历史表达；
- 可读性、非关键 refactor、低价值便利项；
- 尚无用户或工程证据支持的“也许更漂亮”改动。

P3 的处理规则：

- 使用 `Luna Max`；
- 不得阻止 milestone exit；
- 可以批量处理，但不得和 P0 修复混成无法 cherry-pick 的提交；
- 没有空闲并行容量时直接留 backlog。

## 4. 高质量 review 合同

Review 必须高质量，但“高质量”不等于发现越多问题越好。高质量 review 必须：

1. 先声明 active milestone 和本次 review 的目标；
2. 检查真实 Git tree、工作区、分支、提交和远端状态；
3. 对关键结论运行最小充分的真实命令，而不是只读报告；
4. 区分 synthetic tests、真实数据 audit、真实训练和 production evidence；
5. 验证 import 实际来自被 review 的 worktree；
6. 对外部模型/许可证/revision 等可变事实使用官方来源核验；
7. 按 P0–P3 排序，每个 finding 说明对当前里程碑的具体影响；
8. 明确 `GO`、`CONDITIONAL GO`、`CHANGES REQUESTED` 或 `BLOCKED`；
9. P0 必须提供复现证据和关闭条件；
10. P1/P2/P3 不得通过升级措辞伪装成 P0；
11. 已知瑕疵如果不阻止当前 exit gate，应进入 backlog，而不是反复打回；
12. 不因代码是“pre-existing”就忽略会使当前分支不可运行的 P0。

Review 输出最少包含：

```text
Active milestone
Review target / commit
Verdict
P0 findings
P1 findings
P2 findings
P3 findings
Commands/evidence actually verified
Milestone exit decision
Deferred backlog
Next prompt model routing
```

### 4.1 Review 决策规则

| 最高 finding | 默认结论 | 是否可进入下一里程碑 |
| --- | --- | --- |
| P0 | `CHANGES_REQUESTED` 或 `BLOCKED` | 否 |
| P1 | `CONDITIONAL_GO_WITH_QUALITY_BACKLOG` | 当前 exit gate 不要求时可以 |
| P2 | `GO_WITH_TOOLING_BACKLOG` | 可以 |
| P3 | `GO` | 可以 |
| 无 finding | `GO` | 可以 |

安全、数据身份、评估泄漏和未经授权的外部动作不适用“有瑕疵先跑起来”的豁免。

## 5. 重要/紧急四象限

P 等级衡量对当前里程碑的影响；四象限决定调度顺序。二者相关但不等价。

### 第一象限 — 重要且紧急

立即占用主线；通常是当前 P0：

- clean checkout/source completeness；
- owner 对候选模型、revision、下载、运行时和资源的授权；
- 唯一 tokenizer/input contract；
- 第一个 Encoder 数据加载、训练、保存、reload/infer 闭环；
- 阻止 audit、fit 或 checkpoint reload 的错误。

### 第二象限 — 重要但不紧急

排入下一里程碑，或在主线 P0 稳定后并行：

- 多候选、多 seed、partial/full fine-tuning；
- 独立 Gold 和 OOD；
- rare-class、关键边界和 calibration 优化；
- 历史 exact reproduction 的真实 reference environment 执行；
- production acceptance 设计。

### 第三象限 — 紧急但重要性较低

只有确实影响当前开发节奏时才插队：

- CI/脚本/错误输出导致 agent 无法继续；
- 分支、worktree、artifact 路径的短期集成问题；
- 即将阻止交接的工具故障。

解决后立即回到主线，不借机扩展为大规模工具重构。

### 第四象限 — 不重要且不紧急

默认不做：

- GUI、dashboard 和服务化外观；
- 没有测量依据的性能微调；
- LoRA 或更多模型家族的探索；
- 生产 49,054 推理之前的展示层；
- 纯风格性 refactor；
- 在 Mac 上反复模拟不被政策承认的 exact reproduction。

## 6. Agent 模型路由和提示词标题

### 6.1 默认路由

| Priority | 默认模型 | Reasoning effort | 标题前缀 |
| --- | --- | --- | --- |
| P0 | `gpt-5.6-terra` | `max` | `【P0｜Terra Max】` |
| P1 | `gpt-5.6-luna` | `max` | `【P1｜Luna Max】` |
| P2 | `gpt-5.6-luna` | `max` | `【P2｜Luna Max】` |
| P3 | `gpt-5.6-luna` | `max` | `【P3｜Luna Max】` |

模型选择由任务最高未解决 priority 决定。一个 prompt 同时含 P0 和 P2 时，按 P0 使用 Terra Max，但必须把 P2 写成 deferred/optional，避免 Terra 任务被工具优化吞没。

### 6.2 Sol Max 的例外

不得因为任务“代码很多”或“看起来困难”就建议 Sol Max。只有满足以下条件之一才允许建议：

- 同时涉及不可变数据契约、Schema、训练架构和验收政策的高代价不可逆决策；
- Terra Max 已完成一轮证据充分的尝试，仍留下无法关闭的 P0 架构冲突；
- 多仓库或多系统迁移中，错误决策会破坏已有 provenance，且无法通过拆分任务降低风险；
- owner 明确要求使用 Sol Max。

标题必须在最前面显式写：

```text
【建议使用 Sol Max｜P0｜原因：一句话说明为什么 Terra Max 不足】
```

没有这个标题，就按 Terra/Luna 默认路由，不得在正文中悄悄升级模型。

### 6.3 标准提示词头

每个后续 prompt 应以如下信息开始：

```text
【P0｜Terra Max】Milestone M1 — 任务名称

Active milestone:
Priority:
Important/Urgent quadrant:
Why this priority:
Dependencies already satisfied:
Blocking dependencies:
In scope:
Out of scope:
Exit criteria:
Required evidence:
Branch/worktree:
```

P1–P3 将标题和模型替换为 Luna Max。Prompt 必须把优化 backlog 与本轮 exit criteria 分开。

## 7. 全局里程碑

### M0 — 可审计数据与可运行 Classical 基线

状态：`CLOSED_IN_M1_INTEGRATION_CLOSEOUT`

阶段重点：建立可信、可执行的训练地基，不追求生产模型质量。

已确认：

- immutable data/reference package 已通过真实审计；
- Schema、split、quarantine、field weights、Anchor 和 baseline oracles 已冻结；
- macOS comparable Classical run 存在；
- side 已验证并提交缺失 Classical source 的字节一致修复；
- M1 integration merge `adb2fc4` 正常合入了 Side exact-environment gate，
  并保留 Main 的 M1 provenance-bound evidence；
- clean integration checkout 的完整测试（141 passed）、canonical audit、编译和依赖检查通过。

当前 P0：无。M0 source-completeness 已在 integration worktree 的 tracked
`src/semantic_model/models/` 中验证；根级 `/models/` 忽略规则不会再吞掉
该 Python package。

M0 exit evidence：

```text
tracked Classical source exists
clean checkout imports only tracked source
full pytest passes
audit_data passes on immutable package
audit_reference classifies current environment truthfully
one comparable diagnostic train path remains executable
no production claim
```

不阻止 M0 exit 的 backlog：

- Linux exact reproduction；
- exact-environment execution tooling；
- Classical 算法优化；
- dashboard 和运行浏览工具。

### M1 — 第一个可运行的 Encoder 七头训练闭环

状态：`CLOSED_M1_EXIT_ACCEPTED_PROVENANCE_BOUND_DIAGNOSTIC_ONLY`

阶段重点：先让正确的训练路径真实跑起来，而不是先获得最优模型。

M1 P0 critical path 已全部关闭：

1. 集成 M0 source-completeness 和 canonical audit gate；
2. owner 选择一个首跑 Encoder 的 exact model ID/revision/license；
3. owner 明确授权 artifact 下载、项目 runtime 和可用 CPU/MPS/CUDA 资源；
4. 冻结无歧义的 tokenizer/input builder，包括 special tokens、token types、padding 和 truncation；
5. 实现 immutable Train/Dev 加载、七头输出和 `sample_id × head` 权重；
6. 先运行最小 smoke fit，确认 forward/backward/checkpoint/reload；
7. 再运行一个完整 Train 1,822 的单 seed diagnostic run；
8. 保存版本化 checkpoint、class order、config、artifact hashes、metrics 和环境 manifest；
9. 在同一 input builder 上完成 reload inference smoke test；
10. review 关闭所有 M1 P0。

推荐的最小实验范围：

- 一个 owner 批准的 BERT-class Chinese Encoder；
- frozen Encoder + 七个训练 head 作为第一阶段；
- one seed；
- Train 拟合，Dev 用于 early stopping/diagnostic；
- Test 不参与模型选择；
- weak-label 指标只称 diagnostic；
- 不以 Gold/OOD 缺失阻止首跑。

M1 exit evidence：

```text
one authorized exact Encoder artifact identity
one deterministic input builder
one successful seven-head smoke fit
one successful full Train-1822 diagnostic run
checkpoint saved and hash-addressed
reload inference succeeds
Dev metrics and convergence/resource facts recorded
full tests pass
no production approval
```

M1 closeout evidence is the accepted artifact
`b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58` from
implementation commit `88f90b11a4c81fa3b7d356d980be01d261df7cd3`, recorded
by Main evidence commit `4b2210d416ee1006e5fad24a0f5bb88a750c26dd`. The
integration branch normally merged Side exact-gate commit
`2265682af132752c0a70d821b3c39dc0db475f59` in merge commit `adb2fc4`; it
adds a strict no-write exact-environment audit and receipt chain without
altering the accepted M1 artifact. On macOS arm64, that audit exits 2 with
`BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH` and `training_invoked=false`.
The merged M1 closeout regression suite recorded 141 passed tests; this is
integration evidence and does not alter the historical M1 run report.

M1 closeout does not promote weak-label diagnostics to model selection, Gold,
OOD, Test, or production evidence. Remaining M1 backlog is non-blocking:

- **P1 / M2:** multi-seed stability, approved candidate comparison,
  single-task controls, partial/full-unfreeze controls, rare-class and
  negative-transfer analysis, calibration, abstention, and governed
  disagreement review;
- **P2 / historical side lane:** provision the frozen Linux x86_64 / AMD EPYC
  reference environment and rerun the integrated exact-environment gate;
  macOS must remain a comparable-only, no-fit result.

明确 defer 到 M2/M3：

- 三个 seeds；
- BERT/MacBERT/RBT3 全候选比较；
- partial/full fine-tuning；
- single-task versus multi-task；
- 100–500 Gold；
- 完整 OOD/abstention；
- production thresholds；
- 49,054 inference。

### M2 — Encoder 质量与稳定性迭代

状态：`CURRENT_PRIMARY_MILESTONE`

阶段重点：把“能训练”提升为“相比 Classical 有明确、稳定、可解释的收益”。

主要 P1：

- frozen、partial unfreeze 和有依据的 full fine-tune；
- approved BERT/MacBERT/RBT3 或轻量候选比较；
- 至少三个 seeds，报告 mean/dispersion/worst seed；
- single-task 与 shared multi-task，检查 negative transfer；
- rare classes、class imbalance、Reasoning tags 和关键语义边界；
- tokenizer max length/truncation 实测；
- 100–200 条独立 adjudicated Gold 的第一阶段；
- Classical/Encoder disagreement 驱动的定向 review；
- calibration、abstention 和第一版 OOD challenge。

M2 已解锁工作线（不得将 D-023 的单次 M1 授权扩展为新下载、训练或
生产动作）：

- single-task 与 shared multi-task 的受控比较；
- 多 seed 稳定性、frozen/partial/full-unfreeze 阶段和 approved candidate
  比较的合同设计；
- rare class、negative transfer、tokenizer、calibration 与资源诊断；
- Classical/Encoder disagreement 的只读分析与经独立 scope 批准后的
  Gold/OOD/有限 review protocol。

M2 exit evidence：

```text
selected candidate/stage based on frozen Dev policy
meaningful per-head improvement over Classical control
no hidden critical-boundary regression
three-seed stability reported
resource cost measured
known weak-label limitations retained
candidate accepted for formal validation or explicitly rejected
```

### M3 — Production-candidate Gold/OOD 验收

状态：`LOCKED_UNTIL_M2_SELECTS_A_CANDIDATE`

阶段重点：从“有明显效果”升级为“有足够独立证据进入受控生产试点”。

在 M3 中，下列项目升级为 P0：

- 足够、独立、版本化、未用于选择的 adjudicated Gold；
- 明确的一次性 Test 开封政策；
- OOD recall、in-domain false rejection 和 high-confidence OOD error gate；
- calibration、selective risk 和 per-head abstention；
- 关键语义边界错误率；
- CPU inference、artifact reload、license 和依赖复现；
- 模型卡、限制、回滚和禁止自动 Gold promotion。

M3 exit 只能产生：

- `PRODUCTION_CANDIDATE_ACCEPTED_FOR_CONTROLLED_PILOT`；或
- `PRODUCTION_CANDIDATE_REJECTED_WITH_EVIDENCE`。

它不能直接授权无监控的全量生产。

### M4 — 49,054 条受控生产试点

状态：`LOCKED_UNTIL_M3_OWNER_APPROVAL`

阶段重点：小批量、可回滚、可观察地验证真实输入分布和运行成本。

M4 P0：

- owner 明确批准数据范围和输出用途；
- 批处理、幂等、resume、failure isolation 和 artifact binding；
- OOD/abstention 不被调用方覆盖；
- 不把帖子语义预测直接变成投资建议；
- 监控输入漂移、高置信错误和失败率；
- rollback 和版本隔离；
- 输出仍通过 inference schema。

未经 M3/M4 owner gate，不得运行 49,054 条。

### M5 — 生产后优化与平台化

状态：`LOCKED_UNTIL_CONTROLLED_PILOT`

阶段重点：在已有真实瓶颈证据后优化，而不是提前猜测。

候选 P1/P2/P3：

- latency、memory、quantization 和 batch 优化；
- dashboard、运行索引、review queue 工具；
- 有限 LLM review 的成本/收益优化；
- 自动漂移报告；
- 多平台或更多来源的泛化；
- downstream group-state aggregation；
- 服务化和运营工具。

## 8. 可并行 side lanes

### R — 历史 exact reproduction

当前全局等级：`P2 / Luna Max / 重要但不紧急`

说明：历史精确复现保护 provenance，但不阻止 M1 Encoder 首跑。其 source completeness 已解决；exact environment execution gate 可以作为工具 side lane 并行。真正 exact fit 必须等待真实 reference environment。

允许并行的条件：

- 不修改 Encoder 输入合同；
- 不修改 immutable data、Schema、split 或 baseline algorithm；
- 不占用 M1 所需的唯一硬件/agent 资源；
- 不在 Mac 上用容器/QEMU 冒充 exact；
- 提交保持独立可 cherry-pick。

当前建议提示词标题：

```text
【P2｜Luna Max】Milestone R1 — Exact Environment Execution Gate
```

### G — Gold/OOD 数据与验收设计

M1 期间可以并行做协议设计和采样计划，但不应让大规模标注阻止首跑。M1 产生真实错误和 disagreement 后，Gold/OOD 的定向选择进入 M2 P1。

不得自动把模型、LLM 或两模型一致结果升级成 Gold。

### L — 有限 LLM review

M1 前保持锁定，因为还没有 Encoder disagreement/low-confidence 队列。M1 后可以用 Luna Max 建立 P1/P2 的有限 review protocol；外部调用、预算、隐私和 provider 仍需 owner 授权。

### T — 工具、CI 与文档

P2/P3 可以在 M1 P0 稳定后并行，但不能与主线同时修改同一文件所有权。每个工具任务必须证明不会扩大训练范围或改变冻结合同。

## 9. 并行门禁

任务只有同时满足以下条件才可并行：

1. 不修改同一 worktree；
2. 不修改同一冻结数据/Schema/input/split 合同；
3. 不依赖对方尚未产生的 artifact；
4. 不竞争唯一硬件、预算、外部授权或 owner 决策；
5. 可以用独立 commit/branch 合并；
6. 任一失败不会让另一任务的结果失去身份；
7. 已定义谁拥有最后的 integration review。

以下任务默认不可并行：

- tokenizer/input contract 仍在变化时开始正式 Encoder 训练；
- split/Gold 身份仍在变化时开始模型选择；
- 两个 agent 同时编辑同一个 machine-readable contract；
- source-completeness 未集成时，把 primary worktree 的 ignored source 当作成功；
- owner 尚未选定 artifact/revision 时，由多个 agent 各自下载不同模型；
- Test 已被一个分支打开时，另一个分支仍宣称 one-shot Test。

## 10. 当前任务板

Snapshot：2026-08-28。执行前必须验证 live branches，不得只信任本表。

| 顺序 | 任务 | Active milestone | Priority | 象限 | 模型 | 并行性 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | M0 source-completeness、M1 provenance run 与 Side exact gate integration closeout | M0–M1 | P0 | 重要紧急 | Terra Max | CLOSED — `adb2fc4` |
| 2 | M2 candidate/stage/Dev-policy contract and approved-scope planning | M2 | P0 | 重要紧急 | Terra Max | 当前串行入口；不得隐含新下载或训练授权 |
| 3 | 多候选/partial unfreeze/三 seeds/negative transfer | M2 | P1 | 重要不紧急 | Luna Max | M2 scope 与 artifact authorization 冻结后并行 |
| 4 | exact-environment execution gate on frozen Linux x86_64 reference runtime | R1 | P2 | 重要不紧急 | Luna Max | 已集成；macOS fail-closed，不阻断 M2 |
| 5 | 定向 Gold/OOD/abstention | M2–M3 | P1→P0 | 重要不紧急→重要紧急 | Luna Max；升级 P0 后 Terra Max | 仅协议/计划可并行；外部或新数据动作另行授权 |
| 6 | CI、报告、dashboard、工具优化 | T | P2/P3 | 第三/第四象限 | Luna Max | 有空闲槽且无文件冲突时 |

## 11. Milestone closeout

每个里程碑关闭时必须记录：

- final status；
- owner decision；
- merged commit；
- data/model/config/artifact IDs；
- tests and real-run evidence；
- remaining P1/P2/P3 backlog；
- newly unlocked parallel lanes；
- next active milestone；
- next P0 prompt title and model routing。

只要 exit gate 没有要求，P1/P2/P3 backlog 不得阻止 closeout；但它们必须留在 Git，而不是只存在于对话。

## 12. 任何新训练前的仓库操作收口

这是一项由 owner 指示、由 agent 一次性执行的仓库清理工作，不是训练代码中的可信校验、授权系统或 runtime gate。

在任何新的 training/fit 命令启动前：

1. review 并合并本轮要保留的 Main、Side、bugfix 和 feature 产出；
2. 保留一个统一训练分支，删除其他本地及远端 feature/bugfix 临时分支；
3. 删除全部附属 linked worktree，只保留统一分支所在的主工作目录；
4. 删除前只做避免丢失未合并工作的只读 inventory，并先迁移训练仍需使用的 ignored runtime、cache 和 artifact。

不得为此增加 owner receipt、签名、身份认证、allowlist、可信分支、可信远端、remote-sync receipt 或其他单人项目不需要的授权/可信框架。owner 在对话中的直接指令就是是否执行训练的依据。
