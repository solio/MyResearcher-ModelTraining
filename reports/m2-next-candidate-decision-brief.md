# M2 下一候选路线决策简报

状态：`DECISION_BRIEF_PENDING_OWNER_DECISION`

基线提交：`cc29abd0850616e9a9caae9b3af4d5e65ea3ab85`

范围：只做官方资料的只读比较；本简报没有下载模型、创建 runtime、训练、推理，也没有读取 Test、Anchor、Gold、OOD 或 reference predictions。

## 结论

`RECOMMENDED_NEXT_CANDIDATE = RBT3_REASONING_CORRECTIVE_V1`

先保留已经验证过的 `hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`，建立一个新的、独立命名的 reasoning-specific corrective lineage。第一轮只做 reasoning head 的三 seed 诊断：RBT3 的 embeddings 和前两个 Transformer blocks 冻结，只训练最后一个 Transformer block 与 `reasoning_tags` head；六个其他 head 不参与这轮反向传播。纠正变量只有“reasoning-only last-block adaptation”，不同时改变 tokenizer、split、seed、阈值或数据权重。

归因必须从同一初始状态开始：每个 seed 都从官方 RBT3 固定 revision 的原始权重初始化，绝不从 S1、S2 或 S3 checkpoint 继续训练；`reasoning_tags` head 的初始化 RNG、参数初始化函数、Train 数据顺序和 batch 划分必须与 S3 的同 seed 完全一致。这样候选相对 S3 的唯一差异才是 last-block adaptation，相对 S2 才能检验移除其他六头梯度的效果，也不会把现有 S1/S2/S3 artifact 改写成新结果。

`FALLBACK = hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460`

仅当推荐路线按下面的 reasoning 门槛失败，且 owner 另行同意新模型下载后，才考虑 LERT-small。它是同一 HFL/BERT 接口族的轻量新 lineage，能以较低的磁盘和计算成本检验“语言学增强预训练是否比 RBT3 表征更适合弱标签推理”。MacBERT-base 是有吸引力的第三候选，但不作为 fallback：它应在 LERT-small 或新的资源测量之后再单独申请，避免一次引入更大的 cache、内存和训练时间变量。

## 已知 M2 证据（只作为 Dev 诊断上下文）

- S1 frozen RBT3 control 已完成，三 seed 为 35/71/107，`selected_candidate=false`，artifact content address 为 `04a23d76413049e57ff083655f80ad8c3dfc7ed90702a3c9ed66bcfd79f377f6`。
- S2 partial-last-block 的七头均值全部提升，但 `emotion_primary:CALM` 与 `reasoning_tags:NO_REASON_GIVEN` 关键边界回归，结论为 `ACCEPT_EVIDENCE_BUT_DO_NOT_PROMOTE`。
- S3 frozen single-task 诊断显示：`emotion_primary` 有改善，但 `reasoning_tags` mean Macro-F1 为 `0.149804`，相对 S1 三 seed 均下降；`NO_REASON_GIVEN` support 86，在 seed 107 的 S3−S1 F1 delta 为 `-0.130704`。因此 RBT3 的 M2 梯度已经结束，本简报不启动 S4。
- 这些数值均来自已接受的 Dev-only S1/S2/S3 evidence；它们不是 Gold、Test、OOD 或生产结论。

## 四组对照与归因边界

本次 corrective 只输出 `reasoning_tags`，因此四组对照的比较单位全部是
Reasoning 的 Dev 诊断；`Classical` 不参与梯度归因。

| 对照 | 固定身份与可训练部分 | 本次比较要回答的问题 | 可归因结论 |
| --- | --- | --- | --- |
| S3 matching-seed | 官方 RBT3 初始化；Encoder 全冻结；只训练 reasoning head；seed 35/71/107 的 head 初始化与数据顺序固定 | 在相同 single-task 条件下，加入 last-block adaptation 是否改善 reasoning | corrective − S3 是 last-block adaptation 的单变量对照；不得混入其他 head 梯度 |
| S2 matching-seed | 官方 RBT3 初始化；只解冻最后一个 block，但同时训练七个 heads；head LR 与 early-stopping 也不同 | 描述 shared-stage 结果与 corrective 的差异 | corrective − S2 仅作描述性比较；由于 head LR 和 early-stopping 也不同，不能单独证明共享七头梯度是因果解释 |
| S1 matching-seed | 官方 RBT3 初始化；Encoder 全冻结；训练共享七个 heads | 原始 frozen shared control 的基线是多少 | corrective − S1 是总的 reasoning 变化，必须同时拆解报告 S3/S2 对照，不能单独归因 |
| Classical Dev control | 冻结 baseline-v0.3.5 的 Dev reference predictions/weak labels；不反向传播 | 质量背景与最终 Classical gate 的位置 | 只作质量背景，不用来推断 Encoder 梯度来源，也不替代 S1/S2/S3 matching-seed 归因 |

### 三组 Encoder 对照的现有逐 seed Reasoning 证据

下表复录现有 S1/S2/S3 Dev evidence，供未来 corrective run 使用同一报告形状。
它们都是 weak-label Dev 诊断，不是 Test、Gold 或生产指标。S1/S2/S3 的 seed
顺序统一为 `35 / 71 / 107`；未来 corrective 必须在每个 seed 逐项报告
`Macro-F1`、`Micro-F1`、`exact-set accuracy`、15 个标签 F1 及相对三个对照的
delta。

| 对照 | Seed | Macro-F1 | Micro-F1 | Exact-set accuracy | `NO_REASON_GIVEN` F1 (support 86) |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 frozen shared | 35 | 0.156848 | 0.273333 | 0.129464 | 0.267857 |
| S1 frozen shared | 71 | 0.154548 | 0.275410 | 0.131696 | 0.297521 |
| S1 frozen shared | 107 | 0.150258 | 0.285714 | 0.145089 | 0.295652 |
| S2 shared partial-last-block | 35 | 0.231104 | 0.398810 | 0.196429 | 0.267857 |
| S2 shared partial-last-block | 71 | 0.204976 | 0.363450 | 0.180804 | 0.295652 |
| S2 shared partial-last-block | 107 | 0.217861 | 0.388211 | 0.178571 | 0.228571 |
| S3 frozen single-task | 35 | 0.152158 | 0.267559 | 0.125000 | 0.252252 |
| S3 frozen single-task | 71 | 0.152405 | 0.270627 | 0.131696 | 0.297521 |
| S3 frozen single-task | 107 | 0.144848 | 0.291347 | 0.138393 | 0.164948 |

十五个 Reasoning 标签的 Dev F1（同一列顺序为 seed 35 / 71 / 107）如下；粗体行
是本轮唯一重点边界。support 由冻结 Dev weak labels 提供，任何低 support 标签仍
必须报告，不能被均值隐藏。

| Label | Support | S1 F1 (35/71/107) | S2 F1 (35/71/107) | S3 F1 (35/71/107) |
| --- | ---: | --- | --- | --- |
| CROSS_STOCK_REFERENCE | 44 | 0.040816 / 0.000000 / 0.000000 | 0.142857 / 0.115385 / 0.148148 | 0.041667 / 0.000000 / 0.000000 |
| FLOW_POSITIONING | 64 | 0.155844 / 0.170732 / 0.138889 | 0.289157 / 0.250000 / 0.300000 | 0.153846 / 0.170732 / 0.177215 |
| FUNDAMENTAL | 24 | 0.571429 / 0.558140 / 0.540541 | 0.612245 / 0.638298 / 0.640000 | 0.571429 / 0.558140 / 0.564103 |
| MACRO_POLICY | 16 | 0.117647 / 0.117647 / 0.117647 | 0.117647 / 0.111111 / 0.105263 | 0.117647 / 0.117647 / 0.117647 |
| NEWS_EVENT | 32 | 0.210526 / 0.162162 / 0.162162 | 0.210526 / 0.195122 / 0.205128 | 0.162162 / 0.162162 / 0.114286 |
| **NO_REASON_GIVEN** | **86** | **0.267857 / 0.297521 / 0.295652** | **0.267857 / 0.295652 / 0.228571** | **0.252252 / 0.297521 / 0.164948** |
| RELATIVE_PERFORMANCE | 71 | 0.054795 / 0.080000 / 0.027778 | 0.442105 / 0.369565 / 0.489796 | 0.054795 / 0.054795 / 0.080000 |
| RUMOR | 6 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| SARCASM_IRONY | 33 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| SOCIAL_PROOF | 34 | 0.057143 / 0.057143 / 0.057143 | 0.111111 / 0.000000 / 0.000000 | 0.057143 / 0.057143 / 0.057143 |
| TECHNICAL_PRICE | 180 | 0.512281 / 0.510490 / 0.549669 | 0.636364 / 0.617363 / 0.639752 | 0.507042 / 0.503546 / 0.587500 |
| THEME_NARRATIVE | 62 | 0.173913 / 0.173913 / 0.173913 | 0.454545 / 0.300000 / 0.329114 | 0.173913 / 0.173913 / 0.119403 |
| UNKNOWN | 14 | 0.133333 / 0.133333 / 0.133333 | 0.125000 / 0.125000 / 0.125000 | 0.133333 / 0.133333 / 0.133333 |
| VALUATION | 14 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| WORDPLAY | 34 | 0.057143 / 0.057143 / 0.057143 | 0.057143 / 0.057143 / 0.057143 | 0.057143 / 0.057143 / 0.057143 |

本轮不得加入 `emotion_primary` 输出；因此 `emotion_primary:CALM` 只能作为上面
S2/S3 历史背景，不能成为 corrective 的 gate、early-stopping 指标、成功条件或
失败条件。corrective 的唯一 primary 是 `reasoning_tags` Macro-F1；Micro-F1、
exact-set accuracy 和 15 标签 F1 是同一 reasoning head 的辅助诊断。

## 候选的官方身份与可执行性

| 候选 | 官方 model id / immutable revision | License | 官方架构资料 | tokenizer | 官方文件规模（规划参考） | 本地可行性 |
| --- | --- | --- | --- | --- | --- | --- |
| RBT3 reasoning corrective | `hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c` | Apache-2.0 | 3-layer RoBERTa-wwm-ext；config 为 hidden 768、3 layers、12 attention heads；项目已记录约 38M transformer 参数 | `AutoTokenizer`/BERT WordPiece 兼容（`vocab.txt` + `tokenizer.json`） | 固定页列出 repo 约 465 MB、PyTorch 权重约 156 MB | 无新下载；复用已验证 cache；现有 S2 每 seed 109–139 s、约 28.6 MB checkpoint 是本地上界的实测参考 |
| LERT-small fallback | `hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460` | Apache-2.0 | 12 layers、hidden 256、4 attention heads、约 15M transformer 参数；LERT 官方说明其主体是 BERT 结构 | `BertTokenizer` + `BertModel`（也可经 `AutoTokenizer`/`AutoModel` 加载） | 固定页约 144 MB repo、PyTorch 权重约 60.8 MB，`tokenizer.json` 约 269 kB、`vocab.txt` 约 110 kB | 需要一次新下载和新 cache identity；预计仍可复用现有 Transformers/PyTorch 依赖，不预期需要改依赖版本；先冻结 Encoder 只训练七头以控制成本 |
| MacBERT-base stronger candidate | `hfl/chinese-macbert-base@a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` | Apache-2.0 | 12 layers、hidden 768、12 attention heads、约 102M 参数；MacBERT 的 MLM-as-correction、WWM、N-gram masking、SOP 预训练旨在缩小预训练与微调差异 | 官方要求 `BertTokenizer` + `BertModel` | 固定页约 1.3 GB repo、PyTorch 权重约 412 MB，`tokenizer.json` 约 269 kB、`vocab.txt` 约 110 kB | 需要新下载；同一标准 Transformers/PyTorch 接口但 hidden size/层数不同，需新 config/model identity；在 10 GiB/2 h 约束下只能先做冻结 Encoder 诊断，不承诺 partial/full unfreeze 可行 |

官方链接（均为只读资料）：[RBT3 固定 revision 与文件页](https://huggingface.co/hfl/rbt3/tree/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c)、[RBT3 config](https://huggingface.co/hfl/rbt3/blob/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c/config.json)、[LERT-small 固定 revision 与文件页](https://huggingface.co/hfl/chinese-lert-small/tree/69e3e69ba258be5b301b26937e5b55a076c90460)、[LERT 官方架构/下载说明](https://github.com/ymcui/LERT/blob/main/README_EN.md)、[MacBERT-base 固定 revision 与文件页](https://huggingface.co/hfl/chinese-macbert-base/tree/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1)、[MacBERT 官方架构/下载说明](https://github.com/ymcui/MacBERT/blob/master/README_EN.md)、[MacBERT-base config](https://huggingface.co/hfl/chinese-macbert-base/blob/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1/config.json)。

### 对七头任务的适配

三种 Encoder 都可以接入现有六个 single-label heads 加一个 15-label `reasoning_tags` head；差异只在 pooled hidden size 和 tokenizer/model class 的 provenance。LERT-small 的 256 维表示会降低 head 参数和激活占用，但也可能限制跨句关系容量；MacBERT-base 的 768 维、12 层上下文容量较大，代价是更高的 MPS 内存和 CPU fallback 时间。任何候选都必须继续使用现有 `input builder`、四个 special tokens、`HEAD_TAIL`、max length 256、Train/Dev immutable loader、每个 `sample_id × head` weights 和现有七头 metric implementation，不得借换模型引入新的 split 或标签解释。

### 对 reasoning / `NO_REASON_GIVEN` 的可证据化假设

- RBT3 corrective：S2 的 shared update 同时改善均值却伤害关键边界，S3 的 reasoning-only frozen head 又没有恢复 `NO_REASON_GIVEN`。只解冻最后一个 block、只更新 reasoning head 是一个可证伪的中间假设：若负迁移来自共享 head 梯度，应该在保持其他头不变的情况下改善 reasoning；若 seed 107 仍退化，则更可能是 weak-label 边界/类别不平衡而不是共享梯度。
- LERT-small：官方 LERT 论文/README 将其定位为注入 POS/NER/DEP 等语言学特征的 BERT 结构，并报告在多个中文 NLU 任务上相对可比基线的提升。这里的用途只是支持“语言学信息可能帮助隐含理由、否定和语义关系”的先验；官方任务结果不是本项目的 Dev/Test 证据，不能直接替代本项目三 seed 诊断。
- MacBERT-base：官方说明 MLM-as-correction、WWM、N-gram masking 和 SOP 用于减少预训练与微调的差异。该先验可能帮助纠正表达变体、否定和上下文关系，但不能证明它能修复 `NO_REASON_GIVEN`；小样本和弱标签仍可能让更大 Encoder 过拟合或产生新的负迁移。

## 推荐路线的最小诊断合同（未来另行授权后）

`RBT3_REASONING_CORRECTIVE_V1` 不是当前合同、不是 S4，也不是 selected candidate。它需要一个新的、明确的 owner 决定和新的 M2 lineage；本次只提出方案。

1. 三个 run units：seed 35、71、107；只用 Train 1,822 拟合，Dev 448 做 `reasoning_tags.macro_f1` early stopping/diagnostic；不读 Test、Anchor、Gold、OOD、reference predictions。
2. 初始化来源固定为官方 `hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c` 原始权重，不从 S1/S2/S3 checkpoint warm-start。每个 seed 的 reasoning head 参数初始化 RNG、初始化函数、Train 数据顺序和 batch 划分必须逐项复用 S3 matching seed。
3. Encoder embeddings 与前两个 Transformer blocks 冻结，只解冻最后一个 block 和 `reasoning_tags` head；六个其他 head 不实例化为可训练参数，也不参与 loss 或 optimizer。建议 block LR `1e-5`、reasoning-head LR `5e-4`、AdamW weight decay `0.01`。
4. 保持 max length 256、batch 16、最多 12 epochs、patience 3、gradient clipping 1.0、MPS-first/CPU-fallback、每 run 120 分钟和新增磁盘 10 GiB；所有数值必须在 fit 前冻结。
5. 每个 seed 保存 block+head checkpoint，做 CPU reload 并检查 Macro-F1、Micro-F1、exact-set accuracy 和全部 15 label 的有限输出；任何非有限 loss/gradient/logit、超时、磁盘超限或 reload 失败都停止且不聚合。
6. 这是 reasoning-only diagnostic。通过仍只产生 reasoning evidence，不是七头候选、不是 `M2_SELECTED_CANDIDATE`；只有三 seed 完成且门槛通过，才可申请另一个“七头正式候选”合同。失败即停止，LERT 仍需另行授权。

## Fallback：LERT-small 的最小诊断合同（需新的模型授权）

若推荐路线失败，fallback 只允许以新 lineage `LERT_SMALL_FROZEN_SEVEN_HEAD_V1` 提交一次新的 owner 授权。第一轮固定三 seed 35/71/107、Train/Dev 角色、现有七头和相同输入/metric contract，Encoder 全冻结，只训练七个 heads；这样先隔离模型预训练收益，不把“新模型”和“解冻策略”混在一起。只有完整三 seed 的 Dev evidence 通过 stability 和 Classical gate，才有理由申请后续 partial-unfreeze；失败即拒绝，不改下载 revision、不自动尝试另一个模型。该路线需要下载并记录完整 cache/tokenizer hashes；不应修改现有依赖版本，若标准接口不兼容则停止并报告技术 blocker。

## 资源估算与停止条件

以下是规划范围，不是实测承诺。RBT3 数值以当前本地 S2/S3 evidence 为锚：S2 每 seed 109–139 秒、约 28.6 MB checkpoint；S3 每 seed 63–81 秒。LERT/MacBERT 只引用官方文件大小并按相同 batch/length 做保守工程估算，未执行模型加载。

| 路线 | 三 seed MPS 诊断预计 | CPU fallback 预计 | 新增磁盘规划 | 硬上限 |
| --- | --- | --- | --- | --- |
| RBT3 corrective（reasoning-only last block） | 约 4–12 分钟总计 | 约 15–45 分钟总计 | 约 0.2–1.0 GiB（不含已存在 cache） | 每 run 2 h；总新增 10 GiB |
| LERT-small frozen seven-head | 约 4–15 分钟总计 | 约 20–60 分钟总计 | 约 0.3–1.5 GiB（含约 144 MB cache repo 规划量） | 每 run 2 h；总新增 10 GiB |
| MacBERT-base frozen seven-head | 约 10–35 分钟总计 | 约 45–120 分钟总计 | 约 1.5–4.0 GiB（含约 1.3 GB cache repo 规划量） | 每 run 2 h；总新增 10 GiB |

停止规则覆盖 tokenizer/model load、全部 epochs、最终 metrics、CPU reload 和 manifest 写入；超时或接近磁盘上限时保留可诊断 evidence，不压缩/复制出非必要副本。MacBERT 的 partial/full unfreeze 不在上述预算内，不能由这份简报默示授权。

## 进入正式训练合同的数值门槛

先用 matching-seed S1 和冻结 Classical Dev control 做比较；不允许只看一个 aggregate score。一个候选只有在满足以下条件后，才值得写入新的正式训练合同（仍不等于生产批准）：

- 三 seed 完整且 provenance、CPU reload、资源和 finite-output checks 全部通过；每个 head 报告 mean、sample standard deviation、worst seed 和 matching-seed delta。
- 七头最终候选必须满足冻结 Classical gate：任一 head 的 mean Macro-F1 不得低于 Classical 超过 0.01；任一 worst seed 不得低于 Classical 超过 0.03；至少四个 head 的 mean Macro-F1 高于 Classical 至少 0.01；每个 support ≥20 的 critical label F1 drop 不得超过 0.05。Reasoning 还必须同时报告并通过 Macro-F1、Micro-F1、exact-set accuracy 及 per-label 结果；support <20 只能标 `NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION`，不能伪造 pass。
- 对推荐的 reasoning corrective 诊断，额外要求 reasoning Macro-F1 mean 至少比 S1 高 0.01，三 seed mean Micro-F1 和 exact-set accuracy 不得下降超过 0.01，`NO_REASON_GIVEN`（support 86）mean F1 不低于 S1 且 worst-seed drop 不超过 0.03。该诊断没有 Emotion 输出，因此不定义、也不评价 `emotion_primary:CALM` 或任何其他 Emotion gate。
- 三 seed 的 reasoning head sample standard deviation ≤0.05；candidate 若任何 seed 仍有 `NO_REASON_GIVEN` 的严重回归、非有限输出或 device-stratified mismatch，则拒绝，不进入七头正式合同。
- 通过上述门槛只产生“值得起草新合同”的证据；它不授权新模型下载、partial/full unfreeze、Test/Anchor/Gold/OOD/LLM 或生产推理。每个新增模型/revision 都必须在另一个 owner 决定中明确绑定。

## 为什么不优先其他路线

- 不继续 RBT3 的无约束 S4/full unfreeze：现有 M2 RBT3 梯度已结束；S2/S3 已显示关键边界和 seed 107 的 `NO_REASON_GIVEN` 风险。若 RBT3 corrective 也失败，应停止并等待新的模型合同，而不是在同一 lineage 叠加更多未预注册变量。
- 不先选 MacBERT-large 或其他 24-layer/324M 级模型：官方资料显示其模型约 1.2 GB、324M 参数；在 10 GiB、单次 2 小时和 CPU reload 要求下，首轮更可能把资源风险误当成质量结论。它也不是用户要求的最小可控 fallback。
- 不把普通 Chinese RoBERTa-base 作为 stronger route：同为约 102M/12×768 的 BERT 接口族，但 MacBERT 已有针对预训练—微调差异的 correction 预训练先验；若要承担同等下载和资源成本，MacBERT 的可证伪理由更清楚。
- 不优先 ELECTRA：虽然 HFL 有 ELECTRA-small，但它改变了 discriminator/generator 预训练范式，且不属于本次要求的 BERT/RoBERTa/MacBERT 轻量候选主线；在没有额外适配预算和 owner 决定时会扩大解释变量。
- 不用任何官方 benchmark 的 Test 数字替代本项目 Dev 选型；官方结果只用于模型机制/资源的先验说明。

## Owner authorization text（不超过十行）

```text
同意启动 RECOMMENDED_NEXT_CANDIDATE：RBT3_REASONING_CORRECTIVE_V1。
固定 hfl/rbt3@0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c，复用现有本地 cache，不下载新模型。
只运行 reasoning_tags 三 seed 35/71/107；Train 1,822 拟合，Dev 448 以 Macro-F1 early stopping/诊断。
每个 seed 从官方原始权重开始；reasoning head 初始化 RNG、函数、数据顺序和 batch 划分匹配 S3。
冻结 embeddings 与前两个 Transformer blocks，只解冻最后一个 block 和 reasoning head；六个其他 head 不训练。
固定 max_length 256、batch 16、12 epochs、patience 3、AdamW、block LR 1e-5、head LR 5e-4。
MPS 优先、CPU 回退；每 run 2 小时、总新增磁盘 10 GiB，并必须 CPU reload。
禁止 Test/Anchor/Gold/OOD/reference predictions/LLM/云/生产推理；不产生 Emotion/CALM 输出。
通过也只能是 reasoning diagnostic，不是七头候选；失败停止，LERT-small fallback 需另行授权。
```

当前未获得授权的动作：任何 RBT3 corrective fit、任何新模型下载或 cache 创建、LERT/MacBERT 训练、partial/full unfreeze、Test/Anchor/Gold/OOD/LLM/生产路径。当前 `selected_candidate=false` 保持不变。
