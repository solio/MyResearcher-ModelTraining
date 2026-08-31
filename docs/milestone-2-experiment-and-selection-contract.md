# Milestone 2 — Encoder quality experiment and Dev selection contract

Status: `M2_EXPERIMENT_AND_SELECTION_CONTRACT_FROZEN_PENDING_OWNER_EXECUTION_AUTHORIZATION`
Lineage: `myresearcher-encoder-m2-rbt3-quality-v1`
Base: `91d740b1db407d04a090e5fbbeab659598150e16`
Evidence vocabulary: `CONFIRMED`, `PROVISIONAL`, `HYPOTHESIS`, `REJECTED`, `BLOCKED`

## 1. Decision and boundary

The recommended minimum M2 experiment is a **three-seed frozen shared
seven-head control** which reuses the already verified local `hfl/rbt3` cache
at revision `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`. It is deliberately the
same model/tokenizer/input family as the accepted M1 diagnostic, so the first
M2 comparison isolates training-stage effects rather than introducing a model
download, tokenizer, license, or runtime variable.

This is a new M2 weak-label quality-experiment lineage. It is neither the
frozen Classical `baseline-v0.3.5` lineage nor the accepted M1 artifact. It
must never be reported as a production candidate, an M1 replacement, a Gold
result, a Test result, or a 49,054-row inference result.

This document and its companion machine contract do **not** authorize a fit.
D-023/D-024 were exhausted by the accepted single M1 run. Until an owner grants
the exact M2 execution bundle, the M2 state is fail closed: no cache use for
training, no dependency action, no model load for an experiment, no fit, and no
new output directory.

## 2. Immutable controls

The two controls must be carried in every M2 report without mutation:

| Control | Immutable identity | Role in M2 |
| --- | --- | --- |
| Classical | `baseline-v0.3.5`; original model SHA-256 `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`; config SHA-256 `92436eff13c4c67a6dec8a3d645c4e40a4ce8c927d6cb522720709d980cff617` | Regression and disagreement control only. The current macOS lineage remains comparable-diagnostic-only. |
| M1 frozen Encoder | Content address `b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58`; M1 manifest SHA-256 `da9738e4eb7aecaa457fe82d429cdfd5b2c1e2062d4abe231223b4441abd45c2`; checkpoint SHA-256 `e64f71a0b323ac0a7a513b6ae4fddf0e6418b4fdb11f337699e5687da1981cd6` | Read-only one-seed frozen Encoder diagnostic anchor. It is not itself an M2 selection winner. |

### Frozen Classical Dev reference for final M2 selection

The authoritative Classical selection control is not a new fit. It is the
immutable baseline-reference package content ID
`828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`, using
only these frozen Dev artifacts:

| Artifact | Relative path | SHA-256 |
| --- | --- | --- |
| Dev reference predictions | `predictions/dev_reference_predictions_v0.3.5.jsonl` | `833ae4139a99f088986b8b551ae1bc42017844a6215fdf738075faa3ce1174c5` |
| Accepted recomputed metrics | `metrics/recomputed_from_original_model_v0.3.5.json` | `5e6d9fd186d39e3891733dcf35f00898c1373a187d6dc77aa5720b4ac6779595` |
| Frozen Dev weak labels | `splits/labels/teacher_dev_448_v0.3.5.jsonl` | `c71f4ae3e3a7ac8fed73d93b635dfd71f8132e25df858f599908ac262e02d37e` |

The 448 Dev prediction records were independently recomputed against only the
corresponding frozen Dev labels: `sample_id` order and every truth value match,
and all accepted scalar/Reasoning metrics match exactly. This verification did
not read Test or Anchor predictions, unpickle a model/joblib, or execute the
original source.

| Head | Classical Dev Macro-F1 |
| --- | ---: |
| target_mode | 0.32587545082033675 |
| stance | 0.3090808276936797 |
| emotion_primary | 0.20679286503516198 |
| emotion_target | 0.27263592379286355 |
| action_tendency | 0.18546185442009192 |
| context_dependency | 0.4705763666581384 |
| reasoning_tags | 0.41025605014132455 |

The machine contract additionally freezes every Classical per-class/per-label
F1 and support. Classical Reasoning micro-F1 is `0.48633879781420764`; its
Dev-only independently recomputed exact-set accuracy is
`0.12723214285714285`.

The M1 artifact's recorded Dev weak-label macro-F1 values are target mode
0.326448, stance 0.375895, emotion primary 0.167174, emotion target 0.258074,
action tendency 0.132341, context dependency 0.330764, and Reasoning 0.152158
(micro-F1 0.267559; exact-set accuracy 0.125). They are diagnostic context—not
Gold, Test, production, or a substitute for the M2 three-seed control.

The accepted M1 artifact and the retained rejected historical artifact
`3cd8d53cae5ec7346595163c227b4cef8abfd90c5e63c37578c8fc48dc147685` are
immutable. No M2 run may write into, relabel, or overwrite either directory.

## 3. Data role and sealing rules

| Population | Count | M2 role |
| --- | ---: | --- |
| Train | 1,822 | The only fitting population; use frozen `sample_id × head` weights. |
| Dev | 448 | Weak-label early stopping, diagnostics, and the predeclared M2 stage comparison only. |
| Test | 467 | Sealed: not loaded for configuration, metrics, or selection. |
| Anchor50 | 50 | Sealed for this M2 contract; not Gold and not evaluated. |
| Gold | — | No creation or evaluation. |
| OOD | — | No creation or evaluation. |
| LLM / cloud / external API | — | No call or data transfer. |
| Production | 49,054 rows | No inference and no production approval. |

The only M2 selection inputs are Dev predictions/labels created by an
independently authorized run and the predeclared M2 metric rules below. Test,
Anchor, Gold, OOD, LLM output, external facts, and production inputs are not
configuration, model-selection, or diagnostic evidence here.

## 4. Frozen common configuration

The first M2 gradient preserves the M1 input and data interface:

- `encoder-input-builder-v1`; `stock_code`, `stock_name`, and `model_text`;
  exactly four special tokens; no `token_type_ids`.
- Fixed `max_length=256`, code/name caps 8/16, `HEAD_TAIL`, dynamic right
  padding, batch size 16, M1 class order, seven `sample_id × head` weights, and
  head dropout 0.1.
- AdamW with weight decay 0.01, betas 0.9/0.999, epsilon `1e-8`; each stage's
  learning rates are below. Mixed precision is disabled unless separately
  frozen and approved.
- Early stopping uses the mean of the seven declared primary head metrics,
  maximum 12 epochs, patience 3, minimum delta 0, and restore-best-checkpoint.
  It controls epoch choice only; it cannot override a later Dev no-regression
  or seed-stability gate.
- Clip gradient norm at 1.0. Any non-finite loss, gradient, or seven-head CPU
  reload logits stops that run fail closed.
- The configuration is fixed before the first fit. A Dev result can choose only
  among the predeclared stages; it cannot change tokenizer, padding, batch,
  seed set, data role, optimizer family, learning rate, epochs, or gates.

## 5. Recommended minimum experiment gradient

| Order | Stage | Seeds | Trainable parameters | Purpose / trigger |
| ---: | --- | --- | --- | --- |
| S1 | `M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL` | 35, 71, 107 | Seven heads; Encoder frozen | **First recommended M2 execution.** Establish the three-seed control with the fixed M1 cache. |
| S2 | `M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD` | 35, 71, 107 | Seven heads plus only final transformer block | Conditional only after valid S1 evidence and separate S2 authorization. Head LR `3e-4`; encoder LR `1e-5`. |
| S3 | `M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL` | 35, 71, 107 per triggered head | One head; Encoder frozen | Conditional negative-transfer diagnosis only. Run for a head only if its shared-stage mean primary-metric delta is below -0.01 or its critical-boundary gate fails. |

S1 has three run units. S2 has three conditional run units. S3 has three run
units for each affected head (at most 21); it is never a silent all-head
replacement. Full unfreeze is deliberately absent from this contract.

## 6. Dev metrics and two distinct gate types

The primary metric for each of target mode, stance, emotion primary, emotion
target, action tendency, and context dependency is Macro-F1. Reasoning's
primary metric is Macro-F1; its Micro-F1 and exact-set accuracy are mandatory
secondary metrics. Every six-classifier report must also contain accuracy and
per-class precision, recall, F1, and support. Reasoning must contain per-label
precision, recall, F1, and support.

For each stage and each head, report all three seed values, mean, sample
standard deviation, minimum (the worst seed), and maximum. The M2 comparison
unit is a matching seed and stage with identical data, input, and metric
configuration. The aggregate mean of seven head metrics is contextual only;
no single aggregate, best seed, or favorable head can select a stage.

### 6.1 Stage progression gate — relative to S1 only

This gate decides only whether the existing experiment ladder may produce
evidence for the next separately authorized stage. It is **not** the final
M2 candidate-selection gate.

S1 is a frozen three-seed control. Once all three seeds, provenance/resource
checks, seven-head metrics, and seed-stability report complete, S1 may produce
`MAY_REQUEST_S2_OWNER_AUTHORIZATION`. S1 does **not** need to beat Classical,
and it must never be named `M2_SELECTED_CANDIDATE`.

The following label-level boundary proxies are mandatory: target mode
`ON_TARGET/CROSS_TARGET/MARKET_GENERAL/UNKNOWN`; stance
`BULL/BEAR/NEUTRAL/MIXED/UNKNOWN`; emotion primary
`CALM/NONE_EXPLICIT/FEAR/FOMO/UNKNOWN`; emotion target
`PRICE/POSITION/NOT_APPLICABLE/UNKNOWN`; action
`BUY/ADD/REDUCE/SELL/WATCH/NO_ACTION_SIGNAL/UNKNOWN`; every context-dependency
class; and Reasoning `NO_REASON_GIVEN`, `TECHNICAL_PRICE`, `FUNDAMENTAL`,
`VALUATION`, `SARCASM_IRONY`, and `UNKNOWN`.

A label with fewer than 20 Dev examples is reported as
`NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION`, never suppressed and never used to
claim a boundary pass. This accurately distinguishes insufficient weak-label
support from success.

For S2 or S3 against matching S1 seeds, all of the following are required:

- no mean primary-metric decrease greater than 0.01 for any head;
- no worst-seed primary-metric decrease greater than 0.03 for any head;
- no listed critical label with support at least 20 loses more than 0.05 F1;
- every head's sample standard deviation is at most 0.05;
- at least two heads improve by at least 0.01 in matching-seed mean primary
  metric, and no more than two are flat or lower.

A failure rejects that stage for progression. If the failure indicates
shared-head negative transfer, only the predeclared S3 control may be
considered, and only after a separate authorization. It never permits
configuration drift, Test inspection, a new model, or full fine-tuning.

### 6.2 Final M2 candidate selection / exit gate — relative to Classical

Only a complete authorized S2 seven-head candidate can enter this gate; S1 is
a control and S3 is per-head diagnostic evidence, never a selected seven-head
candidate. Passing S2/S3 matching-seed stage gates is necessary but not
sufficient. The final candidate must report, for all seven heads, every seed,
mean, and worst-seed delta against the frozen Classical Dev control above.

The final Classical numerical gate is:

- no head's **mean** Macro-F1 may be more than 0.01 below Classical;
- no head's **worst seed** Macro-F1 may be more than 0.03 below Classical;
- at least **four of seven** heads must have mean Macro-F1 at least 0.01 above
  Classical;
- for every critical class/label with Classical Dev support ≥20, candidate F1
  may not be more than 0.05 below Classical;
- a support <20 critical class/label is explicitly
  `NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION`; it cannot be shown as `PASS` or
  used to satisfy an improvement claim;
- Reasoning must report and pass the equivalent mean/worst tolerance for
  Macro-F1, Micro-F1, and exact-set accuracy, plus every per-label result.

Any Classical-gate failure produces `CANDIDATE_REJECTED_NOT_M2_SELECTED`.
When the authorized S1/S2/S3 ladder has no candidate that passes both the stage
and Classical gates, the only valid result is
`RBT3_M2_LINEAGE_REJECTED_PENDING_NEW_CANDIDATE_CONTRACT_AND_OWNER_AUTHORIZATION`.
It does not permit downloading or training a different model.

## 7. Device, resource, and stop rules

After separate authorization, use MPS first and CPU only when MPS is unavailable.
Do not aggregate MPS and CPU values into one stage result without an explicit
device-stratified report. Every completed seed checkpoint requires an offline
CPU reload/inference smoke using the fixed local cache and finite outputs for
all seven heads.

The M1 observed anchor is MPS, 12 epochs in 58.036 seconds, 157,525,306 cache
bytes, and 765,782 artifact bytes. Therefore S1 has an estimated raw fit time
of roughly three minutes for three seeds; reserve 15–30 minutes end-to-end and
0.1–1.0 GiB new artifact capacity. This is a planning estimate, not an M2
authorization or performance claim. Partial-unfreeze and CPU-training duration
are unmeasured and require separate measurement under hard stops.

The proposed ceilings for any authorized M2 run are 120 minutes per run and 10
GiB total new local disk. On either limit, stop, preserve only immutable
diagnostic evidence, and avoid nonessential copies. Each output must record
device, MPS availability, thread settings, wall time, peak memory if available,
cache/artifact/checkpoint bytes, and CPU reload result.

## 8. Provenance and authorization gates

Before each M2 fit, the runner must pass canonical audit with exit 0,
`training_allowed=true`, and no blocker codes; verify data/reference binding;
verify a clean tracked-source checkout; hash the fixed local cache; and hash
the frozen M2 contract/config. Its content-addressed manifest must record Git
HEAD, critical-source hashes, contract/config hashes, canonical audit ID,
data/reference/Schema IDs, cache/model/tokenizer hashes, M1 control identities,
environment/device, seed, metrics, resource evidence, checkpoint, and CPU
reload result.

The following require independent owner authorization, even if S1 succeeds:

- any M2 fit or cache use beyond this planning task;
- a new model/revision/license/download/hash plan;
- full fine-tuning after partial-unfreeze evidence and a new frozen contract;
- Gold or OOD creation/evaluation; LLM, cloud, or external API calls;
- Test unseal/metrics, Anchor evaluation, or production inference.

M2 is ready for an owner decision, not for execution. The companion
[`encoder-m2-experiment-contract-v1.json`](../manifests/encoder-m2-experiment-contract-v1.json)
is the machine-readable source of the same fail-closed rules.
