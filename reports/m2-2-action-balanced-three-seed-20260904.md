# M2.2 Action 类别不平衡稳定化

状态：**ACTION_BALANCED_THREE_SEED_GATE_PASSED**。这是 Action 头的诊断结果，
不是生产批准；只有 Action 的 seed-35 结果被放入新的 diagnostic unified
candidate，`selected_candidate=false`。

## 实验边界

- 固定模型：`hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460`，Apache-2.0。
- 只训练 `action_tendency`，每个 seed 都从官方原始 LERT snapshot 初始化；不从任何旧 checkpoint warm-start。
- seeds：35、71、107；Train 1,822 拟合；Dev 448 strict Macro-F1 early stopping/诊断。
- Encoder 的共享底部保持冻结；只训练 Action specialist final block 和 Action head。
- hard-label cross entropy；逐样本 Action 权重保持不变，并乘以 Train-only 的
  `clip(sqrt(N/(K*count[c])), 0.5, 4.0)` class weight。
- max length 256、`HEAD_TAIL`、batch 16、max epochs 24、patience 3、AdamW、
  block LR `1e-5`、head LR `5e-4`、weight decay `0.01`、gradient clip `1.0`。
- 本地 MPS 优先；三个 run 实际均使用 MPS；每个 checkpoint 均完成 CPU reload。

Train class counts and frozen class weights：

| Label | BUY | ADD | HOLD | DO_T | REDUCE | SELL | WATCH | NO_ACTION_SIGNAL | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train count | 85 | 25 | 138 | 28 | 23 | 121 | 156 | 1,240 | 6 |
| Class weight | 1.543276 | 2.845659 | 1.211193 | 2.688895 | 2.966805 | 1.293481 | 1.139175 | 0.500000 | 4.000000 |

## 三 seed Dev 结果

| Seed | Best epoch | Device | Accuracy | Macro-F1 | BUY | ADD | HOLD | DO_T | REDUCE | SELL | WATCH | NO_ACTION_SIGNAL | UNKNOWN | Supported predicted classes | Dominant ratio |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 12 | MPS | 0.716518 | 0.329535 | 0.444444 | 0.333333 | 0.352941 | 0.444444 | 0.000000 | 0.280000 | 0.264151 | 0.846498 | 0.000000 | 7 | 0.808036 |
| 71 | 13 | MPS | 0.723214 | 0.351362 | 0.511628 | 0.428571 | 0.377358 | 0.444444 | 0.000000 | 0.321429 | 0.226415 | 0.852410 | 0.000000 | 7 | 0.792411 |
| 107 | 16 | MPS | 0.727679 | 0.344273 | 0.541667 | 0.400000 | 0.407407 | 0.222222 | 0.000000 | 0.363636 | 0.307692 | 0.855828 | 0.000000 | 7 | 0.765625 |

Aggregate: mean Macro-F1 **0.341723**, sample standard deviation **0.011135**,
worst seed **0.329535**. Every seed predicted seven Dev-supported classes and no
seed exceeded the 90% single-class collapse limit. All five M2.2 gate checks passed:
mean ≥ 0.30, sample std ≤ 0.05, worst seed ≥ 0.25, at least five supported
predicted classes per seed, and no collapse.

Each checkpoint was reloaded on CPU with finite output shape `[1, 9]`:

- seed 35: `action_tendency/seed-35/balanced-hard-checkpoint.pt`, 113.242 s;
- seed 71: `action_tendency/seed-71/balanced-hard-checkpoint.pt`, 115.937 s;
- seed 107: `action_tendency/seed-107/balanced-hard-checkpoint.pt`, 131.031 s.

The per-seed resource logs record a combined run time of approximately 360.210 s
(about 6 minutes) and the CPU reload result is finite for all three seeds.

## Fixed comparisons

| Source | Seed 35 Macro-F1 | Seed 71 Macro-F1 | Seed 107 Macro-F1 | Mean | Sample std | Worst |
|---|---:|---:|---:|---:|---:|---:|
| Existing hard-label LERT | 0.333892 | 0.090709 | 0.328857 | 0.251153 | 0.138971 | 0.090709 |
| M2.1 soft-label LERT (seed 35 only) | 0.285851 | — | — | — | — | — |
| Classical frozen estimator | 0.164855 | — | — | — | — | — |
| M2.2 balanced hard-label LERT | 0.329535 | 0.351362 | 0.344273 | 0.341723 | 0.011135 | 0.329535 |

Relative to the existing hard-label LERT, M2.2 changes the Action mean by
`+0.090570` Macro-F1 and removes the seed-71 single-class collapse. Matching-seed
deltas are `-0.004357` (35), `+0.260653` (71), and `+0.015416` (107). Relative
to the M2.1 soft seed-35 comparator, M2.2 seed 35 is `+0.043684` Macro-F1.

## Human-review100 diagnostic

Because the three-seed gate passed, the existing workbook was opened exactly once
after Dev selection. It was not used for fitting, early stopping, class weights,
hyperparameter choice, or checkpoint selection.

| Source | Accuracy | Macro-F1 | adjacent Top-1 semantic | expected semantic | true-label probability |
|---|---:|---:|---:|---:|---:|
| M2.2 balanced Action seed 35 | 0.320 | 0.266118 | 0.343 | 0.307982 | 0.265590 |
| Classical | 0.300 | 0.232986 | 0.362 | 0.267365 | 0.209381 |
| M2.2 − Classical | +0.020 | +0.033133 | -0.019 | +0.040616 | +0.056209 |

The complete per-class F1, confusion matrices, raw probabilities and paired row
outcomes are in `human100-metrics.json`. The Human100 result is diagnostic only and
does not alter the Dev gate or head source decision.

## Diagnostic unified candidate

The new bundle is separate from and does not overwrite the prior unified candidate:

- Root experiment artifact:
  `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-2-action-balanced-three-seed-20260904`
- Root content address:
  `5cc1ddb52d81526c69381b70a0efea0291a72d13b0aaf5c7df9b2def3164f9a1`
- Unified candidate directory:
  `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-2-action-balanced-three-seed-20260904/unified-candidate`
- Unified candidate content address:
  `ca2d1399ab252a14d1fb33f4e6ee2581f2253b0d5c63d84ef0ebb5fdad3caf92`
- Bundle:
  `unified-candidate/unified-runtime-bundle.pt`
- Bundle SHA-256:
  `838890ad98cc9e8c05c9c108534e295e08fbde6178e25e4b222d79c818f06b89`
- A local CPU bundle smoke loaded the fixed snapshot plus all seven state groups for one Dev sample; all seven outputs were finite with shapes `target_mode [1,4]`, `stance [1,5]`, `emotion_primary [1,11]`, `emotion_target [1,7]`, `action_tendency [1,9]`, `reasoning_tags [1,15]`, and `context_dependency [1,4]`.

Head sources in the diagnostic bundle are:

- `target_mode`, `stance`, `emotion_primary`, `emotion_target`: existing LERT seed-35 sources;
- `action_tendency`: M2.2 balanced hard-label LERT seed-35 checkpoint;
- `context_dependency`: existing Classical frozen estimator;
- `reasoning_tags` labels 14: existing LERT seed-35 source;
- `reasoning_tags/SARCASM_IRONY`: existing Classical frozen estimator with the new bundle's Classical threshold fixed to **0.45**.

The old M2.1 soft-label Action checkpoint remains comparator evidence only and is
not integrated into either the prior or new unified candidate. Other five heads were
not trained or modified in M2.2.

## Repository and verification

- M2.1 status corrected to `SOFTLABEL_NO_GAIN_VS_HARD_LABEL_COMPARATOR`.
- The first post-seed aggregation attempt failed only on a preprocessing-contract
  lookup typo after all three checkpoints had been written; it was preserved at
  `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-2-action-balanced-three-seed-20260904-rejected-runtime-error`.
  The one-line fix was committed as `5eec436`; no model download or dependency
  change occurred.
- Main has one local branch and one primary worktree; the remote was not configured
  or modified.
- Full pytest: 251 passed, 1 skipped (252 collected; the skip is the existing local
  immutable-package/runtime availability check).
- `compileall -q src`: passed.
- `.encoder-venv/bin/python -m pip check`: passed.
- `git diff --check`: passed.
- Root manifest lists every artifact file and all listed SHA-256 values were
  recomputed successfully; the root content address above matches the manifest.
- Root artifact disk usage is 31,215,616 bytes (about 30 MiB); the preserved
  first-attempt rejected evidence is 9,793,536 bytes.  Both are far below the
  10 GiB experiment limit.

No Test, OOD, Anchor, LLM, cloud service, production data or production inference
was accessed. No model was downloaded and no dependency was installed.
