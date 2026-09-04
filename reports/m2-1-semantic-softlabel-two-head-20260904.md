# M2.1 semantic soft-label two-head diagnostic

Status: **SOFTLABEL_NO_GAIN_VS_HARD_LABEL_COMPARATOR** (comparison evidence only; not a seven-head candidate and not production approval).

This run trained exactly `emotion_primary` and `action_tendency` from the fixed local
`hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460` snapshot.  The
shared bottom encoder remained frozen, with one independent final block and output
head per experiment head.  The fixed soft target assigns 0.80 to the observed label
and distributes 0.20 across only the declared adjacent labels by normalized
similarity; labels without a declared neighbor remain one-hot.

## Dev selection (Train 1,822 fit; Dev 448 only)

The adoption decision was frozen before opening the human-review workbook.  A head
is adopted only when expected semantic score improves by at least 0.02 and strict
Macro-F1 does not drop by more than 0.01.

| Head | Current source | Current Macro / expected semantic | Soft Macro / expected semantic | Δ Macro | Δ expected semantic | Decision |
|---|---|---:|---:|---:|---:|---|
| `emotion_primary` | LERT seed 35 | 0.288480 / 0.391029 | 0.285457 / 0.360196 | -0.003023 | -0.030833 | KEEP_CURRENT_SOURCE |
| `action_tendency` | Classical frozen estimator | 0.164855 / 0.230910 | 0.285851 / 0.605933 | +0.120996 | +0.375023 | ADOPT_SOFT_LABEL_HEAD |

Strict Dev details (accuracy, adjacent Top-1 semantic score, true-label
probability) are preserved in the immutable artifact's `dev-selection.json` and
each `seed-metrics.json`.  The adopted action head's Dev accuracy is 0.734375,
adjacent Top-1 semantic score 0.744420, and true-label probability 0.590645.  The
retained emotion head's current-source values are the frozen LERT seed-35 baseline;
the soft run itself reached accuracy 0.419643, adjacent Top-1 0.474330, and
true-label probability 0.284674 but failed the semantic-improvement requirement.

## One post-selection human-review100 diagnostic

The workbook was opened once after the original Dev selection and was not used
for fitting, early stopping, threshold selection, or repeated experimentation.
The rows below preserve the historical M2.1 comparator (current LERT for emotion
and the soft-label head for action) against the Classical estimator.  They are
not an adopted unified candidate; the corrected decision retains Classical
`action_tendency` pending the three-seed hard-label balancing experiment.

| Head | M2.1 comparator Accuracy / Macro-F1 | Classical Accuracy / Macro-F1 | Δ Accuracy / Δ Macro-F1 | Δ expected semantic | Paired (comparator-only, Classical-only, both correct, both wrong) |
|---|---:|---:|---:|---:|---:|
| `emotion_primary` | 0.190 / 0.145957 | 0.180 / 0.141231 | +0.010 / +0.004726 | +0.076942 | (10, 9, 9, 72) |
| `action_tendency` | 0.320 / 0.281719 | 0.300 / 0.232986 | +0.020 / +0.048734 | +0.053678 | (17, 15, 15, 53) |

Per-class F1, confusion matrices, raw probability summaries, semantic auxiliary
scores, and every paired row comparison are in `human100-metrics.json` and
`human100-comparison.jsonl` in the artifact.  This 100-row diagnostic is not a
production acceptance decision.

## Execution evidence

- Artifact: `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-1-semantic-softlabel-two-head-20260904`
- Content address: `8b1ba76c3d5f87f72662545702a077ffaa7497010d3a6249794919832064741f`
- Actual device: MPS (`mps_available=true`) for both heads; policy was MPS-first with CPU fallback.
- `emotion_primary` best epoch 18; its resource log records 184.648 s cumulative from experiment start; checkpoint `emotion_primary/seed-35/softlabel-checkpoint.pt` (3,181,605 bytes); CPU reload finite with output shape `[1, 11]`.
- `action_tendency` best epoch 16; its resource log records 339.087 s cumulative from experiment start (about 154.439 s after the first head); checkpoint `action_tendency/seed-35/softlabel-checkpoint.pt` (3,179,557 bytes); CPU reload finite with output shape `[1, 9]`.
- Artifact directory size: 6,696,960 bytes (about 6.4 MiB); cache snapshot measured 61,155,014 bytes.  Both are below the 10 GiB incremental disk limit.
- Runtime recorded in `runtime.json`: Python 3.12.13, PyTorch 2.8.0, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.5.2.

The corrected M2.1 disposition is: **retain `emotion_primary`'s current LERT
seed-35 source and retain the current Classical `action_tendency` source**.  The
M2.1 soft-label Action checkpoint is a comparator only and is not integrated into
the unified candidate.  The three-seed hard-label balancing experiment is the
sole follow-up test for Action stability; all other five heads remain unchanged.

No Test, OOD, Anchor, reference predictions, LLM, cloud, or production data was
accessed.  No model was downloaded and no dependency was installed or changed.

## Verification

- Full pytest: **245 passed, 1 skipped** (246 collected; skip is the existing local immutable-package/runtime availability check).
- `compileall -q src`: passed.
- `.encoder-venv/bin/python -m pip check`: passed (`No broken requirements found`).
- `git diff --check`: passed.
