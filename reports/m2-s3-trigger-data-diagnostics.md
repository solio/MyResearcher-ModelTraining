# M2 S3 trigger data diagnostics

Status: `COMPLETED_READ_ONLY_TRAIN_DEV_WEAK_LABEL_CONTEXT`.

This Side report is a data-only diagnostic background for the S2-triggered
`emotion_primary:CALM` and `reasoning_tags:NO_REASON_GIVEN` cases. Every label,
count, ratio, F1, support, and delta below is a frozen weak-label observation.
Nothing here is Gold, truth, model-selection evidence, a production result, or
an authorization to start S3.

## Implementation and command

The implementation is in
`tools/m2_s3_trigger_data_diagnostics.py` and uses the existing
`semantic_model.encoder_m1.load_m1_partitions` loader. That loader reads only
Train/Dev split labels and the selected canonical-input and field-weight rows.
The script reads only the existing S1/S2 `seed-*/seed-metrics.json` files for
the two requested Dev metrics. It does not load a model, checkpoint, tokenizer,
cache, prediction payload, or reference package.

The real diagnostic was run with:

```text
PYTHONPATH=src python tools/m2_s3_trigger_data_diagnostics.py \
  --config /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/configs/baseline_v0.3.5.yaml \
  --s1-metrics-root /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s1-first-three-seed-20260831 \
  --s2-metrics-root /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s2-partial-last-one-three-seed-20260831 \
  --output-dir runs/m2-s3-trigger-data-diagnostics
```

Only these two small ignored output files were generated:

```text
runs/m2-s3-trigger-data-diagnostics/aggregate-report.json
runs/m2-s3-trigger-data-diagnostics/summary.md
```

Their SHA-256 values are:

| Output | SHA-256 |
| --- | --- |
| `aggregate-report.json` (47,915 bytes) | `00988f69bca52417b1d113da78bec1c4911abac7e0aff5d0b550bfb6b64a559d` |
| `summary.md` (6,179 bytes) | `2321f1f77eb10e593953edcea425b26e8eef7bd1d036b98adda8ae7bdecf6f60` |

No original text, sample-level prediction, model artifact, or checkpoint is
written. Text is used only to calculate character-length buckets.

## Target prevalence and affected-head weights

The affected union means `emotion_primary == CALM` or
`NO_REASON_GIVEN in reasoning_tags`; the overlap is counted once.

| Population | Rows | CALM | NO_REASON_GIVEN | Affected union | Overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,822 | 158 (8.7%) | 316 (17.3%) | 454 (24.9%) | 20 |
| Dev | 448 | 48 (10.7%) | 86 (19.2%) | 128 (28.6%) | 6 |
| Train + Dev | 2,270 | 206 (9.1%) | 402 (17.7%) | 582 (25.6%) | 26 |

The affected-head weight is the `emotion_primary` weight for CALM rows and the
`reasoning_tags` weight for NO_REASON_GIVEN rows. In Train + Dev:

- CALM emotion-head weights: `n=206`, min `0.126`, mean `0.492223`, p50
  `0.3`, p90/p95 `1.0`, max `1.1`, zero fraction `0.0%`.
- NO_REASON_GIVEN reasoning-head weights: `n=402`, min `0.42`, mean `0.843771`,
  p50/p90/p95 `1.0`, max `1.1`, zero fraction `0.0%`.

Exact value-count distributions for Train, Dev, and Train + Dev are retained in
the aggregate JSON; no global sample weight is substituted for per-head
weights.

## Co-occurrence and cross distribution

For CALM, the aggregate contains all five other scalar heads plus the complete
reasoning head as 15 tag-membership counts. Notable Train + Dev counts include
`target_mode:ON_TARGET=158`, `stance:BULL=60`, `stance:BEAR=49`,
`emotion_target:POSITION=98`, `action_tendency:NO_ACTION_SIGNAL=75`,
`context_dependency:SELF_CONTAINED=169`, and reasoning tags
`TECHNICAL_PRICE=112`, `FUNDAMENTAL=34`, `THEME_NARRATIVE=31`,
`FLOW_POSITIONING=30`, `NO_REASON_GIVEN=26`.

For NO_REASON_GIVEN, all six scalar heads are included. Notable Train + Dev
counts include `target_mode:ON_TARGET=383`, `stance:UNKNOWN=183`,
`emotion_primary:FRUSTRATION=65`, `emotion_target:POSITION=132`,
`action_tendency:NO_ACTION_SIGNAL=231`, and
`context_dependency:SELF_CONTAINED=289`. Every one of the other 14 reasoning
tags has count `0`; this is a weak-label data fact, not a semantic truth claim.

| Population | CALM ∩ NO_REASON_GIVEN | CALM only | NO_REASON_GIVEN only | Neither |
| --- | ---: | ---: | ---: | ---: |
| Train | 20 | 138 | 296 | 1,368 |
| Dev | 6 | 42 | 80 | 320 |
| Train + Dev | 26 | 180 | 376 | 1,688 |

## Character-length buckets

Buckets are character counts of `model_text`: `0-19`, `20-49`, `50-99`,
`100-199`, `200-399`, and `400+`. No text is emitted.

| Population | Affected rows by bucket | Remaining rows by bucket |
| --- | --- | --- |
| Train | `365, 56, 7, 23, 3, 0` | `851, 381, 36, 93, 7, 0` |
| Dev | `96, 27, 2, 2, 0, 1` | `209, 76, 11, 21, 3, 0` |
| Train + Dev | `461, 83, 9, 25, 3, 1` | `1060, 457, 47, 114, 10, 0` |

The bucket order is the order listed above. In Train + Dev, 79.2% of affected
rows are in `0-19`, versus 62.8% of remaining rows; this is a distributional
observation only.

## Matching S1/S2 seed metrics

These values are read from the existing three S1/S2 Dev metric files. Support
is unchanged because the weak-label Dev population is fixed at 448 rows.

| Trigger | Seed | S1 F1 / support | S2 F1 / support | Delta (S2−S1) |
| --- | ---: | ---: | ---: | ---: |
| `emotion_primary:CALM` | 35 | 0.107143 / 48 | 0.160000 / 48 | +0.052857 |
| `emotion_primary:CALM` | 71 | 0.000000 / 48 | 0.000000 / 48 | +0.000000 |
| `emotion_primary:CALM` | 107 | 0.155844 / 48 | 0.072727 / 48 | -0.083117 |
| `reasoning_tags:NO_REASON_GIVEN` | 35 | 0.267857 / 86 | 0.267857 / 86 | +0.000000 |
| `reasoning_tags:NO_REASON_GIVEN` | 71 | 0.297521 / 86 | 0.295652 / 86 | -0.001869 |
| `reasoning_tags:NO_REASON_GIVEN` | 107 | 0.295652 / 86 | 0.228571 / 86 | -0.067081 |

Mean F1 is `0.087662 → 0.077576` (delta `-0.010087`) for CALM and
`0.287010 → 0.264027` (delta `-0.022983`) for NO_REASON_GIVEN. These are
weak-label Dev diagnostics, not a model-selection decision.

## HYPOTHESIS (bounded, testable in S3)

1. **HYPOTHESIS_LABEL_COUPLING** — The 26-row overlap may indicate a correlated
   weak-label bundle. S3 would support this only if CALM and reasoning
   single-task results both improve their corresponding Dev weak-label F1,
   especially after separating overlap rows; no improvement or a disappearance
   of the gain after separation would weaken/deny it.
2. **HYPOTHESIS_TEXT_LENGTH_SHIFT** — The affected rows have a different length
   mix, with a larger short-text share. S3 would support this only if gains
   concentrate in the affected buckets while remaining buckets stay stable;
   flat or matching unaffected-row patterns would weaken/deny it.
3. **HYPOTHESIS_AFFECTED_HEAD_WEIGHT** — The observed affected-head weight
   distributions may contribute to trigger behavior even though neither target
   has zero weight. S3 would support this only if trigger-label recall/F1 gains
   track the lower-weight buckets under the fixed evaluation boundary; no such
   relationship would weaken/deny it.

These are hypotheses for later controlled S3 analysis, not root-cause claims.

## Validation and limitations

- Focused synthetic suite: **7 passed** in
  `tests/test_m2_s3_trigger_data_diagnostics.py`.
- Scope-safe repository suite: **163 passed** with
  `pytest -q -m 'not real_data'`; 4 real-data tests were excluded by marker.
- `compileall` over `tools`, `tests`, and `src`: passed.
- `git diff --check`: passed.
- The script does not train, fit, infer, load a checkpoint/model/cache, call an
  LLM/cloud service, download anything, or open Test, Anchor, Gold, OOD, or
  reference predictions.
- The existing S1/S2 metrics are treated as weak-label Dev diagnostics only;
  this report cannot establish truth, generalization, production quality, or
  authorize model selection/S3.
