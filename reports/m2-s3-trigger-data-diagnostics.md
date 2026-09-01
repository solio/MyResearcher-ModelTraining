# M2 S3 trigger data diagnostics

Status: `COMPLETED_READ_ONLY_TRAIN_DEV_WEAK_LABEL_CONTEXT_WITH_S3_DISPOSITION`.

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
The script reads only the existing S1/S2 `seed-*/seed-metrics.json` files and,
when `--s3-metrics-root` is supplied, the two S3 head-specific
`seed-*/seed-metrics.json` files plus
`s3-vs-s1-matching-seed-report.json`. It does not load a model, checkpoint,
tokenizer, cache, prediction payload, or reference package.

The real diagnostic was run with:

```text
PYTHONPATH=src python tools/m2_s3_trigger_data_diagnostics.py \
  --config /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/configs/baseline_v0.3.5.yaml \
  --s1-metrics-root /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s1-first-three-seed-20260831 \
  --s2-metrics-root /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s2-partial-last-one-three-seed-20260831 \
  --s3-metrics-root /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-s3-frozen-single-task-triggered-heads-20260831 \
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
| `aggregate-report.json` (50,219 bytes) | `c51f24a3244b528a2eb7ba6a60fdd3b51fc854767bee09604f6bad519843a5ce` |
| `summary.md` (5,903 bytes) | `03b6f605a6120ccaaf9567c78a3300117a003cfa1baf8b86d71e40b131e080f9` |

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

## Matching S1/S2/S3 seed metrics

These values are read from the existing three S1/S2 Dev metric files and the
three S3 per-head Dev metric files. The S3 matching report was also read and
cross-checked against the seed files. Support is unchanged because the
weak-label Dev population is fixed at 448 rows.

| Trigger | Seed | S1 F1 / support | S2 F1 / support | S3 F1 / support | S3−S1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `emotion_primary:CALM` | 35 | 0.107143 / 48 | 0.160000 / 48 | 0.075472 / 48 | -0.031671 |
| `emotion_primary:CALM` | 71 | 0.000000 / 48 | 0.000000 / 48 | 0.103448 / 48 | +0.103448 |
| `emotion_primary:CALM` | 107 | 0.155844 / 48 | 0.072727 / 48 | 0.166667 / 48 | +0.010823 |
| `reasoning_tags:NO_REASON_GIVEN` | 35 | 0.267857 / 86 | 0.267857 / 86 | 0.252252 / 86 | -0.015605 |
| `reasoning_tags:NO_REASON_GIVEN` | 71 | 0.297521 / 86 | 0.295652 / 86 | 0.297521 / 86 | +0.000000 |
| `reasoning_tags:NO_REASON_GIVEN` | 107 | 0.295652 / 86 | 0.228571 / 86 | 0.164948 / 86 | -0.130704 |

At the full-head level, S3 emotion_primary Macro-F1 relative to S1 has mean
delta `+0.021769`, while reasoning_tags has mean delta `-0.004081`; therefore
the two single-task heads did not simultaneously improve. CALM has partial
recovery across seeds, while NO_REASON_GIVEN seed 107 did not recover. These
are weak-label Dev diagnostics, not a model-selection decision.

## HYPOTHESIS dispositions

1. **HYPOTHESIS_LABEL_COUPLING — `NOT_SUPPORTED_AS_A_SHARED_EXPLANATION`**

   The S3 emotion_primary Macro-F1 mean delta versus S1 is `+0.021769`, while
   the reasoning_tags mean delta is `-0.004081`. The two single-task heads did
   not both improve, so the observed CALM × NO_REASON_GIVEN overlap is not
   supported as a shared explanation by this S3 result.

2. **HYPOTHESIS_TEXT_LENGTH_SHIFT — `UNRESOLVED`**

   The current S3 artifact contains no per-sample character-length result, so
   the existing affected-versus-remaining length distribution cannot be
   confirmed or denied as an explanation.

3. **HYPOTHESIS_AFFECTED_HEAD_WEIGHT — `UNRESOLVED`**

   The current S3 artifact contains no per-sample weight-bucket result, so the
   affected-head weight distribution cannot be confirmed or denied as an
   explanation.

All three dispositions remain weak-label Dev diagnostics and are not root-cause
or model-route conclusions.

## Validation and limitations

- Focused synthetic suite: **9 passed** in
  `tests/test_m2_s3_trigger_data_diagnostics.py`.
- Scope-safe repository suite: **163 passed** with
  `pytest -q -m 'not real_data'`; 4 real-data tests were excluded by marker.
- `compileall` over `tools`, `tests`, and `src`: passed.
- `git diff --check`: passed.
- The script does not train, fit, infer, load a checkpoint/model/cache, call an
  LLM/cloud service, download anything, or open Test, Anchor, Gold, OOD, or
  reference predictions.
- The existing S1/S2/S3 metrics are treated as weak-label Dev diagnostics only;
  this report cannot establish truth, generalization, production quality, or
  authorize model selection or any further S3 execution.
