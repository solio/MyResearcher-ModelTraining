# M3-B Gold100 model-effect evaluation

This is a one-time, owner-authorized evaluation of the frozen M2 head-selected
candidate on the 100 rows in the supplied human-review workbook. It is trend
evidence only; it is not production acceptance and it did not change the
candidate, thresholds, labels, or training artifacts.

## Frozen sources and mapping

- Candidate: `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-head-selected-unified-candidate-20260902`, content address `1a55776e95ccd6c95010bd849638db06d17eecb031a8d1b00987a8a4c56eb658`.
- Workbook: `human-review-artifacts/m3b-gold100-20260902/m3b-gold100-review-pack.xlsx`, SHA-256 `61335766ba314d4e15897335edab42c4c474b1facd566e9083b6876891545ec4`.
- Candidate LERT heads: `target_mode`, `stance`, `emotion_primary`, `emotion_target`, and 14 reasoning labels from seed 35; fixed model `hfl/chinese-lert-small@69e3e69ba258be5b301b26937e5b55a076c90460`.
- Candidate Classical fallback: `action_tendency`, `context_dependency`, and `SARCASM_IRONY`; the Classical estimator was loaded from the byte-preserved candidate copy (`original_model_v0.3.5.joblib`, SHA-256 `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`).
- Human `human_target_mode`, `human_stance`, `human_emotion_primary`, `human_emotion_target`, and `human_action_tendency` were used as entered. `human_context_dependency` was interpreted as the single primary reasoning tag. Empty `human_reasoning_tags` was ignored. `context_dependency=SELF_CONTAINED` was not scored as per-row Gold. Shared-string index 20 was decoded as an empty value.

## Strict single-label results (n=100)

| Head | Candidate Accuracy / Macro-F1 | Classical Accuracy / Macro-F1 | Candidate − Classical | Paired outcomes (candidate-only / Classical-only / both wrong) |
|---|---:|---:|---:|---:|
| target_mode | 0.850 / 0.332 | 0.540 / 0.281 | +0.310 / +0.051 | 39 / 8 / 7 |
| stance | 0.420 / 0.302 | 0.320 / 0.265 | +0.100 / +0.037 | 22 / 12 / 46 |
| emotion_primary | 0.190 / 0.146 | 0.180 / 0.141 | +0.010 / +0.005 | 10 / 9 / 72 |
| emotion_target | 0.360 / 0.206 | 0.160 / 0.096 | +0.200 / +0.110 | 24 / 4 / 60 |
| action_tendency | 0.300 / 0.233 | 0.300 / 0.233 | +0.000 / +0.000 | 0 / 0 / 70 |

Per-class F1 and support, full confusion matrices, raw top-1 and true-label
probability summaries, per-row correctness outcomes, and paired bootstrap CIs
are in the immutable artifact `metrics.json` and `comparison.jsonl`.

## Primary reasoning-tag results

Reasoning is reported only as primary-tag Top-1/Top-3, human-label probability
and rank, and prediction-set inclusion. No full multi-label exact-set accuracy
was computed, and unselected tags were not treated as negative labels.

| Model | Top-1 | Top-3 | Set contains human | Human-label probability | Mean rank | Top-1 semantic similarity |
|---|---:|---:|---:|---:|---:|---:|
| Candidate hybrid | 0.470 | 0.640 | 0.580 | 0.505 | 4.40 | 0.484 |
| Classical | 0.340 | 0.580 | 0.530 | 0.453 | 4.47 | 0.340 |
| Candidate − Classical | +0.130 | +0.060 | +0.050 | +0.051 | −0.07 | +0.144 |

The fixed semantic auxiliary matrix uses same-label similarity 1, undeclared
relations 0, and exactly the owner-specified symmetric relations. It is always
reported beside, never substituted for, strict exact-label metrics.

## Deterministic uncertainty and interpretation

Paired row bootstrap uses 10,000 resamples, seed `20260903`, percentile 95%
intervals. Accuracy-delta intervals are `[+0.19,+0.43]` for `target_mode`,
`[-0.01,+0.21]` for `stance`, `[-0.08,+0.10]` for `emotion_primary`,
`[+0.10,+0.30]` for `emotion_target`, and `[0.00,0.00]` for
`action_tendency`. Reasoning Top-1 delta is `[+0.05,+0.22]`.

Therefore, `target_mode`, `emotion_target`, and primary reasoning show the
clearest improvement over Classical on this small human sample. `stance` is a
promising but uncertain positive trend; `emotion_primary` is effectively a
tie at this sample size; `action_tendency` is an exact tie because it is the
Classical fallback by construction. No evaluated head regressed on the point
estimates.

The result supports continuing the current frozen head-selection route as a
diagnostic direction and makes an additional human sample batch worthwhile,
especially to resolve `stance`/`emotion_primary` uncertainty and verify the
reasoning trend. It does not authorize retuning, retraining, production
inference, or a production approval decision.

## Access and artifact evidence

Output artifact: `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m3b-gold100-human-evaluation-20260903` with content address
`b9bb547c71c2ecc48b219e5d8779b2bd4fec21ad8b9bcaea08a590fb065d5d0a`.

Only the workbook, frozen candidate checkpoint/threshold files, candidate
Classical estimator, fixed local tokenizer/model snapshot, Schema, contract,
and preprocessing contract were read. Test 467, M3-A Test predictions,
Anchor, reference predictions, Gold/OOD packages, LLM/cloud services, and
production data were not accessed. No training, threshold tuning, model
download, dependency change, or artifact overwrite occurred.
