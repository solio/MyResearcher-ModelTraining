# M3-G frozen candidate Test 467 closeout

Status: `M3_IN_DISTRIBUTION_WEAK_LABEL_TREND_POSITIVE_DIAGNOSTIC_ONLY`
`production_approval=false`
Technical run record: `M3G_TEST_467_COMPLETED_DIAGNOSTIC`
This is one frozen weak-label Test generalization evaluation. It does not select a seed, tune a threshold, modify routing or preprocessing, retrain, or create a new candidate. The result is not Gold or production acceptance.

The Test partition had already been used by the earlier M3-A candidate evaluation. This closeout therefore records `REUSED_TEST_DIAGNOSTIC_NOT_PRISTINE_HOLDOUT`; it is diagnostic evidence only and must not be used for further tuning.

## Fixed inputs and execution

- Candidate: `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904`
- Candidate content address: `8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf`
- Model/cache and all head sources were loaded from that immutable candidate and its fixed local snapshot.
- Test role: `teacher_test_467_v0.3.5.jsonl` weak labels joined to canonical inputs by `sample_id`; exactly 467 rows.
- Device: CPU. Training invoked: false. Candidate modified: false.
- Reasoning thresholds came from the selected artifact. No Test threshold selection or “best threshold” was computed.
- Anchor, Embargo, Gold, Human100, OOD, reference predictions, LLM, cloud, and production data were not read.

The saved M3-G summary and run manifest are unchanged. Scalar reports use two explicit aggregation conventions: `macro_f1_supported` averages only classes whose truth support is greater than zero, while `macro_f1_schema_all` averages every class in the frozen Schema order. The historical `macro_f1` field in the evaluator remains an alias for `macro_f1_schema_all` for backward compatibility. For every scalar head except `context_dependency`, the Test truth has support in every Schema class, so the two values are equal. For `context_dependency`, supported Macro-F1 is `0.450779`, schema-all Macro-F1 is `0.338084`; the candidate predictions are identical to the same-runtime Classical predictions.

Command:

```text
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.evaluate_m3g_test \
  --config configs/baseline_v0.3.5.yaml \
  --artifact-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904 \
  --cache-dir .encoder-artifacts/hf-cache-lert-small \
  --output-dir runs/m3g-test467-20260907
```

## Scalar heads

| head | accuracy | macro_f1_supported | macro_f1_schema_all |
|---|---:|---:|---:|
| target_mode | 0.847966 | 0.455382 | 0.455382 |
| stance | 0.471092 | 0.365111 | 0.365111 |
| emotion_primary | 0.419700 | 0.274827 | 0.274827 |
| emotion_target | 0.479657 | 0.319525 | 0.319525 |
| action_tendency | 0.715203 | 0.289737 | 0.289737 |
| context_dependency | 0.603854 | 0.450779 | 0.338084 |

Per-class values are `precision / recall / F1 / support`:

- `target_mode`: CROSS_TARGET `0.7571/0.6235/0.6839/85`; MARKET_GENERAL `0.2857/0.1818/0.2222/11`; ON_TARGET `0.8744/0.9606/0.9154/355`; UNKNOWN `0/0/0/16`.
- `stance`: BEAR `0.4817/0.5486/0.5130/144`; BULL `0.5250/0.3134/0.3925/67`; MIXED `0/0/0/5`; NEUTRAL `0.3520/0.3929/0.3713/112`; UNKNOWN `0.5507/0.5468/0.5487/139`.
- `emotion_primary`: ANGER `0.4211/0.4286/0.4248/56`; ANXIETY `0.3462/0.4615/0.3956/39`; CALM `0.4444/0.1111/0.1778/36`; EXCITEMENT `0.3077/0.0952/0.1455/42`; FEAR `0.3333/0.2500/0.2857/20`; FOMO `0/0/0/2`; FRUSTRATION `0.4451/0.6864/0.5400/118`; HOPE `0.4000/0.1875/0.2553/32`; NONE_EXPLICIT `0.4505/0.4808/0.4651/104`; REGRET `0.3077/0.3636/0.3333/11`; UNKNOWN `0/0/0/7`.
- `emotion_target`: COMPANY `0.4889/0.3667/0.4190/60`; MARKET `0.5000/0.1304/0.2069/23`; NOT_APPLICABLE `0.4333/0.3750/0.4021/104`; OTHER `0.1875/0.1111/0.1395/27`; POSITION `0.5000/0.4375/0.4667/80`; PRICE `0.5083/0.7394/0.6025/165`; UNKNOWN `0/0/0/8`.
- `action_tendency`: ADD `0.1667/0.1000/0.1250/10`; BUY `0.4118/0.4118/0.4118/17`; DO_T `0.5000/0.2308/0.3158/13`; HOLD `0.7273/0.2963/0.4211/27`; NO_ACTION_SIGNAL `0.7942/0.9177/0.8515/328`; REDUCE `0/0/0/4`; SELL `0.3548/0.3438/0.3492/32`; UNKNOWN `0/0/0/2`; WATCH `0.2727/0.0882/0.1333/34`.
- `context_dependency`: EXTERNAL_CONTEXT_REQUIRED `0.3125/0.2381/0.2703/21`; PARTIAL_CONTEXT `0.2848/0.4563/0.3507/103`; SELF_CONTAINED `0.8042/0.6706/0.7313/343`; UNKNOWN `0/0/0/0`.

The complete row/column confusion matrices (actual → predicted counts) are retained in `test-467-summary.json`; the non-zero cells are also reproducible from the 467-row JSONL. For example, the dominant cells are `target_mode ON_TARGET→ON_TARGET=341`, `stance BEAR→BEAR=79`, `emotion_primary FRUSTRATION→FRUSTRATION=81`, `emotion_target PRICE→PRICE=122`, `action_tendency NO_ACTION_SIGNAL→NO_ACTION_SIGNAL=301`, and `context_dependency SELF_CONTAINED→SELF_CONTAINED=230`.

## Candidate versus same-runtime Classical

The following deltas join the saved M3-G candidate metrics with the saved same-runtime Classical Test metrics from M3-A (`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m3a-head-selected-test-evaluation-20260902/metrics.json`, SHA-256 `f1bb56a9fb0831894a487099d0e2a98f01ed601a98f8dc2ad9bfc00074ad89cb`). No Test rows or prediction files were reread for this correction. Scalar deltas are candidate minus Classical; for `context_dependency`, both supported and schema-all conventions are shown.

| head | candidate accuracy | Classical accuracy | Δ accuracy | candidate macro_f1_supported | Classical macro_f1_supported | Δ supported | candidate macro_f1_schema_all | Classical macro_f1_schema_all | Δ schema-all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_mode | 0.847966 | 0.591006 | +0.256960 | 0.455382 | 0.349940 | +0.105442 | 0.455382 | 0.349940 | +0.105442 |
| stance | 0.471092 | 0.372591 | +0.098501 | 0.365111 | 0.307857 | +0.057254 | 0.365111 | 0.307857 | +0.057254 |
| emotion_primary | 0.419700 | 0.239829 | +0.179871 | 0.274827 | 0.190240 | +0.084587 | 0.274827 | 0.190240 | +0.084587 |
| emotion_target | 0.479657 | 0.301927 | +0.177730 | 0.319525 | 0.226453 | +0.093072 | 0.319525 | 0.226453 | +0.093072 |
| action_tendency | 0.715203 | 0.267666 | +0.447537 | 0.289737 | 0.141130 | +0.148607 | 0.289737 | 0.141130 | +0.148607 |
| context_dependency | 0.603854 | 0.603854 | +0.000000 | 0.450779 | 0.450779 | +0.000000 | 0.338084 | 0.338084 | +0.000000 |
| reasoning_tags | — | — | — | 0.514047 | 0.379239 | +0.134808 | — | — | — |

Reasoning same-runtime deltas are Macro-F1 `+0.134808`, Micro-F1 `+0.117453`, and exact-set accuracy `+0.141327`. `SARCASM_IRONY` is the fixed Classical fallback and its same-runtime delta is exactly `0.000000`; a difference observed in a reference environment is not a model degradation here.

## Reasoning tags

- Macro-F1: `0.514047`
- Micro-F1: `0.593540`
- Exact-set accuracy: `0.282655`

Per-label values are `precision / recall / F1 / support`:

| label | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| FUNDAMENTAL | 0.8485 | 0.6512 | 0.7368 | 43 |
| VALUATION | 0.2000 | 0.1538 | 0.1739 | 13 |
| TECHNICAL_PRICE | 0.6957 | 0.6845 | 0.6900 | 187 |
| FLOW_POSITIONING | 0.6628 | 0.6786 | 0.6706 | 84 |
| NEWS_EVENT | 0.6098 | 0.6410 | 0.6250 | 39 |
| RUMOR | 1.0000 | 0.2000 | 0.3333 | 5 |
| SOCIAL_PROOF | 0.6111 | 0.6286 | 0.6197 | 35 |
| MACRO_POLICY | 0.6250 | 0.6818 | 0.6522 | 22 |
| THEME_NARRATIVE | 0.7347 | 0.6429 | 0.6857 | 56 |
| NO_REASON_GIVEN | 0.4488 | 0.6404 | 0.5278 | 89 |
| RELATIVE_PERFORMANCE | 0.6129 | 0.7170 | 0.6609 | 53 |
| CROSS_STOCK_REFERENCE | 0.6000 | 0.3571 | 0.4478 | 42 |
| SARCASM_IRONY | 0.1034 | 0.1111 | 0.1071 | 27 |
| WORDPLAY | 0.2593 | 0.2258 | 0.2414 | 31 |
| UNKNOWN | 0.6364 | 0.4667 | 0.5385 | 15 |

`SARCASM_IRONY` remains the fixed Classical fallback in the frozen candidate; the evaluation did not alter its source or threshold. Full probability vectors and predicted tag sets are in `test-467-predictions.jsonl`.

## Artifact

Ignored run directory: `/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/runs/m3g-test467-20260907`  
Evaluation content address: `8e38ab9cb05c22d610d17d84174187edc0b41fb29e8d1c5dbc1618b346ba3aa1`  
Files: 467 prediction rows, summary metrics, and a content-addressed manifest. The manifest and every listed file hash were independently recomputed successfully.

This closes the current model stage with status `M3_IN_DISTRIBUTION_WEAK_LABEL_TREND_POSITIVE_DIAGNOSTIC_ONLY`. No tuning, retraining, candidate replacement, or production approval follows from these weak-label Test results; `production_approval=false`.
