# M4 Human100 supervision correction diagnostic

Status: **HUMAN_SUPERVISION_NO_GAIN_DIAGNOSTIC**

This was one fixed seed (`35`) treatment. The five trained heads appended all
100 reviewed rows exactly once with weight `1.0`; the original Train 1,822
rows and their per-head weights were retained. `context_dependency` and
`reasoning_tags` were reused unchanged from the selected candidate. Human48
was read only after all five checkpoints were frozen and was not used for fit,
early stopping, threshold selection, or checkpoint selection.

Artifact: `/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m4-human-supervision-20260908-retry`

Content address: `25ef864056e105d33de20115c55bfcf78d6e7e4f8364c057623b6c6434a434af`

| head | baseline Human48 | treatment Human48 | delta | baseline supported Macro-F1 | treatment supported Macro-F1 |
|---|---:|---:|---:|---:|---:|
| target_mode | 28/48 | 27/48 | -1 | 0.315343 | 0.271721 |
| stance | 16/48 | 16/48 | 0 | 0.281091 | 0.289149 |
| emotion_primary | 12/48 | 13/48 | +1 | 0.166905 | 0.180383 |
| emotion_target | 10/48 | 10/48 | 0 | 0.142489 | 0.162865 |
| action_tendency | 20/48 | 19/48 | -1 | 0.159957 | 0.144864 |
| **total / 240** | **86** | **85** | **-1** | — | — |

The predeclared gate failed: treatment did not reach `96/240`, fewer than
three heads improved, and the total decreased by one (the per-head decrease
limit itself was not violated). Therefore this single Human100 addition is
insufficient to correct the observed errors. No candidate replacement is
formed; the existing selected candidate remains the source for all seven
logical heads.

All five runs used MPS (`seed-35`), with elapsed times 46.445s, 33.829s,
33.847s, 33.837s, and 55.708s (total 203.666s). The output occupied about
37.6 MiB; each checkpoint passed the required CPU reload finite-output smoke.

Reproduction command:

```text
PYTHONPATH=src .encoder-venv/bin/python -m semantic_model.encoder_m4_human_supervision \
  --config configs/baseline_v0.3.5.yaml \
  --cache-dir .encoder-artifacts/hf-cache-lert-small \
  --baseline-artifact /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904 \
  --workbook /Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m3b-gold100-20260902/m3b-gold100-review-pack.xlsx \
  --review-export /tmp/m3c-shadow500-review50-20260908-all.48741.jsonl \
  --shadow-input runs/m3c-shadow500-20260907/shadow500-input.jsonl \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m4-human-supervision-20260908-retry
```

The full shared review export was 350 rows (50 samples × 7 heads); the two
`EXCLUDE_SEMANTIC` sample IDs were excluded as whole samples, leaving 48
Human48 rows. The run read no Test, Anchor, OOD, LLM, cloud, or production
data, and did not alter the workbook or any prior artifact.
