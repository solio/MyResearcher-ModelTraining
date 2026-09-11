# M6B — v0.3 sparse auxiliary pilot preparation

Status: **prepared, not imported** (annotation only; no answers and no training).

The package is derived from the finalized Human Train100 pool
(`runs/feedback-loop-v3-finalized-20260910/human-train-100-seven-head.jsonl`).
The corresponding V1/V2/FL3 `selection-train.jsonl` rows were used only to
recover the canonical text and safe metadata for those same 100 sample IDs.
No Check, Challenge/Human48, Test, Gold, OOD, or model artifact was read.

## Frozen package

- Batch ID: `m6b-v03-auxiliary-pilot-20260911`
- Package: `/Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m6b-v03-auxiliary-pilot-20260911/auxiliary-pilot-tasks.jsonl`
- Summary: `/Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m6b-v03-auxiliary-pilot-20260911/summary.json`
- Samples: **20** (20 distinct `exact_content_group` values)
- Sparse assignments: **43**
- Length coverage: 3 short, 13 medium, 4 long
- Source coverage: V1 2, V2 6, FL3 12

Assignment counts are:

| auxiliary factor | tasks |
| --- | ---: |
| `expectation_phase` | 10 |
| `arousal_level` | 4 |
| `desired_price_direction` | 13 |
| `market_scope` | 4 |
| `discourse_focus` | 12 |

Each JSONL row contains only `sample_id`, full `text`, a title (the existing
stock-name title convention), safe metadata, and the list of factors that have
textual applicability cues.  The existing seven top-level answers, auxiliary
answers, model predictions, probabilities, and thresholds are intentionally
omitted.  Cues choose which question is worth asking; they never prefill an
answer.  Blank, punctuation-only, and known placeholder inputs are excluded;
short but semantic text remains eligible.

## Import prerequisite

An operator must review the answer-free JSONL and explicitly convert/import it
through the existing labeling application.  Import has not been run and no
batch or assignment was created by this preparation command.  When imported,
each listed factor must become its own sparse v0.3 assignment; unanswered or
inapplicable factors remain absent (`loss_mask=0`), while an explicit human
`UNKNOWN` answer is a real final answer (`loss_mask=1`).  The resulting labels
remain revisable human labels, not Gold data.

Reproducible preparation command:

```bash
.venv/bin/python tools/prepare_v03_auxiliary_pilot.py
```
