# M6B — v0.3 sparse auxiliary pilot preparation

Status: **prepared, not imported** (annotation only; no answers and no training).

This closeout regenerated the original unimported directory in place after a
selection-rule review.  The previous selection is rejected; only the current
JSONL below is to be considered for a future import.

The package is derived from the finalized Human Train100 pool
(`runs/feedback-loop-v3-finalized-20260910/human-train-100-seven-head.jsonl`).
The corresponding V1/V2/FL3 `selection-train.jsonl` rows were used only to
recover the canonical text and safe metadata for those same 100 sample IDs.
No Check, Challenge/Human48, Test, Gold, OOD, or model artifact was used.

## Frozen package

- Batch ID: `m6b-v03-auxiliary-pilot-20260911`
- Package: `/Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m6b-v03-auxiliary-pilot-20260911/auxiliary-pilot-tasks.jsonl`
- Summary: `/Users/mac/Documents/trae_projects/MyResearcher/human-review-artifacts/m6b-v03-auxiliary-pilot-20260911/summary.json`
- Samples: **20** (20 distinct `exact_content_group` values)
- Sparse assignments: **34**
- Length coverage: 4 short, 12 medium, 4 long
- Source coverage: V1 2, V2 4, FL3 14
- Truncated rows excluded: **11** (all old 40-character cuts plus the known ID)

Assignment counts are:

| auxiliary factor | tasks |
| --- | ---: |
| `expectation_phase` | 10 |
| `arousal_level` | 4 |
| `desired_price_direction` | 7 |
| `market_scope` | 1 |
| `discourse_focus` | 10 |

Each JSONL row contains only `sample_id`, full `text`, a title (the existing
stock-name title convention), safe metadata, and the list of factors that have
textual applicability cues.  The finalized Human Train100 seven-head answers
are retained in memory only to determine applicability: `market_scope` is
emitted only when `emotion_target=MARKET`, and `arousal_level` requires a
non-sentinel `emotion_primary` plus explicit affect evidence.  The answers are
never serialized.  Model predictions, probabilities, and thresholds are also
omitted.  Price-direction tasks require an actual price/result cue; author
actions such as 买入、卖出、清仓、加仓 or 接回 alone do not qualify.  Discourse
tasks require a participant, claim, or narrative cue; sentence-final particles
alone do not qualify.  Blank, punctuation-only, known placeholders, and old
40-character boundary cuts are excluded; short but semantic text remains
eligible.

The four previous `market_scope` rows were replaced because their Human Train
parent was not `MARKET`:

- removed: `fl1-c00ec59f89abaf37f537db47215072854a135f0d6b6df73e8d98cb8b6ce02354`
- removed: `fl1-34ee949ab855b6a92f3648fb9f4567207e6292ca72134b5d33ed4df01b7ecf2d`
- removed: `fl1-6e6cb42336ca0a41ce2fb4106998affa6e446b78994a083773ee565341e4abaf`
- removed: `fl1-ffc95267910aff57bc84453d3b0fe377c2e733ea01163fdd29225b8f43080ca9`
- replacement: `fl1-f2bb9c41addb1b396723b28ecd002ee625b6c7f291706ddb393ad272143b66c7`

The replacement is the only current `market_scope` task and its finalized
Human Train parent is `emotion_target=MARKET`.

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
