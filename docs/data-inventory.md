# Local Data Inventory

Snapshot date: 2026-08-27. Source root was scanned read-only:

`/Users/mac/Documents/trae_projects/MyResearcher/produce-docs/MyResearcher_Semantic_Sampling_Local_Pipeline_Formal_v0.1/formal_run`

The machine-readable record is `manifests/local-data-inventory.json`; hashes
below are SHA-256. Paths embedded in the old upstream `run_manifest.json` are
not treated as live paths.

## Confirmed canonical upstream inputs

| Artifact | Rows / unique IDs | SHA-256 | Role and permission | Evidence |
| --- | ---: | --- | --- | --- |
| `clean_posts.csv` | 49,054 / 49,054 | `ee4adae632c3e5d22ede6d5fd072817fdb1ffc83b0b1cfbaac7ab1c61ba4100c` | Canonical cleaned source; not labels; no Milestone 1 production inference | CONFIRMED — parsed CSV |
| `semantic_pilot_inputs.jsonl` | 3,000 / 3,000 | `8623953d1b89506deac9f9e98676422e0fcefa2bc11b65d04f2a1f57a338576b` | Canonical model input; may supply all split times | CONFIRMED — parsed JSONL |
| `semantic_pilot.csv` | 3,000 / 3,000 | `81b853877b39aca32710aebde7a1168d747230bfccfd0866b7b7cf63b14217c3` | Canonical rich metadata; not labels | CONFIRMED — parsed CSV |
| `gold_candidates.csv` | 400 / 400 | `0aa7215fd59bb6c1c9e5f4deedde7078edb1ee385551458a9291e6e01425bc45` | Gold Candidate only; forbidden as final Gold/Test | CONFIRMED — parsed CSV and `review_status=UNREVIEWED` sample |
| `semantic_stage_v0_1.db` | `clean_posts=49,054`, `semantic_pilot=3,000`, `gold_candidates=400`, no teacher-label table | `fb22b0073dd1d4350b44a25d2946a288f4b97502f015c0456599d536e5645ee5` | Upstream sampling snapshot; not a label store | CONFIRMED — SQLite read-only catalog/counts |
| `run_manifest.json` | declares 49,054 clean, 3,000 pilot, 400 candidates, 16 stocks, 1 source | `64e9a9de938cc4dbcfa24b82151f70386edb43998bfd909cff5971dbfd018be8` | Upstream run facts; old `/workspace/scratch/...` path is non-live | CONFIRMED — JSON inspection |

All 49,054 cleaned rows are from `eastmoney_guba`, across 16 stocks. There is no
Xueqiu source, so cross-platform generalization is not supported.

## Visible labels and review materials — not canonical training truth

| Artifact | Rows / unique IDs | Schema | SHA-256 | Allowed role | Discrepancy |
| --- | ---: | --- | --- | --- | --- |
| Teacher A 400 `teacher_A_gold_400_v0.2.jsonl` | 400 / 400 | `semantic-schema-calibrated-v0.2` | `826d9a9461b8cbfddd8c4583d8c4408ac109d736b0491c42285987326ac5cbc2` | Teacher candidate / disagreement analysis | Filename `gold` does not confer Gold; only 400 |
| Teacher B 400 `teacher_B_gold_400_v0.2.jsonl` | 400 / 400 | `semantic-schema-calibrated-v0.2` | `7fbeaf375149fe86ed46dbe1d78d63bb9d1ad7f5c6ea0731cbb4c892594af80f` | Teacher candidate / disagreement analysis | Same |
| Teacher A blind100 `blind_100_labels_v0.2.2.jsonl` | 100 / 100 | `semantic-schema-calibrated-v0.2.1` | `0e716c0cdecf577408da63025dfd6b5db865729409f6415f3e2c8481141dcc60` | Teacher candidate / regression source | All 100 repeated times mismatch canonical input; all 100 are numeric Excel serial values |
| Teacher B blind100 `blind_100_labels_v0.2.2.jsonl` | 100 / 100 | `semantic-schema-calibrated-v0.2.1` | `855e09e65d6c25349ca18c874ca99a20cad1245adf6c5b31b70d9281870e19b0` | Teacher candidate / regression source | All 100 repeated times mismatch canonical input representation/timezone |
| `MyResearcher_Human_Gold_Calibrated_v0.2.xlsx` | 35 / 35 | workbook calibration schema | `b0433d460cc524873c29887ef5104885bdbb6d42004c3bee2d54e1c9048c20aa` | Calibration/reviewed material | 8 Human Adjudicated + 5 Rule Propagated + 22 Expert Accepted; not Anchor50 |
| `human/MyResearcher_Human_Gold_Review_100_v0.1.xlsx` | 100 review-task rows visible | old schema | `3a0d26a0f46077e53eeaddc7503920cedc4f874cb02558bc0793826c5e370366` | Historical review task | Task workbook does not prove 100 completed Gold records |

The label/input join root is `sample_id`. Repeated label metadata cannot replace
canonical input fields and must fail closed when inconsistent.

## Schema

`MyResearcher_Fresh_Blind100_A_Validation_v0.2.3.xlsx` SHA-256 is
`22a82a30aa1f08ad48554cd1ee054e522fda30e292cd386bda30f8e019d8eee8`.
Sheet `04_Schema`, range `A1:D66`, was exported to the versioned JSON under
`schema/`; its class order is regression-tested. The legacy
`semantic_schema_candidate_v0.1.json` SHA-256 is
`8b38f421ec644bbdda2f84188ad38e4bea43ba58d7d676827e3dfd0c546b5cc4`
and is explicitly not a runtime truth source.

## Missing canonical artifacts

No machine-readable artifact matching the required identity/provenance was
found for the following. Similar names, subsets, spreadsheets, or reconstructed
files do not satisfy the contract.

| Expected artifact | Expected facts | Blocker |
| --- | --- | --- |
| Frozen teacher labels | 3,000 unique labels with exact Schema/provenance | `BLOCKED_MISSING_FROZEN_TEACHER_LABELS` |
| Evidence quarantine manifest | exactly 21 identities and rule violations | `BLOCKED_MISSING_QUARANTINE_MANIFEST` |
| v0.3.5 split manifest | Train 1,822 / Dev 448 / Test 467 / Embargo 242 | `BLOCKED_MISSING_SPLIT_V0_3_5` |
| Anchor50 manifest/labels | 50 identities; 11 human-confirmed + 39 Expert weak Gold with provenance | `BLOCKED_MISSING_ANCHOR_50` |
| Field drift/weight map | 2,979 identities × seven heads; CALM/WATCH/Gate120 provenance | `BLOCKED_MISSING_DRIFT_WEIGHT_MAP` |
| Original baseline report bundle | metrics, tolerance, config, preprocessing, hashes, class order, evaluation code | `BLOCKED_MISSING_BASELINE_REPORT` |
| Original preprocessing contract | exact v0.3.5 text construction/version | `BLOCKED_MISSING_PREPROCESSING_CONTRACT_V0_3_5` |

Top-level state: `BLOCKED_MISSING_CANONICAL_ARTIFACTS` and
`BASELINE_V0_3_5_REPRODUCTION_BLOCKED`.

