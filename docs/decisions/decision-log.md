# Decision Log

Format: `D-<NNN> | date | decision | rationale | status | evidence level`.

| ID | Date | Decision | Rationale | Status | Evidence Level |
| --- | --- | --- | --- | --- | --- |
| D-001 | 2026-08-27 | Persist contracts, manifests, hashes, blockers, and acceptance evidence in Git rather than conversation state. | Repository bootstrap must be reproducible and auditable. | ACTIVE | CONFIRMED — milestone instruction |
| D-002 | 2026-08-27 | Keep canonical input, teacher candidate, weak label, reviewed label, Human Gold, Anchor, prediction, challenge, quarantine, and embargo as separate roles. | Confidence, filename, agreement, and model output do not establish truth provenance. | ACTIVE | CONFIRMED — specification + upstream contract review |
| D-003 | 2026-08-27 | Use `sample_id` as the only label/input join root and fail on repeated metadata disagreement. | The observed Teacher A blind100 timestamps are Excel serial values and all observed blind100 timestamps disagree with canonical input representation. | ACTIVE | CONFIRMED — read-only snapshot audit |
| D-004 | 2026-08-27 | Export frozen Schema once from workbook sheet `04_Schema`; all runtime stages read versioned JSON. | Runtime Excel parsing would make class order and behavior environment-dependent. | ACTIVE | CONFIRMED — workbook inspection + milestone instruction |
| D-005 | 2026-08-27 | Treat the board-context preprocessing contract as `PROVISIONAL/ENGINEERING_ONLY` until the original v0.3.5 contract is handed off. | No machine-readable v0.3.5 preprocessing/config artifact was found. | ACTIVE | PROVISIONAL — exhaustive local filename/text scan |
| D-006 | 2026-08-27 | Never reconstruct the missing 1,822/448/467/242 split, 21-row quarantine, Anchor50, field weights, or 3,000 frozen labels from visible A/B subsets. | Doing so would fabricate provenance and invalidate the reproduction claim. | ACTIVE | CONFIRMED — milestone instruction |
| D-007 | 2026-08-27 | A missing-canonical audit is a valid blocked result, while synthetic validators and harness tests may reach `TESTED`. | Test success does not establish real-data reproduction. | ACTIVE | CONFIRMED — project protocol |
| D-008 | 2026-08-27 | TF-IDF + Logistic Regression remains a diagnostic baseline and can never be named production or used for 49,054-row production inference in Milestone 1. | The task explicitly limits scope and evidence claims. | ACTIVE | CONFIRMED — specification |

## Two Repair Rule

If the same root cause survives two local implementation repairs, record the
root cause and return to the Schema, data, label-policy, split, or architecture
owner. Do not add a third implicit migration or compatibility branch.

