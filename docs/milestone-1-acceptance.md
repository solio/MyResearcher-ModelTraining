# Milestone 1 Acceptance — frozen before implementation

Status: `ACTIVE`; frozen 2026-08-27. Synthetic acceptance establishes at most
`TESTED`. Real v0.3.5 reproduction additionally requires every canonical input
listed in `docs/data-handoff-required.md`.

## Capability acceptance

| ID | Acceptance | Required evidence |
| --- | --- | --- |
| AC-01 | Exactly 3,000 canonical inputs may join exactly 3,000 frozen teacher-label records; exactly 21 quarantined identities leave exactly 2,979 trainable identities. | Audit JSON plus deterministic validator test |
| AC-02 | Split counts are Train 1,822, Dev 448, Test 467, Embargo 242 and sum to 2,979. | Frozen split-manifest validation |
| AC-03 | Train, Dev, Test, Embargo, Anchor, and adjudicated Gold have zero `sample_id` overlap. | Leakage report with set intersections |
| AC-04 | Quarantined identities appear in no trainable split and receive zero weight for every affected head. | Quarantine/split/weight validation |
| AC-05 | Duplicate/missing/extra `sample_id` blocks the run. No fuzzy or metadata-derived join exists. | Negative tests and stable error codes |
| AC-06 | Repeated teacher `stock_code`, `stock_name`, and `published_at` agree with canonical input or block. Canonical input time defines all split checks. | Metadata mismatch tests, including Excel serial timestamp |
| AC-07 | Only one exact Schema version is accepted. v0.1/v0.2/v0.2.1 never mix silently, and class order matches frozen JSON. | Schema regression tests |
| AC-08 | `UNKNOWN≠NEUTRAL`, `NONE_EXPLICIT≠CALM`, and `WATCH≠NO_ACTION_SIGNAL` remain separate ordered classes. | Schema and challenge tests |
| AC-09 | `reasoning_tags` is multi-hot multi-label with the frozen 15-class order. | Encoding regression test |
| AC-10 | Weight identity is `sample_id × prediction_head`; one global sample weight is rejected. | Weight matrix tests |
| AC-11 | Evidence dependencies and canonical-text substring rules are training gates; violations produce quarantine output, not warnings. | Evidence negative tests |
| AC-12 | `build_model_input` is byte-identical across prepare, train, evaluate, export/infer paths. | Parity test |
| AC-13 | Every head stores fixed class order, probabilities, calibration metadata, and abstention threshold. Abstention is never remapped to an ordinary label. | Export/infer tests |
| AC-14 | 100% of inference records validate against `schema/inference-output.schema.json`. | JSON Schema test |
| AC-15 | Same files, Schema, config, and seed yield the same content-addressed manifest and prepared matrix identity. | Determinism replay |
| AC-16 | `audit_data` is read-only, network-free, download-free, emits JSON, and exits non-zero with stable blocker codes when canonical artifacts are absent. | CLI integration test |
| AC-17 | Blocked `prepare` and `train` create no run/model/split/label artifact. | Filesystem negative test |
| AC-18 | Classical baseline trains six independent single-label Logistic Regression heads and transparent binary heads for the 15 reasoning tags, all using per-field weights. | Synthetic end-to-end test |
| AC-19 | Evaluation reports field/class metrics, confusion matrices, confidence, calibration, thresholds, and error rows; run manifests include data/config/code/environment identity. | Synthetic run artifact assertions |
| AC-20 | CPU export/inference executes without PyTorch, an Encoder download, external search, or network access. | CPU end-to-end test |
| AC-21 | Large source files, workbooks, databases, runs, checkpoints, and model weights are absent from ordinary Git history. | `git status`/tracked-file audit |
| AC-22 | Real baseline is reported as reproduced only when original config, preprocessing, split, hashes, class order, evaluation contract/report, and canonical data all validate. | Reproduction-claim gate |

## Frozen negative cases

| Case | Expected result |
| --- | --- |
| 3,000 inputs with only 2,979 labels | `LABEL_COVERAGE_MISMATCH`; no guessed labels |
| 21 quarantine records placed in Train/Dev/Test/Embargo | `QUARANTINE_SPLIT_LEAKAGE` |
| Any split count differs from 1,822/448/467/242 | `SPLIT_COUNT_MISMATCH` |
| One identity in two partitions/Anchor/Gold | `SPLIT_IDENTITY_LEAKAGE` |
| Duplicate input or label identity | `DUPLICATE_SAMPLE_ID` |
| Label identity absent from input | `LABEL_SAMPLE_ID_NOT_FOUND` |
| Required input has no label | `MISSING_LABEL_SAMPLE_ID` |
| Repeated metadata differs | `CANONICAL_METADATA_MISMATCH` |
| Teacher time is an Excel serial | `NON_CANONICAL_LABEL_TIMESTAMP`; split still reads input time |
| Mixed Schema version | `SCHEMA_VERSION_MISMATCH` |
| A normal class replaces abstention | `ABSTENTION_LABEL_COLLISION` |
| Forbidden evidence on a sentinel or non-substring evidence | `EVIDENCE_DEPENDENCY_VIOLATION` / `EVIDENCE_NOT_SUBSTRING` |
| A single `sample_weight` column substitutes for head weights | `FIELD_WEIGHT_CONTRACT_VIOLATION` |
| Canonical handoff artifact absent | Artifact-specific blocker plus top-level `BLOCKED_MISSING_CANONICAL_ARTIFACTS` |

## Reproduction evidence states

- `BASELINE_HARNESS_TESTED`: synthetic end-to-end baseline passes.
- `BASELINE_V0_3_5_REPRODUCTION_BLOCKED`: harness passes but one or more
  canonical/reproduction artifacts are missing or invalid.
- `BASELINE_V0_3_5_REPRODUCED`: reserved for an accepted real run within a
  declared tolerance against the immutable reference report. Tests alone can
  never set this state.
