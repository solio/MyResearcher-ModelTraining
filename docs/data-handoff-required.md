# Canonical Data Handoff Required

The semantic Expert/data owner must deliver one immutable package. Every file
must use UTF-8 JSON/JSONL, include an exact schema/version field, and be covered
by a package manifest containing byte size, SHA-256, logical artifact name,
record count, unique `sample_id` count, producer, creation time, and provenance.

Place the package at the paths configured in
`configs/baseline_v0.3.5.yaml` or update the config in a reviewed commit. Do not
rename an existing visible artifact to satisfy this contract.

## Minimum package

1. `frozen_teacher_labels_v0.3.5.jsonl`: exactly 3,000 unique `sample_id`
   records, one exact frozen Schema version, field labels, field-level
   confidence, Evidence, source batch/Gate120/drift provenance, and repeated
   metadata suitable for fail-closed parity checks. These remain frozen teacher
   labels, not Gold.
2. `quarantine_manifest_v0.3.5.json`: exactly 21 unique identities with Evidence
   rule IDs, affected fields, source-label hashes, and zero-training permission.
3. `split_manifest_v0.3.5.json`: assignments for exactly 2,979 non-quarantined
   identities; Train 1,822, Dev 448, Test 467, Embargo 242. Include canonical
   input/label/quarantine hashes, grouping keys for stock/time/event,
   duplicate/echo group, author isolation status, boundaries, seed, and split
   rationale.
4. `field_weights_v0.3.5.jsonl`: exactly 2,979 identities with a numeric weight
   for each of the seven V1 heads, plus confidence/drift/Gate120/quarantine rule
   provenance and weighting-config version. No single global weight substitute.
5. `anchor50_v0.3.5.jsonl` and `anchor50_manifest_v0.3.5.json`: exactly 50 unique
   identities, labels, role, source hashes, no train/dev/test/embargo overlap,
   and explicit provenance separating the 11 human-confirmed records from 39
   Expert weak Gold records.
6. `baseline_report_v0.3.5.json`: original per-head/per-class metrics, confusion
   matrices, confidence/calibration, exact evaluation code/version, tolerance,
   config and dependency versions, class order, and every input artifact hash.
7. `preprocessing_contract_v0.3.5.json`: exact board-context/model-text
   construction, normalization, missing-field behavior, contract version, and
   regression examples used by the reported baseline.
8. `package_manifest_v0.3.5.json`: content-addressed package root whose
   `artifacts` object maps every logical name to byte size, rows, unique IDs and
   SHA-256; it references every file above and the canonical
   `semantic_pilot_inputs.jsonl` SHA-256
   `8623953d1b89506deac9f9e98676422e0fcefa2bc11b65d04f2a1f57a338576b`.

## Acceptance of the handoff

The package is accepted only after `python -m semantic_model.audit_data` reports
no blocker, verifies all hashes/counts/joins/roles/Evidence dependencies, and
confirms zero split/Anchor/Gold leakage. A spreadsheet, prose claim, subset,
fresh random split, inferred label, or two-teacher agreement is insufficient.
