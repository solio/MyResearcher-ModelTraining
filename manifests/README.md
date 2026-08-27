# Manifests

`immutable-package-audit-v0.3.5.json` records the accepted canonical handoff
without committing any payload data.
`baseline-reference-audit-v0.3.5.json` records the bound original-model,
environment, diagnostics, prediction-oracle, tolerance, and local-environment
classification without committing the model or predictions.
`local-data-inventory.json` is retained as an immutable historical scan of the
earlier `formal_run` snapshot; its missing artifact blockers describe that
snapshot and are superseded by the two separate immutable package audits.

Committed manifests contain only logical identities, hashes, counts, roles, and
provenance. Large source/training/model artifacts remain external. A manifest ID
is SHA-256 over canonical JSON (UTF-8, sorted keys, compact separators) after
omitting that manifest's own ID field.
