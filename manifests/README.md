# Manifests

Committed manifests contain only logical identities, hashes, counts, roles, and
provenance. Large source/training/model artifacts remain external. A manifest ID
is SHA-256 over canonical JSON (UTF-8, sorted keys, compact separators) after
omitting that manifest's own ID field.

