# Milestone 1 Verification

Verification date: 2026-08-27. Evidence state: `CONFIRMED` for commands and
observed local bytes; capability maturity: `TESTED`; real v0.3.5 reproduction:
`BLOCKED`.

## Engineering verification

The following command completed on CPython 3.12/CPU:

```bash
.venv/bin/python -m pytest
```

Result: `62 passed in 1.23s`. The suite includes a complete synthetic canonical
package that passes audit, produces the same prepared-manifest identity on
replay, trains all seven task heads, calibrates Dev thresholds, evaluates
Dev/Test/Anchor, verifies immutable run hashes, exports a local bundle, executes
CPU inference, and validates every output against the frozen inference JSON
Schema. It also covers all blocked/negative contracts in
`docs/milestone-1-acceptance.md`.

`python -m compileall -q src tests` also completed successfully. No PyTorch,
Transformers, Encoder, network search, model download, or 49,054-row inference
was used.

## Real read-only audit

Commands:

```bash
.venv/bin/python -m semantic_model.audit_data --config configs/baseline_v0.3.5.yaml
.venv/bin/python -m semantic_model.prepare --config configs/baseline_v0.3.5.yaml
.venv/bin/python -m semantic_model.train --config configs/baseline_v0.3.5.yaml
```

All three returned exit code `2`, status
`BLOCKED_MISSING_CANONICAL_ARTIFACTS`, and created no `runs/` directory. Audit
ID was `9b239bdd31f4eebc28e3fa637e8914ae113bca3ba0111e3dd0a08f356b57729c`.

The audit reconfirmed:

- 3,000 canonical inputs / 3,000 unique IDs, SHA-256
  `8623953d1b89506deac9f9e98676422e0fcefa2bc11b65d04f2a1f57a338576b`;
- 3,000 canonical metadata rows / 3,000 unique IDs, SHA-256
  `81b853877b39aca32710aebde7a1168d747230bfccfd0866b7b7cf63b14217c3`;
- frozen JSON Schema SHA-256
  `8e06356279e7d735c2bb6acf25cbac1208f901e762c63349db70583d73a466fd`.

Stable blockers:

- `BLOCKED_MISSING_FROZEN_TEACHER_LABELS`
- `BLOCKED_MISSING_QUARANTINE_MANIFEST`
- `BLOCKED_MISSING_SPLIT_V0_3_5`
- `BLOCKED_MISSING_DRIFT_WEIGHT_MAP`
- `BLOCKED_MISSING_ANCHOR_50`
- `BLOCKED_MISSING_BASELINE_REPORT`
- `BLOCKED_MISSING_PREPROCESSING_CONTRACT_V0_3_5`
- `BLOCKED_MISSING_CANONICAL_PACKAGE_MANIFEST`

Therefore no real training, real evaluation, model export, or inference was
performed, and `BASELINE_V0_3_5_REPRODUCED` was not claimed.

