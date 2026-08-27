# MyResearcher-ModelTraining

Reproducible, testable, auditable local engineering for the MyResearcher
post-level semantic student model.

## Current truth

The immutable v0.3.5 data handoff is accepted. Its ZIP SHA-256 is
`c5ff639954fe71d8bc780175584406c6f5c84998c39d0040fdae830134a95378`;
the content-manifest SHA-256 is
`cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`.
All 28 payload hashes and byte sizes pass, as do these semantic gates:

- 3,000 immutable teacher labels;
- exactly 21 forbidden-Evidence rows quarantined;
- 2,979 repaired, trainable weak-label rows;
- Train 1,822 / Embargo-1 131 / Dev 448 / Embargo-2 111 / Test 467;
- one explicit seven-head weight vector for every trainable row;
- Anchor50 with 11 human-confirmed and 39 Expert-preadjudicated rows and zero
  Teacher3000 overlap;
- exact Train-only TF-IDF feature counts: char 11,945 / word 313 / total
  12,258.

The data package is sufficient for a diagnostic run, but not for an exact
baseline-reproduction claim. The embedded reference report and script omit the
reference Python, NumPy, SciPy, scikit-learn, CPU/platform, and convergence
provenance. In the pinned local environment, all six `saga` scalar heads reach
`max_iter=2000` without convergence; their Anchor metrics differ from the
reference, while Reasoning micro-F1 matches exactly. The stable state is
`BASELINE_V0_3_5_REPRODUCTION_BLOCKED_REFERENCE_ENVIRONMENT`, with blocker
`BLOCKED_MISSING_REFERENCE_ENVIRONMENT`—never `REPRODUCED`.

This remains an engineering-only TF-IDF diagnostic baseline. It is not a
production model, does not authorize Encoder/LoRA work, and must not run the
49,054-row production inference.

See [data inventory](docs/data-inventory.md),
[handoff status](docs/data-handoff-required.md), and
[Milestone 1 acceptance](docs/milestone-1-acceptance.md).

## Environment

Use CPython 3.12 and the pinned CPU-only dependency set:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

Encoder dependencies are optional and must not be installed for Milestone 1.

## Install the local immutable package

Canonical data stays outside Git. Extract the content-addressed handoff without
renaming or editing its files:

```bash
unzip MyResearcher_Semantic_Immutable_Data_v0.3.5_cf7a10f25d951d79.zip \
  -d data/local
```

The resulting directory must be
`data/local/MyResearcher_Semantic_Immutable_Data_v0.3.5/`. The audit pins the
manifest hash in `configs/baseline_v0.3.5.yaml`, rejects extra/missing payloads,
symlinks, path escape, duplicate identities, hash changes, and semantic
relationship changes.

## Commands

```bash
python -m semantic_model.audit_data --config configs/baseline_v0.3.5.yaml
python -m semantic_model.prepare --config configs/baseline_v0.3.5.yaml
python -m semantic_model.train --config configs/baseline_v0.3.5.yaml
python -m semantic_model.evaluate --run <run_dir>
python -m semantic_model.export --run <run_dir>
python -m semantic_model.infer --model <model_dir> --input <jsonl> --output <jsonl>
```

`audit_data` is read-only and network-free. `prepare` and `train` call it first.
An invalid/missing package blocks before model fitting. Runs and exports are
content-addressed, immutable, local, and ignored by Git. Exported `joblib`
bundles are trusted local artifacts only; inference verifies bundle hashes and
must never load an untrusted external pickle/joblib file.

## Tests

```bash
pytest
python -m pip check
git diff --check
```

The suite covers legacy fail-closed fixtures and the native v0.3.5 package
format, including byte-level manifest checks, Evidence repair, role separation,
split/weight/Anchor relations, preprocessing parity, feature order, immutable
artifacts, export, CPU inference, and reproduction-claim gating.

Frozen run evidence is recorded in
[`reports/milestone-1-verification.md`](reports/milestone-1-verification.md).
