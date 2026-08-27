# MyResearcher-ModelTraining

Reproducible, testable, auditable local engineering for the MyResearcher
post-level semantic student model.

## Current truth

Milestone 1 is engineering-only and currently blocked from reproducing the
v0.3.5 diagnostic baseline because the canonical 3,000-label package, the
21-row quarantine manifest, the 2,979-row field-weight map, the frozen
1,822/448/467/242 split, Anchor50 provenance, original baseline contract, and
content-addressed package manifest are not present in the observed upstream
snapshot. The repository must return
`BLOCKED_MISSING_CANONICAL_ARTIFACTS` until those immutable artifacts arrive.

The visible Teacher A/B files remain teacher candidates. The visible 400-row
file remains Gold Candidate. The calibrated workbook contains 35 records, not
the specified Anchor50.

See [data inventory](docs/data-inventory.md), [required handoff](docs/data-handoff-required.md),
and [Milestone 1 acceptance](docs/milestone-1-acceptance.md).

## Environment

Use CPython 3.11 or 3.12; the locked/tested path is CPython 3.12. Python 3.12 is
chosen because the pinned scikit-learn/SciPy stack supports it and it leaves a
compatible optional path for later PyTorch/Transformers work without making
those packages Milestone 1 dependencies.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

Encoder dependencies are an optional extra and must not be installed for this
milestone.

## Commands

```bash
python -m semantic_model.audit_data --config configs/baseline_v0.3.5.yaml
python -m semantic_model.prepare --config configs/baseline_v0.3.5.yaml
python -m semantic_model.train --config configs/baseline_v0.3.5.yaml
python -m semantic_model.evaluate --run <run_dir>
python -m semantic_model.export --run <run_dir>
python -m semantic_model.infer --model <model_dir> --input <jsonl> --output <jsonl>
```

`audit_data` and blocked `prepare`/`train` do not download models, access the
network, invent data, or write training artifacts. Once the canonical handoff
is present, the same config supplies all pipeline stages with the same frozen
Schema, preprocessing, split, and field-weight contracts.

## Tests

```bash
pytest
git diff --check
```

Synthetic tests establish `TESTED`, not real-data reproduction. The local
read-only audit is separate and intentionally exits non-zero while canonical
artifacts are absent.
