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

The separate immutable baseline-reference handoff is also accepted. Its ZIP
SHA-256 is
`78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410`;
its content-manifest SHA-256 is
`828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`.
All 17 payloads / 11,439,730 bytes pass. The package binds to the data content
address above and freezes the original model, estimator diagnostics, reference
runtime, 2,787 per-row predictions/probabilities, metrics recomputation, and
acceptance policy. The original model SHA-256 is
`4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`.

The historical baseline was produced under Python 3.12.13, Linux x86_64,
NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, and OpenBLAS
0.3.30/pthreads. Its six `saga` scalar heads also stopped at `n_iter=2000`
without convergence; all 15 `liblinear` Reasoning heads converged. Recomputing
the historical metrics from the frozen original model has maximum absolute
difference `0.0`. The earlier local run was therefore not trained incorrectly;
it is an environment-sensitive comparison run.

The pinned development runtime is macOS arm64 with scikit-learn 1.7.2, so its
allowed state is `COMPARABLE_DIAGNOSTIC_RUN_ONLY`, with
`BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`. Exact reproduction requires the same
reference environment, exact labels on all Train/Dev/Test/Anchor50 rows,
metrics tolerance `1e-12`, and probability tolerance `1e-10`. Even a passing
exact run may only be named
`BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`.

This remains an engineering-only TF-IDF diagnostic baseline. It is not a
production model, does not authorize Encoder/LoRA work, and must not run the
49,054-row production inference.

See [data inventory](docs/data-inventory.md),
[handoff status](docs/data-handoff-required.md), and
[Milestone 1 acceptance](docs/milestone-1-acceptance.md). Every future model
executor must also read the owner-aligned
[architecture handoff and model roadmap](docs/architecture-handoff-and-model-roadmap.md):
it defines the planned BERT-class Encoder as the future primary model, retains
the classical baseline as a permanent control, limits generative LLMs to
offline review/verification, and keeps OOD distinct from `UNKNOWN`.

Execution order is governed separately by the owner-aligned
[milestone, priority, and agent-routing policy](docs/milestone-priority-and-agent-routing.md).
M1 is closed with its accepted provenance-bound weak-label diagnostic artifact
`b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58`; the
current primary milestone is M2 Encoder quality and stability. M2 activation
does not itself authorize a download, fit, new data, external service, or
production action. That policy also freezes the repository's P0–P3 review
meanings, important/urgent quadrants, parallelization gates, and Terra Max /
Luna Max / exceptional Sol Max prompt routing.

## Environment

Use CPython 3.12 and the pinned CPU-only dependency set:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

Encoder dependencies may not be installed without explicit owner authorization.
That authorization was exercised only for the named isolated M1 `hfl/rbt3`
runtime at revision `0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c`. Its CPython
3.12.13 environment contains torch 2.8.0, Transformers 4.57.6, tokenizers
0.22.2, and NumPy 2.5.2; MPS was validated and used, and the mandatory CPU
checkpoint reload/inference smoke passed. The real entry point first requires a
passing canonical data/reference audit, exact owner contract, and clean tracked
source identity. The Classical baseline `.venv` and ambient Anaconda runtime
remain untouched.

The accepted provenance-bound weak-label diagnostic artifact is in
`.encoder-artifacts/m1-rbt3-0aa0527f-provenance-88f90b1/` (content address
`b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58`). The
earlier `3cd8d53cae5ec7346595163c227b4cef8abfd90c5e63c37578c8fc48dc147685`
artifact remains immutable historical evidence but is
`REJECTED_M1_PROVENANCE_INCOMPLETE`. Neither artifact is a production model,
Gold/OOD result, Test result, or authorization for 49,054-row inference.

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

Install the reference handoff beside it, also without renaming or editing any
payload:

```bash
unzip MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip \
  -d data/local
```

The resulting directory must be
`data/local/MyResearcher_Semantic_Baseline_Reference_v0.3.5/`. The reference
audit never executes the supplied script or loads the external `joblib`; it
validates hashes, JSON/JSONL contracts, data binding, estimator fingerprints,
and per-row prediction relations.

## Commands

```bash
python -m semantic_model.audit_reference \
  --config configs/baseline_v0.3.5.yaml \
  --archive /path/to/MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip
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

The suite covers legacy fail-closed fixtures and both native v0.3.5 packages,
including ZIP/path/manifest checks, Evidence repair, role separation,
split/weight/Anchor relations, preprocessing parity, feature order, immutable
artifacts, reference environment classification, 2,787-row prediction binding,
export, CPU inference, and reproduction-claim gating.

Frozen run evidence is recorded in
[`reports/milestone-1-verification.md`](reports/milestone-1-verification.md).
