# Milestone 1 Verification

Verification date: 2026-08-27. Evidence level: `CONFIRMED` for the observed
package bytes, source commit, commands, and generated artifacts. Final local
capability state: `DATA_AND_REFERENCE_VALIDATED_COMPARABLE_ONLY`.

The immutable data package and the content-addressed baseline reference package
both pass integrity and semantic validation. A clean real training, evaluation,
export, CPU inference, and immutable replay also pass. The local run is
`COMPARABLE_DIAGNOSTIC_RUN_ONLY`, not an exact historical reproduction, because
it ran on macOS arm64 with scikit-learn 1.7.2 instead of the frozen Linux x86_64
reference environment with scikit-learn 1.8.0.

Even a future exact match may only be labelled
`BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`. The historical baseline is a
diagnostic model with six non-converged scalar estimators and is not authorized
for production use.

## Immutable data package integrity

Source archive:
`MyResearcher_Semantic_Immutable_Data_v0.3.5_cf7a10f25d951d79.zip`.

| Check | Result |
| --- | --- |
| ZIP SHA-256 | `c5ff639954fe71d8bc780175584406c6f5c84998c39d0040fdae830134a95378` |
| ZIP CRC test | PASS |
| Unsafe absolute/parent/backslash paths | 0 |
| Duplicate ZIP entries | 0 |
| Symlinks | 0 |
| `CONTENT_MANIFEST.json` SHA-256 | `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b` |
| Payload entries | 28 |
| Payload bytes | 14,992,255 |
| Payload hashes/sizes verified | 28/28 |
| Unexpected/missing extracted payloads | 0/0 |

The package was extracted only under ignored `data/local/`; no package ZIP,
payload, model, export, or run artifact is tracked by Git.

## Baseline reference package integrity

Source archive:
`MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip`.

| Check | Result |
| --- | --- |
| ZIP SHA-256 | `78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410` |
| ZIP CRC test | PASS |
| Unsafe absolute/parent/backslash paths | 0 |
| Duplicate ZIP entries | 0 |
| Symlinks | 0 |
| Content-addressed manifest SHA-256 | `828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85` |
| Payload entries | 17 |
| Payload bytes | 11,439,730 |
| Payload hashes/sizes verified | 17/17 |
| Unexpected/missing extracted payloads | 0/0 |
| Bound immutable data package ID | `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b` |
| Original `model.joblib` SHA-256 | `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a` |

The package source file is byte-identical to the corresponding source in the
immutable data package. Its reference predictions cover exactly 2,787 rows:
Train 1,822, Dev 448, Test 467, and Anchor50 50. Canonical ordering, IDs, truth
labels, text hashes, all scalar class probabilities, all Reasoning probabilities,
and Reasoning thresholds are present and validated.

Repository audit manifest ID:
`fa7efb96f46a448caecf9efa55a91187868bb747fe139141a5322f21b20d0b89`.
Reference package audit ID:
`6a2fe3f4a532341e9891581548840701086a07f7a4ce8864c98ddfed36926793`.

## Reference provenance and historical model finding

The frozen reference runtime is:

| Component | Reference value |
| --- | --- |
| Python | 3.12.13 |
| Operating system | Linux x86_64 / Ubuntu 24.04.3 |
| CPU | AMD EPYC |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| OpenBLAS | 0.3.30 / pthreads |

The environment inventory was retrospectively captured from the same persistent
runtime that stored and loaded the original model without version warnings. It
was not automatically emitted by the original fit command. That limitation is
retained explicitly; it does not weaken the separately frozen original model,
predictions, metrics, estimator parameters, and coefficient/intercept
fingerprints.

Inspection of the original model confirms:

- all six scalar heads use SAGA, record `n_iter_ = 2000` and
  `max_iter = 2000`, and are not converged;
- all 15 one-vs-rest Reasoning heads use liblinear and are converged;
- metrics recomputed from the original model's 2,787 frozen predictions have a
  maximum absolute difference of `0.0` from the historical metrics.

The exact-reproduction policy is therefore frozen as follows:

- same reference environment: every per-row predicted label must match exactly;
- aggregate metric absolute tolerance: `1e-12`;
- probability absolute tolerance: `1e-10`;
- cross-platform or different scikit-learn versions: no exact-reproduction
  tolerance is authorized; the result is only a comparable diagnostic run.

## Semantic package audits

Commands:

```bash
.venv/bin/python -m semantic_model.audit_reference \
  --config configs/baseline_v0.3.5.yaml \
  --archive /Users/mac/Documents/Codex/2026-08-27/MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip

.venv/bin/python -m semantic_model.audit_data \
  --config configs/baseline_v0.3.5.yaml
```

Reference audit result:

- status: `REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY`;
- all reference payloads, package bindings, diagnostics, predictions, and
  historical metric recomputation pass;
- exact reproduction is blocked only by
  `BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`.

Data audit result:

- audit ID:
  `5fab05d633c509122bb8bbddd95b5d79f8d76a660b284f5cb20120df2865e414`;
- status: `READY_FOR_COMPARABLE_DIAGNOSTIC_RUN`;
- maturity: `DATA_AND_REFERENCE_VALIDATED_COMPARABLE_ONLY`;
- training allowed: `true`;
- exact-reproduction claim allowed in the current environment: `false`.

Verified data relations include:

- 3,000 canonical inputs = 3,000 frozen label IDs = 3,000 repaired label IDs;
- repaired labels remove exactly 21 forbidden Evidence objects and change zero
  semantic labels;
- quarantine 21 and trainable 2,979 are disjoint and union to frozen 3,000;
- Train 1,822 / Embargo-1 131 / Dev 448 / Embargo-2 111 / Test 467 are pairwise
  disjoint and union to trainable 2,979;
- all split dates and source-order projections are consistent with the frozen
  contracts;
- all 20,853 sample-by-head weights match the declared weighting function;
- Anchor50 has 11 `HUMAN_CONFIRMED` and 39 `EXPERT_PREADJUDICATED` rows with no
  Teacher3000 overlap;
- repository, data package, and reference package class orders agree.

## Deterministic preparation and feature contract

Command:

```bash
.venv/bin/python -m semantic_model.prepare \
  --config configs/baseline_v0.3.5.yaml
```

Result:

- prepared manifest ID:
  `b49335f8c1297b06bd4c41867319c87801787bd678f3c36ba471bea92809d853`;
- Train-only fitted features: char 11,945 / word 313 / total 12,258;
- observed sparse matrix equals an independently constructed
  `hstack([char, word])` reference matrix with difference `nnz=0` and maximum
  absolute difference `0.0`.

Local filesystem paths are retained in audit reports but excluded from content
identity. Moving identical bytes to another directory does not change the
audit or preparation content IDs.

## Clean local training run

Command:

```bash
.venv/bin/python -m semantic_model.train \
  --config configs/baseline_v0.3.5.yaml
```

Run identity:

- run ID:
  `49f67b0476c4f439f1d867476d5c850316048fe09e859d652f2cf4225b31c4db`;
- run manifest ID:
  `784ed578cfc71b98df021e251380a99fda7724907dd42e1b61e8e5b1bd6c8dd2`;
- model manifest ID:
  `b9554bd04ffa048be9140389e22dca1d47c8091ac33b69d54c070e928d288a8c`;
- source Git commit: `0f366a8cb7488006e5d4afd2b3e2224497292712`;
- branch: `feat/milestone-1-reproducible-baseline`;
- Git dirty at training time: `false`;
- status: `COMPARABLE_DIAGNOSTIC_RUN_ONLY`;
- maturity: `DATA_AND_REFERENCE_VALIDATED_COMPARABLE_ONLY`;
- blocker: `BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`;
- elapsed time: 37.708 seconds.

Observed local runtime:

- CPython 3.12.13 on macOS arm64, CPU-only;
- NumPy 2.3.3;
- SciPy 1.16.2;
- scikit-learn 1.7.2;
- joblib 1.5.2.

Run artifact identities:

| Artifact or identity | SHA-256 / content ID |
| --- | --- |
| `run_manifest.json` | `424630826062111c80a891bfecf29a11b82fdffd9a4800cf13eb6220812c2a0a` |
| `model_manifest.json` | `12dd8ac78769b3c3f3537bc403ea8efa465dca0aaace16e90593b1c96af45929` |
| local `model.joblib` | `fe03d685fcfe3158f0b87a358144e5c991dfe16e1f909276a402a583c7b3c11c` |
| `metrics.json` | `dc35675928ce8cb3eab0a0594e0a718bdf7fdc223de2f019476bf6528076da38` |
| `baseline_comparison.json` | `6fba45ab7aadc20035fbfb52a4fc51532fd6a8ff6f2e5ec2ed771204795a37d9` |
| `training_diagnostics.json` | `11b18d13de6a2233b7b4ef1f4e76f43543ba93ca33c0f2a6c4eea1b63596160d` |
| `thresholds.json` | `7b53803e6d786f05dbd493b5441776bad0a10fc9a01e44302f1ed7e514407ecd` |
| `errors.jsonl` | `1d386a99e6b1418f2f089bf820e9604621c20cdec7416e937ac89106d16b7655` |

The local model has the same convergence shape as the original: all six scalar
SAGA heads reached 2,000 iterations without convergence; all 15 Reasoning
liblinear heads converged.

## Full 2,787-row reference comparison

The comparison covers seven output heads per row: six scalar heads plus the
multi-label Reasoning output. There are 19,509 sample-by-head label comparisons.

| Head | Exact label matches | Match rate | Maximum probability delta |
| --- | ---: | ---: | ---: |
| action_tendency | 1,276 / 2,787 | 45.784% | 0.9999999905 |
| context_dependency | 1,912 / 2,787 | 68.604% | 0.9925679974 |
| emotion_primary | 2,443 / 2,787 | 87.657% | 0.4278651178 |
| emotion_target | 2,455 / 2,787 | 88.088% | 0.5416560024 |
| reasoning_tags | 2,787 / 2,787 | 100.000% | 0.0000084075 |
| stance | 2,218 / 2,787 | 79.584% | 0.9976849272 |
| target_mode | 1,786 / 2,787 | 64.083% | 0.9346780591 |

Aggregate result:

- 14,877 / 19,509 individual head labels match;
- 486 / 2,787 rows match on all seven outputs;
- all-seven exact rows by split: Train 413 / 1,822, Dev 38 / 448,
  Test 32 / 467, Anchor50 3 / 50;
- maximum probability absolute delta: `0.9999999905491448`;
- 84 aggregate metric values compared;
- maximum aggregate metric absolute delta: `0.14561027837259105`;
- all 12 aggregate Reasoning metrics across the four evaluated splits match
  exactly, while Reasoning probabilities exceed the frozen `1e-10` tolerance;
- exact reproduction: `false`;
- production authorized: `false`.

These differences are expected evidence of the already-declared solver and
environment sensitivity. They are not evidence that the local pipeline trained
the wrong estimator family or used the wrong data: class order, feature
contract, estimator configuration, convergence shape, row identity, and
comparison semantics are all checked independently.

## Evaluation, export, inference, and immutable replay

Evaluation reverified ten run artifacts by hash and reported
`reproduced=false`, `production=false`.

- export ID:
  `fa1b59b340db49a7b89ffa4a4dedcac963c22060717c256d9ac01f0313fcd5c8`;
- export manifest ID:
  `1081cc29e4af06fb8120a64aa6f621881f68b433b169cd294ffb735832208aa6`;
- two-row CPU inference: `INFERENCE_COMPLETE`, seven heads per row, JSON Schema
  valid;
- inference output SHA-256:
  `6ad9c68c26796515f41217c9c64d3558278d04d27a0b8a4ce47b6cab4dde14bd`;
- repeated training returned `EXISTING_IMMUTABLE_RUN` with the same run ID.

## Engineering verification

The following checks are required on the implementation commit and this report:

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
git diff --check
```

The test suite covers the fail-closed synthetic harness and both real native
packages, including content hashes, archive safety, portable identities, native
Evidence, repair invariants, explicit weights, Schema class order, Anchor-only
compatibility, preprocessing, feature order/counts, calibration, inference
schema, original estimator diagnostics and fingerprints, full 2,787-row
prediction comparison, reference-environment classification, immutable replay,
export, and CPU inference.

No Encoder, PyTorch, Transformers, model download, 49,054-row inference, GUI,
service, production integration, or production authorization was performed.

## Acceptance decision and next gate

Milestone 1 is accepted at the `COMPARABLE_DIAGNOSTIC_RUN_ONLY` evidence level:
the data, reference model, historical metrics, training pipeline, comparison
logic, export, and CPU inference are all executable and auditable.

Exact v0.3.5 reproduction is not yet accepted. The next bounded gate is to run
the same committed pipeline in the frozen Linux x86_64 reference runtime and
apply the frozen row-label, metric, and probability tolerances. Passing that
gate changes the status only to
`BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`; it does not make the baseline a
production model.
