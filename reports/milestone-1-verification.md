# Milestone 1 Verification

Verification date: 2026-08-27. Evidence level: `CONFIRMED` for the observed
local bytes and commands. Final capability state:
`DATA_VALIDATED_REFERENCE_ENVIRONMENT_MISSING`.

The canonical data handoff, real diagnostic training, evaluation, export, and
CPU inference pass. Exact historical v0.3.5 reproduction is not claimed and is
blocked by `BLOCKED_MISSING_REFERENCE_ENVIRONMENT`.

## Immutable package integrity

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

The package was extracted only under ignored `data/local/`; no payload byte,
ZIP, model, export, or run artifact is tracked by Git.

## Semantic data audit

Command:

```bash
.venv/bin/python -m semantic_model.audit_data \
  --config configs/baseline_v0.3.5.yaml
```

Result:

- status: `READY_FOR_DIAGNOSTIC_BASELINE_RUN`;
- training allowed: `true`;
- reproduction claim allowed: `false`;
- audit ID:
  `1219d99fcffc949ae49758cc09fa9f0e99c672f84c7a38130954af1fff27a553`;
- command-output SHA-256:
  `f24660b9795fb8f8bb138ed8eb731535fda9888e9cb9487d010f085cc6037c1a`.

Verified relations:

- 3,000 canonical inputs = 3,000 frozen label IDs = 3,000 repaired label IDs;
- repaired labels remove exactly 21 forbidden Evidence objects and change zero
  semantic labels;
- quarantine 21 and trainable 2,979 are disjoint and union to frozen 3,000;
- Train 1,822 / Embargo-1 131 / Dev 448 / Embargo-2 111 / Test 467 are pairwise
  disjoint and union to trainable 2,979;
- every split date equals canonical `published_at[:10]` under the declared
  no-timezone-conversion policy;
- all five split-label files are exact source-order views of trainable labels;
- all 20,853 explicit sample×head weights equal the embedded reference
  weighting function;
- Anchor50 has 11 `HUMAN_CONFIRMED` + 39 `EXPERT_PREADJUDICATED` rows and zero
  overlap with Teacher3000;
- repository and package Schema class orders match exactly;
- all 28 manifest payloads remain byte-identical.

## Deterministic preparation and feature contract

Command:

```bash
.venv/bin/python -m semantic_model.prepare \
  --config configs/baseline_v0.3.5.yaml
```

Result:

- prepared manifest ID:
  `02e4fddf116a2ae721644cced9f8429df820c31e09c830fe740785d15b7aeb1b`;
- command-output SHA-256:
  `157a54a8feceac039b76f6c52e30004d24b7e16482378dc63410ca6e1b537598`;
- Train-only fitted features: char 11,945 / word 313 / total 12,258;
- observed sparse matrix equals the independently constructed reference
  `hstack([char, word])` matrix with difference `nnz=0` and max absolute
  difference `0.0`.

Local filesystem paths are retained in audit reports but removed from the
content-identity projection. Moving identical bytes to another directory does
not change audit/prepare content IDs.

## Clean real diagnostic run

Command:

```bash
.venv/bin/python -m semantic_model.train \
  --config configs/baseline_v0.3.5.yaml
```

Run identity:

- run ID:
  `1d3fc6be9ddc81381b2af53774b75152a2df91f372c04ace7bb4e4783113cc69`;
- run manifest ID:
  `4bdd9c83ea26ca8cef4ee080b816b1a674f29243641b0b9cd16bba741fe85f4a`;
- Git commit: `e2348833cd6e751b87d91005a2a0bc68e1b31645`;
- branch: `feat/milestone-1-reproducible-baseline`;
- Git dirty at training time: `false`;
- status: `BASELINE_V0_3_5_REPRODUCTION_BLOCKED_REFERENCE_ENVIRONMENT`;
- blocker: `BLOCKED_MISSING_REFERENCE_ENVIRONMENT`;
- command-output SHA-256:
  `99e8ee44e7348df69fb4dc2794682e5f749cb65afffdf8338b436c3f189a95c3`.

Environment recorded by the run:

- CPython 3.12.13, CPU-only;
- NumPy 2.3.3;
- SciPy 1.16.2;
- scikit-learn 1.7.2;
- joblib 1.5.2;
- PyYAML 6.0.2;
- jsonschema 4.25.1;
- PyTorch/Metal/CUDA unused.

Run artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `run_manifest.json` | `eff097ef18f6c165beea8393305a654fe653fbf11eb4dc75cbf66fc37613e6c0` |
| `model_manifest.json` | `3219673d6adc8cf9f7e3220768c1836dea216d0e7c85ae606c3acdb88222353d` |
| `metrics.json` | `dc35675928ce8cb3eab0a0594e0a718bdf7fdc223de2f019476bf6528076da38` |
| `baseline_comparison.json` | `ec6467e6958801e2486e372189bea324388a799c65de465e883d0d44ecfb9353` |
| `training_diagnostics.json` | `11b18d13de6a2233b7b4ef1f4e76f43543ba93ca33c0f2a6c4eea1b63596160d` |

## Anchor50 comparison

Declared local comparison tolerance is `1e-12`, but the comparison cannot
authorize reproduction because the reference environment is missing.

| Head | Reference | Observed | Absolute delta | Within tolerance |
| --- | ---: | ---: | ---: | --- |
| target_mode Macro-F1 | 0.3493754879 | 0.3221462748 | 0.0272292131 | No |
| stance Macro-F1 | 0.4146918097 | 0.3564183185 | 0.0582734912 | No |
| emotion_primary Macro-F1 | 0.2560903149 | 0.1911552500 | 0.0649350649 | No |
| emotion_target Macro-F1 | 0.2091770041 | 0.2603174603 | 0.0511404562 | No |
| action_tendency Macro-F1 | 0.1111111111 | 0.0931085885 | 0.0180025226 | No |
| context_dependency Macro-F1 | 0.3779461279 | 0.3888127854 | 0.0108666574 | No |
| reasoning_tags micro-F1 | 0.5070422535 | 0.5070422535 | 0.0000000000 | Yes |

All six scalar `saga` estimators record `n_iter=[2000]`,
`max_iter=2000`, and `converged=false`; six `ConvergenceWarning` entries are
stored. All 15 `liblinear` Reasoning estimators record `converged=true`.

The package does not state the reference Python, NumPy, SciPy, scikit-learn,
platform/CPU, or convergence state. scikit-learn 1.7 changed how balanced class
weights incorporate sample weights. A controlled Python 3.12 / scikit-learn
1.6.1 probe moved target_mode to `0.3439580133` but still did not match the
reference `0.3493754879`. Version guessing is therefore evidence of sensitivity,
not valid provenance. The reference owner must supply a content-addressed
environment/prediction handoff before the state can become `REPRODUCED`.

## Evaluation, export, inference, and immutable replay

Evaluation verified these ten run artifacts by hash:

`baseline_comparison.json`, `errors.jsonl`, `inference-output.schema.json`,
`metrics.json`, `model.joblib`, `model_manifest.json`,
`preprocessing_contract.json`, `schema.json`, `thresholds.json`, and
`training_diagnostics.json`.

- evaluation command-output SHA-256:
  `b763778f3b8d91d051e2f24b18ee8cb9dfdf900593564979ada959d72fc0a22b`;
- export ID:
  `414e5a3ffa7d506a61a6d7c1dd2a67c35d2e33e0a8829e85d0c6fdb41e6bd7c9`;
- export command-output SHA-256:
  `2ba81c958619c1052cac8bb05de90e3f56d5c53ac79ce7acaa62a43e4cce4219`;
- two-row CPU inference: `INFERENCE_COMPLETE`, seven heads per row, JSON Schema
  valid;
- inference output SHA-256:
  `6ad9c68c26796515f41217c9c64d3558278d04d27a0b8a4ce47b6cab4dde14bd`;
- inference command-output SHA-256:
  `8aa406709d7c632c55be8b006efeea0151a1f5f9f1117c6feeca037e46694e6d`;
- repeated `train` returned `EXISTING_IMMUTABLE_RUN` with the same run ID;
- replay command-output SHA-256:
  `17a1135e050edd874e9fa7f48ca1dc1b934cef751d4927c50793bd834b7a4b5f`.

## Engineering verification

The following completed successfully on the code commit above:

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
git diff --check
```

Result: `72 passed`; no broken requirements. The suite covers the synthetic
fail-closed harness and real native-package audit, including missing-package
preflight, content hashes, portable identities, native Evidence, mechanical
repair, explicit weights, compact Schema, Anchor-only compatibility,
preprocessing, feature order/counts, calibration, inference schema, convergence
diagnostics, reference-environment blocking, immutable replay, export, and CPU
inference.

No Encoder, PyTorch, Transformers, model download, 49,054-row inference, GUI,
service, or production integration was performed.
