# Milestone 1A — v0.3.5 Exact Reference Reproduction

**Final status:** `BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE`
**Date:** 2026-08-28
**Scope:** diagnostic-only reproduction of the frozen v0.3.5 classical
baseline. No model algorithm, tolerance, frozen package, reference package,
main branch, existing run, or existing export was modified.

## Decision

This worktree does **not** have the required reference execution environment:
it is macOS on Apple Silicon, not Linux x86_64 on the recorded AMD EPYC host.
The installed interpreter is also CPython 3.13.9 rather than CPython 3.12.13,
and SciPy is 1.17.1 rather than 1.17.0. This is a hard pre-fit blocker under
the frozen policy; it cannot be converted into an exact-reproduction claim by
using a similar dependency set, a container/emulator, or a relaxed tolerance.

In addition, the shared baseline commit is missing the tracked
`src/semantic_model/models/` package that the audit and training chain imports.
Consequently, neither repository audit command can begin its own checks on this
checkout. This is an implementation/checkout defect, **not** evidence that the
immutable reference package is damaged. A separate, read-only standard-library
ZIP verification passed for both supplied archives; details follow.

No local fit was run. Therefore, no run ID, model manifest ID, model artifact,
evaluation, export, inference output, replay result, or 2,787-row model
comparison exists for this attempt.

## Git isolation and evidence start

| Field | Recorded value |
| --- | --- |
| Source repository | `/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining` |
| Initial source branch | `feat/milestone-1-reproducible-baseline` |
| Common baseline commit | `83ec0eb12b7890d2dce5185975ee293b1688b45c` |
| New isolated worktree | `/Users/mac/Documents/trae_projects/MyResearcher/worktrees/MyResearcher-ModelTraining-exact-reproduction` |
| Delivery branch | `chore/milestone-1a-exact-reference-reproduction` |
| Worktree HEAD at start | `83ec0eb12b7890d2dce5185975ee293b1688b45c` |
| Worktree dirty at start | `false` |
| Last three commits at start | `83ec0eb docs: align encoder and review model roadmap`; `4fc8535 docs: record reference-aware baseline verification`; `0f366a8 feat: validate immutable baseline reference` |

The immutable packages were extracted only under this worktree's ignored
`data/local/` directory. `git check-ignore -v` confirms that directory is
ignored; no package payload, ZIP, original `joblib`, run, or export is staged.

## Immutable evidence audit

### Supplied data package

| Check | Result |
| --- | --- |
| ZIP path | `/Users/mac/Documents/Codex/2026-08-27/MyResearcher_Semantic_Immutable_Data_v0.3.5_cf7a10f25d951d79.zip` |
| Expected and observed ZIP SHA-256 | `c5ff639954fe71d8bc780175584406c6f5c84998c39d0040fdae830134a95378` |
| Content-manifest ID | `cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b` |
| ZIP CRC | `PASS` |
| Duplicate entries / unsafe entries | `0 / 0` |
| Payload count / verified bytes | `28 / 14,992,255` |
| Payload hash mismatches | `[]` |

### Supplied baseline-reference package

| Check | Result |
| --- | --- |
| ZIP path | `/Users/mac/Documents/Codex/2026-08-27/MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip` |
| Expected and observed ZIP SHA-256 | `78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410` |
| Content-manifest ID | `828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85` |
| ZIP CRC | `PASS` |
| Duplicate entries / unsafe entries | `0 / 0` |
| Payload count / verified bytes | `17 / 11,439,730` |
| Payload hash mismatches | `[]` |
| Binding to data content address | `sha256:cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b` — exact match |
| Original model SHA-256, hashed only and never loaded | `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a` |
| Frozen reference source SHA-256, not executed | `5efed038a99de3f5331a8c7ea29198bab12cd11c5e80cd6aede6fc44a7105183` |

The archive checks used Python's standard-library `zipfile` and `hashlib` after
the outer ZIP hashes had passed. They verified CRC, duplicate and unsafe paths,
manifest count/bytes, every manifest payload size, every manifest payload hash,
and the reference-to-data content-address binding. This supplementary check is
explicitly **not** represented as the repository's `audit_reference` result.
It did not execute the reference source or load/unpickle the external original
`model.joblib`.

## Mandatory repository audits

The following commands were attempted with `PYTHONPATH=src` so that the
worktree source was the only project source imported.

| Command | Exit | Result / audit ID |
| --- | ---: | --- |
| `python -m semantic_model.audit_reference --config configs/baseline_v0.3.5.yaml --archive …Reference…zip` | 1 | No `reference_audit_id` emitted. Import stopped at `ModuleNotFoundError: No module named 'semantic_model.models'`. |
| `python -m semantic_model.audit_data --config configs/baseline_v0.3.5.yaml` | 1 | No `audit_id` emitted. The same import failure occurred before package validation. |

The traceback path is
`audit_data -> immutable_package -> reference_package -> models.classical`.
`git ls-tree -r HEAD` and the on-disk source tree contain no
`src/semantic_model/models/` directory or `classical.py`, even though
`reference_package.py`, `train.py`, `infer.py`, and the affected tests import
it. The audit program therefore did **not** return an incorrect
exact-environment acceptance on this non-reference host; it failed closed
before deciding a status.

## Environment pre-flight audit

### Actual host (machine-generated)

| Item | Actual observation |
| --- | --- |
| `uname -a` | `Darwin MacBook-Pro.local 25.5.0 … RELEASE_ARM64_T6020 arm64` |
| `uname -m` | `arm64` |
| `/etc/os-release` | Unavailable: non-Linux host |
| `lscpu` | Unavailable: non-Linux host |
| Python executable / version | `/opt/homebrew/anaconda3/bin/python` / `Python 3.13.9` |
| `platform.platform()` | `macOS-26.5.2-arm64-arm-64bit-Mach-O` |
| `platform.machine()` / `platform.processor()` | `arm64` / `arm` |
| NumPy / SciPy / scikit-learn / joblib / threadpoolctl | `2.2.6` / `1.17.1` / `1.8.0` / `1.5.3` / `3.6.0` |
| `threadpoolctl.threadpool_info()` after NumPy operation | `[]` — no BLAS path, BLAS version, threading layer, architecture, or thread count available from this interpreter |
| `python -m pip check` | Exit 1; four unrelated global-environment conflicts: `vllm` ↔ `openai`, and `streamlit` ↔ `cachetools`, `pandas`, and `pyarrow`. |

`python -m pip freeze` was run as part of the recorded pre-flight audit. It is
the global Anaconda environment, not a Milestone 1 environment, and contains
many unrelated application packages; it was deliberately neither changed nor
committed. Its relevant installed versions are listed in the table above.

### Frozen reference environment (immutable evidence)

| Item | Reference observation |
| --- | --- |
| OS / release | Linux / Ubuntu 24.04.3 LTS, glibc 2.39 |
| Kernel platform | `Linux-6.18.35-x86_64-with-glibc2.39` |
| CPU | AMD EPYC 9V74 80-Core Processor; 9 logical execution threads recorded |
| Architecture / processor | `x86_64` / `x86_64` |
| Python | CPython `3.12.13`, Linux `x86_64` platform tag |
| NumPy / SciPy / scikit-learn / joblib / threadpoolctl | `2.3.5` / `1.17.0` / `1.8.0` / `1.5.3` / `3.6.0` |
| BLAS | OpenBLAS `0.3.30`, `pthreads`, `SkylakeX`; both NumPy and SciPy libraries recorded |
| OpenMP | Built with OpenMP; `libgomp`, 9 threads recorded |
| Thread environment | `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` all recorded as `null` |
| Original artifact loading | `artifact_unpickle_warnings: []` in the same recorded persistent runtime |

The full frozen `pip freeze` is an immutable reference payload:
`environment/pip_freeze_full_v0.3.5.txt` (SHA-256
`6149ae0425c48a7e0ddd21a8adeca2a5101c76710195c2c5230fc163d818b210`).
Its complete contents are reproduced below as immutable evidence. The companion
`numpy_show_config` and `sklearn_show_versions` evidence files record the
OpenBLAS and OpenMP facts above and are each covered by the verified reference
content manifest.

```text
PyMuPDF==1.26.6
PyYAML==6.0.3
annotated-types==0.8.0
artifact_tool_v2 @ file:///tmp/codex-primary-runtime-fT6EqF/generated-source--oai-artifact-tool/lib/agent/tools/artifact_tool_v2
cffi==1.17.1
charset-normalizer==3.4.4
contourpy==1.3.3
cryptography==46.0.0
cycler==0.12.1
et_xmlfile==2.0.0
fonttools==4.61.1
joblib==1.5.3
kiwisolver==1.4.9
lxml==6.1.1
matplotlib==3.10.8
numpy==2.3.5
openpyxl==3.1.5
packaging==26.3
pandas==2.2.3
pdf2image==1.17.0
pdfminer.six==20251107
pdfplumber==0.11.8
pillow==12.3.0
pycparser==2.23
pydantic==2.13.4
pydantic_core==2.46.4
pyhumps==3.8.0
pyparsing==3.3.2
pypdf==6.10.0
pypdfium2==5.3.0
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-pptx==1.0.2
pytz==2026.3.post1
reportlab==4.4.9
scikit-learn==1.8.0
scipy==1.17.0
seaborn==0.13.2
setuptools==83.0.0
six==1.17.0
threadpoolctl==3.6.0
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.3
uv==0.11.33
websockets==16.0
wheel==0.47.0
xlrd==2.0.1
xlsxwriter==3.2.9
zstandard==0.25.0
```

### Exact-environment comparison

| Required comparison | Result |
| --- | --- |
| Linux / Ubuntu vs Darwin / macOS | **Mismatch** |
| x86_64 / AMD EPYC vs arm64 / Apple Silicon | **Mismatch** |
| CPython 3.12.13 vs 3.13.9 | **Mismatch** |
| NumPy 2.3.5 vs 2.2.6 | **Mismatch** |
| SciPy 1.17.0 vs 1.17.1 | **Mismatch** |
| scikit-learn 1.8.0 vs 1.8.0 | Match only; insufficient |
| joblib 1.5.3 vs 1.5.3 | Match only; insufficient |
| threadpoolctl 3.6.0 vs 3.6.0 | Match only; insufficient |
| OpenBLAS 0.3.30 / pthreads / SkylakeX / 9 threads vs actual report | **Mismatch / unavailable** |
| OpenMP `libgomp` vs actual report | **Mismatch / unavailable** |

The local result is not an emulated or containerized approximation. In any
case, an Apple-Silicon Docker or QEMU environment would not authorize an exact
claim under the frozen policy.

### Provenance limitation retained verbatim in substance

The reference environment manifest states that it is a **retrospective capture
from the same persistent runtime bundle currently holding and loading the
original artifact**, and that the original training script did **not** emit an
environment manifest at fit time. It is therefore not represented as an
automatic contemporaneous fit-time snapshot. This limitation remains attached
to the environment evidence; no stronger provenance claim is made here.

## Frozen model and oracle facts

| Item | Frozen result |
| --- | --- |
| Feature contract | char `11,945`; word `313`; total `12,258` |
| Reference scalar estimators | 6 SAGA; each `n_iter_ = max_iter = 2000`; all `converged = false` |
| Reference Reasoning estimators | 15 liblinear; all converged |
| Reference metric recomputation | maximum absolute difference `0.0` |
| Reference prediction rows | Train `1,822`; Dev `448`; Test `467`; Anchor50 `50`; total `2,787` |
| Exact label criterion | every final label on every row and head must match |
| Exact metric criterion | maximum absolute delta `<= 1e-12` |
| Exact probability criterion | maximum absolute delta `<= 1e-10` |

The original source and original model were treated strictly as frozen evidence.
Neither was executed nor loaded. The reference JSON/JSONL files were inspected
only as oracle evidence; no reference script was used as an execution
instruction.

## Training and comparison disposition

| Required item | This attempt |
| --- | --- |
| `prepare` execution / prepare ID | Not invoked / none — environment gate already failed and repository audit import is broken. |
| Feature result / parity (`difference nnz`, `max abs diff`) | Not measured; no prepare run. |
| `train` execution / run ID / run manifest / model manifest | Not invoked / none — no local model was fit. |
| Local model hash and elapsed time | Not applicable. |
| Local convergence diagnostics or model warnings | Not applicable. |
| 2,787-row final-label comparison | **Not performed**; 0 of 2,787 rows were compared from a newly trained model. |
| Metric maximum absolute delta | Not applicable; no comparison was produced. |
| Probability maximum absolute delta | Not applicable; no comparison was produced. |
| Evaluation / export / two-row inference / immutable-run replay | Not performed; no training artifact exists. |
| 49,054-row inference | Not run and remains forbidden. |

This is not an oracle mismatch outcome: no fitting or oracle comparison was
allowed to start in an accepted reference environment. It is therefore neither
`BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY` nor
`REJECTED_EXACT_REPRODUCTION_MISMATCH`.

## Validation commands

| Command | Exit | Result |
| --- | ---: | --- |
| `PYTHONPATH=src python -m compileall -q src tests` | 0 | Syntax compilation passed; it cannot detect a missing imported module. |
| `PYTHONPATH=src python -m pytest -q` | 2 | Seven collection errors, all rooted in missing `semantic_model.models`. |
| `python -m pip check` | 1 | Global Anaconda environment has the four conflicts documented above. |
| `git diff --check` | 0 | Passed after the report was written. |

## Production gate

| Gate | Value |
| --- | --- |
| Final Milestone 1A status | `BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE` |
| Exact reproduced | `false` |
| `production_approval` | `false` |
| `production_inference_49054_allowed` | `false` |
| `encoder_training_allowed` | `false` |
| Production inference run | `false` |

## Required next conditions (not performed by this task)

1. Obtain a real, auditable Linux x86_64 environment that matches the frozen
   Ubuntu/AMD EPYC/Python/NumPy/SciPy/scikit-learn/joblib/OpenBLAS/OpenMP and
   threadpool contract. No paid cloud resource was provisioned for this task.
2. Restore the missing `semantic_model.models.classical` implementation through
   the owning source/checkout remediation, then rerun the repository's
   `audit_reference` and `audit_data` gates from that environment.
3. Only if those gates authorize the same reference environment may the frozen
   `prepare`, `train`, full 2,787-row comparison, evaluation, export, smoke
   inference, and immutable replay be run. The comparison must retain the
   exact-label, `1e-12` metric, and `1e-10` probability criteria unchanged.

No Encoder, PyTorch/Transformers component, Gold set, OOD version, or model
algorithm change was started in this Milestone 1A diagnostic task.
