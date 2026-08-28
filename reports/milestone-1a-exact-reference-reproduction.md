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

At the initial review, the shared baseline commit was missing the tracked
`src/semantic_model/models/` package that the audit and training chain imports.
Consequently, neither repository audit command could begin its own checks on the
initial clean checkout. This was an implementation/checkout defect, **not** evidence that the
immutable reference package is damaged. A separate, read-only standard-library
ZIP verification passed for both supplied archives; details follow.

No local fit was run. Therefore, no run ID, model manifest ID, model artifact,
evaluation, export, inference output, replay result, or 2,787-row model
comparison exists for this attempt.

## Follow-up side-review remediation — source completeness

This section was added after the side review identified the missing tracked
classical-model package. It preserves the original blocker observations below
and records the subsequent remediation and verification in chronological order.

### Initial blockers

- `BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE` — active and unresolved. The
  available host is macOS arm64, not the accepted Linux x86_64 / AMD EPYC
  reference environment.
- `BLOCKED_MISSING_TRACKED_CLASSICAL_MODEL_IMPLEMENTATION` — confirmed as an
  independent P1 source-completeness problem in the initial clean worktree.
  It made `semantic_model.models` unavailable and caused seven test modules to
  fail collection, while the primary worktree could import its ignored local
  copy.

### Source provenance and remediation

The source candidate was not copied until provenance had been verified against
the formal macOS comparable run
`49f67b0476c4f439f1d867476d5c850316048fe09e859d652f2cf4225b31c4db`.
That run's manifest ID is
`784ed578cfc71b98df021e251380a99fda7724907dd42e1b61e8e5b1bd6c8dd2`; its
`run_manifest.json` SHA-256 is
`424630826062111c80a891bfecf29a11b82fdffd9a4800cf13eb6220812c2a0a`.
The run's `code_manifest` explicitly records both required source paths.

| Path | Primary ignored source SHA-256 | Formal run `code_manifest` SHA-256 | Side worktree SHA-256 | Result |
| --- | --- | --- | --- | --- |
| `src/semantic_model/models/__init__.py` | `9eb280ac6f0c590dc6a31022ba4c5ef268ade4345dccf57149e62c652e249ba5` | `9eb280ac6f0c590dc6a31022ba4c5ef268ade4345dccf57149e62c652e249ba5` | `9eb280ac6f0c590dc6a31022ba4c5ef268ade4345dccf57149e62c652e249ba5` | Exact three-way match |
| `src/semantic_model/models/classical.py` | `184c24799e2491de3ce645853e80254086c3e238cea2c779b9d629d15c6e9728` | `184c24799e2491de3ce645853e80254086c3e238cea2c779b9d629d15c6e9728` | `184c24799e2491de3ce645853e80254086c3e238cea2c779b9d629d15c6e9728` | Exact three-way match |

The recorded primary-worktree mtimes (`2026-08-27T11:40:29Z` for
`__init__.py`, `2026-08-27T14:59:09Z` for `classical.py`) were considered only
supporting evidence; the byte hashes above were the admission criterion.

The root cause was verified as the unanchored `.gitignore` rule
`.gitignore:28:models/`. Git confirmed that this rule ignored both nested Python
files, while `git ls-tree -r 83ec0eb… -- src/semantic_model` confirmed neither
path existed in the baseline Git tree. The rule now reads `/models/`, which
continues to ignore root-level model artifacts but no longer ignores the Python
package. `git check-ignore -v` now returns no match for either source file and
matches `.gitignore:28:/models/` for a root `models/example-artifact` probe.

The admitted source is a transparent local TF-IDF + scikit-learn Logistic
Regression implementation. Static review found no secrets, API tokens,
credentials, absolute local paths, network/LLM imports, model weights, or
untrusted serialized-data loading. Its `ClassicalMultiHeadModel` import and
public calls are consistent with `train.py`, `infer.py`, and
`reference_package.py`.

The verified source was added byte-for-byte, without an algorithm change, in
the independently cherry-pickable commit:

```text
022a8b7af8b3a93ec39aef641f67b13300275e92
fix: track classical model package
```

That commit contains exactly these three files and no report, model, data,
run, or export artifact:

```text
.gitignore
src/semantic_model/models/__init__.py
src/semantic_model/models/classical.py
```

`git ls-files src/semantic_model/models` now lists both source files. The scan
for other ignored `src/**/*.py` or `tests/**/*.py` files returned no output;
only ignored caches and immutable local-data material remain outside Git.

The preserved, run-manifest-matching `__init__.py` has a historical blank line
at EOF. `git diff --cached --check` reported that one whitespace warning while
the source-fix commit was staged. It was deliberately not removed because doing
so would change its SHA-256 and violate the formal run-code provenance. The
post-commit clean-worktree `git diff --check` exits 0; no policy or source
behavior was changed to hide this provenance fact.

### Post-remediation developer validation

The following project validation environment is distinct from the initial
global Anaconda shell and from the frozen Linux reference environment:

| Field | Post-remediation project developer environment |
| --- | --- |
| Interpreter | `/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining/.venv/bin/python` |
| Python / host | CPython `3.12.13`; macOS `arm64` |
| NumPy / SciPy | `2.3.3` / `1.16.2` |
| scikit-learn / joblib / threadpoolctl | `1.7.2` / `1.5.2` / `3.6.0` |
| Imported source | This side worktree's `src/semantic_model/models/classical.py` |
| CPU threading observed | `libomp.dylib`, OpenMP, 12 threads; no accepted reference OpenBLAS entry |

All commands use `PYTHONPATH` pointing explicitly at this side worktree's
`src` directory; no import falls back to the ignored primary-worktree source.

| Command | Exit | Result |
| --- | ---: | --- |
| `python -m compileall -q src tests` | 0 | Passed. |
| `python -m pytest -q` | 0 | All 76 collected tests passed; no `semantic_model.models` collection error. |
| `python -m pip check` | 0 | `No broken requirements found` in the project `.venv`. |
| `git diff --check` | 0 | Clean post-commit worktree. |

### Post-remediation package audits and deterministic preparation

| Operation | Exit / ID | Result |
| --- | --- | --- |
| `audit_reference` | 0 / `6a2fe3f4a532341e9891581548840701086a07f7a4ce8864c98ddfed36926793` | `REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY` |
| `audit_data` | 0 / `5fab05d633c509122bb8bbddd95b5d79f8d76a660b284f5cb20120df2865e414` | `READY_FOR_COMPARABLE_DIAGNOSTIC_RUN` |
| `prepare` | 0 / `b49335f8c1297b06bd4c41867319c87801787bd678f3c36ba471bea92809d853` | `PREPARED`; immutable ignored preparation artifact only |

Both audits validated the data content ID
`cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b`
and reference content ID
`828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85`.
Reference audit validated 17 payloads / 11,439,730 bytes, the original-model
hash `4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a`,
and all 2,787 frozen prediction rows. Data audit validated the 28-payload
immutable data package and returned no missing artifacts.

The post-remediation reference audit correctly reports only
`COMPARABLE_DIAGNOSTIC_RUN_ONLY` and
`BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`. Its recorded mismatches are Darwin
versus Linux, arm64 versus x86_64, NumPy 2.3.3 versus 2.3.5, SciPy 1.16.2
versus 1.17.0, scikit-learn 1.7.2 versus 1.8.0, joblib 1.5.2 versus 1.5.3,
and absence of the required OpenBLAS 0.3.30/pthreads record. It did **not**
return an exact-environment status.

Preparation used the frozen 1,822-row Train partition and produced char
features `11,945`, word features `313`, and total features `12,258`. An
independent `hstack([char, word])` reconstruction matched the fitted feature
matrix exactly: `difference nnz = 0`, `max abs diff = 0.0`. The prepare
manifest ID is the same ID reported above.

### Resolved and remaining blockers

```text
BLOCKED_MISSING_TRACKED_CLASSICAL_MODEL_IMPLEMENTATION
→ RESOLVED_BY_TRACKED_SOURCE_COMMIT

BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE
→ ACTIVE
```

No exact-environment fit was run. `train`, evaluation, export, inference, and
immutable-run replay were intentionally not run in this follow-up. **0 of
2,787 rows** were compared from a newly trained exact-environment model.

### Final follow-up state

```yaml
final_status: BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE
exact_reproduced: false
production_approval: false
production_inference_49054_allowed: false
```

Restoring clean-source completeness does not alter the frozen exact-reproduction
gate. The only eligible next execution environment remains the audited Linux
x86_64 / AMD EPYC reference environment with the documented dependency, BLAS,
OpenMP, and threadpool contract.


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

## Initial repository-audit attempts (before source remediation)

The following commands were attempted with `PYTHONPATH=src` so that the
worktree source was the only project source imported.

| Command | Exit | Result / audit ID |
| --- | ---: | --- |
| `python -m semantic_model.audit_reference --config configs/baseline_v0.3.5.yaml --archive …Reference…zip` | 1 | No `reference_audit_id` emitted. Import stopped at `ModuleNotFoundError: No module named 'semantic_model.models'`. |
| `python -m semantic_model.audit_data --config configs/baseline_v0.3.5.yaml` | 1 | No `audit_id` emitted. The same import failure occurred before package validation. |

The traceback path is
`audit_data -> immutable_package -> reference_package -> models.classical`.
`git ls-tree -r 83ec0eb -- src/semantic_model` and the initial worktree source tree contain no
`src/semantic_model/models/` directory or `classical.py`, even though
`reference_package.py`, `train.py`, `infer.py`, and the affected tests import
it. The audit program therefore did **not** return an incorrect
exact-environment acceptance on this non-reference host; it failed closed
before deciding a status.

## Initial environment pre-flight audit

### Initial global shell environment (machine-generated)

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

## Initial training and comparison disposition

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

## Initial validation commands

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

## Initial required next conditions (partially superseded by source remediation)

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
