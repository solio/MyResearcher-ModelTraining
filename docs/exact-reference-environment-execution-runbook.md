# Exact Reference Environment — Execution Runbook

Status: `MILESTONE_1A_EXACT_ENVIRONMENT_EXECUTION_GATE_READY`.

This is a handoff runbook for a future executor who has the actual frozen
Linux x86_64 / AMD EPYC reference runtime. It does not authorize a training
run on the current macOS arm64 workstation. The current state remains
`BLOCKED_REFERENCE_ENVIRONMENT_UNAVAILABLE`, `exact_reproduced=false`,
`production_approval=false`, and
`production_inference_49054_allowed=false`.

The only preflight authority is the read-only command below. A zero exit means
only that the environment identity is exact enough to request the next owner
authorization. It does not run `prepare`, does not fit a model, and does not
approve production.

```bash
PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.audit_exact_environment \
  --config "$REPO/configs/baseline_v0.3.5.yaml" \
  --reference-archive "$REFERENCE_ZIP"
```

Its JSON is sorted and has the stable `exact_environment_audit_id`. On an
unchanged runtime, rerunning it produces the same ID. A non-zero exit—most
notably `BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH`—is a terminal workflow
failure: do not proceed to `prepare` or `train`.

## Frozen environment identity

The gate compares the whole frozen execution identity, not merely `pip freeze`:

- OS, kernel/release, distribution evidence, architecture, processor, glibc;
- logical CPU count and CPU model/family;
- CPython implementation, exact version, compiler, and platform tag;
- NumPy, SciPy, scikit-learn, joblib, and threadpoolctl versions;
- all normalized threadpool records, including BLAS implementation/version/
  threading layer/architecture and the OpenMP runtime;
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and the recorded runtime
  bundle version.

The reference evidence is deliberately limited: it was captured
retrospectively from the same persistent runtime that retained and loaded the
original model without warnings. It was not contemporaneous telemetry emitted
by the historical fit command. That limitation remains part of the audit
output and cannot be upgraded.

Docker on Apple Silicon, QEMU, matching Python package versions with a
different BLAS, the same Linux distribution on a different CPU, a similar AMD
cloud machine, or only a matching `pip freeze` are all non-matches. Do not
rename any of these into an exact reference environment.

## Future executor: fail-fast preflight

Set only task-scoped variables. Do not set or overwrite `HOME`, `home`, or
`CODEX_HOME`.

```bash
set -euo pipefail

REPO=/path/to/MyResearcher-ModelTraining
PYTHON=/path/to/reference/python
DATA_ZIP=/path/to/MyResearcher_Semantic_Immutable_Data_v0.3.5_cf7a10f25d951d79.zip
REFERENCE_ZIP=/path/to/MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip
EXPECTED_SOURCE_COMMIT=/approved/exact-execution-gate-commit

test -d "$REPO"
test -x "$PYTHON"
test -f "$DATA_ZIP"
test -f "$REFERENCE_ZIP"

# 1. Verify the exact source commit and a clean execution worktree.
git -C "$REPO" status --short
test -n "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"

# 2. Verify both immutable archive content addresses before any installation.
test "$(shasum -a 256 "$DATA_ZIP" | awk '{print $1}')" = \
  "c5ff639954fe71d8bc780175584406c6f5c84998c39d0040fdae830134a95378"
test "$(shasum -a 256 "$REFERENCE_ZIP" | awk '{print $1}')" = \
  "78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410"

# Installation/extraction is an independently authorized immutable handoff
# action. It must preserve the pinned package directory names and bytes.

# 3. Audit the already extracted canonical and reference packages. These are
# read-only: neither command executes reference source nor unpickles its joblib.
PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.audit_data \
  --config "$REPO/configs/baseline_v0.3.5.yaml"
PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.audit_reference \
  --config "$REPO/configs/baseline_v0.3.5.yaml" \
  --archive "$REFERENCE_ZIP"

# 4. The strict preflight is the final no-write gate. set -e makes a mismatch
# stop here; its JSON remains in the shell variable rather than a report file.
exact_preflight_json="$(PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.audit_exact_environment \
  --config "$REPO/configs/baseline_v0.3.5.yaml" \
  --reference-archive "$REFERENCE_ZIP")"
printf '%s\n' "$exact_preflight_json"
EXACT_PREFLIGHT_JSON="$exact_preflight_json" "$PYTHON" - <<'PY'
import json
import os

result = json.loads(os.environ["EXACT_PREFLIGHT_JSON"])
if result.get("status") != "EXACT_REFERENCE_ENVIRONMENT_READY":
    raise SystemExit("strict preflight did not authorize the next workflow step")
if result.get("exact_environment_ready") is not True:
    raise SystemExit("exact_environment_ready must be true")
if result.get("training_invoked") is not False:
    raise SystemExit("preflight must never invoke training")
if result.get("production_approval") is not False:
    raise SystemExit("this baseline never receives production approval")
PY
```

The older `audit_reference` command may correctly exit zero with
`REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY`. That confirms a
trusted package, not a trusted execution environment; it can never bypass the
strict preflight in step 4.

## Explicitly gated write phase (future only)

The following is a template, not a command to run on the current workstation.
It defaults to dry-run. `EXECUTE_WRITES=1` permits only preparation after all
six preflight gates have passed; training has its own separate owner-controlled
flag. Do not collapse the flags or let a non-zero preflight continue.

```bash
set -euo pipefail

EXECUTE_WRITES="${EXECUTE_WRITES:-0}"
EXECUTE_TRAIN="${EXECUTE_TRAIN:-0}"

if [ "$EXECUTE_WRITES" != "1" ]; then
  printf '%s\n' 'Dry run complete: no prepare, train, export, or inference was invoked.'
  exit 0
fi

# Requires recorded owner authorization for PREPARE.
PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.prepare \
  --config "$REPO/configs/baseline_v0.3.5.yaml"

# Preparation only establishes an immutable feature manifest. It is not a
# fitting authorization and does not establish exact reproduction.
PYTHONPATH="$REPO/src" "$PYTHON" -m pytest -q \
  "$REPO/tests/test_preprocessing_parity.py" \
  "$REPO/tests/test_prepare_pipeline.py"

if [ "$EXECUTE_TRAIN" != "1" ]; then
  printf '%s\n' 'Prepare completed; train remains blocked pending separate owner authorization.'
  exit 0
fi

# Requires separate recorded owner authorization for TRAIN. Capture the JSON
# response, identify RUN_DIR, then follow the state contract sequentially.
PYTHONPATH="$REPO/src" "$PYTHON" -m semantic_model.train \
  --config "$REPO/configs/baseline_v0.3.5.yaml"
```

The complete state order, ownership requirements, write/fit permissions, and
content-address expectations are frozen in
[`exact-reproduction-execution-contract-v0.3.5.json`](../manifests/exact-reproduction-execution-contract-v0.3.5.json).
The mandatory sequence is:

1. `VERIFY_SOURCE_COMMIT` through `AUDIT_EXACT_ENVIRONMENT` (the first six
   read-only terminal gates);
2. `PREPARE`, then `VERIFY_FEATURE_PARITY`;
3. separately authorized `TRAIN`, then convergence/warning capture;
4. 2,787-row prediction, metrics, and probability oracle comparison;
5. evaluate, separately authorized local export, two-row non-production smoke
   test, immutable replay, and final diagnostic-only status.

## Frozen exact success criteria

Every condition below is mandatory; none may be relaxed:

- Train/Dev/Test/Anchor50 reference rows are exactly 1,822 / 448 / 467 / 50,
  totaling 2,787.
- Every scalar final label and every Reasoning-tag final label matches the
  reference row-for-row.
- Every scalar class order and every Reasoning threshold matches exactly.
- Maximum absolute metric delta is at most `1e-12`; maximum absolute
  probability delta is at most `1e-10`.
- The six original scalar SAGA estimators have
  `n_iter_=2000`, `max_iter=2000`, and `converged=false`; all 15 Reasoning
  liblinear estimators converge.

Only then may the run be finalized as
`BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY`. That status still leaves
`production_approval=false` and
`production_inference_49054_allowed=false`. No 49,054-row inference, encoder
work, network access, external LLM call, reference-source execution, or
external-joblib unpickle is permitted by this runbook.
