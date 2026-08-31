"""Content-addressed evidence required by an exact v0.3.5 write workflow.

The receipt is intentionally an external handoff artifact: this module can
verify a supplied owner authorization receipt but never creates or infers one.
Ordinary comparable diagnostic execution does not use this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ContractError
from .hashes import content_addressed_id


STRICT_PREFLIGHT_RECEIPT_SCHEMA = "semantic-exact-preflight-receipt-v0.3.5"
OWNER_AUTHORIZATION_RECEIPT_SCHEMA = "semantic-owner-authorization-receipt-v0.3.5"

_STRICT_RECEIPT_ID_KEY = "strict_preflight_receipt_id"
_OWNER_RECEIPT_ID_KEY = "owner_authorization_receipt_id"
_OBSERVATION_KEY = "observation"
_STRICT_RECEIPT_KEYS = {
    "receipt_schema_version",
    "accepted_source_commit",
    "source_worktree_clean",
    "canonical_data_audit_id",
    "data_package_manifest_id",
    "reference_audit_id",
    "reference_package_manifest_id",
    "original_model_sha256",
    "reference_environment_evidence_sha256",
    "current_strict_runtime_capture",
    "exact_environment_audit_id",
    "exact_environment_ready",
    "environment_policy_eligible_for_exact_reproduction",
    "owner_prepare_authorized",
    "owner_train_authorized",
    "production_approval",
    "production_inference_49054_allowed",
    _OBSERVATION_KEY,
    _STRICT_RECEIPT_ID_KEY,
}


def _contains_local_path(value: Any) -> bool:
    """Reject location-bearing strict-receipt content, including nested fields."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {"path", "filepath", "file_path", "local_path"}:
                return True
            if _contains_local_path(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_local_path(item) for item in value)
    if isinstance(value, str):
        return value.startswith("/") or (
            len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}
        )
    return False


def _require_mapping(value: Any, *, code: str, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(code, message)
    return value


def _load_receipt(value: str | Path | Mapping[str, Any], *, kind: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_INVALID",
            f"unable to read {kind} receipt",
        ) from exc
    return _require_mapping(
        loaded,
        code="EXACT_EXECUTION_RECEIPT_INVALID",
        message=f"{kind} receipt must be a JSON object",
    )


def _verify_receipt_id(
    receipt: Mapping[str, Any], *, id_key: str, kind: str
) -> str:
    observed = receipt.get(id_key)
    expected = content_addressed_id(
        receipt, omit_keys={id_key, _OBSERVATION_KEY}
    )
    if observed != expected:
        raise ContractError(
            "CONTENT_ADDRESS_MISMATCH",
            f"{kind} receipt content address differs",
            observed=observed,
            expected=expected,
        )
    return expected


def _require_nonempty_string(receipt: Mapping[str, Any], key: str, *, kind: str) -> str:
    value = receipt.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_INVALID",
            f"{kind} receipt requires {key}",
        )
    return value


def build_strict_preflight_receipt(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Create a content-addressed receipt from an exact, clean preflight only."""

    if (
        preflight.get("status") != "EXACT_REFERENCE_ENVIRONMENT_READY"
        or preflight.get("environment_policy_eligible_for_exact_reproduction")
        is not True
        or preflight.get("exact_environment_ready") is not True
    ):
        raise ContractError(
            "EXACT_EXECUTION_ENVIRONMENT_NOT_ELIGIBLE",
            "a strict receipt requires an exact-reference-environment preflight",
        )
    source = _require_mapping(
        preflight.get("source_identity"),
        code="EXACT_EXECUTION_SOURCE_IDENTITY_INVALID",
        message="strict preflight lacks source identity",
    )
    if source.get("source_worktree_clean") is not True:
        raise ContractError(
            "EXACT_EXECUTION_SOURCE_DIRTY",
            "a strict receipt requires a clean source worktree",
        )
    receipt: dict[str, Any] = {
        "receipt_schema_version": STRICT_PREFLIGHT_RECEIPT_SCHEMA,
        "accepted_source_commit": _require_nonempty_string(
            source, "accepted_source_commit", kind="strict preflight"
        ),
        "source_worktree_clean": True,
        "canonical_data_audit_id": _require_nonempty_string(
            preflight, "canonical_data_audit_id", kind="strict preflight"
        ),
        "data_package_manifest_id": _require_nonempty_string(
            preflight, "data_package_manifest_id", kind="strict preflight"
        ),
        "reference_audit_id": _require_nonempty_string(
            preflight, "reference_package_audit_id", kind="strict preflight"
        ),
        "reference_package_manifest_id": _require_nonempty_string(
            preflight, "reference_package_manifest_id", kind="strict preflight"
        ),
        "original_model_sha256": _require_nonempty_string(
            preflight, "original_model_sha256", kind="strict preflight"
        ),
        "reference_environment_evidence_sha256": _require_nonempty_string(
            preflight,
            "reference_environment_evidence_sha256",
            kind="strict preflight",
        ),
        "current_strict_runtime_capture": dict(
            _require_mapping(
                preflight.get("current_environment_capture"),
                code="EXACT_EXECUTION_RECEIPT_INVALID",
                message="strict preflight lacks current runtime capture",
            )
        ),
        "exact_environment_audit_id": _require_nonempty_string(
            preflight, "exact_environment_audit_id", kind="strict preflight"
        ),
        "exact_environment_ready": True,
        "environment_policy_eligible_for_exact_reproduction": True,
        "owner_prepare_authorized": False,
        "owner_train_authorized": False,
        "production_approval": False,
        "production_inference_49054_allowed": False,
        "observation": {
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    receipt[_STRICT_RECEIPT_ID_KEY] = content_addressed_id(
        receipt, omit_keys={_STRICT_RECEIPT_ID_KEY, _OBSERVATION_KEY}
    )
    return receipt


def verify_strict_preflight_receipt(
    value: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify intrinsic receipt integrity without trusting a caller boolean."""

    receipt = _load_receipt(value, kind="strict preflight")
    if receipt.get("receipt_schema_version") != STRICT_PREFLIGHT_RECEIPT_SCHEMA:
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_INVALID",
            "unsupported strict preflight receipt schema",
        )
    if set(receipt) != _STRICT_RECEIPT_KEYS or _contains_local_path(receipt):
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_INVALID",
            "strict receipt has unexpected fields or local path content",
        )
    _verify_receipt_id(receipt, id_key=_STRICT_RECEIPT_ID_KEY, kind="strict preflight")
    if (
        receipt.get("exact_environment_ready") is not True
        or receipt.get("environment_policy_eligible_for_exact_reproduction")
        is not True
        or receipt.get("source_worktree_clean") is not True
        or receipt.get("production_approval") is not False
        or receipt.get("production_inference_49054_allowed") is not False
        or receipt.get("owner_prepare_authorized") is not False
        or receipt.get("owner_train_authorized") is not False
    ):
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_INVALID",
            "strict receipt has invalid execution eligibility or approval fields",
        )
    for key in (
        "accepted_source_commit",
        "canonical_data_audit_id",
        "data_package_manifest_id",
        "reference_audit_id",
        "reference_package_manifest_id",
        "original_model_sha256",
        "reference_environment_evidence_sha256",
        "exact_environment_audit_id",
    ):
        _require_nonempty_string(receipt, key, kind="strict preflight")
    _require_mapping(
        receipt.get("current_strict_runtime_capture"),
        code="EXACT_EXECUTION_RECEIPT_INVALID",
        message="strict receipt lacks current runtime capture",
    )
    return receipt


def verify_current_strict_preflight_receipt(
    receipt_value: str | Path | Mapping[str, Any],
    *,
    config_path: str | Path,
    reference_archive: str | Path,
) -> Mapping[str, Any]:
    """Re-run strict preflight and reject source/data/reference/runtime drift."""

    receipt = verify_strict_preflight_receipt(receipt_value)
    # Deferred import prevents a module cycle while still using the one strict
    # preflight implementation as the current-environment verifier.
    from .audit_exact_environment import run_exact_environment_audit

    current, exit_code = run_exact_environment_audit(config_path, reference_archive)
    if exit_code != 0 or current.get("strict_preflight_receipt") is None:
        raise ContractError(
            "EXACT_EXECUTION_PREFLIGHT_FAILED",
            "current strict preflight did not produce an exact receipt",
            status=current.get("status"),
        )
    current_receipt = verify_strict_preflight_receipt(
        _require_mapping(
            current.get("strict_preflight_receipt"),
            code="EXACT_EXECUTION_PREFLIGHT_FAILED",
            message="current strict preflight did not emit a receipt",
        )
    )
    if current_receipt.get(_STRICT_RECEIPT_ID_KEY) != receipt.get(
        _STRICT_RECEIPT_ID_KEY
    ):
        raise ContractError(
            "EXACT_EXECUTION_RECEIPT_STALE",
            "source, package, environment evidence, or runtime identity drifted",
            observed=current_receipt.get(_STRICT_RECEIPT_ID_KEY),
            expected=receipt.get(_STRICT_RECEIPT_ID_KEY),
        )
    return receipt


def verify_owner_authorization_receipt(
    value: str | Path | Mapping[str, Any],
    *,
    required_scope: str,
    strict_preflight_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate an externally supplied owner authorization for one write scope."""

    receipt = _load_receipt(value, kind="owner authorization")
    if receipt.get("receipt_schema_version") != OWNER_AUTHORIZATION_RECEIPT_SCHEMA:
        raise ContractError(
            "OWNER_AUTHORIZATION_RECEIPT_INVALID",
            "unsupported owner authorization receipt schema",
        )
    _verify_receipt_id(receipt, id_key=_OWNER_RECEIPT_ID_KEY, kind="owner authorization")
    if receipt.get("scope") != required_scope or receipt.get("approved") is not True:
        raise ContractError(
            "OWNER_AUTHORIZATION_SCOPE_DENIED",
            f"owner authorization does not grant {required_scope}",
        )
    if receipt.get("strict_preflight_receipt_id") != strict_preflight_receipt.get(
        _STRICT_RECEIPT_ID_KEY
    ):
        raise ContractError(
            "OWNER_AUTHORIZATION_BINDING_MISMATCH",
            "owner authorization targets a different strict receipt",
        )
    for key in ("authorized_by", "external_record_id"):
        _require_nonempty_string(receipt, key, kind="owner authorization")
    if receipt.get("production_approval") is not False:
        raise ContractError(
            "OWNER_AUTHORIZATION_RECEIPT_INVALID",
            "exact baseline owner authorization cannot grant production approval",
        )
    return receipt


def validate_exact_write_authorization(
    *,
    config_path: str | Path,
    reference_archive: str | Path | None,
    strict_preflight_receipt: str | Path | Mapping[str, Any] | None,
    owner_authorization_receipt: str | Path | Mapping[str, Any] | None,
    required_scope: str,
) -> dict[str, Any]:
    """Revalidate strict evidence and one owner scope before any write path."""

    if strict_preflight_receipt is None:
        raise ContractError(
            "BLOCKED_EXACT_EXECUTION_RECEIPT_MISSING",
            "exact-mode execution requires a strict preflight receipt",
        )
    if reference_archive is None:
        raise ContractError(
            "BLOCKED_EXACT_EXECUTION_REFERENCE_ARCHIVE_MISSING",
            "exact-mode execution requires the pinned reference archive",
        )
    if owner_authorization_receipt is None:
        raise ContractError(
            "BLOCKED_OWNER_AUTHORIZATION_RECEIPT_MISSING",
            f"exact-mode {required_scope} requires an external owner receipt",
        )
    strict_receipt = verify_current_strict_preflight_receipt(
        strict_preflight_receipt,
        config_path=config_path,
        reference_archive=reference_archive,
    )
    owner_receipt = verify_owner_authorization_receipt(
        owner_authorization_receipt,
        required_scope=required_scope,
        strict_preflight_receipt=strict_receipt,
    )
    return {
        "strict_preflight_receipt": dict(strict_receipt),
        "strict_preflight_receipt_id": strict_receipt[
            _STRICT_RECEIPT_ID_KEY
        ],
        "accepted_source_commit": strict_receipt["accepted_source_commit"],
        "source_worktree_clean": strict_receipt["source_worktree_clean"],
        "canonical_data_audit_id": strict_receipt["canonical_data_audit_id"],
        "data_package_manifest_id": strict_receipt["data_package_manifest_id"],
        "reference_audit_id": strict_receipt["reference_audit_id"],
        "reference_package_manifest_id": strict_receipt[
            "reference_package_manifest_id"
        ],
        "original_model_sha256": strict_receipt["original_model_sha256"],
        "exact_environment_audit_id": strict_receipt["exact_environment_audit_id"],
        "reference_environment_evidence_sha256": strict_receipt[
            "reference_environment_evidence_sha256"
        ],
        "current_strict_runtime_capture": strict_receipt[
            "current_strict_runtime_capture"
        ],
        "owner_authorization_receipt_id": owner_receipt[_OWNER_RECEIPT_ID_KEY],
        "owner_authorization_scope": required_scope,
        "production_approval": False,
        "production_inference_49054_allowed": False,
    }
