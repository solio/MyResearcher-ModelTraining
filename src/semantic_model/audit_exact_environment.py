"""Fail-closed, read-only authorization gate for exact v0.3.5 execution.

This module intentionally performs no preparation, fitting, evaluation, export,
inference, network access, archive extraction, or reference-artifact loading.
It verifies the canonical data and reference-package evidence already present
on disk, then determines whether the current runtime is the frozen reference
environment.  A passing result authorizes only the next workflow decision; it
never invokes training and never authorizes production.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .audit_data import audit_config
from .audit_reference import build_reference_audit_result
from .config import ProjectConfig
from .errors import ContractError
from .exact_execution_receipt import build_strict_preflight_receipt
from .hashes import content_addressed_id, without_local_paths
from .reference_package import (
    audit_reference_archive,
    capture_runtime_environment,
    compare_runtime_to_reference,
    load_reference_environment,
    reference_environment_evidence_sha256,
)


EXACT_ENVIRONMENT_AUDIT_SCHEMA = "semantic-exact-environment-audit-v0.3.5"
_RUNTIME_MISMATCH_STATUS = "BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"

_NON_IDENTITY_OUTPUT_FIELDS = {
    "exact_environment_audit_id",
    "source_identity",
    "environment_policy_eligible_for_exact_reproduction",
    "owner_prepare_authorized",
    "owner_train_authorized",
    "prepare_execution_authorized",
    "train_execution_authorized",
    "next_required_decision",
    "strict_preflight_receipt",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _identity_error(error: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Retain a stable error class without placing local paths in the audit ID."""

    if error is None:
        return None
    return {"code": error.get("code")}


def _audit_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a location- and time-independent audit identity payload."""

    identity = dict(without_local_paths(result))
    for key in _NON_IDENTITY_OUTPUT_FIELDS:
        identity.pop(key, None)
    error = identity.get("error")
    if isinstance(error, Mapping):
        identity["error"] = _identity_error(error)
    return identity


def _finalize(result: dict[str, Any]) -> dict[str, Any]:
    result["exact_environment_audit_id"] = content_addressed_id(_audit_identity(result))
    return result


def _empty_result() -> dict[str, Any]:
    """Return the fixed output shape used even when preflight fails early."""

    return {
        "audit_schema_version": EXACT_ENVIRONMENT_AUDIT_SCHEMA,
        "status": "BLOCKED_EXACT_ENVIRONMENT_PREFLIGHT_INVALID",
        "exact_environment_ready": False,
        "exact_reproduction_authorized": False,
        "environment_policy_eligible_for_exact_reproduction": False,
        "owner_prepare_authorized": False,
        "owner_train_authorized": False,
        "prepare_execution_authorized": False,
        "train_execution_authorized": False,
        "training_invoked": False,
        "production_approval": False,
        "production_inference_49054_allowed": False,
        "data_package_manifest_id": None,
        "reference_package_manifest_id": None,
        "original_model_sha256": None,
        "canonical_data_audit_id": None,
        "reference_package_audit_id": None,
        "reference_environment_evidence_sha256": None,
        "current_environment_capture": None,
        "mismatches": [],
        "blocker_codes": [],
        "package_audit_success": False,
        "reference_archive_audit_success": False,
        "canonical_data_audit_status": None,
        "reference_package_audit_status": None,
        "reference_environment_provenance_limitation": None,
        "source_identity": None,
        "strict_preflight_receipt": None,
        "next_authorized_action": "REMEDIATE_PREFLIGHT_CONTRACT_ERROR_AND_RERUN",
        "next_required_decision": "REMEDIATE_PREFLIGHT_CONTRACT_ERROR_AND_RERUN",
    }


def capture_source_identity(config: ProjectConfig) -> dict[str, Any]:
    """Read the commit and cleanliness of the execution worktree."""

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=config.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ContractError(
                "EXACT_EXECUTION_SOURCE_IDENTITY_UNAVAILABLE",
                "unable to capture the source commit/worktree state",
            )
        return completed.stdout.strip()

    return {
        "accepted_source_commit": git("rev-parse", "HEAD"),
        "source_worktree_clean": git("status", "--porcelain") == "",
    }


def _result_from_contract_error(exc: ContractError) -> tuple[dict[str, Any], int]:
    result = _empty_result()
    if exc.code.startswith("REFERENCE_"):
        result["status"] = "BLOCKED_INVALID_REFERENCE_PACKAGE"
    elif exc.code.startswith("CONFIG_"):
        result["status"] = "BLOCKED_EXACT_ENVIRONMENT_PREFLIGHT_INVALID"
    else:
        result["status"] = "BLOCKED_CANONICAL_DATA_AUDIT"
    result["blocker_codes"] = [result["status"], exc.code]
    result["error"] = exc.as_dict()
    return _finalize(result), 3


def _result_from_data_audit(data_audit: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    result = _empty_result()
    result.update(
        {
            "status": "BLOCKED_CANONICAL_DATA_AUDIT",
            "data_package_manifest_id": _mapping(
                data_audit.get("validation_summary")
            ).get("package_manifest_id"),
            "canonical_data_audit_id": data_audit.get("audit_id"),
            "canonical_data_audit_status": data_audit.get("status"),
            "blocker_codes": [
                "BLOCKED_CANONICAL_DATA_AUDIT",
                *_mapping(data_audit).get("blocker_codes", []),
            ],
            "next_authorized_action": "RESTORE_CANONICAL_DATA_PACKAGE_AND_RERUN",
        }
    )
    return _finalize(result), 2


def _validated_result(
    *,
    data_audit: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
    reference_environment: Mapping[str, Any],
    environment_evidence_sha256: str,
    current_environment: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    reference_package = _mapping(reference_audit.get("reference_package"))
    comparison = compare_runtime_to_reference(
        current_environment, reference_environment, strict=True
    )
    matches_reference = bool(comparison["matches_reference"])
    result = _empty_result()
    result.update(
        {
            "status": (
                "EXACT_REFERENCE_ENVIRONMENT_READY"
                if matches_reference
                else _RUNTIME_MISMATCH_STATUS
            ),
            "exact_environment_ready": matches_reference,
            # Environment eligibility and owner authorization are deliberately
            # separate. A preflight can never grant either write permission.
            "exact_reproduction_authorized": False,
            "environment_policy_eligible_for_exact_reproduction": matches_reference,
            "data_package_manifest_id": _mapping(
                data_audit.get("validation_summary")
            ).get("package_manifest_id"),
            "reference_package_manifest_id": reference_package.get(
                "package_manifest_id"
            ),
            "original_model_sha256": reference_package.get("original_model_sha256"),
            "canonical_data_audit_id": data_audit.get("audit_id"),
            "reference_package_audit_id": reference_audit.get("reference_audit_id"),
            "reference_environment_evidence_sha256": environment_evidence_sha256,
            "current_environment_capture": dict(current_environment),
            "mismatches": comparison["mismatches"],
            "blocker_codes": (
                [] if matches_reference else [_RUNTIME_MISMATCH_STATUS]
            ),
            "package_audit_success": True,
            "reference_archive_audit_success": True,
            "canonical_data_audit_status": data_audit.get("status"),
            "reference_package_audit_status": reference_audit.get("status"),
            "reference_environment_provenance_limitation": reference_environment.get(
                "capture_scope"
            ),
            "source_identity": dict(source_identity),
            "next_authorized_action": (
                "AWAIT_OWNER_AUTHORIZATION_TO_RUN_PREPARE"
                if matches_reference
                else "PROVISION_THE_FROZEN_LINUX_X86_64_AMD_EPYC_REFERENCE_ENVIRONMENT_AND_RERUN_PREFLIGHT"
            ),
            "next_required_decision": (
                "OWNER_PREPARE_AUTHORIZATION_RECEIPT_REQUIRED"
                if matches_reference
                else "PROVISION_THE_FROZEN_LINUX_X86_64_AMD_EPYC_REFERENCE_ENVIRONMENT_AND_RERUN_PREFLIGHT"
            ),
        }
    )
    finalized = _finalize(result)
    if matches_reference and source_identity.get("source_worktree_clean") is True:
        finalized["strict_preflight_receipt"] = build_strict_preflight_receipt(
            finalized
        )
    elif matches_reference:
        finalized["next_required_decision"] = "CLEAN_SOURCE_WORKTREE_AND_RERUN_PREFLIGHT"
    return finalized, 0 if matches_reference else 2


def run_exact_environment_audit(
    config_path: str | Path, reference_archive: str | Path
) -> tuple[dict[str, Any], int]:
    """Run the six read-only gates and authorize exact execution only on a match."""

    try:
        config = ProjectConfig.load(config_path)

        # Gate order is intentional: data must pass before reference evidence can
        # be considered and neither audit creates a run or loads the joblib.
        data_audit = audit_config(config)
        if not data_audit.get("training_allowed"):
            return _result_from_data_audit(data_audit)

        archive = audit_reference_archive(config, reference_archive)
        reference_audit = build_reference_audit_result(archive, data_audit)
        reference_environment = load_reference_environment(config)
        environment_evidence_sha256 = reference_environment_evidence_sha256(config)
        current_environment = capture_runtime_environment()
        source_identity = capture_source_identity(config)
        return _validated_result(
            data_audit=data_audit,
            reference_audit=reference_audit,
            reference_environment=reference_environment,
            environment_evidence_sha256=environment_evidence_sha256,
            current_environment=current_environment,
            source_identity=source_identity,
        )
    except ContractError as exc:
        return _result_from_contract_error(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only, fail-closed exact reference-environment preflight"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-archive", required=True)
    args = parser.parse_args(argv)
    result, exit_code = run_exact_environment_audit(
        args.config, args.reference_archive
    )
    sys.stdout.write(f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
