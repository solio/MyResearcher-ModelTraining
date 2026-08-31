"""Fail-closed M2-S1 frozen shared-seven-head runner.

This module deliberately contains no top-level Torch, Transformers, Hub, or
cache imports.  M2 remains unauthorized in the frozen experiment contract;
only a separately supplied, content-addressed owner receipt can enter the
preflight path.  A missing or invalid receipt therefore cannot load a model,
inspect a cache, create an output directory, or invoke an optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from . import encoder_m1 as m1
from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL"
FROZEN_CONTRACT_COMMIT = "df12078b90f21c5942f838fb2175b636bc20a5db"
RECEIPT_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-owner-authorization-receipt.v2"
OWNER_DECISION_RECORD_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-owner-decision-record.v1"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-control-run.v1"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")
OWNER_RECEIPT_RELATIVE_PATH = Path("manifests/owner-decisions/m2-s1-owner-authorization-receipt.json")
OWNER_RECEIPT_SCHEMA_RELATIVE_PATH = Path("manifests/owner-decisions/m2-s1-owner-authorization-receipt-schema.json")
OWNER_DECISION_RECORD_RELATIVE_PATH = Path("manifests/owner-decisions/m2-s1-owner-decision-record.json")
OWNER_DECISION_RECORD_SCHEMA_RELATIVE_PATH = Path("manifests/owner-decisions/m2-s1-owner-decision-record-schema.json")
SEEDS = (35, 71, 107)
MAX_WALL_TIME_SECONDS = 120 * 60
MAX_NEW_DISK_GIB = 10

# The M1 implementation remains the single source of truth for input
# construction, immutable Train/Dev loading, per-sample/head weights, metrics,
# and the CPU reload smoke test.  These paths add the M2 entry point and retain
# the existing M1 input/data/schema dependencies in evidence provenance.
CRITICAL_SOURCE_PATHS = (
    "src/semantic_model/encoder_m2_s1.py",
    *m1.CRITICAL_SOURCE_PATHS,
)


def _require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        raise ContractError(code, message, **details)


def _mapping(value: Any, code: str, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object")
    return value


def _fixed_sha256(value: Any, code: str, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        code,
        f"{name} must be a lowercase SHA-256 value",
        observed=value,
    )
    return str(value)


def _read_json(path: str | Path, *, missing_code: str, invalid_code: str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(missing_code, "required JSON file is missing", path=str(candidate)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(invalid_code, "required JSON file is invalid", path=str(candidate), detail=str(exc)) from exc
    _require(isinstance(value, dict), invalid_code, "JSON root must be an object", path=str(candidate))
    return value


def _parse_expiry(value: Any) -> str:
    _require(isinstance(value, str) and value.endswith("Z"), "M2_S1_OWNER_RECEIPT_INVALID", "expires_at_utc must be an ISO-8601 UTC string")
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("M2_S1_OWNER_RECEIPT_INVALID", "expires_at_utc is not ISO-8601", observed=value) from exc
    _require(expiry.tzinfo is not None, "M2_S1_OWNER_RECEIPT_INVALID", "expires_at_utc must include a UTC offset")
    _require(expiry > datetime.now(UTC), "M2_S1_OWNER_RECEIPT_EXPIRED", "owner authorization receipt has expired", expires_at_utc=value)
    return value


def _contract_requirements(contract_path: str | Path) -> dict[str, Any]:
    """Read only the frozen M2 fields that an S1 receipt is allowed to bind."""

    path = Path(contract_path).resolve()
    contract = _read_json(path, missing_code="M2_S1_CONTRACT_MISSING", invalid_code="M2_S1_CONTRACT_INVALID")
    _require(
        contract.get("manifest_schema_version") == "myresearcher.encoder-m2-experiment-contract.v1",
        "M2_S1_CONTRACT_INVALID",
        "M2 runner requires the frozen M2 experiment contract",
    )
    m2_authorization = _mapping(contract.get("m2_execution_authorization"), "M2_S1_CONTRACT_INVALID", "m2_execution_authorization")
    _require(
        m2_authorization.get("authorization_granted") is False
        and m2_authorization.get("training_allowed") is False,
        "M2_S1_CONTRACT_AUTHORIZATION_STATE_INVALID",
        "the frozen M2 planning contract must remain fail closed; authorization belongs only in a receipt",
    )
    recommended = _mapping(contract.get("recommended_first_execution"), "M2_S1_CONTRACT_INVALID", "recommended_first_execution")
    model = _mapping(recommended.get("recommended_model"), "M2_S1_CONTRACT_INVALID", "recommended_model")
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S1_CONTRACT_INVALID", "frozen_input_and_common_training_configuration")
    resources = _mapping(contract.get("resource_and_stop_contract"), "M2_S1_CONTRACT_INVALID", "resource_and_stop_contract")
    hard_stops = _mapping(resources.get("proposed_hard_stops_requiring_owner_approval"), "M2_S1_CONTRACT_INVALID", "proposed_hard_stops_requiring_owner_approval")
    data_roles = _mapping(contract.get("data_role_and_seal"), "M2_S1_CONTRACT_INVALID", "data_role_and_seal")
    stages = contract.get("minimal_experiment_gradient")
    _require(isinstance(stages, list), "M2_S1_CONTRACT_INVALID", "minimal_experiment_gradient must be a list")
    stage = next((item for item in stages if isinstance(item, Mapping) and item.get("stage_id") == STAGE_ID), None)
    _require(stage is not None, "M2_S1_STAGE_MISSING", "frozen contract does not define M2-S1")
    _require(
        stage.get("encoder_state") == "FROZEN"
        and stage.get("architecture") == "ONE_SHARED_ENCODER_WITH_SEVEN_HEADS"
        and stage.get("seeds") == list(SEEDS),
        "M2_S1_STAGE_CONTRACT_INVALID",
        "M2-S1 must remain the frozen shared seven-head three-seed control",
    )
    _require(
        model.get("model_id") == m1.MODEL_ID
        and model.get("revision") == m1.REVISION
        and model.get("license") == m1.LICENSE
        and model.get("trust_remote_code") is False
        and model.get("new_download_allowed_by_this_contract") is False,
        "M2_S1_MODEL_CONTRACT_INVALID",
        "M2-S1 must use the one fixed local RBT3 model without download",
    )
    _require(
        common.get("max_length") == 256
        and common.get("batch_size") == 16
        and common.get("truncation") == "HEAD_TAIL"
        and common.get("token_type_ids") == "NOT_EMITTED"
        and common.get("per_sample_per_head_weights_required") is True,
        "M2_S1_TRAINING_CONTRACT_INVALID",
        "M2-S1 common training configuration differs from the frozen contract",
    )
    early_stopping = _mapping(common.get("early_stopping"), "M2_S1_CONTRACT_INVALID", "early_stopping")
    optimizer = _mapping(common.get("optimizer"), "M2_S1_CONTRACT_INVALID", "optimizer")
    gradient = _mapping(common.get("gradient_controls"), "M2_S1_CONTRACT_INVALID", "gradient_controls")
    _require(
        early_stopping.get("max_epochs") == 12
        and early_stopping.get("patience_epochs") == 3
        and optimizer.get("name") == "AdamW"
        and optimizer.get("head_learning_rate") == 0.0005
        and optimizer.get("weight_decay") == 0.01
        and gradient.get("gradient_clipping_max_norm") == 1.0,
        "M2_S1_OPTIMIZATION_CONTRACT_INVALID",
        "M2-S1 optimization controls differ from the frozen contract",
    )
    train = _mapping(data_roles.get("train"), "M2_S1_CONTRACT_INVALID", "train")
    dev = _mapping(data_roles.get("dev"), "M2_S1_CONTRACT_INVALID", "dev")
    _require(train.get("rows") == 1822 and dev.get("rows") == 448, "M2_S1_DATA_CONTRACT_INVALID", "M2-S1 requires Train 1822 and Dev 448")
    pre_training = _mapping(contract.get("m2_s1_pre_training_execution_contract"), "M2_S1_CONTRACT_INVALID", "m2_s1_pre_training_execution_contract")
    paths = _mapping(pre_training.get("required_tracked_owner_files"), "M2_S1_CONTRACT_INVALID", "required_tracked_owner_files")
    expected_paths = {
        "receipt_relative_path": OWNER_RECEIPT_RELATIVE_PATH.as_posix(),
        "receipt_schema_relative_path": OWNER_RECEIPT_SCHEMA_RELATIVE_PATH.as_posix(),
        "owner_decision_record_relative_path": OWNER_DECISION_RECORD_RELATIVE_PATH.as_posix(),
        "owner_decision_record_schema_relative_path": OWNER_DECISION_RECORD_SCHEMA_RELATIVE_PATH.as_posix(),
    }
    _require(paths == expected_paths, "M2_S1_OWNER_PATH_CONTRACT_INVALID", "M2-S1 owner files must remain at their fixed tracked paths")
    repository_gate = _mapping(pre_training.get("d026_repository_consolidation_gate"), "M2_S1_CONTRACT_INVALID", "d026_repository_consolidation_gate")
    _require(
        pre_training.get("expected_unified_training_branch") == "feat/m2-s1-runner"
        and repository_gate.get("failure_status") == "BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED",
        "M2_S1_REPOSITORY_GATE_CONTRACT_INVALID",
        "M2-S1 requires the D-026 repository consolidation gate",
    )
    snapshot = _mapping(pre_training.get("fixed_local_cache_snapshot"), "M2_S1_CONTRACT_INVALID", "fixed_local_cache_snapshot")
    snapshot_files = snapshot.get("files")
    _require(isinstance(snapshot_files, list) and len(snapshot_files) == 8, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "M2-S1 must freeze all eight accepted M1 snapshot files")
    for row in snapshot_files:
        item = _mapping(row, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "snapshot file")
        _require(isinstance(item.get("path"), str) and isinstance(item.get("bytes"), int) and item["bytes"] >= 0, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "snapshot file needs path and byte count")
        _fixed_sha256(item.get("sha256"), "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "snapshot file sha256")
    runtime_identity = _mapping(pre_training.get("frozen_runtime_identity_from_accepted_m1_environment"), "M2_S1_CONTRACT_INVALID", "frozen_runtime_identity_from_accepted_m1_environment")
    m1_controls = _mapping(pre_training.get("accepted_m1_control_evidence"), "M2_S1_CONTRACT_INVALID", "accepted_m1_control_evidence")
    _require(m1_controls.get("artifact_content_address") == "b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58", "M2_S1_M1_CONTROL_CONTRACT_INVALID", "M2-S1 must retain the accepted M1 control identity")
    return {
        "contract": contract,
        "contract_path": path,
        "contract_sha256": sha256_file(path),
        "model": {
            "model_id": m1.MODEL_ID,
            "revision": m1.REVISION,
            "license": m1.LICENSE,
            "model_weight_sha256": _fixed_sha256(model.get("model_weight_sha256"), "M2_S1_CONTRACT_INVALID", "model_weight_sha256"),
            "tokenizer_json_sha256": _fixed_sha256(model.get("tokenizer_json_sha256"), "M2_S1_CONTRACT_INVALID", "tokenizer_json_sha256"),
            "vocab_txt_sha256": _fixed_sha256(model.get("vocab_txt_sha256"), "M2_S1_CONTRACT_INVALID", "vocab_txt_sha256"),
        },
        "train_rows": 1822,
        "dev_rows": 448,
        "per_run_wall_time_minutes": hard_stops.get("per_run_wall_time_minutes"),
        "total_new_local_disk_gib": hard_stops.get("total_new_local_disk_gib"),
        "pre_training": pre_training,
        "repository_gate": repository_gate,
        "snapshot": {"required_relative_directory": snapshot.get("required_relative_directory"), "files": snapshot_files, "content_address": content_addressed_id({"files": snapshot_files})},
        "runtime_identity": runtime_identity,
        "m1_controls": m1_controls,
    }


def _git_output(worktree: Path, arguments: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(worktree), *arguments], check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise ContractError("M2_S1_GIT_PROVENANCE_UNAVAILABLE", "unable to inspect required Git provenance", arguments=list(arguments), stderr=result.stderr.strip())
    return result.stdout.strip()


def _require_tracked_at_head(worktree: Path, relative: Path, code: str) -> Path:
    _require(not relative.is_absolute() and ".." not in relative.parts, code, "owner path must be repository-relative")
    candidate = worktree / relative
    _require(candidate.is_file(), code, "required tracked owner file is missing", path=relative.as_posix())
    _git_output(worktree, ["ls-files", "--error-unmatch", "--", relative.as_posix()])
    _git_output(worktree, ["cat-file", "-e", f"HEAD:{relative.as_posix()}"])
    return candidate


def _validate_schema(value: Mapping[str, Any], schema_path: Path, code: str) -> None:
    schema = _read_json(schema_path, missing_code=code, invalid_code=code)
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value), key=lambda error: list(error.absolute_path))
    _require(not errors, code, "JSON does not satisfy its strict tracked schema", detail=errors[0].message if errors else None)


def validate_owner_authorization_receipt(
    *,
    worktree: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Validate only the tracked receipt plus independent tracked allowlist.

    A self-hashed file at an arbitrary path is deliberately not an owner
    authorization.  Both the receipt and the separately content-addressed
    decision record must be tracked at HEAD and pass their strict schemas.
    """

    root = Path(worktree).resolve()
    receipt_path = _require_tracked_at_head(root, OWNER_RECEIPT_RELATIVE_PATH, "M2_S1_OWNER_RECEIPT_PATH_INVALID")
    receipt_schema_path = _require_tracked_at_head(root, OWNER_RECEIPT_SCHEMA_RELATIVE_PATH, "M2_S1_OWNER_RECEIPT_SCHEMA_MISSING")
    record_path = _require_tracked_at_head(root, OWNER_DECISION_RECORD_RELATIVE_PATH, "M2_S1_OWNER_DECISION_RECORD_PATH_INVALID")
    record_schema_path = _require_tracked_at_head(root, OWNER_DECISION_RECORD_SCHEMA_RELATIVE_PATH, "M2_S1_OWNER_DECISION_RECORD_SCHEMA_MISSING")
    receipt = _read_json(receipt_path, missing_code="M2_S1_OWNER_RECEIPT_MISSING", invalid_code="M2_S1_OWNER_RECEIPT_INVALID")
    record = _read_json(record_path, missing_code="M2_S1_OWNER_DECISION_RECORD_MISSING", invalid_code="M2_S1_OWNER_DECISION_RECORD_INVALID")
    _validate_schema(receipt, receipt_schema_path, "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID")
    _validate_schema(record, record_schema_path, "M2_S1_OWNER_DECISION_RECORD_SCHEMA_INVALID")
    _require(record.get("record_schema_version") == OWNER_DECISION_RECORD_SCHEMA_VERSION, "M2_S1_OWNER_DECISION_RECORD_INVALID", "owner decision record schema version is unsupported")
    _require(
        record.get("record_content_address") == content_addressed_id(record, omit_keys={"record_content_address"}),
        "M2_S1_OWNER_DECISION_RECORD_CONTENT_ADDRESS_MISMATCH",
        "owner decision record content address does not match its payload",
    )
    _require(receipt.get("authorization_granted") is True, "M2_S1_OWNER_AUTHORIZATION_NOT_GRANTED", "M2-S1 fit requires an explicit granted owner receipt")
    _require(record.get("authorization_granted") is True, "M2_S1_OWNER_DECISION_NOT_AUTHORIZED", "independent tracked owner decision record remains ungranted")
    _require(receipt.get("owner_decision_id") == record.get("owner_decision_id"), "M2_S1_OWNER_DECISION_ID_MISMATCH", "receipt must bind the allowlisting owner decision record")
    observed_address = receipt.get("receipt_content_address")
    expected_address = content_addressed_id(receipt, omit_keys={"receipt_content_address"})
    _require(observed_address == expected_address, "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH", "receipt content address does not match the receipt payload", observed=observed_address, expected=expected_address)
    allowlist = record.get("authorized_receipt_content_addresses")
    _require(isinstance(allowlist, list) and observed_address in allowlist, "M2_S1_OWNER_RECEIPT_NOT_ALLOWLISTED", "self-hashed receipt is not allowlisted by the tracked owner decision record")
    _parse_expiry(receipt.get("expires_at_utc"))

    frozen = _contract_requirements(contract_path)
    _require(receipt.get("frozen_contract_commit") == FROZEN_CONTRACT_COMMIT, "M2_S1_FROZEN_CONTRACT_COMMIT_MISMATCH", "receipt must bind the frozen M2 contract commit")
    _require(
        receipt.get("contract_sha256") == frozen["contract_sha256"],
        "M2_S1_CONTRACT_SHA256_MISMATCH",
        "receipt does not bind the exact frozen M2 contract bytes",
    )
    _require(receipt.get("stage_id") == STAGE_ID, "M2_S1_STAGE_MISMATCH", "receipt does not authorize M2-S1")
    repository = _mapping(receipt.get("unified_repository"), "M2_S1_OWNER_RECEIPT_INVALID", "unified_repository")
    _require(
        repository.get("branch") == frozen["pre_training"]["expected_unified_training_branch"]
        and repository.get("remote_branch") == f"origin/{repository.get('branch')}"
        and isinstance(repository.get("commit"), str)
        and len(repository["commit"]) == 40
        and all(character in "0123456789abcdef" for character in repository["commit"]),
        "M2_S1_UNIFIED_REPOSITORY_IDENTITY_MISMATCH",
        "receipt must bind the single expected branch, remote branch, and exact commit",
    )
    _require(receipt.get("canonical_config_sha256") == frozen["pre_training"]["receipt_rules"]["receipt_must_bind_canonical_config_sha256"], "M2_S1_CONFIG_SHA256_MISMATCH", "receipt does not bind the frozen canonical config identity")
    _require(receipt.get("cache_snapshot_content_address") == frozen["snapshot"]["content_address"], "M2_S1_CACHE_SNAPSHOT_IDENTITY_MISMATCH", "receipt does not bind the complete eight-file cache snapshot")
    _require(receipt.get("runtime_environment_sha256") == frozen["runtime_identity"]["environment_sha256"], "M2_S1_RUNTIME_IDENTITY_MISMATCH", "receipt does not bind the frozen runtime environment identity")
    model = _mapping(receipt.get("model"), "M2_S1_OWNER_RECEIPT_INVALID", "model")
    for key, code in (
        ("model_id", "M2_S1_MODEL_ID_MISMATCH"),
        ("revision", "M2_S1_MODEL_REVISION_MISMATCH"),
        ("license", "M2_S1_MODEL_LICENSE_MISMATCH"),
        ("model_weight_sha256", "M2_S1_MODEL_HASH_MISMATCH"),
        ("tokenizer_json_sha256", "M2_S1_MODEL_HASH_MISMATCH"),
        ("vocab_txt_sha256", "M2_S1_MODEL_HASH_MISMATCH"),
    ):
        _require(model.get(key) == frozen["model"][key], code, "receipt model identity differs from the frozen M2 contract", field=key)
    cache_policy = _mapping(receipt.get("cache_policy"), "M2_S1_OWNER_RECEIPT_INVALID", "cache_policy")
    _require(
        cache_policy.get("local_files_only") is True and cache_policy.get("no_download") is True,
        "M2_S1_CACHE_POLICY_MISMATCH",
        "M2-S1 receipt must require fixed local files only and prohibit downloads",
    )
    execution = _mapping(receipt.get("execution"), "M2_S1_OWNER_RECEIPT_INVALID", "execution")
    _require(execution.get("seeds") == list(SEEDS), "M2_S1_SEEDS_MISMATCH", "receipt must authorize exactly seeds 35, 71, and 107")
    _require(
        execution.get("train_rows") == frozen["train_rows"]
        and execution.get("dev_rows") == frozen["dev_rows"]
        and execution.get("train_role") == "TRAIN_ONLY_FIT"
        and execution.get("dev_role") == "EARLY_STOPPING_AND_DIAGNOSTIC_ONLY",
        "M2_S1_DATA_ROLE_MISMATCH",
        "receipt must bind Train 1822 fit and Dev 448 diagnostics only",
    )
    _require(
        execution.get("device_policy") == "MPS_FIRST_CPU_FALLBACK",
        "M2_S1_DEVICE_POLICY_MISMATCH",
        "receipt must bind MPS-first with CPU fallback",
    )
    _require(
        execution.get("per_run_wall_time_minutes") == frozen["per_run_wall_time_minutes"]
        and execution.get("total_new_local_disk_gib") == frozen["total_new_local_disk_gib"],
        "M2_S1_RESOURCE_LIMIT_MISMATCH",
        "receipt resource limits differ from the frozen contract",
    )
    prohibitions = _mapping(execution.get("prohibitions"), "M2_S1_OWNER_RECEIPT_INVALID", "execution.prohibitions")
    required_prohibitions = {
        "test": False,
        "anchor": False,
        "gold": False,
        "ood": False,
        "llm": False,
        "cloud_or_external_api": False,
        "production": False,
        "model_download": False,
        "dependency_install": False,
        "full_unfreeze": False,
        "s2_or_s3": False,
    }
    _require(
        all(prohibitions.get(key) is expected for key, expected in required_prohibitions.items()),
        "M2_S1_PROHIBITION_MISMATCH",
        "receipt must explicitly keep all out-of-scope data and execution paths prohibited",
    )
    return {"receipt": receipt, "owner_decision_record": record, "frozen_contract": frozen, "owner_receipt_path": OWNER_RECEIPT_RELATIVE_PATH.as_posix(), "owner_decision_record_path": OWNER_DECISION_RECORD_RELATIVE_PATH.as_posix()}


def _blocked(code: str, message: str, *, phase: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "M2_S1_BLOCKED_FAIL_CLOSED",
        "phase": phase,
        "blocker_codes": [code],
        "message": message,
        "details": details,
        "training_invoked": False,
        "model_loaded": False,
        "cache_accessed": False,
        "output_created": False,
        "aggregate_created": False,
        "selected_candidate": False,
    }


def _git_has_ancestor(worktree: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(worktree), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        result.returncode == 0,
        "M2_S1_FROZEN_CONTRACT_COMMIT_NOT_IN_HEAD",
        "current training source must descend from the frozen M2 contract commit",
        frozen_contract_commit=commit,
    )


def validate_repository_consolidation(worktree: str | Path, receipt_authorization: Mapping[str, Any]) -> dict[str, Any]:
    """Apply D-026 before canonical audit, cache, runtime, or output activity."""

    root = Path(worktree).resolve()
    try:
        receipt = _mapping(receipt_authorization.get("receipt"), "M2_S1_OWNER_RECEIPT_INVALID", "receipt")
        repository = _mapping(receipt.get("unified_repository"), "M2_S1_OWNER_RECEIPT_INVALID", "unified_repository")
        expected_branch = str(receipt_authorization["frozen_contract"]["pre_training"]["expected_unified_training_branch"])
        expected_remote = f"origin/{expected_branch}"
        branch = _git_output(root, ["branch", "--show-current"])
        head = _git_output(root, ["rev-parse", "--verify", "HEAD"])
        local_branches = [item for item in _git_output(root, ["for-each-ref", "--format=%(refname:short)", "refs/heads"]).splitlines() if item]
        remote_branches = [item for item in _git_output(root, ["for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"]).splitlines() if item]
        worktree_rows = [item.removeprefix("worktree ") for item in _git_output(root, ["worktree", "list", "--porcelain"]).splitlines() if item.startswith("worktree ")]
        status = _git_output(root, ["status", "--porcelain=v1", "--untracked-files=all"])
        upstream = _git_output(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        counts = _git_output(root, ["rev-list", "--left-right", "--count", "@{u}...HEAD"]).split()
        _require(branch == expected_branch == repository.get("branch"), "M2_S1_REPOSITORY_BRANCH_MISMATCH", "current branch is not the owner-declared unified branch")
        _require(head == repository.get("commit"), "M2_S1_REPOSITORY_HEAD_MISMATCH", "current HEAD does not equal the receipt's unified commit")
        _require(local_branches == [expected_branch], "M2_S1_REPOSITORY_LOCAL_BRANCHES_NOT_CONSOLIDATED", "D-026 requires exactly one local unified branch", local_branches=local_branches)
        _require(remote_branches == [expected_remote], "M2_S1_REPOSITORY_REMOTE_BRANCHES_NOT_CONSOLIDATED", "D-026 requires exactly one matching origin branch", remote_branches=remote_branches)
        _require(worktree_rows == [str(root)], "M2_S1_REPOSITORY_WORKTREES_NOT_CONSOLIDATED", "D-026 requires exactly one primary worktree", worktrees=worktree_rows)
        _require(status == "", "M2_S1_REPOSITORY_WORKTREE_NOT_CLEAN", "D-026 requires a clean worktree", status_entries=status.splitlines())
        _require(upstream == expected_remote and len(counts) == 2 and counts == ["0", "0"], "M2_S1_REPOSITORY_UPSTREAM_NOT_SYNCED", "D-026 requires exactly matching upstream with ahead/behind 0/0", upstream=upstream, counts=counts)
        return {
            "status": "D026_REPOSITORY_CONSOLIDATED",
            "branch": branch,
            "head": head,
            "upstream": upstream,
            "upstream_behind": 0,
            "upstream_ahead": 0,
            "local_branch_count": len(local_branches),
            "matching_remote_branch_count": len(remote_branches),
            "worktree_count": len(worktree_rows),
            "worktree": str(root),
        }
    except ContractError as exc:
        raise ContractError(
            "BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED",
            "D-026 repository consolidation gate did not pass",
            cause=exc.code,
            **exc.details,
        ) from exc


def validate_fixed_cache_snapshot(cache_dir: Path, frozen: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Hash all eight accepted M1 snapshot files before a model is imported."""

    relative = Path(str(frozen["snapshot"]["required_relative_directory"]))
    _require(not relative.is_absolute() and ".." not in relative.parts, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "cache snapshot directory must remain relative")
    snapshot = (cache_dir / relative).resolve()
    _require(snapshot.is_dir() and snapshot.name == m1.REVISION, "M2_S1_FIXED_REVISION_CACHE_MISSING", "M2-S1 requires the fixed local M1 snapshot directory", snapshot=str(snapshot))
    rows: list[dict[str, Any]] = []
    for expected in frozen["snapshot"]["files"]:
        relative_file = Path(str(expected["path"]))
        _require(not relative_file.is_absolute() and ".." not in relative_file.parts, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "snapshot file path must be relative")
        candidate = snapshot / relative_file
        _require(candidate.is_file(), "M2_S1_CACHE_SNAPSHOT_FILE_MISSING", "required fixed-cache snapshot file is missing", path=relative_file.as_posix())
        observed = {"path": relative_file.as_posix(), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
        _require(observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"], "M2_S1_CACHE_SNAPSHOT_FILE_MISMATCH", "fixed-cache snapshot file bytes or SHA-256 differ from accepted M1 evidence", path=relative_file.as_posix(), observed=observed, expected=expected)
        rows.append(observed)
    identity = {"required_relative_directory": relative.as_posix(), "files": rows, "content_address": content_addressed_id({"files": rows})}
    _require(identity["content_address"] == frozen["snapshot"]["content_address"], "M2_S1_CACHE_SNAPSHOT_IDENTITY_MISMATCH", "full fixed-cache snapshot content identity differs from the receipt/contract")
    return snapshot, identity


def observe_runtime_identity(runtime: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
    """Observe package versions after runtime import but before any model load."""

    np, torch, _AutoModel, _AutoTokenizer = runtime
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "tokenizers", "numpy", "huggingface-hub", "safetensors"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "packages": packages,
        "torch_runtime_version": getattr(torch, "__version__", None),
        "numpy_runtime_version": getattr(np, "__version__", None),
    }


def validate_runtime_identity(runtime: tuple[Any, Any, Any, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    observed = observe_runtime_identity(runtime)
    expected = frozen["runtime_identity"]
    _require(observed["python"] == expected["python"], "M2_S1_RUNTIME_IDENTITY_MISMATCH", "Python runtime differs from the frozen accepted M1 environment", observed=observed["python"], expected=expected["python"])
    _require(observed["packages"] == expected["packages"], "M2_S1_RUNTIME_IDENTITY_MISMATCH", "runtime package versions differ from the frozen accepted M1 environment", observed=observed["packages"], expected=expected["packages"])
    _require(observed["torch_runtime_version"] == expected["packages"]["torch"] and observed["numpy_runtime_version"] == expected["packages"]["numpy"], "M2_S1_RUNTIME_IDENTITY_MISMATCH", "runtime module versions differ from frozen package versions")
    return {"environment_sha256": expected["environment_sha256"], "observed": observed, "device_policy": expected["device_policy"]}


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_authorized_output_dir(output_dir: str | Path, cache_dir: str | Path, *, worktree: Path, receipt: Mapping[str, Any], frozen: Mapping[str, Any]) -> Path:
    root = Path(output_dir).expanduser().resolve()
    authorized = Path(str(receipt.get("output_dir", ""))).expanduser().resolve()
    _require(root == authorized, "M2_S1_OUTPUT_DIR_NOT_AUTHORIZED", "output_dir must equal the unique directory authorized in the fixed receipt", output_dir=str(root), authorized_output_dir=str(authorized))
    _require(not root.exists(), "M2_S1_OUTPUT_REPLAY_OR_EXISTS", "an owner receipt cannot be replayed into an existing output directory", output_dir=str(root))
    cache = Path(cache_dir).expanduser().resolve()
    _require(not _is_within(root, cache), "M2_S1_OUTPUT_PROTECTED_PATH", "output_dir cannot be inside the fixed model cache", output_dir=str(root))
    for relative in frozen["pre_training"]["protected_output_roots"]:
        protected = (worktree / str(relative)).resolve()
        _require(not _is_within(root, protected), "M2_S1_OUTPUT_PROTECTED_PATH", "output_dir cannot be inside a protected repository path", output_dir=str(root), protected_path=str(protected))
    _require(
        not any(ancestor.name in {".encoder-artifacts", ".encoder-venv"} for ancestor in (root, *root.parents)),
        "M2_S1_OUTPUT_PROTECTED_PATH",
        "output_dir cannot be inside any historical M1 artifact or Encoder runtime directory",
        output_dir=str(root),
    )
    return root


def _enforce_resource_limits(start: float, root: Path, *, phase: str, seed: int | None = None) -> None:
    _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_S1_WALL_TIME_LIMIT_EXCEEDED", "M2-S1 exceeded its 120-minute wall-time limit", phase=phase, seed=seed)
    _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_S1_DISK_LIMIT_EXCEEDED", "M2-S1 outputs exceeded the 10 GiB new-local-disk limit", phase=phase, seed=seed)


def validate_m2_s1_preflight(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    worktree: str | Path,
    receipt_authorization: Mapping[str, Any],
    repository_consolidation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the ordered no-runtime gates after receipt validation succeeds."""

    root = Path(worktree).resolve()
    # Canonical audit validates exit=0, training_allowed=true, blocker_codes=[],
    # data/reference binding, config identity, and schema identity before any
    # cache or model activity.
    canonical = m1.validate_canonical_audit(config_path)
    source = m1.validate_source_provenance(root, critical_sources=CRITICAL_SOURCE_PATHS)
    _require(
        "src/semantic_model/encoder_m2_s1.py" in source["critical_source_sha256"],
        "M2_S1_ENTRYPOINT_NOT_TRACKED_AT_HEAD",
        "current HEAD must contain the real M2-S1 training entry point",
    )
    _git_has_ancestor(root, FROZEN_CONTRACT_COMMIT)
    frozen = receipt_authorization["frozen_contract"]
    _require(
        receipt_authorization["receipt"].get("canonical_config_sha256") == canonical["config_sha256"],
        "M2_S1_CONFIG_SHA256_MISMATCH",
        "receipt canonical config identity differs from the audited config",
    )
    # Re-hash the on-disk frozen contract immediately before cache verification.
    _require(
        sha256_file(frozen["contract_path"]) == frozen["contract_sha256"],
        "M2_S1_CONTRACT_SHA256_MISMATCH",
        "frozen M2 contract changed after receipt validation",
    )
    snapshot, snapshot_identity = validate_fixed_cache_snapshot(Path(cache_dir).resolve(), frozen)
    identity = {
        "git_head": source["git_head"],
        "critical_source_sha256": source["critical_source_sha256"],
        "contract_relative_path": CONTRACT_RELATIVE_PATH.as_posix(),
        "contract_sha256": frozen["contract_sha256"],
        "config_sha256": canonical["config_sha256"],
        "canonical_audit_id": canonical["canonical_audit_id"],
        "data_package_content_id": canonical["data_package_content_id"],
        "reference_package_content_id": canonical["reference_package_content_id"],
        "reference_binding_data_package_content_address": canonical["reference_binding_data_package_content_address"],
        "schema_version": canonical["schema_version"],
        "schema_sha256": canonical["schema_sha256"],
        "owner_receipt_content_address": receipt_authorization["receipt"]["receipt_content_address"],
        "owner_decision_id": receipt_authorization["receipt"]["owner_decision_id"],
        "frozen_contract_commit": FROZEN_CONTRACT_COMMIT,
        "model_id": frozen["model"]["model_id"],
        "revision": frozen["model"]["revision"],
        "license": frozen["model"]["license"],
        "unified_branch": repository_consolidation["branch"],
        "unified_commit": repository_consolidation["head"],
        "owner_decision_record_content_address": receipt_authorization["owner_decision_record"]["record_content_address"],
    }
    return {"identity": identity, "canonical": canonical, "frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": snapshot_identity, "repository_consolidation": dict(repository_consolidation), "owner_receipt": receipt_authorization["receipt"], "owner_decision_record": receipt_authorization["owner_decision_record"]}


def _frozen_training_config(contract: Mapping[str, Any], class_order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S1_CONTRACT_INVALID", "frozen_input_and_common_training_configuration")
    optimizer = _mapping(common.get("optimizer"), "M2_S1_CONTRACT_INVALID", "optimizer")
    stopping = _mapping(common.get("early_stopping"), "M2_S1_CONTRACT_INVALID", "early_stopping")
    return {
        "configuration_schema_version": "myresearcher.encoder-m2-s1-frozen-config.v1",
        "stage_id": STAGE_ID,
        "configuration_frozen_before_fit": True,
        "model_id": m1.MODEL_ID,
        "revision": m1.REVISION,
        "license": m1.LICENSE,
        "trust_remote_code": False,
        "local_files_only": True,
        "input_builder_version": common["input_builder_version"],
        "stock_code_token_cap": common["stock_code_token_cap"],
        "stock_name_token_cap": common["stock_name_token_cap"],
        "max_length": common["max_length"],
        "truncation": common["truncation"],
        "padding": common["padding"],
        "token_type_ids": common["token_type_ids"],
        "batch_size": common["batch_size"],
        "head_dropout": common["head_dropout"],
        "class_order": {head: list(class_order[head]) for head in V1_HEADS},
        "optimizer": {"name": optimizer["name"], "learning_rate": optimizer["head_learning_rate"], "weight_decay": optimizer["weight_decay"], "betas": optimizer["betas"], "epsilon": optimizer["epsilon"]},
        "stopping": {"max_epochs": stopping["max_epochs"], "patience": stopping["patience_epochs"], "minimum_delta": stopping["minimum_delta"], "early_stopping_metric": stopping["metric"], "wall_time_limit_seconds": MAX_WALL_TIME_SECONDS},
        "gradient_clipping_max_norm": common["gradient_controls"]["gradient_clipping_max_norm"],
        "reasoning_probability_threshold": 0.5,
        "encoder_state": "FROZEN",
        "trainable_heads": 7,
        "fit_population": "Train_1822_only",
        "dev_role": "early_stopping_and_diagnostic_only",
        "test_role": "not_loaded_not_used",
    }


def _ensure_finite(torch: Any, value: Any, code: str, message: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), code, message)


def _checkpoint_heads(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in model.heads.state_dict().items()}


def _critical_boundary_report(contract: Mapping[str, Any], dev_metrics: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    metric_contract = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_S1_CONTRACT_INVALID", "dev_metrics_and_no_regression")
    proxies = metric_contract.get("critical_boundary_proxies")
    _require(isinstance(proxies, list) and len(proxies) == 7, "M2_S1_CONTRACT_INVALID", "M2-S1 requires seven frozen critical-boundary definitions")
    report: dict[str, Any] = {}
    for proxy in proxies:
        item = _mapping(proxy, "M2_S1_CONTRACT_INVALID", "critical boundary proxy")
        head = str(item["head"])
        observed_head = _mapping(dev_metrics.get(head), "M2_S1_METRICS_MISSING", f"Dev metrics {head}")
        labels = observed_head.get("per_label") if head == "reasoning_tags" else observed_head.get("per_class")
        labels = _mapping(labels, "M2_S1_METRICS_MISSING", f"critical labels {head}")
        report[head] = {
            label: {
                "support": labels.get(label, {}).get("support") if isinstance(labels.get(label), Mapping) else None,
                "f1": labels.get(label, {}).get("f1") if isinstance(labels.get(label), Mapping) else None,
                "status": "REPORTED_ONLY_S1_CONTROL_NOT_A_SELECTION_OR_PRODUCTION_GATE",
            }
            for label in item["labels"]
        }
    return {"stage_id": STAGE_ID, "seed": seed, "scope": "DEV_WEAK_LABEL_DIAGNOSTIC_ONLY", "selected_candidate": False, "critical_boundaries": report}


def _execute_one_seed(
    *,
    runtime: tuple[Any, Any, Any, Any],
    seed: int,
    root: Path,
    snapshot: Path,
    schema: Any,
    train: Sequence[m1.M1Record],
    dev: Sequence[m1.M1Record],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    m2_contract: Mapping[str, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    """Execute one authorized frozen seed using M1's shared primitives."""

    np, torch, AutoModel, AutoTokenizer = runtime
    seed_root = root / f"seed-{seed}"
    seed_root.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    _enforce_resource_limits(start, root, phase="before_model_load", seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    _enforce_resource_limits(start, root, phase="after_model_load", seed=seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_trainable = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    _require(encoder_trainable == 0 and len(model.heads) == 7, "M2_S1_FREEZE_OR_HEAD_CONTRACT_VIOLATION", "S1 requires a frozen Encoder and seven trainable heads")
    optimizer = torch.optim.AdamW(trainable, lr=float(config["optimizer"]["learning_rate"]), weight_decay=float(config["optimizer"]["weight_decay"]), betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best_score, best_epoch, stale = -1.0, 0, 0
    log_path = seed_root / "training-log.jsonl"
    for epoch in range(1, int(config["stopping"]["max_epochs"]) + 1):
        _enforce_resource_limits(start, root, phase="before_epoch", seed=seed)
        epoch_start = time.monotonic()
        model.train()
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), int(config["batch_size"])):
            _enforce_resource_limits(start, root, phase="during_epoch", seed=seed)
            batch = m1._as_batch(torch, tokenizer, shuffled[offset:offset + int(config["batch_size"])], config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            for head in V1_HEADS:
                _ensure_finite(torch, logits[head], "M2_S1_NONFINITE_LOGITS", "non-finite logits stop S1 fail closed")
            loss = m1._weighted_loss(torch, logits, batch)
            _ensure_finite(torch, loss, "M2_S1_NONFINITE_LOSS", "non-finite loss stops S1 fail closed")
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _ensure_finite(torch, parameter.grad, "M2_S1_NONFINITE_GRADIENT", "non-finite gradient stops S1 fail closed")
            torch.nn.utils.clip_grad_norm_(trainable, float(config["gradient_clipping_max_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _enforce_resource_limits(start, root, phase="after_epoch", seed=seed)
        dev_metrics = m1.diagnostic_metrics(torch, model, tokenizer, dev, config, device)
        _enforce_resource_limits(start, root, phase="after_epoch_dev_metrics", seed=seed)
        score = float(dev_metrics["diagnostic_score"])
        improved = score > best_score + float(config["stopping"]["minimum_delta"])
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "seed": seed, "model_id": m1.MODEL_ID, "revision": m1.REVISION, "frozen_config": config, "provenance": provenance, "heads_state_dict": _checkpoint_heads(model)}, seed_root / "heads-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log_path, {"seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6) if losses else None, "dev_diagnostic_score": score, "dev_macro_f1_by_head": {head: dev_metrics[head]["macro_f1"] for head in V1_HEADS}, "epoch_seconds": round(time.monotonic() - epoch_start, 3), "elapsed_seconds": round(time.monotonic() - start, 3), "improved": improved, "stale_epochs": stale})
        if stale >= int(config["stopping"]["patience"]):
            break
    checkpoint_path = seed_root / "heads-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_S1_CHECKPOINT_MISSING", "S1 seed completed without an immutable checkpoint", seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    best_model.heads.load_state_dict(checkpoint["heads_state_dict"])
    train_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, train, config, device)
    dev_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, dev, config, device)
    _enforce_resource_limits(start, root, phase="after_final_metrics", seed=seed)
    try:
        smoke = m1.cpu_reload_and_inference_smoke(torch, lambda: m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU checkpoint reload/inference smoke failed", seed=seed, cause=exc.code) from exc
    _require(smoke.get("all_logits_finite") is True, "M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU checkpoint reload/inference must have finite logits", seed=seed)
    _enforce_resource_limits(start, root, phase="after_cpu_reload", seed=seed)
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "seed": seed, "best_epoch": best_epoch, "early_stopping_score": best_score, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(seed_root / "seed-metrics.json", metrics)
    critical_boundary = _critical_boundary_report(m2_contract, dev_metrics, seed=seed)
    m1._json_dump(seed_root / "critical-boundary-report.json", critical_boundary)
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - start, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(seed_root), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": platform.python_version(), "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
    m1._json_dump(seed_root / "resource-log.json", resource)
    _enforce_resource_limits(start, root, phase="after_seed_evidence", seed=seed)
    return {"seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "critical_boundary_report_sha256": sha256_file(seed_root / "critical-boundary-report.json"), "output_dir": str(seed_root), "model_loaded": True}


def aggregate_s1_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only a complete exactly-once 35/71/107 S1 control."""

    observed_seeds = [item.get("seed") for item in seed_results]
    _require(observed_seeds == list(SEEDS), "M2_S1_INCOMPLETE_SEEDS", "aggregate requires all three S1 seeds in frozen order", observed_seeds=observed_seeds)
    devices = [
        _mapping(item.get("resource"), "M2_S1_RESOURCE_LOG_MISSING", "seed resource").get("actual_device")
        for item in seed_results
    ]
    _require(
        len(set(devices)) == 1,
        "M2_S1_MIXED_DEVICE",
        "mixed-device seeds cannot generate a normal aggregate or S2 request",
        device_stratified_seed_devices={str(item["seed"]): device for item, device in zip(seed_results, devices, strict=True)},
    )
    heads: dict[str, Any] = {}
    for head in V1_HEADS:
        values: list[float] = []
        for result in seed_results:
            metrics = _mapping(result.get("metrics"), "M2_S1_METRICS_MISSING", "seed metrics")
            dev = _mapping(metrics.get("dev"), "M2_S1_METRICS_MISSING", "seed Dev metrics")
            head_metrics = _mapping(dev.get(head), "M2_S1_METRICS_MISSING", f"{head} metrics")
            value = head_metrics.get("macro_f1")
            _require(isinstance(value, (int, float)) and math.isfinite(float(value)), "M2_S1_METRICS_MISSING", "each seed requires a finite seven-head Dev primary metric", head=head, seed=result.get("seed"))
            values.append(float(value))
        heads[head] = {"per_seed_values": values, "mean": sum(values) / len(values), "sample_standard_deviation": statistics.stdev(values), "minimum_worst_seed": min(values), "maximum": max(values)}
    stability_passed = all(item["sample_standard_deviation"] <= 0.05 for item in heads.values())
    return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "actual_device": devices[0], "all_seeds_complete": True, "per_head_primary_macro_f1": heads, "seed_stability_gate_passed": stability_passed, "allowed_output": "MAY_REQUEST_S2_OWNER_AUTHORIZATION" if stability_passed else "S1_REJECTED_OR_BLOCKED_EVIDENCE", "selected_candidate": False}


def _authorized_failure(
    root: Path | None,
    exc: ContractError,
    *,
    preflight: Mapping[str, Any],
    output_created: bool,
    model_loaded: bool,
    cache_accessed: bool,
    training_invoked: bool,
    phase: str,
) -> dict[str, Any]:
    evidence = {
        "status": "S1_REJECTED_OR_BLOCKED_EVIDENCE",
        "stage_id": STAGE_ID,
        "phase": phase,
        "blocker_codes": [exc.code],
        "details": exc.details,
        "training_invoked": training_invoked,
        "model_loaded": model_loaded,
        "cache_accessed": cache_accessed,
        "output_created": output_created,
        "aggregate_created": False,
        "selected_candidate": False,
    }
    if exc.code == "M2_S1_MIXED_DEVICE":
        evidence["device_stratified_rejected_evidence"] = exc.details.get("device_stratified_seed_devices", {})
    if output_created and root is not None:
        aggregate_path = root / "stage-aggregate.json"
        if aggregate_path.is_file():
            m1._json_dump(
                aggregate_path,
                {
                    "status": "INVALIDATED_NOT_AN_M2_S1_AGGREGATE",
                    "stage_id": STAGE_ID,
                    "blocker_codes": [exc.code],
                    "aggregate_created": False,
                    "selected_candidate": False,
                },
            )
        m1._json_dump(root / "blocked-evidence.json", evidence)
        rejected = m1._write_content_manifest(
            root,
            {
                "manifest_schema_version": "myresearcher.encoder-m2-s1-rejected-artifact-manifest.v1",
                "status": evidence["status"],
                "stage_id": STAGE_ID,
                "selected_candidate": False,
                "failure": evidence,
                "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"],
                "m1_controls": preflight["frozen_contract"]["m1_controls"],
                "provenance": preflight["identity"],
                "repository_consolidation_receipt": preflight["repository_consolidation"],
                "complete_cache_snapshot": preflight["snapshot_identity"],
                "owner_receipt": {"content_address": preflight["owner_receipt"]["receipt_content_address"], "owner_decision_id": preflight["owner_receipt"]["owner_decision_id"], "decision_record_content_address": preflight["owner_decision_record"]["record_content_address"]},
            },
        )
        evidence["rejected_content_address"] = rejected["content_address"]
    return evidence


def _run_authorized_s1(
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    preflight: Mapping[str, Any],
    runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies,
    seed_executor: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run three independent seeds only after all authorization/preflight gates."""

    stage_started = time.monotonic()
    root: Path | None = None
    results: list[Mapping[str, Any]] = []
    runtime_identity: dict[str, Any] | None = None
    seed_execution_entered = False
    try:
        # This is intentionally the first dynamic-runtime import in the module.
        runtime = runtime_loader()
        runtime_identity = validate_runtime_identity(runtime, preflight["frozen_contract"])
        root = validate_authorized_output_dir(
            output_dir,
            cache_dir,
            worktree=Path(preflight["repository_consolidation"].get("worktree", Path(__file__).resolve().parents[2])).resolve(),
            receipt=preflight["owner_receipt"],
            frozen=preflight["frozen_contract"],
        )
        root.mkdir(parents=True, exist_ok=False)
        _enforce_resource_limits(stage_started, root, phase="before_data_load")
        config = ProjectConfig.load(config_path)
        schema, train, dev = m1.load_m1_partitions(config)
        _require(len(train) == 1822 and len(dev) == 448, "M2_S1_DATA_CONTRACT_INVALID", "M2-S1 loader must expose exactly Train 1822 and Dev 448")
        frozen_config = _frozen_training_config(preflight["frozen_contract"]["contract"], schema.class_order)
        m1._json_dump(root / "training-config.json", frozen_config)
        m1._json_dump(root / "class-order.json", {"schema_version": schema.schema_version, "class_order": frozen_config["class_order"]})
        execute = seed_executor or _execute_one_seed
        for seed in SEEDS:
            _enforce_resource_limits(stage_started, root, phase="before_seed", seed=seed)
            # A failure before a seed returns no result can still occur after
            # the runtime/model path has been entered.  Record that boundary
            # conservatively rather than make a false no-load/no-fit claim.
            seed_execution_entered = True
            result = execute(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=schema, train=train, dev=dev, config=frozen_config, provenance=preflight["identity"], m2_contract=preflight["frozen_contract"]["contract"], cache_dir=Path(cache_dir).resolve())
            _require(result.get("seed") == seed, "M2_S1_SEED_EXECUTOR_IDENTITY_MISMATCH", "seed executor returned an unexpected seed", expected=seed, observed=result.get("seed"))
            results.append(result)
        _enforce_resource_limits(stage_started, root, phase="before_stage_aggregate")
        aggregate = aggregate_s1_seed_results(results)
        m1._json_dump(root / "stage-aggregate.json", aggregate)
        _enforce_resource_limits(stage_started, root, phase="before_final_manifest")
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s1-artifact-manifest.v2", "stage_id": STAGE_ID, "diagnostic_only": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"], "m1_controls": preflight["frozen_contract"]["m1_controls"], "unified_branch_and_commit": {"branch": preflight["repository_consolidation"]["branch"], "commit": preflight["repository_consolidation"]["head"]}, "repository_consolidation_receipt": preflight["repository_consolidation"], "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_identity": {"actual_device": aggregate["actual_device"], "policy": "MPS_FIRST_CPU_FALLBACK"}, "owner_receipt": {"content_address": preflight["owner_receipt"]["receipt_content_address"], "owner_decision_id": preflight["owner_receipt"]["owner_decision_id"], "decision_record_content_address": preflight["owner_decision_record"]["record_content_address"]}, "critical_boundary_report": {str(item["seed"]): item["critical_boundary_report_sha256"] for item in results}, "model_id": m1.MODEL_ID, "resolved_revision": m1.REVISION, "license": m1.LICENSE, "provenance": preflight["identity"], "training_config_sha256": sha256_file(root / "training-config.json"), "class_order_sha256": sha256_file(root / "class-order.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "seed_checkpoints": {str(item["seed"]): item["checkpoint_sha256"] for item in results}, "per_seed_metrics_and_resources": {str(item["seed"]): {"seed_metrics_sha256": sha256_file(root / f"seed-{item['seed']}" / "seed-metrics.json"), "resource_log_sha256": sha256_file(root / f"seed-{item['seed']}" / "resource-log.json")} for item in results}})
        _enforce_resource_limits(stage_started, root, phase="after_final_manifest")
        return {"status": "M2_S1_CONTROL_COMPLETED", "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate}
    except ContractError as exc:
        return _authorized_failure(root, exc, preflight=preflight, output_created=root is not None and root.exists(), model_loaded=seed_execution_entered, cache_accessed=True, training_invoked=seed_execution_entered, phase="AUTHORIZED_EXECUTION")
    except Exception as exc:
        failure = ContractError("M2_S1_RUNTIME_EXCEPTION", "runtime or OOM exception stopped M2-S1 fail closed", exception_type=type(exc).__name__, detail=str(exc))
        return _authorized_failure(root, failure, preflight=preflight, output_created=root is not None and root.exists(), model_loaded=seed_execution_entered, cache_accessed=True, training_invoked=seed_execution_entered, phase="AUTHORIZED_EXECUTION")


def run_m2_s1(
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    worktree: str | Path | None = None,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """M2-S1 entry point.  Receipt validation is the first operation by design."""

    root = Path(worktree).resolve() if worktree is not None else Path(__file__).resolve().parents[2]
    frozen_contract_path = Path(contract_path).resolve() if contract_path is not None else root / CONTRACT_RELATIVE_PATH
    try:
        # Do not move this gate: no audit/data/cache/output/runtime work precedes it.
        receipt_authorization = validate_owner_authorization_receipt(worktree=root, contract_path=frozen_contract_path)
    except ContractError as exc:
        return _blocked(exc.code, exc.message, phase="OWNER_RECEIPT", **exc.details)
    try:
        repository_consolidation = validate_repository_consolidation(root, receipt_authorization)
    except ContractError as exc:
        return _blocked(exc.code, exc.message, phase="D026_REPOSITORY_CONSOLIDATION", **exc.details)
    try:
        preflight = validate_m2_s1_preflight(config_path, cache_dir, worktree=root, receipt_authorization=receipt_authorization, repository_consolidation=repository_consolidation)
    except ContractError as exc:
        return _blocked(exc.code, exc.message, phase="CANONICAL_OR_PROVENANCE_PREFLIGHT", **exc.details)
    return _run_authorized_s1(config_path, output_dir, cache_dir, preflight=preflight)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an independently owner-authorized M2-S1 frozen three-seed control")
    parser.add_argument("--config", required=True, help="Immutable project config; only Train/Dev M2 paths are read")
    parser.add_argument("--output-dir", required=True, help="New immutable M2-S1 output directory")
    parser.add_argument("--cache-dir", required=True, help="Existing local fixed-revision cache; no download is permitted")
    args = parser.parse_args(argv)
    result = run_m2_s1(args.config, args.output_dir, args.cache_dir)
    stream = sys.stdout if result.get("status") == "M2_S1_CONTROL_COMPLETED" else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("status") == "M2_S1_CONTROL_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
