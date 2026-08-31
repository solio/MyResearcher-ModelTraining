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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import encoder_m1 as m1
from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL"
FROZEN_CONTRACT_COMMIT = "0d2f64cf0ce26953e83b17d043da4441f4930dc0"
RECEIPT_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-owner-authorization-receipt.v1"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-control-run.v1"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")
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
    }


def validate_owner_authorization_receipt(
    receipt_path: str | Path,
    *,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Validate a future S1 receipt before reading data, cache, or runtime.

    The frozen contract deliberately remains `authorization_granted=false`.
    This validator accepts no ambient or inferred authorization: it requires a
    separately content-addressed receipt for exactly one M2-S1 scope.
    """

    receipt = _read_json(
        receipt_path,
        missing_code="M2_S1_OWNER_RECEIPT_MISSING",
        invalid_code="M2_S1_OWNER_RECEIPT_INVALID",
    )
    _require(
        receipt.get("receipt_schema_version") == RECEIPT_SCHEMA_VERSION,
        "M2_S1_OWNER_RECEIPT_INVALID",
        "receipt schema version is not supported",
    )
    _require(
        receipt.get("authorization_granted") is True,
        "M2_S1_OWNER_AUTHORIZATION_NOT_GRANTED",
        "M2-S1 fit requires an explicit granted owner receipt",
    )
    _require(
        isinstance(receipt.get("owner_decision_id"), str) and receipt["owner_decision_id"].strip(),
        "M2_S1_OWNER_DECISION_ID_MISSING",
        "receipt must identify the owner decision authorizing this run",
    )
    observed_address = receipt.get("receipt_content_address")
    expected_address = content_addressed_id(receipt, omit_keys={"receipt_content_address"})
    _require(
        observed_address == expected_address,
        "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH",
        "receipt content address does not match the receipt payload",
        observed=observed_address,
        expected=expected_address,
    )
    _parse_expiry(receipt.get("expires_at_utc"))

    frozen = _contract_requirements(contract_path)
    _require(
        receipt.get("frozen_contract_commit") == FROZEN_CONTRACT_COMMIT,
        "M2_S1_FROZEN_CONTRACT_COMMIT_MISMATCH",
        "receipt must bind the frozen M2 contract commit",
    )
    _require(
        receipt.get("contract_sha256") == frozen["contract_sha256"],
        "M2_S1_CONTRACT_SHA256_MISMATCH",
        "receipt does not bind the exact frozen M2 contract bytes",
    )
    _require(receipt.get("stage_id") == STAGE_ID, "M2_S1_STAGE_MISMATCH", "receipt does not authorize M2-S1")
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
    return {"receipt": receipt, "frozen_contract": frozen}


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
    result = m1.subprocess.run(
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


def validate_m2_s1_preflight(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    worktree: str | Path,
    receipt_authorization: Mapping[str, Any],
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
    # Re-hash the on-disk frozen contract immediately before cache verification.
    _require(
        sha256_file(frozen["contract_path"]) == frozen["contract_sha256"],
        "M2_S1_CONTRACT_SHA256_MISMATCH",
        "frozen M2 contract changed after receipt validation",
    )
    owner_contract_for_cache = {
        "model_artifact_sha256": frozen["model"]["model_weight_sha256"],
        "tokenizer_json_sha256": frozen["model"]["tokenizer_json_sha256"],
        "vocab_txt_sha256": frozen["model"]["vocab_txt_sha256"],
    }
    snapshot = m1._validated_fixed_snapshot(Path(cache_dir).resolve(), owner_contract_for_cache)
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
    }
    return {"identity": identity, "canonical": canonical, "frozen_contract": frozen, "snapshot": snapshot}


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
    cache_dir: Path,
) -> dict[str, Any]:
    """Execute one authorized frozen seed using M1's shared primitives."""

    np, torch, AutoModel, AutoTokenizer = runtime
    seed_root = root / f"seed-{seed}"
    seed_root.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_trainable = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    _require(encoder_trainable == 0 and len(model.heads) == 7, "M2_S1_FREEZE_OR_HEAD_CONTRACT_VIOLATION", "S1 requires a frozen Encoder and seven trainable heads")
    optimizer = torch.optim.AdamW(trainable, lr=float(config["optimizer"]["learning_rate"]), weight_decay=float(config["optimizer"]["weight_decay"]), betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best_score, best_epoch, stale = -1.0, 0, 0
    log_path = seed_root / "training-log.jsonl"
    for epoch in range(1, int(config["stopping"]["max_epochs"]) + 1):
        _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_S1_WALL_TIME_LIMIT_EXCEEDED", "seed exceeded its 120-minute wall-time limit", seed=seed)
        _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_S1_DISK_LIMIT_EXCEEDED", "M2-S1 artifacts exceeded the 10 GiB limit", seed=seed)
        epoch_start = time.monotonic()
        model.train()
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), int(config["batch_size"])):
            _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_S1_WALL_TIME_LIMIT_EXCEEDED", "seed exceeded its 120-minute wall-time limit", seed=seed)
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
        dev_metrics = m1.diagnostic_metrics(torch, model, tokenizer, dev, config, device)
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
    try:
        smoke = m1.cpu_reload_and_inference_smoke(torch, lambda: m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU checkpoint reload/inference smoke failed", seed=seed, cause=exc.code) from exc
    _require(smoke.get("all_logits_finite") is True, "M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU checkpoint reload/inference must have finite logits", seed=seed)
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "seed": seed, "best_epoch": best_epoch, "early_stopping_score": best_score, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(seed_root / "seed-metrics.json", metrics)
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - start, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(seed_root), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": platform.python_version(), "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
    m1._json_dump(seed_root / "resource-log.json", resource)
    return {"seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "output_dir": str(seed_root), "model_loaded": True}


def aggregate_s1_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only a complete exactly-once 35/71/107 S1 control."""

    observed_seeds = [item.get("seed") for item in seed_results]
    _require(observed_seeds == list(SEEDS), "M2_S1_INCOMPLETE_SEEDS", "aggregate requires all three S1 seeds in frozen order", observed_seeds=observed_seeds)
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
    return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "all_seeds_complete": True, "per_head_primary_macro_f1": heads, "seed_stability_gate_passed": stability_passed, "allowed_output": "MAY_REQUEST_S2_OWNER_AUTHORIZATION" if stability_passed else "S1_REJECTED_OR_BLOCKED_EVIDENCE", "selected_candidate": False}


def _authorized_failure(root: Path, exc: ContractError, *, output_created: bool, model_loaded: bool, cache_accessed: bool) -> dict[str, Any]:
    evidence = {"status": "S1_REJECTED_OR_BLOCKED_EVIDENCE", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "training_invoked": True, "model_loaded": model_loaded, "cache_accessed": cache_accessed, "output_created": output_created, "aggregate_created": False, "selected_candidate": False}
    if output_created:
        m1._json_dump(root / "blocked-evidence.json", evidence)
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

    # This is intentionally the first dynamic-runtime import in the module.
    runtime = runtime_loader()
    root = Path(output_dir).resolve()
    _require(not root.exists(), "M2_S1_OUTPUT_ALREADY_EXISTS", "M2-S1 requires a new immutable output directory", output_dir=str(root))
    root.mkdir(parents=True, exist_ok=False)
    config = ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_S1_DATA_CONTRACT_INVALID", "M2-S1 loader must expose exactly Train 1822 and Dev 448")
    frozen_config = _frozen_training_config(preflight["frozen_contract"]["contract"], schema.class_order)
    m1._json_dump(root / "training-config.json", frozen_config)
    m1._json_dump(root / "class-order.json", {"schema_version": schema.schema_version, "class_order": frozen_config["class_order"]})
    results: list[Mapping[str, Any]] = []
    execute = seed_executor or _execute_one_seed
    try:
        for seed in SEEDS:
            result = execute(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=schema, train=train, dev=dev, config=frozen_config, provenance=preflight["identity"], cache_dir=Path(cache_dir).resolve())
            _require(result.get("seed") == seed, "M2_S1_SEED_EXECUTOR_IDENTITY_MISMATCH", "seed executor returned an unexpected seed", expected=seed, observed=result.get("seed"))
            results.append(result)
    except ContractError as exc:
        return _authorized_failure(root, exc, output_created=True, model_loaded=bool(results) or exc.code not in {"M2_S1_OUTPUT_ALREADY_EXISTS"}, cache_accessed=True)
    aggregate = aggregate_s1_seed_results(results)
    m1._json_dump(root / "stage-aggregate.json", aggregate)
    manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s1-artifact-manifest.v1", "stage_id": STAGE_ID, "diagnostic_only": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "model_id": m1.MODEL_ID, "resolved_revision": m1.REVISION, "license": m1.LICENSE, "fixed_model_and_tokenizer_sha256": preflight["frozen_contract"]["model"], "provenance": preflight["identity"], "training_config_sha256": sha256_file(root / "training-config.json"), "class_order_sha256": sha256_file(root / "class-order.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "seed_checkpoints": {str(item["seed"]): item["checkpoint_sha256"] for item in results}, "per_seed_metrics_and_resources": {str(item["seed"]): {"seed_metrics_sha256": sha256_file(root / f"seed-{item['seed']}" / "seed-metrics.json"), "resource_log_sha256": sha256_file(root / f"seed-{item['seed']}" / "resource-log.json")} for item in results}})
    return {"status": "M2_S1_CONTROL_COMPLETED", "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate}


def run_m2_s1(
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    receipt_path: str | Path,
    *,
    worktree: str | Path | None = None,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """M2-S1 entry point.  Receipt validation is the first operation by design."""

    root = Path(worktree).resolve() if worktree is not None else Path(__file__).resolve().parents[2]
    frozen_contract_path = Path(contract_path).resolve() if contract_path is not None else root / CONTRACT_RELATIVE_PATH
    try:
        # Do not move this gate: no audit/data/cache/output/runtime work precedes it.
        receipt_authorization = validate_owner_authorization_receipt(receipt_path, contract_path=frozen_contract_path)
    except ContractError as exc:
        return _blocked(exc.code, exc.message, phase="OWNER_RECEIPT", **exc.details)
    try:
        preflight = validate_m2_s1_preflight(config_path, cache_dir, worktree=root, receipt_authorization=receipt_authorization)
    except ContractError as exc:
        return _blocked(exc.code, exc.message, phase="CANONICAL_OR_PROVENANCE_PREFLIGHT", **exc.details)
    return _run_authorized_s1(config_path, output_dir, cache_dir, preflight=preflight)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an independently owner-authorized M2-S1 frozen three-seed control")
    parser.add_argument("--config", required=True, help="Immutable project config; only Train/Dev M2 paths are read")
    parser.add_argument("--output-dir", required=True, help="New immutable M2-S1 output directory")
    parser.add_argument("--cache-dir", required=True, help="Existing local fixed-revision cache; no download is permitted")
    parser.add_argument("--owner-authorization-receipt", required=True, help="Separately issued content-addressed M2-S1 receipt")
    args = parser.parse_args(argv)
    result = run_m2_s1(args.config, args.output_dir, args.cache_dir, args.owner_authorization_receipt)
    stream = sys.stdout if result.get("status") == "M2_S1_CONTROL_COMPLETED" else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("status") == "M2_S1_CONTROL_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
