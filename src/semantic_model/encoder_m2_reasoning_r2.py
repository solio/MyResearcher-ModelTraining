"""Train-only per-label threshold calibration for the accepted RBT3 corrective runs.

R2 never changes model parameters.  It reloads the three immutable corrective
checkpoints, selects one threshold per reasoning label from Train only, and
then evaluates the frozen thresholds on Dev.  No probability table is written
to disk and no data role beyond Train/Dev is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import encoder_m1 as m1
from . import encoder_m2_reasoning_corrective as corrective
from . import encoder_m2_s1 as s1
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "M2-R2-REASONING-TRAIN-ONLY-THRESHOLD-CALIBRATION"
TARGET_HEAD = "reasoning_tags"
SEEDS = (35, 71, 107)
THRESHOLD_GRID = tuple(round(index / 100, 2) for index in range(5, 96))
PARENT_CONTENT_ADDRESS = "0654d2ad1e537ef0637f71e8604ea02943fd08d2ffdae8900ed9f1c70eaf4238"
MODEL_ID = corrective.MODEL_ID
REVISION = corrective.REVISION
LICENSE = corrective.LICENSE
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
DEFAULT_PARENT_ARTIFACT = corrective.DEFAULT_ARTIFACT_ROOT / "m2-rbt3-reasoning-corrective-v1-20260901-retry"
DEFAULT_OUTPUT = corrective.DEFAULT_ARTIFACT_ROOT / "m2-r2-reasoning-threshold-calibration-20260901"

# These are the fixed Classical feasibility thresholds supplied by the M2
# contract.  They are not recomputed from any Test/Anchor/reference file.
CLASSICAL_GATE = {
    "macro_mean_min": 0.400256050141,
    "macro_worst_seed_min": 0.380256050141,
    "micro_mean_min": 0.476338797814,
    "micro_worst_seed_min": 0.456338797814,
    "exact_mean_min": 0.117232142857,
    "exact_worst_seed_min": 0.097232142857,
    "macro_sample_std_max": 0.05,
    "critical_label_f1_drop_max": 0.05,
}
CRITICAL_LABELS = ("NO_REASON_GIVEN", "TECHNICAL_PRICE", "FUNDAMENTAL", "SARCASM_IRONY")


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _mapping(value: Any, code: str, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object")
    return value


def _read_json(path: str | Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(code, "required JSON file is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(code, "required JSON file is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), code, "required JSON root must be an object", path=str(path))
    return value


def _contract_and_parent(contract_path: str | Path, parent_artifact: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = s1._contract_requirements(contract_path)
    contract = frozen["contract"]
    _require(contract.get("prohibitions", {}).get("test_evaluation") is False, "M2_R2_CONTRACT_INVALID", "Test evaluation must remain prohibited")
    _require(contract.get("prohibitions", {}).get("anchor_evaluation") is False, "M2_R2_CONTRACT_INVALID", "Anchor evaluation must remain prohibited")
    _require(contract.get("prohibitions", {}).get("gold_evaluation_or_creation") is False, "M2_R2_CONTRACT_INVALID", "Gold must remain prohibited")
    _require(contract.get("prohibitions", {}).get("ood_evaluation_or_creation") is False, "M2_R2_CONTRACT_INVALID", "OOD must remain prohibited")
    controls = _mapping(contract.get("immutable_controls"), "M2_R2_CONTRACT_INVALID", "immutable controls")
    classical = _mapping(controls.get("classical_v0_3_5_control"), "M2_R2_CONTRACT_INVALID", "Classical control")
    frozen_dev = _mapping(classical.get("frozen_dev_metrics"), "M2_R2_CONTRACT_INVALID", "frozen Classical Dev metrics")
    reasoning = _mapping(frozen_dev.get(TARGET_HEAD), "M2_R2_CONTRACT_INVALID", "Classical reasoning metrics")
    _require(reasoning.get("primary_macro_f1") == 0.41025605014132455, "M2_R2_CLASSICAL_REFERENCE_INVALID", "Classical Macro-F1 identity changed")
    _require(reasoning.get("micro_f1") == 0.48633879781420764, "M2_R2_CLASSICAL_REFERENCE_INVALID", "Classical Micro-F1 identity changed")
    _require(reasoning.get("exact_set_accuracy") == 0.12723214285714285, "M2_R2_CLASSICAL_REFERENCE_INVALID", "Classical exact-set identity changed")
    parent = corrective._load_comparator(parent_artifact, PARENT_CONTENT_ADDRESS, stage="S1")
    return frozen, {"classical": reasoning, "parent": parent}


def validate_r2_preflight(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    worktree: str | Path,
    contract_path: str | Path,
    parent_artifact: str | Path,
) -> dict[str, Any]:
    frozen, controls = _contract_and_parent(contract_path, parent_artifact)
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_R2_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, snapshot_identity = s1.validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {
        "frozen_contract": frozen,
        "snapshot": snapshot,
        "snapshot_identity": snapshot_identity,
        "schema": schema,
        "train": train,
        "dev": dev,
        "parent": controls["parent"],
        "classical_reasoning": controls["classical"],
        "identity": {"contract_sha256": frozen["contract_sha256"], "parent_content_address": PARENT_CONTENT_ADDRESS, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version, "data_scope": "Train_1822_and_Dev_448_only"},
        "worktree": str(Path(worktree).resolve()),
    }


def choose_thresholds(labels: Sequence[Sequence[int]], probabilities: Sequence[Sequence[float]], class_order: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    """Choose each threshold by Train F1, with the frozen tie-break order."""

    _require(len(labels) == len(probabilities), "M2_R2_CALIBRATION_INPUT_INVALID", "label/probability row counts differ")
    _require(len(class_order) == 15, "M2_R2_CALIBRATION_INPUT_INVALID", "reasoning head must have 15 labels")
    thresholds: dict[str, float] = {}
    details: dict[str, Any] = {}
    for index, label_name in enumerate(class_order):
        truth = [int(row[index]) for row in labels]
        scores = [float(row[index]) for row in probabilities]
        candidates: list[tuple[float, float]] = []
        for threshold in THRESHOLD_GRID:
            predicted = [int(score >= threshold) for score in scores]
            f1 = _binary_f1(truth, predicted)
            candidates.append((threshold, f1))
        best_f1 = max(f1 for _, f1 in candidates)
        tied = [threshold for threshold, f1 in candidates if abs(f1 - best_f1) <= 1e-12]
        selected = max(tied, key=lambda value: (-abs(value - 0.50), value))
        thresholds[label_name] = selected
        raw_f1 = _binary_f1(truth, [int(score >= 0.50) for score in scores])
        details[label_name] = {"threshold": selected, "train_f1": best_f1, "raw_0_50_f1": raw_f1, "tie_break": "closest_to_0_50_then_higher_threshold", "support": sum(truth)}
    return thresholds, details


def _binary_f1(truth: Sequence[int], predicted: Sequence[int]) -> float:
    tp = sum(left == 1 and right == 1 for left, right in zip(truth, predicted, strict=True))
    fp = sum(left == 0 and right == 1 for left, right in zip(truth, predicted, strict=True))
    fn = sum(left == 1 and right == 0 for left, right in zip(truth, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metrics_from_probabilities(labels: Sequence[Sequence[int]], probabilities: Sequence[Sequence[float]], class_order: Sequence[str], thresholds: Mapping[str, float]) -> dict[str, Any]:
    predicted = [[int(float(row[index]) >= float(thresholds[label])) for index, label in enumerate(class_order)] for row in probabilities]
    return corrective.m1._reasoning_metrics(labels, predicted, class_order)


def _predict(runtime: tuple[Any, Any, Any, Any], model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> tuple[Any, Any]:
    np, torch, _auto_model, _auto_tokenizer = runtime
    labels: list[Any] = []
    probabilities: list[Any] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), int(config["batch_size"])):
            batch = m1._as_batch(torch, tokenizer, records[offset : offset + int(config["batch_size"])], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[TARGET_HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "M2_R2_NONFINITE_LOGITS", "non-finite reasoning logits during calibration")
            labels.append(batch["labels"][TARGET_HEAD].detach().cpu().numpy().astype(int))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(labels, axis=0), np.concatenate(probabilities, axis=0)


def _validate_checkpoint(checkpoint: Mapping[str, Any], seed: int, config: Mapping[str, Any]) -> None:
    _require(checkpoint.get("stage_id") == corrective.STAGE_ID, "M2_R2_CHECKPOINT_STAGE_MISMATCH", "checkpoint is not RBT3 corrective evidence", seed=seed)
    _require(checkpoint.get("seed") == seed, "M2_R2_CHECKPOINT_SEED_MISMATCH", "checkpoint seed does not match run unit", seed=seed)
    frozen = _mapping(checkpoint.get("frozen_config"), "M2_R2_CHECKPOINT_INVALID", "checkpoint frozen config")
    for key, expected in (("model_id", MODEL_ID), ("revision", REVISION), ("max_length", 256), ("batch_size", 16), ("truncation", "HEAD_TAIL"), ("local_files_only", True), ("trust_remote_code", False)):
        _require(frozen.get(key) == expected, "M2_R2_CHECKPOINT_CONFIG_MISMATCH", f"checkpoint {key} differs", seed=seed)
    _require(checkpoint.get("last_transformer_block_prefix") == "encoder.encoder.layer.2", "M2_R2_CHECKPOINT_BLOCK_MISMATCH", "checkpoint block identity differs", seed=seed)
    _require(isinstance(checkpoint.get("last_transformer_block_state_dict"), Mapping) and isinstance(checkpoint.get("reasoning_head_state_dict"), Mapping), "M2_R2_CHECKPOINT_STATE_MISSING", "checkpoint lacks block or reasoning head state", seed=seed)
    _require(config["max_length"] == frozen.get("max_length") and config["batch_size"] == frozen.get("batch_size"), "M2_R2_CONFIG_MISMATCH", "calibration config differs from checkpoint", seed=seed)


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {"per_seed_values": values, "mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values), "worst_seed": min(values)}


def aggregate_r2(rows: Sequence[Mapping[str, Any]], *, calibrated: bool) -> dict[str, Any]:
    _require([row["seed"] for row in rows] == list(SEEDS), "M2_R2_INCOMPLETE_SEEDS", "all three calibration seeds are required")
    devices = [row["resource"]["actual_device"] for row in rows]
    _require(len(set(devices)) == 1, "M2_R2_MIXED_DEVICE", "calibration cannot aggregate mixed devices")
    metric_rows = [row["metrics"]["dev_calibrated" if calibrated else "dev_raw"] for row in rows]
    return {"seeds": list(SEEDS), "actual_device": devices[0], "all_seeds_complete": True, "threshold_state": "CALIBRATED" if calibrated else "UNIFORM_0_50", "macro_f1": _aggregate_metric_rows(metric_rows, "macro_f1"), "micro_f1": _aggregate_metric_rows(metric_rows, "micro_f1"), "exact_set_accuracy": _aggregate_metric_rows(metric_rows, "exact_set_accuracy")}


def classical_gate(aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], classical: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "macro_mean": float(aggregate["macro_f1"]["mean"]) >= CLASSICAL_GATE["macro_mean_min"],
        "macro_worst_seed": float(aggregate["macro_f1"]["worst_seed"]) >= CLASSICAL_GATE["macro_worst_seed_min"],
        "micro_mean": float(aggregate["micro_f1"]["mean"]) >= CLASSICAL_GATE["micro_mean_min"],
        "micro_worst_seed": float(aggregate["micro_f1"]["worst_seed"]) >= CLASSICAL_GATE["micro_worst_seed_min"],
        "exact_mean": float(aggregate["exact_set_accuracy"]["mean"]) >= CLASSICAL_GATE["exact_mean_min"],
        "exact_worst_seed": float(aggregate["exact_set_accuracy"]["worst_seed"]) >= CLASSICAL_GATE["exact_worst_seed_min"],
        "macro_sample_std": float(aggregate["macro_f1"]["sample_standard_deviation"]) <= CLASSICAL_GATE["macro_sample_std_max"],
    }
    critical: dict[str, Any] = {}
    classical_labels = _mapping(classical.get("per_label"), "M2_R2_CLASSICAL_REFERENCE_INVALID", "Classical per-label metrics")
    critical_failures: list[str] = []
    for label in CRITICAL_LABELS:
        reference = _mapping(classical_labels.get(label), "M2_R2_CLASSICAL_REFERENCE_INVALID", f"Classical {label}")
        support = int(reference.get("support", 0))
        per_seed = []
        for row in rows:
            observed = float(row["metrics"]["dev_calibrated"]["per_label"][label]["f1"])
            baseline = float(reference["f1"])
            per_seed.append({"seed": row["seed"], "support": support, "classical_f1": baseline, "calibrated_f1": observed, "delta": round(observed - baseline, 6), "status": "EVALUATED" if support >= 20 else "NOT_EVALUABLE_FOR_NUMERICAL_GATE"})
        passed = None if support < 20 else all(value["delta"] >= -CLASSICAL_GATE["critical_label_f1_drop_max"] for value in per_seed)
        if passed is False:
            critical_failures.append(label)
        critical[label] = {"classical_f1": float(reference["f1"]), "support": support, "per_seed": per_seed, "passed": passed}
    checks["critical_labels"] = not critical_failures
    return {"thresholds": CLASSICAL_GATE, "checks": checks, "critical_labels": critical, "critical_failures": critical_failures, "passed": all(checks.values()), "selected_candidate": False, "diagnostic_only": True}


def _seed(
    *,
    runtime: tuple[Any, Any, Any, Any],
    seed: int,
    root: Path,
    snapshot: Path,
    schema: Any,
    train: Sequence[m1.M1Record],
    dev: Sequence[m1.M1Record],
    config: Mapping[str, Any],
    checkpoint_path: Path,
    cache_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint, seed, config)
    # Reproduce the corrective/S3 initialization stream before constructing
    # the model so the stored reasoning-head identity is verifiable.
    corrective._set_seed(torch, np, seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model, block, block_prefix = corrective._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    initial_head_sha = corrective._reasoning_head_state_digest(torch, model)
    _require(initial_head_sha == checkpoint.get("initial_reasoning_head_sha256"), "M2_R2_INITIAL_HEAD_IDENTITY_MISMATCH", "checkpoint reasoning head initialization differs", seed=seed)
    model = model.to(device)
    block.load_state_dict(checkpoint["last_transformer_block_state_dict"])
    model.heads[TARGET_HEAD].load_state_dict(checkpoint["reasoning_head_state_dict"])
    corrective.validate_trainable_parameters(model, block, block_prefix)
    labels_train, probabilities_train = _predict(runtime, model, tokenizer, train, config, device)
    labels_dev, probabilities_dev = _predict(runtime, model, tokenizer, dev, config, device)
    class_order = config["class_order"][TARGET_HEAD]
    raw_thresholds = {label: 0.50 for label in class_order}
    thresholds, train_calibration = choose_thresholds(labels_train.tolist(), probabilities_train.tolist(), class_order)
    train_raw = metrics_from_probabilities(labels_train.tolist(), probabilities_train.tolist(), class_order, raw_thresholds)
    train_calibrated = metrics_from_probabilities(labels_train.tolist(), probabilities_train.tolist(), class_order, thresholds)
    dev_raw = metrics_from_probabilities(labels_dev.tolist(), probabilities_dev.tolist(), class_order, raw_thresholds)
    dev_calibrated = metrics_from_probabilities(labels_dev.tolist(), probabilities_dev.tolist(), class_order, thresholds)
    try:
        smoke = corrective._cpu_reload_corrective(torch, lambda: corrective._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_R2_CPU_RELOAD_SMOKE_FAILED", "R2 CPU reload failed", cause=exc.code) from exc
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "checkpoint_sha256": sha256_file(checkpoint_path), "provenance": provenance}
    m1._json_dump(folder / "thresholds.json", {"stage_id": STAGE_ID, "seed": seed, "threshold_grid": [*THRESHOLD_GRID], "thresholds": thresholds, "selection_rule": "MAX_TRAIN_F1_THEN_CLOSEST_TO_0_50_THEN_HIGHER_THRESHOLD", "labels": train_calibration})
    m1._json_dump(folder / "train-calibration-summary.json", {"metric_scope": "TRAIN_WEAK_LABEL_ONLY_FOR_THRESHOLD_SELECTION", "seed": seed, "raw_uniform_0_50": train_raw, "calibrated": train_calibrated, "per_label_selection": train_calibration})
    m1._json_dump(folder / "dev-metrics.json", {"metric_scope": "DEV_WEAK_LABEL_DIAGNOSTIC_ONLY_THRESHOLDS_FROZEN_ON_TRAIN", "seed": seed, "raw_uniform_0_50": dev_raw, "calibrated": dev_calibrated})
    m1._json_dump(folder / "resource-log.json", resource)
    s1._limits(started, root, "after_seed_calibration", seed)
    return {"seed": seed, "metrics": {"dev_raw": dev_raw, "dev_calibrated": dev_calibrated, "train_raw": train_raw, "train_calibrated": train_calibrated, "thresholds": thresholds}, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "thresholds_sha256": sha256_file(folder / "thresholds.json"), "train_summary_sha256": sha256_file(folder / "train-calibration-summary.json"), "dev_metrics_sha256": sha256_file(folder / "dev-metrics.json")}


def _failure(root: Path | None, exc: ContractError, preflight: Mapping[str, Any] | None, entered: bool) -> dict[str, Any]:
    result = {"status": "CALIBRATED_REASONING_DIAGNOSTIC_REJECTED", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": False, "model_loaded": entered, "cache_accessed": entered, "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False}
    if root and root.exists():
        m1._json_dump(root / "rejected-calibration-evidence.json", result)
        payload: dict[str, Any] = {"manifest_schema_version": "myresearcher.encoder-m2-r2-rejected-artifact-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "selected_candidate": False}
        if preflight:
            payload.update({"model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "parent_content_address": PARENT_CONTENT_ADDRESS, "complete_cache_snapshot": preflight.get("snapshot_identity")})
        result["rejected_content_address"] = m1._write_content_manifest(root, payload)["content_address"]
    return result


def run_r2(
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    parent_artifact: str | Path = DEFAULT_PARENT_ARTIFACT,
    worktree: str | Path | None = None,
    contract_path: str | Path | None = None,
    runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies,
    seed_executor: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / s1.CONTRACT_RELATIVE_PATH
    preflight: dict[str, Any] | None = None
    try:
        preflight = validate_r2_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file, parent_artifact=parent_artifact)
    except ContractError as exc:
        return _failure(None, exc, None, False)
    root: Path | None = None
    entered = False
    rows: list[Mapping[str, Any]] = []
    started = time.monotonic()
    try:
        runtime = runtime_loader()
        runtime_identity = s1.validate_runtime_identity(runtime, preflight["frozen_contract"])
        root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True, exist_ok=False)
        s1._limits(started, root, "before_calibration")
        config = corrective._config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        config.update({"stage_id": STAGE_ID, "max_length": 256, "batch_size": 16, "truncation": "HEAD_TAIL", "threshold_grid": [*THRESHOLD_GRID], "threshold_selection_population": "Train_1822_only", "dev_role": "diagnostic_only_after_threshold_freeze", "calibration_method": "PER_LABEL_TRAIN_F1", "selected_candidate": False})
        m1._json_dump(root / "calibration-config.json", config)
        provenance = {**preflight["identity"], "runtime_identity": runtime_identity, "parent_content_address": PARENT_CONTENT_ADDRESS}
        m1._json_dump(root / "provenance.json", provenance)
        executor = seed_executor or _seed
        checkpoint_root = Path(parent_artifact).resolve()
        for seed in SEEDS:
            entered = True
            rows.append(executor(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, checkpoint_path=checkpoint_root / f"seed-{seed}" / "reasoning-corrective-checkpoint.pt", cache_dir=Path(cache_dir), provenance=provenance))
        aggregate_raw = aggregate_r2(rows, calibrated=False)
        aggregate_calibrated = aggregate_r2(rows, calibrated=True)
        gate = classical_gate(aggregate_calibrated, rows, preflight["classical_reasoning"])
        m1._json_dump(root / "stage-aggregate.json", {"stage_id": STAGE_ID, "raw_uniform_0_50": aggregate_raw, "calibrated": aggregate_calibrated, "selected_candidate": False})
        m1._json_dump(root / "classical-feasibility-gate.json", gate)
        s1._limits(started, root, "after_aggregate_and_gate")
        status = "CALIBRATED_REASONING_DIAGNOSTIC_PASSED" if gate["passed"] else "CALIBRATED_REASONING_DIAGNOSTIC_REJECTED"
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-r2-reasoning-calibration-artifact-manifest.v1", "status": status, "stage_id": STAGE_ID, "diagnostic_only": True, "selected_candidate": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "parent_corrective_artifact": {"content_address": PARENT_CONTENT_ADDRESS, "artifact_path": str(Path(parent_artifact).resolve())}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_identity": {"actual_device": aggregate_calibrated["actual_device"], "policy": "MPS_FIRST_CPU_FALLBACK"}, "aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "gate_sha256": sha256_file(root / "classical-feasibility-gate.json"), "calibration_config_sha256": sha256_file(root / "calibration-config.json"), "provenance_sha256": sha256_file(root / "provenance.json"), "seed_checkpoints": {str(row["seed"]): row["checkpoint_sha256"] for row in rows}, "seed_thresholds": {str(row["seed"]): row["thresholds_sha256"] for row in rows}, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_inference"], "output_scope": "SMALL_DIAGNOSTIC_NO_PER_ROW_PROBABILITY_FILES"})
        s1._limits(started, root, "after_final_manifest")
        return {"status": status, "stage_id": STAGE_ID, "training_invoked": False, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "output_dir": str(root), "content_address": manifest["content_address"], "aggregate_raw": aggregate_raw, "aggregate_calibrated": aggregate_calibrated, "classical_gate": gate}
    except ContractError as exc:
        return _failure(root, exc, preflight, entered)
    except Exception as exc:
        return _failure(root, ContractError("M2_R2_RUNTIME_EXCEPTION", "R2 calibration runtime exception", exception_type=type(exc).__name__, detail=str(exc)), preflight, entered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate RBT3 reasoning thresholds on Train only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-artifact", default=str(DEFAULT_PARENT_ARTIFACT))
    args = parser.parse_args(argv)
    result = run_r2(args.config, args.output_dir, args.cache_dir, parent_artifact=args.parent_artifact)
    stream = sys.stdout if result.get("status") in {"CALIBRATED_REASONING_DIAGNOSTIC_PASSED", "CALIBRATED_REASONING_DIAGNOSTIC_REJECTED"} and result.get("output_created") else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result.get("status") == "CALIBRATED_REASONING_DIAGNOSTIC_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
