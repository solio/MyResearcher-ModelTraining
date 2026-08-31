"""M2-S3 frozen single-task diagnostics for the two triggered heads.

S3 is a negative-transfer diagnostic only.  It never trains the seven-head
model jointly, never promotes a candidate, and never starts another stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import encoder_m1 as m1
from . import encoder_m2_s1 as s1
from . import encoder_m2_s2 as s2
from .errors import ContractError
from .hashes import sha256_file
from .schema import SINGLE_LABEL_HEADS, V1_HEADS


STAGE_ID = "M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL"
TRIGGERED_HEADS = ("emotion_primary", "reasoning_tags")
SEEDS = (35, 71, 107)
HEAD_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 16
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
DEFAULT_S1_ARTIFACT = s2.DEFAULT_S1_ARTIFACT


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _mapping(value: Any, code: str, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object")
    return value


def _contract_requirements(contract_path: str | Path) -> dict[str, Any]:
    frozen = s2._contract_requirements(contract_path)
    contract = frozen["contract"]
    stages = contract.get("minimal_experiment_gradient")
    stage = next((item for item in stages if isinstance(item, Mapping) and item.get("stage_id") == STAGE_ID), None) if isinstance(stages, list) else None
    _require(stage is not None, "M2_S3_STAGE_CONTRACT_INVALID", "S3 stage is missing")
    _require(stage.get("architecture") == "ONE_FROZEN_SHARED_ENCODER_WITH_ONE_TRAINABLE_HEAD_PER_RUN" and stage.get("encoder_state") == "FROZEN" and stage.get("unfrozen_transformer_blocks") == 0, "M2_S3_STAGE_CONTRACT_INVALID", "S3 must be a frozen single-task control")
    _require(stage.get("seeds") == list(SEEDS), "M2_S3_STAGE_CONTRACT_INVALID", "S3 seeds changed")
    _require(stage.get("head_learning_rate") == HEAD_LEARNING_RATE and stage.get("encoder_learning_rate") is None, "M2_S3_OPTIMIZER_CONTRACT_INVALID", "S3 learning rate changed")
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S3_CONTRACT_INVALID", "common training configuration")
    early = _mapping(common.get("early_stopping"), "M2_S3_CONTRACT_INVALID", "early stopping")
    gradient = _mapping(common.get("gradient_controls"), "M2_S3_CONTRACT_INVALID", "gradient controls")
    _require(common.get("max_length") == 256 and common.get("batch_size") == BATCH_SIZE and common.get("truncation") == "HEAD_TAIL" and early.get("max_epochs") == MAX_EPOCHS and early.get("patience_epochs") == PATIENCE and gradient.get("gradient_clipping_max_norm") == 1.0, "M2_S3_TRAINING_CONTRACT_INVALID", "S3 fixed training configuration changed")
    return {**frozen, "stage": stage}


def validate_s3_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    frozen = _contract_requirements(contract_path)
    config = s1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_S3_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, identity = s1.validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": identity, "schema": schema, "train": train, "dev": dev, "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": m1.MODEL_ID, "revision": m1.REVISION, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version}, "worktree": str(Path(worktree).resolve())}


def validate_s3_trainable_parameters(model: Any, target_head: str) -> dict[str, Any]:
    _require(target_head in TRIGGERED_HEADS, "M2_S3_TRIGGER_HEAD_INVALID", "S3 target head is not one of the triggered heads", head=target_head)
    _require(len(model.heads) == 7, "M2_S3_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "S3 model must expose all seven heads")
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"heads.{target_head}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M2_S3_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "only the selected head may require gradients", target_head=target_head, expected=sorted(expected), observed=sorted(actual))
    encoder_trainable = sorted(name for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad)
    _require(not encoder_trainable, "M2_S3_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "Encoder parameters must remain frozen", unexpected=encoder_trainable)
    return {"target_head": target_head, "encoder_trainable": False, "trainable_parameters": sorted(actual)}


def _set_single_head_trainable(model: Any, target_head: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(f"heads.{target_head}.")


def _single_head_loss(torch: Any, logits: Mapping[str, Any], batch: Mapping[str, Any], target_head: str) -> Any:
    if target_head in SINGLE_LABEL_HEADS:
        raw = torch.nn.functional.cross_entropy(logits[target_head], batch["labels"][target_head], reduction="none")
        weight = batch["weights"][target_head]
        return (raw * weight).sum() / weight.sum().clamp_min(1e-12)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits[target_head], batch["labels"][target_head], reduction="none").mean(dim=1)
    weight = batch["weights"][target_head]
    return (raw * weight).sum() / weight.sum().clamp_min(1e-12)


def _seed(*, runtime: tuple[Any, Any, Any, Any], target_head: str, seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / target_head / f"seed-{seed}"
    folder.mkdir(parents=True)
    started = time.monotonic()
    s1._limits(started, root, "before_model_load", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    _set_single_head_trainable(model, target_head)
    parameter_identity = validate_s3_trainable_parameters(model, target_head)
    s1._limits(started, root, "after_model_load", seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best, epoch_best, stale = -1.0, 0, 0
    log = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            s1._limits(started, root, "during_epoch", seed)
            batch = m1._as_batch(torch, tokenizer, shuffled[offset : offset + BATCH_SIZE], config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            for head in V1_HEADS:
                _require(bool(torch.isfinite(logits[head]).all().item()), "M2_S3_NONFINITE_LOGITS", "non-finite S3 logits", head=head)
            loss = _single_head_loss(torch, logits, batch, target_head)
            _require(bool(torch.isfinite(loss).all().item()), "M2_S3_NONFINITE_LOSS", "non-finite S3 loss", head=target_head)
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all().item()), "M2_S3_NONFINITE_GRADIENT", "non-finite S3 gradient", head=target_head)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = m1.diagnostic_metrics(torch, model, tokenizer, dev, config, device)
        score = float(dev_metrics[target_head]["macro_f1"])
        improved = score > best + float(config["stopping"]["minimum_delta"])
        if improved:
            best, epoch_best, stale = score, epoch, 0
            torch.save({"run_schema_version": "myresearcher.encoder-m2-s3-single-task-run.v1", "stage_id": STAGE_ID, "target_head": target_head, "seed": seed, "frozen_config": config, "provenance": provenance, "parameter_identity": parameter_identity, "heads_state_dict": {key: value.detach().cpu() for key, value in model.heads.state_dict().items()}}, folder / "head-checkpoint.pt")
        else:
            stale += 1
        row = {"target_head": target_head, "seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_primary_macro_f1": score, "dev_reasoning_micro_f1": dev_metrics[target_head].get("micro_f1") if target_head == "reasoning_tags" else None, "dev_reasoning_exact_set_accuracy": dev_metrics[target_head].get("exact_set_accuracy") if target_head == "reasoning_tags" else None, "improved": improved, "stale_epochs": stale}
        m1._jsonl_append(log, row)
        s1._limits(started, root, "after_epoch", seed)
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "head-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_S3_CHECKPOINT_MISSING", "S3 checkpoint missing", head=target_head, seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    best_model.heads.load_state_dict(checkpoint["heads_state_dict"])
    _set_single_head_trainable(best_model, target_head)
    validate_s3_trainable_parameters(best_model, target_head)
    train_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, train, config, device)
    dev_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, dev, config, device)
    s1._limits(started, root, "after_final_metrics", seed)
    try:
        smoke = m1.cpu_reload_and_inference_smoke(torch, lambda: m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_S3_CPU_RELOAD_SMOKE_FAILED", "S3 CPU reload failed", cause=exc.code) from exc
    _require(smoke.get("all_logits_finite") is True, "M2_S3_CPU_RELOAD_SMOKE_FAILED", "S3 CPU logits are not finite")
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "target_head": target_head, "seed": seed, "best_epoch": epoch_best, "early_stopping_metric": f"{target_head}.macro_f1", "early_stopping_score": best, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": {target_head: train_metrics[target_head]}, "dev": {target_head: dev_metrics[target_head]}, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(folder / "seed-metrics.json", metrics)
    resource = {"target_head": target_head, "seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    m1._json_dump(folder / "resource-log.json", resource)
    s1._limits(started, root, "after_seed_evidence", seed)
    return {"target_head": target_head, "seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path)}


def aggregate_s3_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(results) == len(TRIGGERED_HEADS) * len(SEEDS), "M2_S3_INCOMPLETE_RUNS", "six S3 run units are required")
    per_head: dict[str, Any] = {}
    devices: dict[str, dict[str, str]] = {}
    for head in TRIGGERED_HEADS:
        rows = [item for item in results if item["target_head"] == head]
        _require([item["seed"] for item in rows] == list(SEEDS), "M2_S3_INCOMPLETE_RUNS", "each S3 head requires seeds in fixed order", head=head)
        devices[head] = {str(item["seed"]): item["resource"]["actual_device"] for item in rows}
        _require(len(set(devices[head].values())) == 1, "M2_S3_MIXED_DEVICE", "S3 cannot aggregate mixed-device seeds", head=head, device_stratified_seed_devices=devices[head])
        primary = [float(item["metrics"]["dev"][head]["macro_f1"]) for item in rows]
        _require(all(math.isfinite(value) for value in primary), "M2_S3_METRICS_MISSING", "non-finite S3 primary metric", head=head)
        values: dict[str, Any] = {"primary_macro_f1": {"per_seed_values": primary, "mean": sum(primary) / 3, "sample_standard_deviation": statistics.stdev(primary), "minimum_worst_seed": min(primary), "maximum": max(primary)}}
        if head == "reasoning_tags":
            for metric in ("micro_f1", "exact_set_accuracy"):
                secondary = [float(item["metrics"]["dev"][head][metric]) for item in rows]
                values[metric] = {"per_seed_values": secondary, "mean": sum(secondary) / 3, "sample_standard_deviation": statistics.stdev(secondary), "minimum_worst_seed": min(secondary), "maximum": max(secondary)}
        per_head[head] = values
    return {"stage_id": STAGE_ID, "triggered_heads": list(TRIGGERED_HEADS), "seeds": list(SEEDS), "all_six_runs_complete": True, "devices": devices, "per_head": per_head, "selected_candidate": False, "promotion": "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC"}


def matching_s1_report(contract: Mapping[str, Any], s1_control: Mapping[str, Any], results: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    critical_reports: dict[str, Any] = {}
    proxies = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_S3_CONTRACT_INVALID", "metric contract")["critical_boundary_proxies"]
    for head in TRIGGERED_HEADS:
        rows = [item for item in results if item["target_head"] == head]
        s1_values = [float(s1_control["seed_metrics"][seed]["dev"][head]["macro_f1"]) for seed in SEEDS]
        s3_values = [float(item["metrics"]["dev"][head]["macro_f1"]) for item in rows]
        delta = [round(right - left, 6) for left, right in zip(s1_values, s3_values, strict=True)]
        reports[head] = {"s1_per_seed": s1_values, "s3_per_seed": s3_values, "delta_per_seed": delta, "s1_mean": sum(s1_values) / 3, "s3_mean": sum(s3_values) / 3, "mean_delta": sum(delta) / 3, "worst_seed_delta": min(delta), "s3_sample_standard_deviation": statistics.stdev(s3_values)}
        if head == "reasoning_tags":
            for metric in ("micro_f1", "exact_set_accuracy"):
                s1_secondary = [float(s1_control["seed_metrics"][seed]["dev"][head][metric]) for seed in SEEDS]
                s3_secondary = [float(item["metrics"]["dev"][head][metric]) for item in rows]
                reports[head][metric] = {"s1_per_seed": s1_secondary, "s3_per_seed": s3_secondary, "delta_per_seed": [round(right - left, 6) for left, right in zip(s1_secondary, s3_secondary, strict=True)], "s3_mean": sum(s3_secondary) / 3, "s3_sample_standard_deviation": statistics.stdev(s3_secondary), "s3_worst_seed": min(s3_secondary)}
        target_proxy = next(item for item in proxies if item["head"] == head)
        labels: dict[str, Any] = {}
        key = "per_label" if head == "reasoning_tags" else "per_class"
        for label in target_proxy["labels"]:
            s1_rows = [s1_control["seed_metrics"][seed]["dev"][head][key][label] for seed in SEEDS]
            s3_rows = [item["metrics"]["dev"][head][key][label] for item in rows]
            labels[label] = {"support_per_seed": [int(row["support"]) for row in s1_rows], "s1_f1_per_seed": [float(row["f1"]) for row in s1_rows], "s3_f1_per_seed": [float(row["f1"]) for row in s3_rows], "delta_per_seed": [round(float(right["f1"]) - float(left["f1"]), 6) for left, right in zip(s1_rows, s3_rows, strict=True)], "status": "REPORTED_S3_DIAGNOSTIC_ONLY"}
        critical_reports[head] = labels
    return {"stage_id": STAGE_ID, "comparator": "S1_FROZEN_SHARED_MATCHING_SEED", "per_head": reports, "critical_labels": critical_reports, "selected_candidate": False, "promotion": "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC", "triggered_heads": list(TRIGGERED_HEADS), "stability_gate_passed": all(float(aggregate["per_head"][head]["primary_macro_f1"]["sample_standard_deviation"]) <= 0.05 for head in TRIGGERED_HEADS)}


def _failure(root: Path | None, exc: ContractError, preflight: Mapping[str, Any], entered: bool) -> dict[str, Any]:
    result = {"status": "M2_S3_REJECTED_OR_BLOCKED_EVIDENCE", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": entered, "model_loaded": entered, "cache_accessed": entered, "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False}
    if root and root.exists():
        m1._json_dump(root / "blocked-evidence.json", result)
        result["rejected_content_address"] = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s3-rejected-artifact-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"], "complete_cache_snapshot": preflight["snapshot_identity"]})["content_address"]
    return result


def run_m2_s3(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path, *, s1_artifact: str | Path = DEFAULT_S1_ARTIFACT, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies, seed_executor: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / s1.CONTRACT_RELATIVE_PATH
    try:
        preflight = validate_s3_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
        s1_control = s2.load_s1_control(s1_artifact)
    except ContractError as exc:
        return {"status": "M2_S3_BLOCKED_FAIL_CLOSED", "phase": "TRAIN_DEV_TECHNICAL_PREFLIGHT", "blocker_codes": [exc.code], "training_invoked": False, "model_loaded": False, "cache_accessed": False, "output_created": False, "aggregate_created": False, "selected_candidate": False}
    root: Path | None = None
    results: list[Mapping[str, Any]] = []
    entered = False
    started = time.monotonic()
    try:
        runtime = runtime_loader()
        runtime_identity = s1.validate_runtime_identity(runtime, preflight["frozen_contract"])
        root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True)
        s1._limits(started, root, "before_fit")
        config = s2._config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        config["stage_id"] = STAGE_ID
        config["optimizer"] = {"name": "AdamW", "learning_rate": HEAD_LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "betas": config["optimizer"]["betas"], "epsilon": config["optimizer"]["epsilon"], "trainable_scope": "ONE_HEAD_PER_RUN"}
        config["early_stopping"] = {"metric": "TARGET_HEAD_PRIMARY_MACRO_F1", "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": config["stopping"]["minimum_delta"]}
        config["encoder_state"] = "FROZEN"
        config["triggered_heads"] = list(TRIGGERED_HEADS)
        config["promotion_rule"] = "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC"
        m1._json_dump(root / "training-config.json", config)
        executor = seed_executor or _seed
        for head in TRIGGERED_HEADS:
            for seed in SEEDS:
                entered = True
                results.append(executor(runtime=runtime, target_head=head, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=preflight["identity"], cache_dir=Path(cache_dir)))
        aggregate = aggregate_s3_results(results)
        m1._json_dump(root / "stage-aggregate.json", aggregate)
        report = matching_s1_report(preflight["frozen_contract"]["contract"], s1_control, results, aggregate)
        m1._json_dump(root / "s3-vs-s1-matching-seed-report.json", report)
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s3-artifact-manifest.v1", "stage_id": STAGE_ID, "diagnostic_only": True, "selected_candidate": False, "promotion": "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC", "triggered_heads": list(TRIGGERED_HEADS), "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"], "s1_control": {"content_address": s1_control["manifest"]["content_address"], "artifact_path": str(s1_control["root"])}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_identity": {"per_head": aggregate["devices"], "policy": "MPS_FIRST_CPU_FALLBACK"}, "training_config_sha256": sha256_file(root / "training-config.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "s3_vs_s1_report_sha256": sha256_file(root / "s3-vs-s1-matching-seed-report.json"), "seed_checkpoints": {f"{item['target_head']}:seed-{item['seed']}": item["checkpoint_sha256"] for item in results}})
        s1._limits(started, root, "after_final_manifest")
        return {"status": "M2_S3_DIAGNOSTIC_COMPLETED", "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "promotion": "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC", "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate, "matching_seed_report": report}
    except ContractError as exc:
        return _failure(root, exc, preflight, entered)
    except Exception as exc:
        return _failure(root, ContractError("M2_S3_RUNTIME_EXCEPTION", "S3 runtime/OOM exception", exception_type=type(exc).__name__, detail=str(exc)), preflight, entered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--s1-artifact", default=str(DEFAULT_S1_ARTIFACT))
    args = parser.parse_args(argv)
    result = run_m2_s3(args.config, args.output_dir, args.cache_dir, s1_artifact=args.s1_artifact)
    ok = result.get("status") == "M2_S3_DIAGNOSTIC_COMPLETED"
    (sys.stdout if ok else sys.stderr).write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
