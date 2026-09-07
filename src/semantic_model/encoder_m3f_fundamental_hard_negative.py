"""M3-F FUNDAMENTAL hard-negative, three-seed diagnostic.

Only the FUNDAMENTAL row of the frozen LERT reasoning output is trainable.
The reasoning specialist block and the other fourteen output rows are loaded
from the accepted seed-35 checkpoint and remain frozen.  This is a bounded
Train/Dev diagnostic; it never opens Test, Gold, Human100, OOD, or Shadow500.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import pickle
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import encoder_m1 as m1
from . import encoder_m2_lert_final as lert
from . import encoder_m2_s1 as s1
from . import encoder_m2_s3 as s3
from .errors import ContractError
from .hashes import sha256_file


STAGE_ID = "M3-F-FUNDAMENTAL-HARD-NEGATIVE-V1"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m3f-fundamental-hard-negative-run.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
TARGET_HEAD = "reasoning_tags"
TARGET_LABEL = "FUNDAMENTAL"
SEEDS = (35, 71, 107)
MAX_EPOCHS = 24
PATIENCE = 3
BATCH_SIZE = 16
THRESHOLD = 0.47
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
HARD_NEGATIVE_MULTIPLIER = 3.0
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
HARD_NEGATIVE_PATTERN = re.compile(r"基金|机构|外资|举牌|定增|股价")
INITIALIZATION_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-lert-final-specialist-seven-head-20260902"
INITIALIZATION_CHECKPOINT = INITIALIZATION_ARTIFACT / "reasoning_tags" / "seed-35" / "specialist-checkpoint.pt"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m3f-fundamental-hard-negative-three-seed-20260907"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")

OLD_DEV_PRECISION = 0.727273
OLD_DEV_RECALL = 0.666667
OLD_DEV_F1 = 0.695652
OLD_DEV_SUPPORT = 24
MIN_MEAN_F1 = 0.675652
MIN_MEAN_RECALL = 0.616667


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def is_hard_negative(record: Any) -> bool:
    """Return whether a Train record belongs to the fixed hard-negative set."""

    labels = getattr(record, "label", {})
    reasoning = labels.get(TARGET_HEAD, []) if isinstance(labels, Mapping) else []
    text = str(getattr(record, "model_text", ""))
    return TARGET_LABEL not in reasoning and HARD_NEGATIVE_PATTERN.search(text) is not None


def hard_negative_ids(records: Sequence[Any]) -> set[str]:
    return {str(record.sample_id) for record in records if is_hard_negative(record)}


def effective_sample_weight(record: Any) -> float:
    base = float(record.weights[TARGET_HEAD])
    return base * HARD_NEGATIVE_MULTIPLIER if is_hard_negative(record) else base


def fundamental_binary_metrics(probabilities: Sequence[float], truth: Sequence[int], *, threshold: float = THRESHOLD, boundary_ids: set[str] | None = None, sample_ids: Sequence[str] | None = None) -> dict[str, Any]:
    if len(probabilities) != len(truth):
        raise ValueError("probabilities and truth lengths differ")
    predicted = [int(float(value) >= threshold) for value in probabilities]
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(truth, predicted, strict=True))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(truth, predicted, strict=True))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(truth, predicted, strict=True))
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(truth, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    boundary_fp = None
    if boundary_ids is not None and sample_ids is not None:
        boundary_fp = sum(sample_id in boundary_ids and guess == 1 for sample_id, guess in zip(sample_ids, predicted, strict=True))
    return {
        "threshold": float(threshold),
        "rows": len(truth),
        "support": int(sum(truth)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "boundary_false_positives": boundary_fp,
        "finite": all(math.isfinite(float(value)) for value in probabilities),
    }


def make_fundamental_output_row(torch: Any, base_linear: Any, target_index: int) -> Any:
    """Replace a Linear head with one trainable row and frozen other rows."""

    nn = torch.nn
    functional = nn.functional

    class FundamentalOutputRow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target_index = int(target_index)
            self.register_buffer("frozen_weight", base_linear.weight.detach().clone())
            self.register_buffer("frozen_bias", base_linear.bias.detach().clone())
            self.weight = nn.Parameter(base_linear.weight.detach()[target_index : target_index + 1].clone())
            self.bias = nn.Parameter(base_linear.bias.detach()[target_index : target_index + 1].clone())

        def forward(self, values: Any) -> Any:
            frozen = functional.linear(values, self.frozen_weight, self.frozen_bias)
            target = functional.linear(values, self.weight, self.bias)
            left = frozen[:, : self.target_index]
            right = frozen[:, self.target_index + 1 :]
            return torch.cat((left, target, right), dim=1)

    return FundamentalOutputRow()


def trainable_row_identity(model: Any, target_head: str = TARGET_HEAD) -> dict[str, Any]:
    target = model.heads[target_head]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = {f"heads.{target_head}.weight", f"heads.{target_head}.bias"}
    _require(set(trainable) == expected, "M3F_TRAINABLE_SCOPE_INVALID", "only FUNDAMENTAL output row parameters may require gradients", observed=sorted(trainable), expected=sorted(expected))
    _require(hasattr(target, "frozen_weight") and hasattr(target, "frozen_bias"), "M3F_TRAINABLE_SCOPE_INVALID", "FUNDAMENTAL head must expose frozen non-target rows")
    _require(all(not parameter.requires_grad for parameter in model.encoder.parameters()), "M3F_TRAINABLE_SCOPE_INVALID", "LERT encoder must remain frozen")
    _require(all(not parameter.requires_grad for parameter in model.specialist_blocks[target_head].parameters()), "M3F_TRAINABLE_SCOPE_INVALID", "reasoning specialist block must remain frozen")
    return {"trainable_parameters": sorted(trainable), "target_head": target_head, "target_label": TARGET_LABEL, "frozen_specialist_block": True, "frozen_other_reasoning_rows": True}


def _load_initial_checkpoint(torch: Any, path: Path = INITIALIZATION_CHECKPOINT) -> dict[str, Any]:
    _require(path.is_file(), "M3F_INITIAL_CHECKPOINT_MISSING", "accepted seed-35 reasoning checkpoint is missing", path=str(path))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "M3F_INITIAL_CHECKPOINT_INVALID", "initial checkpoint must be a mapping")
    _require(checkpoint.get("seed") == 35 and checkpoint.get("head") == TARGET_HEAD, "M3F_INITIAL_CHECKPOINT_IDENTITY_INVALID", "initialization must be the accepted reasoning seed-35 checkpoint")
    frozen = checkpoint.get("frozen_config")
    _require(isinstance(frozen, Mapping) and frozen.get("model_id") == MODEL_ID and frozen.get("revision") == REVISION and frozen.get("license") == LICENSE, "M3F_INITIAL_CHECKPOINT_MODEL_INVALID", "initial checkpoint model identity differs")
    _require(isinstance(checkpoint.get("specialist_block_state_dict"), Mapping) and isinstance(checkpoint.get("head_state_dict"), Mapping), "M3F_INITIAL_CHECKPOINT_INVALID", "initial checkpoint lacks specialist block or head state")
    return dict(checkpoint)


def _build_model(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, dropout: float, initial: Mapping[str, Any]) -> Any:
    model = lert._make_model(torch, AutoModel, snapshot, schema, dropout)
    model.specialist_blocks[TARGET_HEAD].load_state_dict(initial["specialist_block_state_dict"])
    model.heads[TARGET_HEAD].load_state_dict(initial["head_state_dict"])
    target_index = list(schema.class_order[TARGET_HEAD]).index(TARGET_LABEL)
    model.heads[TARGET_HEAD] = make_fundamental_output_row(torch, model.heads[TARGET_HEAD], target_index)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.heads[TARGET_HEAD].weight.requires_grad = True
    model.heads[TARGET_HEAD].bias.requires_grad = True
    return model


def _set_training_mode(model: Any) -> None:
    model.train(True)
    model.encoder.eval()
    for block in model.specialist_blocks.values():
        block.eval()
    for head in model.heads.values():
        head.eval()
    model.dropout.train(True)


def _predict_probabilities(torch: Any, model: Any, tokenizer: Any, records: Sequence[Any], config: Mapping[str, Any], device: Any) -> list[float]:
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch_records = records[offset : offset + BATCH_SIZE]
            batch = m1._as_batch(torch, tokenizer, batch_records, config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[TARGET_HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "M3F_NONFINITE_LOGITS", "FUNDAMENTAL evaluation logits are non-finite")
            target_index = list(config["class_order"][TARGET_HEAD]).index(TARGET_LABEL)
            probabilities.extend(torch.sigmoid(logits[:, target_index]).detach().cpu().tolist())
    return probabilities


def _loss(torch: Any, logits: Any, batch: Mapping[str, Any], records: Sequence[Any], target_index: int) -> Any:
    labels = batch["labels"][TARGET_HEAD][:, target_index]
    base_weights = batch["weights"][TARGET_HEAD]
    multipliers = torch.tensor([HARD_NEGATIVE_MULTIPLIER if is_hard_negative(record) else 1.0 for record in records], dtype=torch.float32, device=labels.device)
    weights = base_weights * multipliers
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, target_index], labels, reduction="none")
    return (raw * weights).sum() / weights.sum().clamp_min(1e-12)


def _hash_non_target_rows(model: Any) -> str:
    digest = hashlib.sha256()
    head = model.heads[TARGET_HEAD]
    for name, value in (("frozen_weight", head.frozen_weight), ("frozen_bias", head.frozen_bias)):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(torch: Any, path: Path, *, model: Any, seed: int, config: Mapping[str, Any], provenance: Mapping[str, Any], parameter_identity: Mapping[str, Any], source_checkpoint_sha256: str) -> None:
    head = model.heads[TARGET_HEAD]
    payload = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "head": TARGET_HEAD,
        "target_label": TARGET_LABEL,
        "seed": seed,
        "frozen_config": config,
        "provenance": provenance,
        "parameter_identity": parameter_identity,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "specialist_block_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[TARGET_HEAD].state_dict().items()},
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "non_target_reasoning_rows_sha256": _hash_non_target_rows(model),
    }
    torch.save(payload, path)


def _cpu_reload_smoke(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, tokenizer: Any, config: Mapping[str, Any], checkpoint: Mapping[str, Any], record: Any, initial: Mapping[str, Any]) -> dict[str, Any]:
    model = _build_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]), initial)
    model.specialist_blocks[TARGET_HEAD].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[TARGET_HEAD].load_state_dict(checkpoint["head_state_dict"])
    model = model.to(torch.device("cpu")); model.eval()
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    target_index = list(config["class_order"][TARGET_HEAD]).index(TARGET_LABEL)
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])[TARGET_HEAD]
        probability = torch.sigmoid(logits[:, target_index])
    finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(probability).all().item())
    _require(finite, "M3F_CPU_RELOAD_FAILED", "CPU reload produced non-finite FUNDAMENTAL output")
    return {"required": True, "device": "cpu", "sample_id": record.sample_id, "finite": finite, "output_shape": list(logits.shape), "fundamental_probability": float(probability.item())}


def _seed_run(*, runtime: tuple[Any, Any, Any, Any], seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[Any], dev: Sequence[Any], config: Mapping[str, Any], provenance: Mapping[str, Any], initial: Mapping[str, Any], source_checkpoint_sha256: str, run_started: float, cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / TARGET_HEAD / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    s1._limits(run_started, root, "before_model_load", seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _build_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]), initial).to(device)
    parameter_identity = trainable_row_identity(model)
    target_index = list(config["class_order"][TARGET_HEAD]).index(TARGET_LABEL)
    trainable = [model.heads[TARGET_HEAD].weight, model.heads[TARGET_HEAD].bias]
    optimizer = torch.optim.AdamW(trainable, lr=HEAD_LR, weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best = -math.inf; best_epoch = 0; stale = 0; log_path = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        _set_training_mode(model)
        shuffled = list(train); random.Random(seed + epoch).shuffle(shuffled); losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            s1._limits(run_started, root, "during_epoch", seed)
            batch_records = shuffled[offset : offset + BATCH_SIZE]
            batch = m1._as_batch(torch, tokenizer, batch_records, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[TARGET_HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "M3F_NONFINITE_LOGITS", "FUNDAMENTAL training logits are non-finite", seed=seed, epoch=epoch)
            loss = _loss(torch, logits, batch, batch_records, target_index)
            _require(bool(torch.isfinite(loss).all().item()), "M3F_NONFINITE_LOSS", "hard-negative loss is non-finite", seed=seed, epoch=epoch)
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all().item()), "M3F_NONFINITE_GRADIENT", "FUNDAMENTAL gradient is non-finite", seed=seed, epoch=epoch)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_probabilities = _predict_probabilities(torch, model, tokenizer, dev, config, device)
        dev_truth = [int(TARGET_LABEL in record.label[TARGET_HEAD]) for record in dev]
        boundary_ids = {record.sample_id for record in dev if TARGET_LABEL not in record.label[TARGET_HEAD] and HARD_NEGATIVE_PATTERN.search(record.model_text)}
        dev_metric = fundamental_binary_metrics(dev_probabilities, dev_truth, threshold=THRESHOLD, boundary_ids=boundary_ids, sample_ids=[record.sample_id for record in dev])
        score = float(dev_metric["f1"]); improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            _save_checkpoint(torch, folder / "fundamental-hard-negative-checkpoint.pt", model=model, seed=seed, config=config, provenance=provenance, parameter_identity=parameter_identity, source_checkpoint_sha256=source_checkpoint_sha256)
        else:
            stale += 1
        row = {"seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_fundamental_precision": dev_metric["precision"], "dev_fundamental_recall": dev_metric["recall"], "dev_fundamental_f1": dev_metric["f1"], "dev_boundary_false_positives": dev_metric["boundary_false_positives"], "improved": improved, "stale_epochs": stale, "threshold": THRESHOLD}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        s1._limits(run_started, root, "after_epoch", seed)
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "fundamental-hard-negative-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M3F_CHECKPOINT_MISSING", "best FUNDAMENTAL checkpoint was not written", seed=seed)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_model = _build_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]), initial).to(device)
    best_model.specialist_blocks[TARGET_HEAD].load_state_dict(checkpoint["specialist_block_state_dict"])
    best_model.heads[TARGET_HEAD].load_state_dict(checkpoint["head_state_dict"])
    train_probabilities = _predict_probabilities(torch, best_model, tokenizer, train, config, device)
    dev_probabilities = _predict_probabilities(torch, best_model, tokenizer, dev, config, device)
    train_truth = [int(TARGET_LABEL in record.label[TARGET_HEAD]) for record in train]
    dev_truth = [int(TARGET_LABEL in record.label[TARGET_HEAD]) for record in dev]
    boundary_ids = {record.sample_id for record in dev if TARGET_LABEL not in record.label[TARGET_HEAD] and HARD_NEGATIVE_PATTERN.search(record.model_text)}
    train_metric = fundamental_binary_metrics(train_probabilities, train_truth, threshold=THRESHOLD)
    dev_metric = fundamental_binary_metrics(dev_probabilities, dev_truth, threshold=THRESHOLD, boundary_ids=boundary_ids, sample_ids=[record.sample_id for record in dev])
    smoke = _cpu_reload_smoke(torch, AutoModel, snapshot, schema, tokenizer, config, checkpoint, dev[0], initial)
    _require(checkpoint.get("non_target_reasoning_rows_sha256") == _hash_non_target_rows(best_model), "M3F_FROZEN_REASONING_ROWS_CHANGED", "non-FUNDAMENTAL reasoning rows changed")
    metrics = {"metric_scope": "WEAK_LABEL_TRAIN_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "head": TARGET_HEAD, "target_label": TARGET_LABEL, "seed": seed, "best_epoch": best_epoch, "early_stopping_metric": "FUNDAMENTAL_positive_F1_at_fixed_0.47", "sample_counts": {"train": len(train), "dev": len(dev)}, "train": {TARGET_LABEL: train_metric}, "dev": {TARGET_LABEL: dev_metric}, "hard_negative_count": sum(is_hard_negative(record) for record in train), "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER, "cpu_reload_inference_smoke": smoke, "non_target_reasoning_rows_unchanged": True}
    _json_dump(folder / "seed-metrics.json", metrics)
    resource = {"head": TARGET_HEAD, "target_label": TARGET_LABEL, "seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    _json_dump(folder / "resource-log.json", resource)
    s1._limits(run_started, root, "after_seed_evidence", seed)
    return {"head": TARGET_HEAD, "seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "non_target_reasoning_rows_sha256": checkpoint["non_target_reasoning_rows_sha256"]}


def _old_boundary_ids(dev: Sequence[Any]) -> set[str]:
    return {record.sample_id for record in dev if TARGET_LABEL not in record.label[TARGET_HEAD] and HARD_NEGATIVE_PATTERN.search(record.model_text)}


def _aggregate(results: Sequence[Mapping[str, Any]], old_boundary_fp: int) -> dict[str, Any]:
    _require([item.get("seed") for item in results] == list(SEEDS), "M3F_INCOMPLETE_SEEDS", "exact seeds 35/71/107 are required")
    precision = [float(item["metrics"]["dev"][TARGET_LABEL]["precision"]) for item in results]
    recall = [float(item["metrics"]["dev"][TARGET_LABEL]["recall"]) for item in results]
    f1 = [float(item["metrics"]["dev"][TARGET_LABEL]["f1"]) for item in results]
    boundary_fp = [int(item["metrics"]["dev"][TARGET_LABEL]["boundary_false_positives"]) for item in results]
    devices = [item["resource"]["actual_device"] for item in results]
    return {"stage_id": STAGE_ID, "head": TARGET_HEAD, "target_label": TARGET_LABEL, "seeds": list(SEEDS), "devices": devices, "mean": {"precision": statistics.mean(precision), "recall": statistics.mean(recall), "f1": statistics.mean(f1)}, "sample_standard_deviation": {"precision": statistics.stdev(precision), "recall": statistics.stdev(recall), "f1": statistics.stdev(f1)}, "worst_seed": {"precision": min(precision), "recall": min(recall), "f1": min(f1)}, "per_seed": {str(seed): {"precision": p, "recall": r, "f1": f, "boundary_false_positives": b} for seed, p, r, f, b in zip(SEEDS, precision, recall, f1, boundary_fp, strict=True)}, "old_dev": {"precision": OLD_DEV_PRECISION, "recall": OLD_DEV_RECALL, "f1": OLD_DEV_F1, "support": OLD_DEV_SUPPORT, "boundary_false_positives": old_boundary_fp}, "boundary_reduction_seeds": int(sum(value < old_boundary_fp for value in boundary_fp)), "other_14_reasoning_outputs_unchanged": all(item["metrics"].get("non_target_reasoning_rows_unchanged") is True for item in results), "cpu_reload_all_finite": all(item["metrics"]["cpu_reload_inference_smoke"].get("finite") is True for item in results)}


def _gate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    mean = aggregate["mean"]; old = aggregate["old_dev"]
    checks = {
        "mean_precision_above_old": float(mean["precision"]) > OLD_DEV_PRECISION,
        "mean_f1_at_least_floor": float(mean["f1"]) >= MIN_MEAN_F1,
        "mean_recall_at_least_floor": float(mean["recall"]) >= MIN_MEAN_RECALL,
        "at_least_two_boundary_reduction_seeds": int(aggregate["boundary_reduction_seeds"]) >= 2,
        "other_14_reasoning_outputs_unchanged": bool(aggregate["other_14_reasoning_outputs_unchanged"]),
        "cpu_reload_all_finite": bool(aggregate["cpu_reload_all_finite"]),
    }
    return {"stage_id": STAGE_ID, "target_label": TARGET_LABEL, "threshold": THRESHOLD, "passed": all(checks.values()), "checks": checks, "old_boundary_false_positives": old["boundary_false_positives"], "note": "With an old boundary false-positive count of zero, the reduction condition cannot pass; this is reported without altering the gate."}


def validate_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    frozen = lert._contract_requirements(contract_path)
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M3F_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, identity = _validate_lert_cache_snapshot(Path(cache_dir).resolve())
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": identity, "schema": schema, "train": train, "dev": dev, "worktree": str(Path(worktree).resolve()), "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version}}


def _validate_lert_cache_snapshot(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Validate the already-present fixed LERT snapshot without downloading."""

    relative = Path("official-snapshot") / REVISION
    _require(not cache_dir.is_absolute() or cache_dir.exists(), "M3F_CACHE_PATH_INVALID", "cache directory must be a local existing path")
    snapshot = (cache_dir / relative).resolve()
    _require(snapshot.is_dir(), "M3F_FIXED_LERT_CACHE_MISSING", "fixed LERT snapshot is missing", path=str(snapshot))
    required = ("config.json", "pytorch_model.bin", "added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
    rows: list[dict[str, Any]] = []
    for name in required:
        path = snapshot / name
        _require(path.is_file(), "M3F_FIXED_LERT_CACHE_FILE_MISSING", "required fixed LERT cache file is missing", path=name)
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    identity = {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "required_relative_directory": relative.as_posix(), "files": rows, "content_address": _sha256_json({"model_id": MODEL_ID, "revision": REVISION, "files": rows})}
    return snapshot, identity


def _validate_runtime_identity(runtime: tuple[Any, Any, Any, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Check the accepted package set while allowing a Python patch refresh."""

    np, torch, _model, _tokenizer = runtime
    expected = frozen["runtime_identity"]
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "tokenizers", "numpy", "huggingface-hub", "safetensors"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    observed_python = {"implementation": platform.python_implementation(), "version": platform.python_version()}
    expected_python = expected["python"]
    observed = {
        "python": observed_python,
        "packages": packages,
        "torch_runtime_version": getattr(torch, "__version__", None),
        "numpy_runtime_version": getattr(np, "__version__", None),
    }
    _require(observed_python["implementation"] == expected_python["implementation"], "M3F_RUNTIME_IDENTITY_MISMATCH", "Python implementation differs from accepted runtime", observed=observed, expected=expected)
    _require(tuple(int(part) for part in observed_python["version"].split(".")[:2]) == tuple(int(part) for part in expected_python["version"].split(".")[:2]), "M3F_RUNTIME_IDENTITY_MISMATCH", "Python major/minor differs from accepted runtime", observed=observed, expected=expected)
    _require(packages == expected["packages"], "M3F_RUNTIME_IDENTITY_MISMATCH", "Encoder package versions differ from accepted runtime", observed=observed, expected=expected)
    _require(observed["torch_runtime_version"] == expected["packages"]["torch"] and observed["numpy_runtime_version"] == expected["packages"]["numpy"], "M3F_RUNTIME_IDENTITY_MISMATCH", "runtime library version differs from accepted package set", observed=observed, expected=expected)
    return {"environment_sha256": expected["environment_sha256"], "observed": observed, "device_policy": expected["device_policy"], "python_patch_refresh_allowed": True}


def _failure(root: Path | None, exc: ContractError, entered: Sequence[int], preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "M3F_FUNDAMENTAL_HARD_NEGATIVE_REJECTED", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": bool(entered), "model_loaded": bool(entered), "cache_accessed": bool(preflight), "output_created": bool(root and root.exists()), "aggregate_created": False, "replacement_diagnostic": False, "selected_candidate": False, "seeds_started": list(entered)}
    if root and root.exists():
        _json_dump(root / "rejected-diagnostic-evidence.json", result)
        result["rejected_content_address"] = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m3f-rejected-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "selected_candidate": False})["content_address"]
    return result


def run_m3f(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path = DEFAULT_CACHE, *, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / CONTRACT_RELATIVE_PATH
    root: Path | None = None
    entered: list[int] = []
    preflight: dict[str, Any] | None = None
    run_started = time.monotonic()
    try:
        preflight = validate_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
        runtime = runtime_loader()
        runtime_identity = _validate_runtime_identity(runtime, preflight["frozen_contract"])
        initial_np, initial_torch, _, _ = runtime
        initial = _load_initial_checkpoint(initial_torch)
        source_sha = sha256_file(INITIALIZATION_CHECKPOINT)
        boundary_ids = _old_boundary_ids(preflight["dev"])
        # The fixed hard-negative definition is an explicit Train-only check.
        _require(len(hard_negative_ids(preflight["train"])) == 48, "M3F_HARD_NEGATIVE_COUNT_INVALID", "expected exactly 48 Train hard negatives", observed=len(hard_negative_ids(preflight["train"])))
        root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True, exist_ok=False)
        config = lert._config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        config.update({"stage_id": STAGE_ID, "target_head": TARGET_HEAD, "target_label": TARGET_LABEL, "max_length": 256, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "reasoning_probability_threshold": THRESHOLD, "optimizer": {"name": "AdamW", "learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY, "betas": config["optimizer"]["betas"], "epsilon": config["optimizer"]["epsilon"]}, "hard_negative_terms": ["基金", "机构", "外资", "举牌", "定增", "股价"], "hard_negative_count": 48, "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER, "trainable_scope": "reasoning_tags.FUNDAMENTAL_output_row_only", "frozen_scope": "LERT_encoder_reasoning_specialist_block_and_other_14_reasoning_rows"})
        provenance = {**preflight["identity"], "runtime_identity": runtime_identity, "cache_snapshot": preflight["snapshot_identity"], "initialization_artifact": str(INITIALIZATION_ARTIFACT), "initialization_checkpoint": str(INITIALIZATION_CHECKPOINT), "initialization_checkpoint_sha256": source_sha, "old_dev_boundary_sample_count": len(boundary_ids), "old_dev_boundary_false_positives": 0, "forbidden_inputs_not_read": ["Test", "Gold", "Human100", "OOD", "Shadow500", "LLM", "cloud", "production_data"]}
        _json_dump(root / "training-config.json", config); _json_dump(root / "provenance.json", provenance)
        results: list[Mapping[str, Any]] = []
        for seed in SEEDS:
            entered.append(seed)
            results.append(_seed_run(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, initial=initial, source_checkpoint_sha256=source_sha, run_started=run_started, cache_dir=Path(cache_dir)))
        aggregate = _aggregate(results, 0)
        gate = _gate(aggregate)
        status = "M3F_FUNDAMENTAL_REPLACEMENT_DIAGNOSTIC_PASSED" if gate["passed"] else "M3F_FUNDAMENTAL_HARD_NEGATIVE_REJECTED"
        _json_dump(root / "stage-aggregate.json", aggregate); _json_dump(root / "fundamental-replacement-gate.json", gate)
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m3f-fundamental-hard-negative-manifest.v1", "status": status, "stage_id": STAGE_ID, "target_head": TARGET_HEAD, "target_label": TARGET_LABEL, "selected_candidate": False, "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "initialization_checkpoint_sha256": source_sha, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "data_scope": "Train_1822_and_Dev_448_only", "hard_negative_count": 48, "hard_negative_multiplier": HARD_NEGATIVE_MULTIPLIER, "fixed_threshold": THRESHOLD, "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "gate_sha256": sha256_file(root / "fundamental-replacement-gate.json"), "seed_checkpoints": {str(item["seed"]): item["checkpoint_sha256"] for item in results}, "forbidden_inputs": provenance["forbidden_inputs_not_read"]})
        s1._limits(run_started, root, "after_final_manifest")
        return {"status": status, "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "replacement_diagnostic": gate["passed"], "selected_candidate": False, "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate, "gate": gate}
    except ContractError as exc:
        return _failure(root, exc, entered, preflight)
    except Exception as exc:
        return _failure(root, ContractError("M3F_RUNTIME_EXCEPTION", "M3-F runtime exception", exception_type=type(exc).__name__, detail=str(exc)), entered, preflight)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_m3f(args.config, args.output_dir, args.cache_dir)
    (sys.stdout if result.get("status") in {"M3F_FUNDAMENTAL_REPLACEMENT_DIAGNOSTIC_PASSED", "M3F_FUNDAMENTAL_HARD_NEGATIVE_REJECTED"} else sys.stderr).write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    return 0 if result.get("status") == "M3F_FUNDAMENTAL_REPLACEMENT_DIAGNOSTIC_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
