"""M2.2 Action hard-label balancing experiment.

This stage is deliberately limited to the ``action_tendency`` specialist.  It
uses the existing LERT specialist architecture and canonical Train/Dev loader,
but replaces the ordinary hard-label loss with a Train-derived class-weighted
hard cross entropy.  Human-review100 is opened only if the three-seed stability
gate passes, and is then a single post-selection diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file


STAGE_ID = "M2.2-ACTION-BALANCED-HARD-LABEL-THREE-SEED"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-2-action-balanced-run.v1"
MODEL_ID = m21.MODEL_ID
REVISION = m21.REVISION
LICENSE = m21.LICENSE
HEAD = "action_tendency"
SEEDS = (35, 71, 107)
MAX_LENGTH = 256
BATCH_SIZE = 16
MAX_EPOCHS = 24
PATIENCE = 3
BLOCK_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
MIN_MEAN_MACRO = 0.30
MAX_SAMPLE_STD = 0.05
MIN_WORST_SEED = 0.25
MIN_SUPPORTED_PREDICTED_CLASSES = 5
MAX_COLLAPSE_RATIO = 0.90

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-2-action-balanced-three-seed-20260904"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "manifests" / "encoder-m2-experiment-contract-v1.json"
HARD_LABEL_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-lert-classical-hybrid-seven-head-20260902-retry"
M21_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-1-semantic-softlabel-two-head-20260904"
CURRENT_CANDIDATE_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-head-selected-unified-candidate-20260902"
M3B_WORKBOOK = m21.M3B_WORKBOOK


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def train_class_counts(records: Sequence[m1.M1Record], order: Sequence[str]) -> dict[str, int]:
    counts = Counter(str(record.label[HEAD]) for record in records)
    return {label: int(counts.get(label, 0)) for label in order}


def class_weight_map(records: Sequence[m1.M1Record], order: Sequence[str]) -> dict[str, float]:
    """Compute the frozen sqrt-inverse-frequency Train-only class weights."""

    counts = train_class_counts(records, order)
    total = len(records)
    classes = len(order)
    result: dict[str, float] = {}
    for label in order:
        count = counts[label]
        raw = 4.0 if count == 0 else math.sqrt(total / (classes * count))
        result[label] = float(min(4.0, max(0.5, raw)))
    return result


def hard_weighted_loss(torch: Any, logits: Any, labels: Any, sample_weights: Any, class_weights: Any) -> Any:
    """Cross entropy weighted by Train-derived class weight and sample weight."""

    raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = raw * class_weights[labels] * sample_weights
    return weighted.sum() / sample_weights.sum().clamp_min(1e-12)


def prediction_balance_summary(probabilities: np.ndarray, truth: Sequence[str], order: Sequence[str]) -> dict[str, Any]:
    predicted = [order[int(index)] for index in np.argmax(probabilities, axis=1)]
    true_support = Counter(str(value) for value in truth)
    predicted_counts = {label: int(predicted.count(label)) for label in order}
    supported = sorted(label for label in order if predicted_counts[label] > 0 and true_support[label] > 0)
    dominant = max(predicted_counts.values(), default=0)
    ratio = float(dominant / len(predicted)) if predicted else 0.0
    return {
        "predicted_class_counts": predicted_counts,
        "supported_predicted_classes": supported,
        "supported_predicted_class_count": len(supported),
        "dominant_prediction_ratio": ratio,
        "collapse_detected": ratio > MAX_COLLAPSE_RATIO,
    }


def three_seed_gate(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(item["dev"]["macro_f1"]) for item in per_seed]
    mean = float(statistics.fmean(values)) if values else 0.0
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    worst = float(min(values)) if values else 0.0
    support_counts = {str(item["seed"]): int(item["balance"]["supported_predicted_class_count"]) for item in per_seed}
    collapse = {str(item["seed"]): bool(item["balance"]["collapse_detected"]) for item in per_seed}
    checks = {
        "mean_macro_f1_at_least_0_30": mean >= MIN_MEAN_MACRO,
        "sample_std_at_most_0_05": std <= MAX_SAMPLE_STD,
        "worst_seed_at_least_0_25": worst >= MIN_WORST_SEED,
        "each_seed_predicts_at_least_5_supported_classes": all(value >= MIN_SUPPORTED_PREDICTED_CLASSES for value in support_counts.values()),
        "no_seed_over_90_percent_single_class": all(not value for value in collapse.values()),
    }
    return {"passed": bool(checks and all(checks.values())), "checks": checks, "mean_macro_f1": mean, "sample_standard_deviation": std, "worst_seed": worst, "supported_predicted_class_count": support_counts, "collapse_detected": collapse}


def _contract() -> dict[str, Any]:
    frozen = lert._contract_requirements(CONTRACT_PATH)
    contract = frozen["contract"]
    _require(contract["lert_final_specialist_candidate_contract"]["model"]["model_id"] == MODEL_ID, "M2_2_MODEL_CONTRACT_INVALID", "LERT model id changed")
    _require(contract["lert_final_specialist_candidate_contract"]["model"]["revision"] == REVISION, "M2_2_MODEL_CONTRACT_INVALID", "LERT revision changed")
    training = contract["lert_final_specialist_candidate_contract"]["training"]
    _require(training["seeds"] == list(SEEDS) and training["max_length"] == MAX_LENGTH and training["batch_size"] == BATCH_SIZE and training["max_epochs"] == MAX_EPOCHS and training["patience"] == PATIENCE and training["gradient_clip_norm"] == GRADIENT_CLIP, "M2_2_TRAINING_CONTRACT_INVALID", "frozen training configuration changed")
    return frozen


def _preflight(config_path: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    frozen = _contract()
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_2_DATA_SCOPE_INVALID", "expected Train 1,822 and Dev 448")
    snapshot, identity = m21._snapshot(Path(cache_dir).resolve())
    return {"frozen_contract": frozen, "config": config, "schema": schema, "train": train, "dev": dev, "snapshot": snapshot, "snapshot_identity": identity}


def _config(frozen: Mapping[str, Any], schema: Any, counts: Mapping[str, int], weights: Mapping[str, float]) -> dict[str, Any]:
    config = lert._config(frozen["contract"], schema.class_order)
    config.update({"stage_id": STAGE_ID, "run_schema_version": RUN_SCHEMA_VERSION, "head": HEAD, "seed_order": list(SEEDS), "loss": "hard_label_cross_entropy_with_train_only_class_weight_and_sample_weight", "class_weight_rule": {"formula": "clip(sqrt(N/(K*count[c])), 0.5, 4.0)", "population": "Train_1822_only", "counts": dict(counts), "weights": dict(weights)}, "selection_gate": {"mean_macro_f1_min": MIN_MEAN_MACRO, "sample_std_max": MAX_SAMPLE_STD, "worst_seed_min": MIN_WORST_SEED, "min_supported_predicted_classes": MIN_SUPPORTED_PREDICTED_CLASSES, "max_single_class_prediction_ratio": MAX_COLLAPSE_RATIO}, "human100_role": "one_post_three_seed_gate_diagnostic_only", "selected_candidate": False})
    return config


def _seed_train(*, runtime: tuple[Any, Any, Any, Any], seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], counts: Mapping[str, int], weights: Mapping[str, float], cache_dir: Path, experiment_started: float) -> dict[str, Any]:
    np_module, torch, AutoModel, AutoTokenizer = runtime
    seed_started = time.monotonic()
    folder = root / HEAD / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    m21._set_seed(torch, np_module, seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.set_active_head(HEAD)
    model = model.to(device)
    parameter_identity = m21._validate_trainable(model, HEAD)
    trainable_block = list(model.specialist_blocks[HEAD].parameters())
    trainable_head = list(model.heads[HEAD].parameters())
    trainable = [*trainable_block, *trainable_head]
    class_weight_tensor = torch.tensor([float(weights[label]) for label in schema.class_order[HEAD]], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([{"params": trainable_block, "lr": BLOCK_LR}, {"params": trainable_head, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best_score, best_epoch, stale = -math.inf, 0, 0
    log_path = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(True)
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            _limits(experiment_started, root, "during_fit", seed)
            batch_records = shuffled[offset:offset + BATCH_SIZE]
            batch = m1._as_batch(torch, tokenizer, batch_records, config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[HEAD]
            _require(bool(torch.isfinite(logits).all().item()), "M2_2_NONFINITE_LOGITS", "non-finite Action logits", seed=seed)
            loss = hard_weighted_loss(torch, logits, batch["labels"][HEAD], batch["weights"][HEAD], class_weight_tensor)
            _require(bool(torch.isfinite(loss).item()), "M2_2_NONFINITE_LOSS", "non-finite weighted hard-label loss", seed=seed)
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all().item()), "M2_2_NONFINITE_GRADIENT", "non-finite Action gradient", seed=seed)
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_prob = m21._predict(torch, model, tokenizer, dev, config, HEAD, device)
        dev_metrics = m21.strict_and_semantic_metrics(dev_prob, [str(record.label[HEAD]) for record in dev], list(schema.class_order[HEAD]))
        score = float(dev_metrics["macro_f1"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": HEAD, "seed": seed, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "frozen_config": config, "class_weight_rule": config["class_weight_rule"], "parameter_identity": parameter_identity, "specialist_block_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[HEAD].state_dict().items()}, "head_state_dict": {key: value.detach().cpu() for key, value in model.heads[HEAD].state_dict().items()}}, folder / "balanced-hard-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log_path, {"head": HEAD, "seed": seed, "epoch": epoch, "train_weighted_hard_loss": round(sum(losses) / len(losses), 6), "dev": dev_metrics, "early_stopping_metric": "strict_macro_f1", "improved": improved, "stale_epochs": stale})
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "balanced-hard-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_2_CHECKPOINT_MISSING", "balanced hard-label checkpoint missing", seed=seed)
    checkpoint = m21._load_weights(checkpoint_path, torch)
    best_model = m21._model_for_checkpoint(torch, AutoModel, snapshot, schema, config, HEAD, checkpoint, device)
    m21._validate_trainable(best_model, HEAD)
    train_prob = m21._predict(torch, best_model, tokenizer, train, config, HEAD, device)
    dev_prob = m21._predict(torch, best_model, tokenizer, dev, config, HEAD, device)
    order = list(schema.class_order[HEAD])
    train_metrics = m21.strict_and_semantic_metrics(train_prob, [str(record.label[HEAD]) for record in train], order)
    dev_metrics = m21.strict_and_semantic_metrics(dev_prob, [str(record.label[HEAD]) for record in dev], order)
    balance = prediction_balance_summary(dev_prob, [str(record.label[HEAD]) for record in dev], order)
    cpu_model = m21._model_for_checkpoint(torch, AutoModel, snapshot, schema, config, HEAD, checkpoint, torch.device("cpu"))
    cpu_model.eval()
    cpu_batch = m1._as_batch(torch, tokenizer, [dev[0]], config, torch.device("cpu"))
    with torch.no_grad():
        cpu_logits = cpu_model(cpu_batch["input_ids"], cpu_batch["attention_mask"])[HEAD]
    cpu_finite = bool(torch.isfinite(cpu_logits).all().item())
    _require(cpu_finite, "M2_2_CPU_RELOAD_FAILED", "CPU reload emitted non-finite Action logits", seed=seed)
    metrics = {"head": HEAD, "seed": seed, "best_epoch": best_epoch, "early_stopping_metric": "strict_macro_f1", "early_stopping_score": best_score, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "class_weight_rule": {"formula": "clip(sqrt(N/(K*count[c])), 0.5, 4.0)", "population": "Train_1822_only", "counts": dict(counts), "weights": dict(weights)}, "prediction_balance": balance, "cpu_reload": {"required": True, "device": "cpu", "finite": cpu_finite, "output_shape": list(cpu_logits.shape)}}
    _dump(folder / "seed-metrics.json", metrics)
    resource = {"head": HEAD, "seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - seed_started, 3), "experiment_elapsed_seconds": round(time.monotonic() - experiment_started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": metrics["cpu_reload"], "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    _dump(folder / "resource-log.json", resource)
    _limits(experiment_started, root, "after_seed", seed)
    return {"head": HEAD, "seed": seed, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "metrics": metrics, "dev": dev_metrics, "balance": balance, "resource": resource}


def _reference_comparison(schema: Any, dev: Sequence[m1.M1Record], prep: Any) -> dict[str, Any]:
    order = list(schema.class_order[HEAD])
    hard: dict[str, Any] = {}
    for seed in SEEDS:
        path = HARD_LABEL_ARTIFACT / HEAD / f"seed-{seed}" / "seed-metrics.json"
        _require(path.is_file(), "M2_2_HARD_REFERENCE_MISSING", "existing hard-label LERT reference is missing", seed=seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        hard[str(seed)] = {"artifact": str(path), "sha256": sha256_file(path), "best_epoch": payload.get("best_epoch"), "dev": payload["dev"][HEAD], "resource": json.loads((path.parent / "resource-log.json").read_text(encoding="utf-8"))}
    soft_path = M21_ARTIFACT / HEAD / "seed-35" / "seed-metrics.json"
    _require(soft_path.is_file(), "M2_2_SOFT_REFERENCE_MISSING", "M2.1 soft-label Action reference is missing")
    soft_payload = json.loads(soft_path.read_text(encoding="utf-8"))
    records = [{"sample_id": record.sample_id, "stock_code": record.stock_code, "stock_name": record.stock_name, "model_text": record.model_text} for record in dev]
    classical_prob = m21._classical_probabilities(records, schema, prep, CURRENT_CANDIDATE_ARTIFACT / "classical-fallback/original_model_v0.3.5.joblib")[HEAD]
    classical = m21.strict_and_semantic_metrics(classical_prob, [str(record.label[HEAD]) for record in dev], order)
    return {"classical": {"source": "Classical frozen estimator; Dev_448 recomputation", "dev": classical}, "hard_label_lert": hard, "m2_1_soft_label": {"artifact": str(soft_path), "sha256": sha256_file(soft_path), "dev": soft_payload["dev"]}}


def _human_records(rows: Sequence[Mapping[str, str]], schema: Any) -> list[m1.M1Record]:
    return [m1.M1Record(sample_id=row["sample_id"], stock_code=row.get("stock_code", ""), stock_name=row.get("stock_name", ""), model_text=row["model_text"], label={head: schema.class_order[head][0] for head in schema.class_order} | {"reasoning_tags": [schema.class_order["reasoning_tags"][0]]}, weights={head: 1.0 for head in schema.class_order}) for row in rows]


def _human_action_metrics(rows: Sequence[Mapping[str, str]], candidate_prob: np.ndarray, classical_prob: np.ndarray, schema: Any) -> dict[str, Any]:
    order = list(schema.class_order[HEAD])
    truth = [str(row[f"human_{HEAD}"]) for row in rows]
    candidate = m21.strict_and_semantic_metrics(candidate_prob, truth, order)
    classical = m21.strict_and_semantic_metrics(classical_prob, truth, order)
    candidate_labels = [order[int(index)] for index in np.argmax(candidate_prob, axis=1)]
    classical_labels = [order[int(index)] for index in np.argmax(classical_prob, axis=1)]
    candidate_correct = np.asarray(candidate_labels) == np.asarray(truth)
    classical_correct = np.asarray(classical_labels) == np.asarray(truth)
    return {"rows": len(rows), "scope": "human_reviewed_Gold100_post_three_seed_gate_only", "candidate": candidate, "classical": classical, "candidate_minus_classical": {key: candidate[key] - classical[key] for key in ("accuracy", "macro_f1", "semantic_top1_score", "expected_semantic_score", "true_label_probability")}, "paired_outcomes": {"candidate_only_correct": int(np.sum(candidate_correct & ~classical_correct)), "classical_only_correct": int(np.sum(~candidate_correct & classical_correct)), "both_correct": int(np.sum(candidate_correct & classical_correct)), "both_wrong": int(np.sum(~candidate_correct & ~classical_correct))}}


def _assemble_unified_candidate(root: Path, balanced: Mapping[str, Any], runtime: tuple[Any, Any, Any, Any], snapshot_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Create a diagnostic-only seed-35 unified bundle without changing old artifacts."""

    parent_bundle_path = CURRENT_CANDIDATE_ARTIFACT / "unified-runtime-bundle.pt"
    _require(parent_bundle_path.is_file(), "M2_2_PARENT_BUNDLE_MISSING", "current unified candidate bundle is missing")
    candidate_root = root / "unified-candidate"
    candidate_root.mkdir(parents=True, exist_ok=False)
    classical_dir = candidate_root / "classical-fallback"
    classical_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(CURRENT_CANDIDATE_ARTIFACT / "classical-fallback/original_model_v0.3.5.joblib", classical_dir / "original_model_v0.3.5.joblib")
    np_module, torch, _AutoModel, _AutoTokenizer = runtime
    parent_bundle = torch.load(parent_bundle_path, map_location="cpu", weights_only=False)
    bundle = copy.deepcopy(parent_bundle)
    checkpoint_path = Path(str(balanced["checkpoint"]))
    checkpoint = m21._load_weights(checkpoint_path, torch)
    bundle["stage"] = STAGE_ID
    bundle["status"] = "M2_2_DIAGNOSTIC_UNIFIED_CANDIDATE"
    bundle["head_sources"] = dict(bundle["head_sources"])
    bundle["head_sources"][HEAD] = "LERT balanced hard-label seed-35"
    bundle["lert_specialist_block_state_dict"] = dict(bundle["lert_specialist_block_state_dict"])
    bundle["lert_heads_state_dict"] = dict(bundle["lert_heads_state_dict"])
    bundle["lert_specialist_block_state_dict"][HEAD] = checkpoint["specialist_block_state_dict"]
    bundle["lert_heads_state_dict"][HEAD] = checkpoint["head_state_dict"]
    bundle["classical_fallback"] = dict(bundle["classical_fallback"])
    bundle["classical_fallback"]["heads"] = ["context_dependency"]
    if isinstance(bundle["classical_fallback"].get("dev_prediction_table"), dict):
        bundle["classical_fallback"]["dev_prediction_table"] = {sample_id: {key: value for key, value in values.items() if key != HEAD} for sample_id, values in bundle["classical_fallback"]["dev_prediction_table"].items()}
    bundle["reasoning_thresholds"] = dict(bundle["reasoning_thresholds"])
    bundle["reasoning_thresholds"]["SARCASM_IRONY"] = 0.45
    bundle["action_balanced_checkpoint_sha256"] = sha256_file(checkpoint_path)
    bundle["selected_candidate"] = False
    bundle["forbidden_inputs"] = ["Test", "Anchor", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_inference"]
    bundle_path = candidate_root / "unified-runtime-bundle.pt"
    torch.save(bundle, bundle_path)
    selection = json.loads((CURRENT_CANDIDATE_ARTIFACT / "selection-config.json").read_text(encoding="utf-8"))
    selection["stage"] = STAGE_ID
    selection["status"] = "M2_2_DIAGNOSTIC_UNIFIED_CANDIDATE"
    selection["selected_seed"] = 35
    selection["head_sources"] = dict(selection["head_sources"])
    selection["head_sources"][HEAD] = "LERT balanced hard-label seed-35"
    selection["thresholds"] = dict(selection["thresholds"])
    selection["thresholds"]["SARCASM_IRONY"] = 0.45
    selection["action_balanced_checkpoint"] = {"relative_path": "../action_tendency/seed-35/balanced-hard-checkpoint.pt", "sha256": sha256_file(checkpoint_path)}
    selection["selected_candidate"] = False
    _dump(candidate_root / "selection-config.json", selection)
    runtime_evidence = {"status": "DIAGNOSTIC_ONLY", "cpu_reload": {"new_action_tendency": balanced["metrics"]["cpu_reload"], "existing_lert_heads_reused": True, "classical_fallback_heads": ["context_dependency"], "sarcasm_irony_threshold": 0.45}, "bundle_sha256": sha256_file(bundle_path), "classical_estimator_sha256": sha256_file(classical_dir / "original_model_v0.3.5.joblib"), "snapshot_content_address": snapshot_identity["content_address"], "selected_candidate": False}
    _dump(candidate_root / "runtime-load-evidence.json", runtime_evidence)
    manifest = m1._write_content_manifest(candidate_root, {"manifest_schema_version": "myresearcher.m2-2-diagnostic-unified-candidate-manifest.v1", "stage_id": STAGE_ID, "status": "M2_2_DIAGNOSTIC_UNIFIED_CANDIDATE", "selected_candidate": False, "head_sources": selection["head_sources"], "sarcasm_irony_classical_threshold": 0.45, "action_balanced_checkpoint_sha256": sha256_file(checkpoint_path), "parent_candidate_content_address": json.loads((CURRENT_CANDIDATE_ARTIFACT / "content-addressed-manifest.json").read_text(encoding="utf-8"))["content_address"], "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "forbidden_inputs": bundle["forbidden_inputs"]})
    return {"path": str(candidate_root), "content_address": manifest["content_address"], "bundle_path": str(bundle_path), "bundle_sha256": sha256_file(bundle_path), "selection_config": str(candidate_root / "selection-config.json"), "runtime_load_evidence": str(candidate_root / "runtime-load-evidence.json")}


def _limits(started: float, root: Path, phase: str, seed: int | None = None) -> None:
    _require(time.monotonic() - started <= MAX_WALL_TIME_SECONDS, "M2_2_WALL_TIME_LIMIT_EXCEEDED", "two-hour experiment limit exceeded", phase=phase, seed=seed)
    _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_2_DISK_LIMIT_EXCEEDED", "ten GiB experiment output limit exceeded", phase=phase, seed=seed)


def run_balanced(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path = DEFAULT_CACHE) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    _require(not root.exists(), "M2_2_OUTPUT_ALREADY_EXISTS", "output directory is immutable and must not already exist", output_dir=str(root))
    experiment_started = time.monotonic()
    preflight = _preflight(config_path, cache_dir)
    schema, train, dev = preflight["schema"], preflight["train"], preflight["dev"]
    order = list(schema.class_order[HEAD])
    counts = train_class_counts(train, order)
    weights = class_weight_map(train, order)
    root.mkdir(parents=True, exist_ok=False)
    runtime = m21._load_runtime()
    runtime_identity = {"python": platform.python_version(), "torch": runtime[1].__version__, "transformers": __import__("transformers").__version__, "tokenizers": __import__("tokenizers").__version__, "numpy": runtime[0].__version__, "device_policy": "MPS_FIRST_CPU_FALLBACK"}
    config = _config(preflight["frozen_contract"], schema, counts, weights)
    _dump(root / "training-config.json", config)
    _dump(root / "runtime.json", runtime_identity)
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        results.append(_seed_train(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=schema, train=train, dev=dev, config=config, counts=counts, weights=weights, cache_dir=Path(cache_dir).resolve(), experiment_started=experiment_started))
    aggregate = {"stage_id": STAGE_ID, "head": HEAD, "seeds": list(SEEDS), "run_units_complete": len(results) == 3, "per_seed": [{"seed": item["seed"], "best_epoch": item["metrics"]["best_epoch"], "dev": item["dev"], "balance": item["balance"], "resource": item["resource"]} for item in results]}
    gate = three_seed_gate(aggregate["per_seed"])
    aggregate["gate"] = gate
    _dump(root / "stage-aggregate.json", aggregate)
    prep = m21.PreprocessingContract.load(preflight["config"].data_path("preprocessing_contract"))
    comparison = _reference_comparison(schema, dev, prep)
    comparison["balanced_hard_label"] = {str(item["seed"]): {"checkpoint_sha256": item["checkpoint_sha256"], "best_epoch": item["metrics"]["best_epoch"], "dev": item["dev"], "balance": item["balance"]} for item in results}
    _dump(root / "comparisons.json", comparison)
    human_metrics: dict[str, Any] | None = None
    unified: dict[str, Any] | None = None
    if gate["passed"]:
        rows = m21._read_human_workbook(M3B_WORKBOOK)
        human_records = _human_records(rows, schema)
        cp = m21._load_weights(Path(results[0]["checkpoint"]), runtime[1])
        tokenizer = runtime[3].from_pretrained(str(preflight["snapshot"]), local_files_only=True, trust_remote_code=False, use_fast=True)
        model = m21._model_for_checkpoint(runtime[1], runtime[2], preflight["snapshot"], schema, config, HEAD, cp, runtime[1].device("cpu"))
        candidate_prob = m21._predict(runtime[1], model, tokenizer, human_records, config, HEAD, runtime[1].device("cpu"))
        classical_prob = m21._classical_probabilities([{"sample_id": row["sample_id"], "stock_code": row.get("stock_code", ""), "stock_name": row.get("stock_name", ""), "model_text": row["model_text"]} for row in rows], schema, prep, CURRENT_CANDIDATE_ARTIFACT / "classical-fallback/original_model_v0.3.5.joblib")[HEAD]
        human_metrics = _human_action_metrics(rows, candidate_prob, classical_prob, schema)
        _dump(root / "human100-metrics.json", human_metrics)
        unified = _assemble_unified_candidate(root, results[0], runtime, preflight["snapshot_identity"])
    status = "ACTION_BALANCED_THREE_SEED_GATE_PASSED" if gate["passed"] else "ACTION_BALANCING_NO_STABLE_GAIN"
    payload = {"manifest_schema_version": "myresearcher.encoder-m2-2-action-balanced-artifact-manifest.v1", "stage_id": STAGE_ID, "status": status, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "head": HEAD, "seeds": list(SEEDS), "class_weight_rule": config["class_weight_rule"], "data_scope": "Train_1822_fit_and_Dev_448_early_stopping_diagnostic_only", "runtime_identity": runtime_identity, "cache_snapshot": preflight["snapshot_identity"], "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "comparisons_sha256": sha256_file(root / "comparisons.json"), "selected_candidate": False, "human100_evaluated": human_metrics is not None, "unified_candidate": unified, "forbidden_inputs_not_read": ["Test", "Anchor", "Gold", "OOD", "LLM", "cloud", "production"], "m2_1_softlabel_status": "SOFTLABEL_NO_GAIN_VS_HARD_LABEL_COMPARATOR"}
    manifest = m1._write_content_manifest(root, payload)
    _limits(experiment_started, root, "after_manifest")
    return {"status": status, "output_dir": str(root), "content_address": manifest["content_address"], "gate": gate, "results": results, "comparison": comparison, "human100": human_metrics, "unified_candidate": unified, "runtime": runtime_identity}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=m21.CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_balanced(args.config, args.output_dir, args.cache_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] == "ACTION_BALANCED_THREE_SEED_GATE_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
