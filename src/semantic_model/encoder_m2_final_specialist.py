"""M2 final RBT3 candidate: seven independent last-block specialists.

The runner is deliberately a single executable stage.  It trains reasoning
first and fails fast before any other head when the frozen Classical gate is
not met.  All data access is through the immutable Train/Dev loader; the
sealed roles are never opened.
"""

from __future__ import annotations

import argparse
import copy
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
from . import encoder_m2_reasoning_r2 as r2
from . import encoder_m2_s1 as s1
from . import encoder_m2_s3 as s3
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import SINGLE_LABEL_HEADS, V1_HEADS


STAGE_ID = "M2-FINAL-SPECIALIST-SEVEN-HEAD-CANDIDATE"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-final-specialist-run.v1"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")
MODEL_ID = "hfl/rbt3"
REVISION = "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"
LICENSE = "Apache-2.0"
SEEDS = (35, 71, 107)
REASONING_HEAD = "reasoning_tags"
SPECIALIST_HEADS = tuple(V1_HEADS)
MAX_EPOCHS = 24
PATIENCE = 3
BATCH_SIZE = 16
MAX_LENGTH = 256
BLOCK_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-final-specialist-seven-head-20260901"


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
        raise ContractError(code, "required JSON is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), code, "required JSON root must be an object")
    return value


def _plain_json(value: Any) -> Any:
    """Keep checkpoint provenance safe for PyTorch weights-only loading."""

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _contract_requirements(contract_path: str | Path) -> dict[str, Any]:
    frozen = s1._contract_requirements(contract_path)
    final = _mapping(frozen["contract"].get("final_specialist_candidate_contract"), "M2_FINAL_CONTRACT_INVALID", "final specialist contract")
    model = _mapping(final.get("model"), "M2_FINAL_CONTRACT_INVALID", "model")
    _require(final.get("contract_id") == "M2_FINAL_SPECIALIST_SEVEN_HEAD_V1", "M2_FINAL_CONTRACT_INVALID", "wrong final contract id")
    _require(model.get("model_id") == MODEL_ID and model.get("revision") == REVISION and model.get("license") == LICENSE, "M2_FINAL_MODEL_CONTRACT_INVALID", "model identity changed")
    _require(model.get("local_files_only") is True and model.get("trust_remote_code") is False and "FORBIDDEN" in str(model.get("warm_start")), "M2_FINAL_MODEL_CONTRACT_INVALID", "local fixed original initialization required")
    architecture = _mapping(final.get("architecture"), "M2_FINAL_CONTRACT_INVALID", "architecture")
    _require(architecture.get("heads") == list(V1_HEADS) and architecture.get("unified_runtime") == "one_input_returns_all_seven_heads", "M2_FINAL_ARCHITECTURE_INVALID", "seven independent specialists are required")
    data = _mapping(final.get("data_scope"), "M2_FINAL_CONTRACT_INVALID", "data scope")
    _require(data.get("train_rows") == 1822 and data.get("dev_rows") == 448 and data.get("train_role") == "fit_only" and data.get("dev_role") == "early_stopping_and_diagnostic_only", "M2_FINAL_DATA_CONTRACT_INVALID", "Train/Dev scope changed")
    training = _mapping(final.get("training"), "M2_FINAL_CONTRACT_INVALID", "training")
    expected = {"seeds": list(SEEDS), "max_length": MAX_LENGTH, "truncation": "HEAD_TAIL", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "gradient_clip_norm": 1.0}
    _require(all(training.get(key) == value for key, value in expected.items()), "M2_FINAL_TRAINING_CONTRACT_INVALID", "fixed training value changed")
    optimizer = _mapping(training.get("optimizer"), "M2_FINAL_CONTRACT_INVALID", "optimizer")
    _require(optimizer.get("name") == "AdamW" and optimizer.get("block_learning_rate") == BLOCK_LR and optimizer.get("head_learning_rate") == HEAD_LR and optimizer.get("weight_decay") == WEIGHT_DECAY, "M2_FINAL_OPTIMIZER_CONTRACT_INVALID", "optimizer changed")
    _require(training.get("device_policy") == "MPS_FIRST_CPU_FALLBACK" and training.get("per_run_wall_time_seconds") == 7200 and training.get("total_new_disk_gib") == 10, "M2_FINAL_RESOURCE_CONTRACT_INVALID", "resource policy changed")
    _require(_mapping(final.get("selection"), "M2_FINAL_CONTRACT_INVALID", "selection").get("predeclared_m3_seed") == 35, "M2_FINAL_SELECTION_CONTRACT_INVALID", "seed 35 must be predeclared")
    return {**frozen, "final": final}


def validate_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    """Technical Train/Dev-only preflight; it intentionally does no canonical audit."""

    frozen = _contract_requirements(contract_path)
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_FINAL_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, identity = s1.validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": identity, "schema": schema, "train": train, "dev": dev, "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version, "data_scope": "Train_1822_and_Dev_448_only"}, "worktree": str(Path(worktree).resolve())}


def _config(contract: Mapping[str, Any], order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract["frozen_input_and_common_training_configuration"], "M2_FINAL_CONTRACT_INVALID", "common configuration")
    optimizer = _mapping(common["optimizer"], "M2_FINAL_CONTRACT_INVALID", "optimizer")
    return {"stage_id": STAGE_ID, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "local_files_only": True, "trust_remote_code": False, "initialization_source": "OFFICIAL_FIXED_RBT3_ORIGINAL_WEIGHTS_NO_WARM_START", "shared_components": ["tokenizer", "input_builder", "embeddings", "transformer_blocks_0_and_1"], "per_head_components": "independent_transformer_block_2_and_output_head", "max_length": MAX_LENGTH, "truncation": "HEAD_TAIL", "padding": common["padding"], "token_type_ids": "NOT_EMITTED", "batch_size": BATCH_SIZE, "head_dropout": common["head_dropout"], "stock_code_token_cap": common["stock_code_token_cap"], "stock_name_token_cap": common["stock_name_token_cap"], "class_order": {head: list(order[head]) for head in V1_HEADS}, "optimizer": {"name": "AdamW", "block_learning_rate": BLOCK_LR, "head_learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY, "betas": optimizer["betas"], "epsilon": optimizer["epsilon"]}, "stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": 0.0}, "gradient_clipping_max_norm": 1.0, "reasoning_probability_threshold": 0.5, "fit_population": "Train_1822_only", "dev_role": "early_stopping_and_diagnostic_only", "sealed_roles_not_loaded": ["Test", "Anchor", "Gold", "OOD", "reference_predictions"], "selected_candidate": False}


def _set_seed(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def _locate_last_block(encoder: Any) -> tuple[str, Any]:
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    _require(layers is not None and len(layers) >= 3, "M2_FINAL_ENCODER_ARCHITECTURE_INVALID", "RBT3 must expose at least three Transformer blocks")
    return "encoder.encoder.layer.2", layers[2]


def _make_model(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, dropout: float) -> Any:
    """Construct shared bottom layers plus seven independently cloned final blocks."""

    nn = torch.nn

    class SpecialistModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
            hidden_size = int(self.encoder.config.hidden_size)
            _, original_last = _locate_last_block(self.encoder)
            self.dropout = nn.Dropout(dropout)
            self.specialist_blocks = nn.ModuleDict({head: copy.deepcopy(original_last) for head in V1_HEADS})
            self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, len(schema.class_order[head])) for head in V1_HEADS})
            self.last_transformer_block_prefix = "encoder.encoder.layer.2"
            self.active_head: str | None = None
            for block in self.specialist_blocks.values():
                for parameter in block.parameters():
                    parameter.requires_grad = False
            for head, classifier in self.heads.items():
                for parameter in classifier.parameters():
                    parameter.requires_grad = False

        def set_active_head(self, head: str) -> None:
            _require(head in V1_HEADS, "M2_FINAL_HEAD_INVALID", "unknown specialist head", head=head)
            self.active_head = head
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            captured: dict[str, Any] = {}

            def bypass(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], _output: Any) -> tuple[Any, ...]:
                captured["args"] = args
                captured["kwargs"] = kwargs
                return (args[0],)

            _, base_last = _locate_last_block(self.encoder)
            hook = base_last.register_forward_hook(bypass, with_kwargs=True)
            try:
                self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                hook.remove()
            _require("args" in captured, "M2_FINAL_ENCODER_FORWARD_INVALID", "shared trunk did not expose final-block input")
            result: dict[str, Any] = {}
            for head in V1_HEADS:
                layer_output = self.specialist_blocks[head](*captured["args"], **captured["kwargs"])
                hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                result[head] = self.heads[head](self.dropout(hidden[:, 0, :]))
            return result

    return SpecialistModel()


def validate_trainable_parameters(model: Any, head: str) -> dict[str, Any]:
    _require(head in V1_HEADS, "M2_FINAL_HEAD_INVALID", "unknown head", head=head)
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M2_FINAL_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "only selected final block and head may be trainable", head=head, expected=sorted(expected), observed=sorted(actual))
    unexpected_encoder = sorted(name for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad)
    _require(not unexpected_encoder, "M2_FINAL_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "shared embeddings or first blocks are trainable", unexpected=unexpected_encoder)
    return {"active_head": head, "trainable_parameters": sorted(actual), "shared_encoder_frozen": True, "independent_last_block": True}


def _finite(torch: Any, value: Any, code: str, message: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), code, message)


def _predict_reasoning(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> tuple[list[list[int]], list[list[float]]]:
    labels: list[list[int]] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset:offset + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[REASONING_HEAD]
            _finite(torch, logits, "M2_FINAL_NONFINITE_LOGITS", "non-finite reasoning logits")
            labels.extend(batch["labels"][REASONING_HEAD].detach().cpu().int().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return labels, probabilities


def _target_metrics(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str) -> dict[str, Any]:
    all_metrics = m1.diagnostic_metrics(torch, model, tokenizer, records, config, device)
    return all_metrics[head]


def _cpu_reload(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, tokenizer: Any, config: Mapping[str, Any], checkpoint: Mapping[str, Any], record: m1.M1Record, head: str) -> dict[str, Any]:
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.set_active_head(head)
    model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    model = model.to(torch.device("cpu"))
    model.eval()
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    with torch.no_grad():
        outputs = model(batch["input_ids"], batch["attention_mask"])[head]
    finite = bool(torch.isfinite(outputs).all().item())
    _require(finite, "M2_FINAL_CPU_RELOAD_FAILED", "CPU reload emitted non-finite logits", head=head, seed=checkpoint.get("seed"))
    return {"required": True, "device": "cpu", "head": head, "sample_id": record.sample_id, "finite": finite, "output_shape": list(outputs.shape)}


def _limits(start: float, root: Path, phase: str, seed: int | None = None, head: str | None = None) -> None:
    _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_FINAL_WALL_TIME_LIMIT_EXCEEDED", "run exceeded two-hour wall limit", phase=phase, seed=seed, head=head)
    _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_FINAL_DISK_LIMIT_EXCEEDED", "output exceeded ten GiB", phase=phase, seed=seed, head=head)


def _seed_head(*, runtime: tuple[Any, Any, Any, Any], head: str, seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / head / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _limits(started, root, "before_model_load", seed, head)
    _set_seed(torch, np, seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.set_active_head(head)
    model = model.to(device)
    parameter_identity = validate_trainable_parameters(model, head)
    trainable_block = list(model.specialist_blocks[head].parameters())
    trainable_head = list(model.heads[head].parameters())
    trainable = [*trainable_block, *trainable_head]
    optimizer = torch.optim.AdamW([{"params": trainable_block, "lr": BLOCK_LR}, {"params": trainable_head, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best, best_epoch, stale = -math.inf, 0, 0
    log = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(True)
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            _limits(started, root, "during_epoch", seed, head)
            batch = m1._as_batch(torch, tokenizer, shuffled[offset:offset + BATCH_SIZE], config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            _finite(torch, logits[head], "M2_FINAL_NONFINITE_LOGITS", "non-finite selected-head logits")
            loss = s3._single_head_loss(torch, logits, batch, head)
            _finite(torch, loss, "M2_FINAL_NONFINITE_LOSS", "non-finite selected-head loss")
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _finite(torch, parameter.grad, "M2_FINAL_NONFINITE_GRADIENT", "non-finite selected-head gradient")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metric = _target_metrics(torch, model, tokenizer, dev, config, device, head)
        score = float(dev_metric["macro_f1"])
        improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": seed, "frozen_config": config, "provenance": provenance, "parameter_identity": parameter_identity, "specialist_block_prefix": f"specialist_blocks.{head}", "specialist_block_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {key: value.detach().cpu() for key, value in model.heads[head].state_dict().items()}}, folder / "specialist-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log, {"head": head, "seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_primary_macro_f1": score, "improved": improved, "stale_epochs": stale})
        _limits(started, root, "after_epoch", seed, head)
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "specialist-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_FINAL_CHECKPOINT_MISSING", "specialist checkpoint missing", head=head, seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    best_model.set_active_head(head)
    best_model = best_model.to(device)
    best_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
    best_model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    train_metric = _target_metrics(torch, best_model, tokenizer, train, config, device, head)
    dev_metric = _target_metrics(torch, best_model, tokenizer, dev, config, device, head)
    thresholds: dict[str, float] | None = None
    train_summary: dict[str, Any] | None = None
    if head == REASONING_HEAD:
        train_labels, train_probabilities = _predict_reasoning(torch, best_model, tokenizer, train, config, device)
        dev_labels, dev_probabilities = _predict_reasoning(torch, best_model, tokenizer, dev, config, device)
        thresholds, train_summary = r2.choose_thresholds(train_labels, train_probabilities, config["class_order"][REASONING_HEAD])
        calibrated = r2.metrics_from_probabilities(dev_labels, dev_probabilities, config["class_order"][REASONING_HEAD], thresholds)
        dev_metric = calibrated
        m1._json_dump(folder / "reasoning-thresholds.json", {"selection_population": "Train_1822_only", "thresholds": thresholds, "train_summary": train_summary, "dev_metrics_after_frozen_thresholds": calibrated})
    smoke = _cpu_reload(torch, AutoModel, snapshot, schema, tokenizer, config, checkpoint, dev[0], head)
    _limits(started, root, "after_cpu_reload", seed, head)
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "head": head, "seed": seed, "best_epoch": best_epoch, "early_stopping_metric": f"{head}.macro_f1", "sample_counts": {"train": len(train), "dev": len(dev)}, "train": {head: train_metric}, "dev": {head: dev_metric}, "thresholds": thresholds, "train_threshold_summary": train_summary, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(folder / "seed-metrics.json", metrics)
    resource = {"head": head, "seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    m1._json_dump(folder / "resource-log.json", resource)
    _limits(started, root, "after_seed_evidence", seed, head)
    return {"head": head, "seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "thresholds_sha256": sha256_file(folder / "reasoning-thresholds.json") if head == REASONING_HEAD else None}


def _classical_gate(contract: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controls = _mapping(contract.get("immutable_controls"), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", "immutable controls")
    classical = _mapping(controls.get("classical_v0_3_5_control"), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", "Classical control")
    frozen_all = _mapping(classical.get("frozen_dev_metrics"), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", "Classical metrics")
    frozen = _mapping(frozen_all.get("scalar_heads"), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", "Classical scalar metrics")
    reasoning_reference = _mapping(frozen_all.get(REASONING_HEAD), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", "Classical reasoning metrics")
    frozen = {**frozen, REASONING_HEAD: reasoning_reference}
    final_gate = _mapping(contract["final_specialist_candidate_contract"], "M2_FINAL_CONTRACT_INVALID", "final contract").get("final_classical_gate")
    gate = _mapping(final_gate, "M2_FINAL_CONTRACT_INVALID", "final Classical gate")
    per_head: dict[str, Any] = {}
    improved_heads = 0
    failures: list[str] = []
    for head in V1_HEADS:
        rows = [item for item in results if item["head"] == head]
        _require([row["seed"] for row in rows] == list(SEEDS), "M2_FINAL_INCOMPLETE_RUNS", "each head requires three seeds", head=head)
        values = [float(row["metrics"]["dev"][head]["macro_f1"]) for row in rows]
        baseline = float(_mapping(frozen.get(head), "M2_FINAL_CLASSICAL_REFERENCE_INVALID", head)["primary_macro_f1"])
        mean, std, worst = statistics.mean(values), statistics.stdev(values), min(values)
        deltas = [value - baseline for value in values]
        checks = {"mean_drop": mean - baseline >= -float(gate["mean_macro_drop_max"]), "worst_seed_drop": worst - baseline >= -float(gate["worst_seed_macro_drop_max"]), "sample_std": std <= float(gate["per_head_macro_sample_std_max"]), "mean_improvement": mean - baseline >= float(gate["minimum_mean_improvement"])}
        if checks["mean_improvement"]:
            improved_heads += 1
        if not all(checks[key] for key in ("mean_drop", "worst_seed_drop", "sample_std")):
            failures.append(head)
        item: dict[str, Any] = {"classical_macro_f1": baseline, "per_seed": values, "delta_per_seed": [round(value, 6) for value in deltas], "mean": mean, "sample_standard_deviation": std, "worst_seed": worst, "checks": checks}
        if head == REASONING_HEAD:
            for metric in ("micro_f1", "exact_set_accuracy"):
                candidate_values = [float(row["metrics"]["dev"][head][metric]) for row in rows]
                classical_value = float(frozen[head][metric])
                item[metric] = {"classical": classical_value, "per_seed": candidate_values, "mean": statistics.mean(candidate_values), "worst_seed": min(candidate_values), "delta_mean": statistics.mean(candidate_values) - classical_value, "delta_worst": min(candidate_values) - classical_value, "checks": {"mean_drop": statistics.mean(candidate_values) - classical_value >= -0.01, "worst_seed_drop": min(candidate_values) - classical_value >= -0.03}}
                if not all(item[metric]["checks"].values()):
                    failures.append(f"{head}:{metric}")
        per_head[head] = item
    critical: dict[str, Any] = {}
    proxies = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_FINAL_CONTRACT_INVALID", "metric gate").get("critical_boundary_proxies", [])
    for proxy in proxies:
        head = proxy["head"]
        baseline_metrics = frozen[head]
        key = "per_label" if head == REASONING_HEAD else "per_class"
        observed_rows = [item for item in results if item["head"] == head]
        for label in proxy["labels"]:
            # Some frozen scalar heads have no observed support for UNKNOWN;
            # retain the proxy in the report but do not turn an absent class
            # into a fabricated numerical regression.
            base_row = baseline_metrics[key].get(label, {"f1": 0.0, "support": 0})
            support = int(base_row.get("support", 0))
            deltas = [float(item["metrics"]["dev"][head][key][label]["f1"]) - float(base_row["f1"]) for item in observed_rows]
            passed = None if support < 20 else all(delta >= -float(gate["critical_f1_drop_max"]) for delta in deltas)
            critical[f"{head}:{label}"] = {"support": support, "classical_f1": float(base_row["f1"]), "delta_per_seed": [round(delta, 6) for delta in deltas], "passed": passed, "status": "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION" if support < 20 else ("PASS" if passed else "FAIL")}
            if passed is False:
                failures.append(f"{head}:{label}")
    checks = {"all_head_no_regression": not any(name in failures for name in V1_HEADS), "minimum_four_head_improvements": improved_heads >= int(gate["minimum_heads_mean_improvement"]), "critical_boundaries": not any(value["passed"] is False for value in critical.values())}
    return {"classical_control_source": "immutable_controls.classical_v0_3_5_control.frozen_dev_metrics", "per_head": per_head, "critical_boundaries": critical, "improved_head_count": improved_heads, "failures": sorted(set(failures)), "checks": checks, "passed": all(checks.values()), "selected_candidate": False}


def _reasoning_gate(contract: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasoning = [item for item in results if item["head"] == REASONING_HEAD]
    _require([item["seed"] for item in reasoning] == list(SEEDS), "M2_FINAL_REASONING_INCOMPLETE", "reasoning three-seed fail-fast requires all seeds")
    classical_all = contract["immutable_controls"]["classical_v0_3_5_control"]["frozen_dev_metrics"]
    classical = classical_all[REASONING_HEAD]
    values = [float(item["metrics"]["dev"][REASONING_HEAD]["macro_f1"]) for item in reasoning]
    micro = [float(item["metrics"]["dev"][REASONING_HEAD]["micro_f1"]) for item in reasoning]
    exact = [float(item["metrics"]["dev"][REASONING_HEAD]["exact_set_accuracy"]) for item in reasoning]
    spec = _mapping(contract["final_specialist_candidate_contract"]["reasoning_fail_fast"], "M2_FINAL_CONTRACT_INVALID", "reasoning gate")
    g = _mapping(spec["classical_gate"], "M2_FINAL_CONTRACT_INVALID", "reasoning Classical gate")
    critical: dict[str, Any] = {}
    for label in spec["critical_labels"]:
        base = classical["per_label"][label]
        deltas = [float(item["metrics"]["dev"][REASONING_HEAD]["per_label"][label]["f1"]) - float(base["f1"]) for item in reasoning]
        critical[label] = {"support": int(base["support"]), "delta_per_seed": [round(x, 6) for x in deltas], "passed": None if int(base["support"]) < 20 else min(deltas) >= -float(g["critical_label_f1_drop_max"])}
    checks = {"macro_mean": statistics.mean(values) >= float(g["macro_mean_min"]), "macro_worst_seed": min(values) >= float(g["macro_worst_seed_min"]), "micro_mean": statistics.mean(micro) >= float(g["micro_mean_min"]), "micro_worst_seed": min(micro) >= float(g["micro_worst_seed_min"]), "exact_mean": statistics.mean(exact) >= float(g["exact_mean_min"]), "exact_worst_seed": min(exact) >= float(g["exact_worst_seed_min"]), "macro_std": statistics.stdev(values) <= float(g["macro_sample_std_max"]), "critical_labels": all(value["passed"] is not False for value in critical.values())}
    return {"per_seed": {str(item["seed"]): {"macro_f1": float(item["metrics"]["dev"][REASONING_HEAD]["macro_f1"]), "micro_f1": float(item["metrics"]["dev"][REASONING_HEAD]["micro_f1"]), "exact_set_accuracy": float(item["metrics"]["dev"][REASONING_HEAD]["exact_set_accuracy"])} for item in reasoning}, "checks": checks, "critical_labels": critical, "passed": all(checks.values()), "action": "CONTINUE_SIX_SPECIALISTS" if all(checks.values()) else "STOP_BEFORE_OTHER_HEADS", "selected_candidate": False}


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(results) == len(V1_HEADS) * len(SEEDS), "M2_FINAL_INCOMPLETE_RUNS", "all 21 head/seed units are required")
    devices = {head: {str(item["seed"]): item["resource"]["actual_device"] for item in results if item["head"] == head} for head in V1_HEADS}
    _require(all(len(set(values.values())) == 1 for values in devices.values()), "M2_FINAL_MIXED_DEVICE", "a head cannot aggregate mixed devices", devices=devices)
    return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "heads": list(V1_HEADS), "all_run_units_complete": True, "devices": devices, "per_head": {head: {"mean_macro_f1": statistics.mean(float(item["metrics"]["dev"][head]["macro_f1"]) for item in results if item["head"] == head), "sample_standard_deviation": statistics.stdev(float(item["metrics"]["dev"][head]["macro_f1"]) for item in results if item["head"] == head)} for head in V1_HEADS}, "selected_candidate": False}


def _failure(root: Path | None, exc: ContractError, entered: Sequence[str], preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {"status": "RBT3_M2_CANDIDATE_REJECTED", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": bool(entered), "model_loaded": bool(entered), "cache_accessed": bool(entered), "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False, "heads_started": list(entered)}
    if root and root.exists():
        m1._json_dump(root / "rejected-diagnostic-evidence.json", result)
        payload = {"manifest_schema_version": "myresearcher.encoder-m2-final-specialist-rejected-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "selected_candidate": False, "m2_lineage": preflight["frozen_contract"]["final"]["lineage_id"] if preflight else None}
        result["rejected_content_address"] = m1._write_content_manifest(root, payload)["content_address"]
    return result


def _write_unified_bundle(runtime: tuple[Any, Any, Any, Any], root: Path, snapshot: Path, schema: Any, config: Mapping[str, Any], results: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    seed = 35
    selected = [item for item in results if item["seed"] == seed]
    _require(len(selected) == len(V1_HEADS), "M2_FINAL_BUNDLE_INCOMPLETE", "seed-35 unified bundle requires every head")
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    state_blocks: dict[str, Any] = {}
    state_heads: dict[str, Any] = {}
    for item in selected:
        checkpoint = torch.load(root / item["head"] / f"seed-{seed}" / "specialist-checkpoint.pt", map_location="cpu", weights_only=True)
        state_blocks[item["head"]] = checkpoint["specialist_block_state_dict"]
        state_heads[item["head"]] = checkpoint["head_state_dict"]
        model.specialist_blocks[item["head"]].load_state_dict(state_blocks[item["head"]])
        model.heads[item["head"]].load_state_dict(state_heads[item["head"]])
    bundle_dir = root / "unified-bundle"
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = bundle_dir / "seed-35-unified-bundle.pt"
    torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "seed": seed, "provenance": provenance, "shared_bottom": "official_rbt3_embeddings_and_transformer_blocks_0_and_1", "specialist_block_state_dict": state_blocks, "heads_state_dict": state_heads}, bundle_path)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    record = _load_dev_record_for_bundle(results)
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    model = model.to(torch.device("cpu")); model.eval()
    with torch.no_grad():
        unified = model(batch["input_ids"], batch["attention_mask"])
    equivalence: dict[str, Any] = {}
    for head in V1_HEADS:
        checkpoint = torch.load(root / head / "seed-35" / "specialist-checkpoint.pt", map_location="cpu", weights_only=True)
        standalone = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
        standalone.set_active_head(head)
        standalone.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
        standalone.heads[head].load_state_dict(checkpoint["head_state_dict"])
        standalone.eval()
        with torch.no_grad():
            solo = standalone(batch["input_ids"], batch["attention_mask"])[head]
        difference = float((unified[head] - solo).abs().max().item())
        equivalence[head] = {"finite": bool(torch.isfinite(unified[head]).all().item() and torch.isfinite(solo).all().item()), "numerically_equivalent": difference <= 1e-6, "max_abs_diff": difference}
    m1._json_dump(bundle_dir / "cpu-equivalence.json", {"seed": seed, "per_head": equivalence, "all_finite": all(item["finite"] for item in equivalence.values()), "all_numerically_equivalent": all(item["numerically_equivalent"] for item in equivalence.values())})
    return {"bundle_path": str(bundle_path), "bundle_sha256": sha256_file(bundle_path), "equivalence_path": str(bundle_dir / "cpu-equivalence.json"), "equivalence_sha256": sha256_file(bundle_dir / "cpu-equivalence.json"), "equivalence": equivalence}


def _load_dev_record_for_bundle(results: Sequence[Mapping[str, Any]]) -> m1.M1Record:
    # The actual record is injected by run_final; this fallback is replaced at call time.
    record = results[0].get("bundle_record")
    _require(isinstance(record, m1.M1Record), "M2_FINAL_BUNDLE_RECORD_MISSING", "bundle equivalence record missing")
    return record


def run_final(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path, *, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies, seed_executor: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / CONTRACT_RELATIVE_PATH
    preflight: dict[str, Any] | None = None
    try:
        preflight = validate_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
    except ContractError as exc:
        return _failure(None, exc, [], None)
    root: Path | None = None
    started_heads: list[str] = []
    try:
        runtime = runtime_loader()
        runtime_identity = _plain_json(s1.validate_runtime_identity(runtime, preflight["frozen_contract"]))
        root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        config = _config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        m1._json_dump(root / "training-config.json", config)
        provenance = {**preflight["identity"], "runtime_identity": runtime_identity, "contract_sha256": preflight["frozen_contract"]["contract_sha256"], "cache_snapshot": preflight["snapshot_identity"]}
        m1._json_dump(root / "provenance.json", provenance)
        executor = seed_executor or _seed_head
        reasoning_results: list[Mapping[str, Any]] = []
        for seed in SEEDS:
            started_heads.append(REASONING_HEAD)
            reasoning_results.append(executor(runtime=runtime, head=REASONING_HEAD, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, cache_dir=Path(cache_dir)))
        reasoning_gate = _reasoning_gate(preflight["frozen_contract"]["contract"], reasoning_results)
        m1._json_dump(root / "reasoning-fail-fast-gate.json", reasoning_gate)
        if not reasoning_gate["passed"]:
            return _failure(root, ContractError("M2_FINAL_REASONING_CLASSICAL_GATE_FAILED", "reasoning fail-fast Classical gate failed", failures=[name for name, passed in reasoning_gate["checks"].items() if not passed]), started_heads, preflight)
        results = list(reasoning_results)
        for head in V1_HEADS:
            if head == REASONING_HEAD:
                continue
            for seed in SEEDS:
                started_heads.append(head)
                results.append(executor(runtime=runtime, head=head, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, cache_dir=Path(cache_dir)))
        # Attach one Train/Dev-only record for deterministic bundle equivalence.
        for item in results:
            if isinstance(item, dict):
                item["bundle_record"] = preflight["dev"][0]
        aggregate = aggregate_results(results)
        gate = _classical_gate(preflight["frozen_contract"]["contract"], results)
        bundle = _write_unified_bundle(runtime, root, preflight["snapshot"], preflight["schema"], config, results, provenance)
        m1._json_dump(root / "stage-aggregate.json", aggregate)
        m1._json_dump(root / "final-classical-gate.json", gate)
        m1._json_dump(root / "unified-bundle-evidence.json", bundle)
        _limits(started, root, "after_final_metrics")
        status = "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3" if gate["passed"] else "RBT3_M2_CANDIDATE_REJECTED"
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-final-specialist-artifact-manifest.v1", "stage_id": STAGE_ID, "status": status, "m2_lineage": preflight["frozen_contract"]["final"]["lineage_id"], "selected_candidate": gate["passed"], "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_policy": "MPS_FIRST_CPU_FALLBACK", "data_scope": "Train_1822_and_Dev_448_only", "seed35_predeclared_for_m3": True, "reasoning_fail_fast_gate_sha256": sha256_file(root / "reasoning-fail-fast-gate.json"), "final_classical_gate_sha256": sha256_file(root / "final-classical-gate.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "unified_bundle": bundle, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_inference_49054"]})
        _limits(started, root, "after_final_manifest")
        return {"status": status, "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": gate["passed"], "output_dir": str(root), "content_address": manifest["content_address"], "reasoning_gate": reasoning_gate, "aggregate": aggregate, "final_gate": gate, "unified_bundle": bundle}
    except ContractError as exc:
        return _failure(root, exc, started_heads, preflight)
    except Exception as exc:
        return _failure(root, ContractError("M2_FINAL_RUNTIME_EXCEPTION", "final specialist runtime exception", exception_type=type(exc).__name__, detail=str(exc)), started_heads, preflight)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the final seven-head RBT3 specialist candidate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_final(args.config, args.output_dir, args.cache_dir)
    stream = sys.stdout if result.get("status") in {"M2_SELECTED_CANDIDATE_FROZEN_FOR_M3", "RBT3_M2_CANDIDATE_REJECTED"} else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    return 0 if result.get("status") == "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3" else 2


if __name__ == "__main__":
    raise SystemExit(main())
