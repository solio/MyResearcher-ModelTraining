"""M2-S2 partial-unfreeze runner for the fixed local RBT3 lineage.

S2 is deliberately a separate entry point from the S1 frozen control.  It
reuses M1's Train/Dev loader, input builder, weighted loss, metrics, and CPU
reload smoke, while making the final Transformer block the only trainable
Encoder component.
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import encoder_m1 as m1
from . import encoder_m2_s1 as s1
from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD"
SEEDS = (35, 71, 107)
HEAD_LEARNING_RATE = 3e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 16
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
S1_ARTIFACT_CONTENT_ADDRESS = "04a23d76413049e57ff083655f80ad8c3dfc7ed90702a3c9ed66bcfd79f377f6"
DEFAULT_S1_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-s1-first-three-seed-20260831"

# Keep the M1 technical helpers as the single implementation for cache/output
# and runtime checks; local aliases make the dependency explicit and easy to
# replace with synthetic fixtures in tests.
validate_output_dir = s1.validate_output_dir
validate_runtime_identity = s1.validate_runtime_identity


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _mapping(value: Any, code: str, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object")
    return value


def _read_json(path: str | Path, *, missing_code: str, invalid_code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(missing_code, "required JSON file is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(invalid_code, "required JSON file is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), invalid_code, "JSON root must be an object")
    return value


def _contract_requirements(contract_path: str | Path) -> dict[str, Any]:
    frozen = s1._contract_requirements(contract_path)
    contract = frozen["contract"]
    stages = contract.get("minimal_experiment_gradient")
    stage = next((item for item in stages if isinstance(item, Mapping) and item.get("stage_id") == STAGE_ID), None) if isinstance(stages, list) else None
    _require(stage is not None, "M2_S2_STAGE_CONTRACT_INVALID", "S2 stage is missing")
    _require(stage.get("architecture") == "ONE_SHARED_ENCODER_WITH_SEVEN_HEADS", "M2_S2_STAGE_CONTRACT_INVALID", "S2 must use the shared seven-head architecture")
    _require(stage.get("encoder_state") == "PARTIAL_UNFREEZE" and stage.get("unfrozen_transformer_blocks") == 1, "M2_S2_STAGE_CONTRACT_INVALID", "S2 must unfreeze exactly one Transformer block")
    _require(stage.get("unfreeze_rule") == "Only the final transformer block is trainable; embeddings and all earlier transformer blocks remain frozen.", "M2_S2_STAGE_CONTRACT_INVALID", "S2 unfreeze rule changed")
    _require(stage.get("seeds") == list(SEEDS) and stage.get("run_units") == 3, "M2_S2_STAGE_CONTRACT_INVALID", "S2 must use exactly three fixed seeds")
    _require(stage.get("head_learning_rate") == HEAD_LEARNING_RATE and stage.get("encoder_learning_rate") == ENCODER_LEARNING_RATE, "M2_S2_OPTIMIZER_CONTRACT_INVALID", "S2 learning rates changed")
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S2_CONTRACT_INVALID", "common training configuration")
    _require(common.get("max_length") == 256 and common.get("batch_size") == BATCH_SIZE and common.get("truncation") == "HEAD_TAIL", "M2_S2_TRAINING_CONTRACT_INVALID", "S2 input configuration changed")
    early = _mapping(common.get("early_stopping"), "M2_S2_CONTRACT_INVALID", "early stopping")
    gradient = _mapping(common.get("gradient_controls"), "M2_S2_CONTRACT_INVALID", "gradient controls")
    _require(early.get("max_epochs") == MAX_EPOCHS and early.get("patience_epochs") == PATIENCE and gradient.get("gradient_clipping_max_norm") == 1.0, "M2_S2_TRAINING_CONTRACT_INVALID", "S2 stopping or gradient configuration changed")
    return {**frozen, "stage": stage}


def load_s1_control(artifact_dir: str | Path) -> dict[str, Any]:
    """Read and verify the immutable S1 artifact without changing it."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest = _read_json(root / "content-addressed-manifest.json", missing_code="M2_S2_S1_ARTIFACT_MISSING", invalid_code="M2_S2_S1_ARTIFACT_INVALID")
    _require(manifest.get("content_address") == S1_ARTIFACT_CONTENT_ADDRESS, "M2_S2_S1_ARTIFACT_ID_MISMATCH", "S2 requires the accepted S1 artifact")
    _require(manifest.get("content_address") == content_addressed_id(manifest, omit_keys={"content_address"}), "M2_S2_S1_ARTIFACT_ID_MISMATCH", "S1 artifact content address does not recompute")
    _require(manifest.get("files") == m1._hash_tree(root, exclude={"content-addressed-manifest.json"}), "M2_S2_S1_ARTIFACT_HASH_MISMATCH", "S1 artifact file hashes differ")
    seed_metrics: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        metrics = _read_json(root / f"seed-{seed}" / "seed-metrics.json", missing_code="M2_S2_S1_METRICS_MISSING", invalid_code="M2_S2_S1_METRICS_INVALID")
        _require(metrics.get("sample_counts") == {"train": 1822, "dev": 448}, "M2_S2_S1_METRICS_INVALID", "S1 sample counts differ", seed=seed)
        _require(all(head in metrics.get("dev", {}) for head in V1_HEADS), "M2_S2_S1_METRICS_INVALID", "S1 seven-head metrics are incomplete", seed=seed)
        seed_metrics[seed] = metrics
    return {"root": root, "manifest": manifest, "seed_metrics": seed_metrics}


def validate_s2_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    """Load only Train/Dev and the fixed local cache; never run a full audit."""

    frozen = _contract_requirements(contract_path)
    config = ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_S2_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, snapshot_identity = s1.validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": snapshot_identity, "schema": schema, "train": train, "dev": dev, "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": m1.MODEL_ID, "revision": m1.REVISION, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version}, "worktree": str(Path(worktree).resolve())}


def _locate_last_transformer_block(encoder: Any) -> tuple[str, Any, Sequence[Any]]:
    for container_name in ("encoder", "roberta", "bert", "transformer"):
        container = getattr(encoder, container_name, None)
        blocks = getattr(container, "layer", None)
        if blocks is not None and len(blocks) > 0:
            return f"{container_name}.layer.{len(blocks) - 1}", blocks[-1], blocks
    blocks = getattr(encoder, "layer", None)
    if blocks is not None and len(blocks) > 0:
        return f"layer.{len(blocks) - 1}", blocks[-1], blocks
    raise ContractError("M2_S2_TRANSFORMER_BLOCK_NOT_FOUND", "RBT3 Transformer block list was not found")


def _make_model(torch: Any, AutoModel: Any, snapshot_path: Path, schema: Any, dropout: float) -> Any:
    nn = torch.nn

    class PartialUnfrozenEncoderSevenHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False)
            inner_prefix, self.last_block, self._blocks = _locate_last_transformer_block(self.encoder)
            self.last_transformer_block_prefix = f"encoder.{inner_prefix}"
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.last_block.parameters():
                parameter.requires_grad = True
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, len(schema.class_order[head])) for head in V1_HEADS})

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            self.last_block.train(mode)
            self.dropout.train(mode)
            self.heads.train(mode)
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            representation = self.dropout(encoded.last_hidden_state[:, 0, :])
            return {head: classifier(representation) for head, classifier in self.heads.items()}

    return PartialUnfrozenEncoderSevenHead()


def validate_s2_trainable_parameters(model: Any) -> dict[str, Any]:
    """Require exactly the final block and all seven heads to be trainable."""

    block_prefix = f"{model.last_transformer_block_prefix}."
    expected = {name for name, parameter in model.named_parameters() if name.startswith(block_prefix) or name.startswith("heads.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_heads = {name for name, parameter in model.named_parameters() if name.startswith("heads.")}
    _require(len(expected_heads) > 0 and len(model.heads) == 7, "M2_S2_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "all seven heads must be present")
    _require(actual == expected, "M2_S2_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "only the final Transformer block and seven heads may require gradients", expected=sorted(expected), observed=sorted(actual), last_block_prefix=block_prefix)
    unexpected_encoder = sorted(name for name in actual if name.startswith("encoder.") and not name.startswith(block_prefix))
    _require(not unexpected_encoder, "M2_S2_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "an embedding or earlier Encoder parameter is trainable", unexpected=unexpected_encoder)
    return {"last_transformer_block_prefix": block_prefix[:-1], "trainable_parameter_count": len(actual), "trainable_parameters": sorted(actual)}


def _config(contract: Mapping[str, Any], order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S2_CONTRACT_INVALID", "common training configuration")
    optimizer = _mapping(common.get("optimizer"), "M2_S2_CONTRACT_INVALID", "optimizer")
    early = _mapping(common.get("early_stopping"), "M2_S2_CONTRACT_INVALID", "early stopping")
    return {"stage_id": STAGE_ID, "model_id": m1.MODEL_ID, "revision": m1.REVISION, "trust_remote_code": False, "local_files_only": True, "input_builder_version": common["input_builder_version"], "stock_code_token_cap": common["stock_code_token_cap"], "stock_name_token_cap": common["stock_name_token_cap"], "max_length": 256, "truncation": "HEAD_TAIL", "padding": common["padding"], "token_type_ids": "NOT_EMITTED", "batch_size": BATCH_SIZE, "head_dropout": common["head_dropout"], "class_order": {head: list(order[head]) for head in V1_HEADS}, "optimizer": {"name": "AdamW", "head_learning_rate": HEAD_LEARNING_RATE, "encoder_learning_rate": ENCODER_LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "betas": optimizer["betas"], "epsilon": optimizer["epsilon"], "parameter_groups": ["seven_heads", "final_transformer_block"]}, "stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": early["minimum_delta"]}, "gradient_clipping_max_norm": 1.0, "reasoning_probability_threshold": 0.5, "encoder_state": "PARTIAL_UNFREEZE_FINAL_BLOCK_ONLY", "fit_population": "Train_1822_only", "dev_role": "early_stopping_and_diagnostic_only", "test_role": "not_loaded_not_used"}


def _finite(torch: Any, value: Any, code: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), code, "non-finite training value")


def _critical_report(contract: Mapping[str, Any], dev_metrics: Mapping[str, Any], seed: int) -> dict[str, Any]:
    proxies = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_S2_CONTRACT_INVALID", "metric contract").get("critical_boundary_proxies")
    _require(isinstance(proxies, list) and len(proxies) == 7, "M2_S2_CONTRACT_INVALID", "seven critical boundaries required")
    result: dict[str, Any] = {}
    for item in proxies:
        head = item["head"]
        labels = _mapping(dev_metrics[head].get("per_label") if head == "reasoning_tags" else dev_metrics[head].get("per_class"), "M2_S2_METRICS_MISSING", head)
        result[head] = {label: {"support": labels.get(label, {}).get("support"), "f1": labels.get(label, {}).get("f1")} for label in item["labels"]}
    return {"stage_id": STAGE_ID, "seed": seed, "critical_boundaries": result}


def _cpu_reload_s2(torch: Any, model_factory: Callable[[], Any], checkpoint: Mapping[str, Any], tokenizer: Any, record: m1.M1Record, config: Mapping[str, Any]) -> dict[str, Any]:
    def restored_factory() -> Any:
        model = model_factory()
        model.last_block.load_state_dict(checkpoint["last_transformer_block_state_dict"])
        return model

    return m1.cpu_reload_and_inference_smoke(torch, restored_factory, {"heads_state_dict": checkpoint["heads_state_dict"]}, tokenizer, record, config)


def _seed(*, runtime: tuple[Any, Any, Any, Any], seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], m2_contract: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / f"seed-{seed}"
    folder.mkdir()
    started = time.monotonic()
    s1._limits(started, root, "before_model_load", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    parameter_identity = validate_s2_trainable_parameters(model)
    s1._limits(started, root, "after_model_load", seed)
    heads = list(model.heads.parameters())
    block = list(model.last_block.parameters())
    optimizer = torch.optim.AdamW([{"params": heads, "lr": HEAD_LEARNING_RATE}, {"params": block, "lr": ENCODER_LEARNING_RATE}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    _require([group["lr"] for group in optimizer.param_groups] == [HEAD_LEARNING_RATE, ENCODER_LEARNING_RATE], "M2_S2_OPTIMIZER_CONTRACT_INVALID", "S2 optimizer parameter groups changed")
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
            for head_name in V1_HEADS:
                _finite(torch, logits[head_name], "M2_S2_NONFINITE_LOGITS")
            loss = m1._weighted_loss(torch, logits, batch)
            _finite(torch, loss, "M2_S2_NONFINITE_LOSS")
            loss.backward()
            for parameter in [*heads, *block]:
                if parameter.grad is not None:
                    _finite(torch, parameter.grad, "M2_S2_NONFINITE_GRADIENT")
            torch.nn.utils.clip_grad_norm_([*heads, *block], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = m1.diagnostic_metrics(torch, model, tokenizer, dev, config, device)
        score = float(dev_metrics["diagnostic_score"])
        improved = score > best + float(config["stopping"]["minimum_delta"])
        if improved:
            best, epoch_best, stale = score, epoch, 0
            torch.save({"run_schema_version": "myresearcher.encoder-m2-s2-partial-unfreeze-run.v1", "stage_id": STAGE_ID, "seed": seed, "frozen_config": config, "provenance": provenance, "parameter_identity": parameter_identity, "last_transformer_block_prefix": model.last_transformer_block_prefix, "last_transformer_block_state_dict": {key: value.detach().cpu() for key, value in model.last_block.state_dict().items()}, "heads_state_dict": {key: value.detach().cpu() for key, value in model.heads.state_dict().items()}}, folder / "heads-and-last-block-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log, {"seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_diagnostic_score": score, "improved": improved, "stale_epochs": stale})
        s1._limits(started, root, "after_epoch", seed)
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "heads-and-last-block-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_S2_CHECKPOINT_MISSING", "S2 checkpoint missing", seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device)
    best_model.last_block.load_state_dict(checkpoint["last_transformer_block_state_dict"])
    best_model.heads.load_state_dict(checkpoint["heads_state_dict"])
    validate_s2_trainable_parameters(best_model)
    train_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, train, config, device)
    dev_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, dev, config, device)
    s1._limits(started, root, "after_final_metrics", seed)
    try:
        smoke = _cpu_reload_s2(torch, lambda: _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_S2_CPU_RELOAD_SMOKE_FAILED", "S2 CPU reload failed", cause=exc.code) from exc
    _require(smoke.get("all_logits_finite") is True, "M2_S2_CPU_RELOAD_SMOKE_FAILED", "S2 CPU logits are not finite")
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "seed": seed, "best_epoch": epoch_best, "early_stopping_score": best, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(folder / "seed-metrics.json", metrics)
    critical = _critical_report(m2_contract, dev_metrics, seed)
    m1._json_dump(folder / "critical-boundary-report.json", critical)
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    m1._json_dump(folder / "resource-log.json", resource)
    s1._limits(started, root, "after_seed_evidence", seed)
    return {"seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "critical_boundary_report_sha256": sha256_file(folder / "critical-boundary-report.json")}


def aggregate_s2_seed_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require([item.get("seed") for item in results] == list(SEEDS), "M2_S2_INCOMPLETE_SEEDS", "all three S2 seeds are required")
    devices = [str(item["resource"]["actual_device"]) for item in results]
    _require(len(set(devices)) == 1, "M2_S2_MIXED_DEVICE", "S2 cannot aggregate mixed-device seeds", device_stratified_seed_devices={str(item["seed"]): item["resource"]["actual_device"] for item in results})
    per_head: dict[str, Any] = {}
    for head in V1_HEADS:
        values = [float(item["metrics"]["dev"][head]["macro_f1"]) for item in results]
        _require(all(math.isfinite(value) for value in values), "M2_S2_METRICS_MISSING", "non-finite S2 primary metric", head=head)
        per_head[head] = {"per_seed_values": values, "mean": sum(values) / len(values), "sample_standard_deviation": statistics.stdev(values), "minimum_worst_seed": min(values), "maximum": max(values)}
    stable = all(item["sample_standard_deviation"] <= 0.05 for item in per_head.values())
    return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "actual_device": devices[0], "all_seeds_complete": True, "per_head_primary_macro_f1": per_head, "seed_stability_gate_passed": stable, "allowed_output": "S2_STAGE_EVIDENCE_ONLY" if stable else "S2_REJECTED_OR_BLOCKED_EVIDENCE", "selected_candidate": False}


def _matching_seed_report(contract: Mapping[str, Any], s1_control: Mapping[str, Any], results: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(_mapping(contract.get("dev_metrics_and_no_regression"), "M2_S2_CONTRACT_INVALID", "metric contract").get("stage_no_regression_gate"), "M2_S2_CONTRACT_INVALID", "stage no-regression gate")
    head_reports: dict[str, Any] = {}
    failed_heads: list[str] = []
    for head in V1_HEADS:
        s1_values = [float(s1_control["seed_metrics"][seed]["dev"][head]["macro_f1"]) for seed in SEEDS]
        s2_values = [float(item["metrics"]["dev"][head]["macro_f1"]) for item in results]
        deltas = [round(right - left, 6) for left, right in zip(s1_values, s2_values, strict=True)]
        mean_delta = sum(deltas) / len(deltas)
        worst_seed_delta = min(deltas)
        mean_pass = mean_delta >= -float(gate["maximum_mean_primary_metric_drop"])
        worst_pass = worst_seed_delta >= -float(gate["maximum_worst_seed_primary_metric_drop"])
        head_reports[head] = {"s1_per_seed": s1_values, "s2_per_seed": s2_values, "delta_per_seed": deltas, "s1_mean": sum(s1_values) / 3, "s2_mean": sum(s2_values) / 3, "mean_delta": mean_delta, "worst_seed_delta": worst_seed_delta, "s2_sample_standard_deviation": statistics.stdev(s2_values), "mean_no_regression_passed": mean_pass, "worst_seed_no_regression_passed": worst_pass, "no_regression_passed": mean_pass and worst_pass}
        if not (mean_pass and worst_pass):
            failed_heads.append(head)
    critical: dict[str, Any] = {}
    critical_failures: list[str] = []
    proxies = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_S2_CONTRACT_INVALID", "metric contract")["critical_boundary_proxies"]
    critical_limit = float(gate["maximum_critical_label_f1_drop_when_dev_support_at_least_20"])
    for item in proxies:
        head = item["head"]
        labels: dict[str, Any] = {}
        for label in item["labels"]:
            s1_rows = [s1_control["seed_metrics"][seed]["dev"][head]["per_label" if head == "reasoning_tags" else "per_class"][label] for seed in SEEDS]
            s2_rows = [result["metrics"]["dev"][head]["per_label" if head == "reasoning_tags" else "per_class"][label] for result in results]
            supports = [int(row["support"]) for row in s1_rows]
            s1_f1 = [float(row["f1"]) for row in s1_rows]
            s2_f1 = [float(row["f1"]) for row in s2_rows]
            deltas = [round(right - left, 6) for left, right in zip(s1_f1, s2_f1, strict=True)]
            evaluable = min(supports) >= 20
            passed: bool | None = all(delta >= -critical_limit for delta in deltas) if evaluable else None
            if passed is False:
                critical_failures.append(f"{head}:{label}")
            labels[label] = {"support_per_seed": supports, "s1_f1_per_seed": s1_f1, "s2_f1_per_seed": s2_f1, "delta_per_seed": deltas, "status": "EVALUATED" if evaluable else "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION", "no_regression_passed": passed}
        critical[head] = labels
    mean_improvements = [head for head, report in head_reports.items() if report["mean_delta"] >= 0.01]
    flat_or_lower = [head for head, report in head_reports.items() if report["mean_delta"] <= 0.0]
    critical_passed = not critical_failures
    no_regression = not failed_heads and critical_passed
    promotion_passed = bool(aggregate["seed_stability_gate_passed"] and no_regression and len(flat_or_lower) <= 2 and len(mean_improvements) >= 2)
    return {"stage_id": STAGE_ID, "comparator": "S1_FROZEN_SHARED_MATCHING_SEED", "per_head": head_reports, "critical_labels": critical, "critical_label_failures": critical_failures, "failed_heads": failed_heads, "stability_gate_passed": aggregate["seed_stability_gate_passed"], "mean_improvement_heads": mean_improvements, "flat_or_lower_heads": flat_or_lower, "promotion": {"all_no_regression_passed": no_regression, "flat_or_lower_head_count": len(flat_or_lower), "maximum_flat_or_lower_head_count": 2, "at_least_two_heads_mean_improved_by_0.01": len(mean_improvements) >= 2, "passed": promotion_passed}, "s3_triggered_heads": failed_heads, "selected_candidate": False}


def _failure(root: Path | None, exc: ContractError, preflight: Mapping[str, Any], entered: bool) -> dict[str, Any]:
    result = {"status": "M2_S2_REJECTED_OR_BLOCKED_EVIDENCE", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": entered, "model_loaded": entered, "cache_accessed": bool(entered), "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False}
    if root and root.exists():
        m1._json_dump(root / "blocked-evidence.json", result)
        result["rejected_content_address"] = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s2-rejected-artifact-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"], "m1_controls": preflight["frozen_contract"]["m1_controls"], "complete_cache_snapshot": preflight["snapshot_identity"]})["content_address"]
    return result


def run_m2_s2(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path, *, s1_artifact: str | Path = DEFAULT_S1_ARTIFACT, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies, seed_executor: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / s1.CONTRACT_RELATIVE_PATH
    try:
        preflight = validate_s2_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
        s1_control = load_s1_control(s1_artifact)
    except ContractError as exc:
        return {"status": "M2_S2_BLOCKED_FAIL_CLOSED", "phase": "TRAIN_DEV_TECHNICAL_PREFLIGHT", "blocker_codes": [exc.code], "training_invoked": False, "model_loaded": False, "cache_accessed": False, "output_created": False, "aggregate_created": False, "selected_candidate": False}
    started = time.monotonic()
    root: Path | None = None
    results: list[Mapping[str, Any]] = []
    entered = False
    try:
        runtime = runtime_loader()
        runtime_identity = validate_runtime_identity(runtime, preflight["frozen_contract"])
        root = validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True)
        s1._limits(started, root, "before_fit")
        config = _config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        m1._json_dump(root / "training-config.json", config)
        executor = seed_executor or _seed
        for seed in SEEDS:
            entered = True
            results.append(executor(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=preflight["identity"], m2_contract=preflight["frozen_contract"]["contract"], cache_dir=Path(cache_dir)))
        aggregate = aggregate_s2_seed_results(results)
        m1._json_dump(root / "stage-aggregate.json", aggregate)
        report = _matching_seed_report(preflight["frozen_contract"]["contract"], s1_control, results, aggregate)
        m1._json_dump(root / "s2-vs-s1-matching-seed-report.json", report)
        s1._limits(started, root, "after_matching_seed_report")
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-s2-artifact-manifest.v1", "stage_id": STAGE_ID, "diagnostic_only": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "m2_lineage": preflight["frozen_contract"]["contract"]["new_model_lineage"], "m1_controls": preflight["frozen_contract"]["m1_controls"], "s1_control": {"content_address": s1_control["manifest"]["content_address"], "artifact_path": str(s1_control["root"])}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_identity": {"actual_device": aggregate["actual_device"], "policy": "MPS_FIRST_CPU_FALLBACK"}, "training_config_sha256": sha256_file(root / "training-config.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "s2_vs_s1_report_sha256": sha256_file(root / "s2-vs-s1-matching-seed-report.json"), "seed_checkpoints": {str(item["seed"]): item["checkpoint_sha256"] for item in results}, "critical_boundary_report": {str(item["seed"]): item["critical_boundary_report_sha256"] for item in results}})
        s1._limits(started, root, "after_final_manifest")
        return {"status": "M2_S2_CONTROL_COMPLETED" if report["promotion"]["passed"] else "M2_S2_COMPLETED_NOT_PROMOTED", "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "allowed_output": aggregate["allowed_output"], "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate, "matching_seed_report": report}
    except ContractError as exc:
        return _failure(root, exc, preflight, entered)
    except Exception as exc:
        return _failure(root, ContractError("M2_S2_RUNTIME_EXCEPTION", "S2 runtime/OOM exception", exception_type=type(exc).__name__, detail=str(exc)), preflight, entered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--s1-artifact", default=str(DEFAULT_S1_ARTIFACT))
    args = parser.parse_args(argv)
    result = run_m2_s2(args.config, args.output_dir, args.cache_dir, s1_artifact=args.s1_artifact)
    stream = sys.stdout if result.get("status") in {"M2_S2_CONTROL_COMPLETED", "M2_S2_COMPLETED_NOT_PROMOTED"} else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result.get("status") in {"M2_S2_CONTROL_COMPLETED", "M2_S2_COMPLETED_NOT_PROMOTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
