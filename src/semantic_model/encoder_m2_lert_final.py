"""Final, single-attempt LERT-small M2 specialist candidate.

This is a new model lineage.  It shares LERT's embeddings and first eleven
Transformer blocks, while every task owns an independent twelfth block and
head.  Reasoning is always trained first and gates the remaining six heads.
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
from . import encoder_m2_final_specialist as rbt3
from . import encoder_m2_reasoning_r2 as r2
from . import encoder_m2_s1 as s1
from . import encoder_m2_s3 as s3
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "M2-LERT-FINAL-SPECIALIST-SEVEN-HEAD-CANDIDATE"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-lert-final-specialist-run.v1"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
SEEDS = (35, 71, 107)
REASONING_HEAD = "reasoning_tags"
MAX_EPOCHS = 24
PATIENCE = 3
BATCH_SIZE = 16
MAX_LENGTH = 256
BLOCK_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_WALL_TIME_SECONDS = 7200
MAX_NEW_DISK_GIB = 10
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-lert-final-specialist-seven-head-20260902"
DOWNLOAD_PATTERNS = ("README.md", "config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "added_tokens.json")


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
        raise ContractError(code, "required JSON is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(code, "required JSON is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), code, "required JSON root must be an object")
    return value


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _contract_requirements(path: str | Path) -> dict[str, Any]:
    frozen = s1._contract_requirements(path)
    contract = frozen["contract"]
    final = _mapping(contract.get("lert_final_specialist_candidate_contract"), "M2_LERT_CONTRACT_INVALID", "LERT final specialist contract")
    model = _mapping(final.get("model"), "M2_LERT_CONTRACT_INVALID", "model")
    _require(final.get("contract_id") == "M2_LERT_FINAL_SPECIALIST_SEVEN_HEAD_V1", "M2_LERT_CONTRACT_INVALID", "wrong LERT contract id")
    _require(model.get("model_id") == MODEL_ID and model.get("revision") == REVISION and model.get("license") == LICENSE, "M2_LERT_MODEL_CONTRACT_INVALID", "LERT model identity changed")
    _require(model.get("trust_remote_code") is False and "FORBIDDEN" in str(model.get("warm_start")), "M2_LERT_MODEL_CONTRACT_INVALID", "LERT must use original weights without warm-start")
    architecture = _mapping(final.get("architecture"), "M2_LERT_CONTRACT_INVALID", "architecture")
    _require(architecture.get("heads") == list(V1_HEADS), "M2_LERT_ARCHITECTURE_INVALID", "seven heads changed")
    data = _mapping(final.get("data_scope"), "M2_LERT_CONTRACT_INVALID", "data scope")
    _require(data.get("train_rows") == 1822 and data.get("dev_rows") == 448, "M2_LERT_DATA_CONTRACT_INVALID", "Train/Dev counts changed")
    training = _mapping(final.get("training"), "M2_LERT_CONTRACT_INVALID", "training")
    expected = {"seeds": list(SEEDS), "max_length": MAX_LENGTH, "truncation": "HEAD_TAIL", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "gradient_clip_norm": 1.0}
    _require(all(training.get(k) == v for k, v in expected.items()), "M2_LERT_TRAINING_CONTRACT_INVALID", "fixed training configuration changed")
    opt = _mapping(training.get("optimizer"), "M2_LERT_CONTRACT_INVALID", "optimizer")
    _require(opt.get("name") == "AdamW" and opt.get("block_learning_rate") == BLOCK_LR and opt.get("head_learning_rate") == HEAD_LR and opt.get("weight_decay") == WEIGHT_DECAY, "M2_LERT_OPTIMIZER_CONTRACT_INVALID", "optimizer changed")
    _require(training.get("device_policy") == "MPS_FIRST_CPU_FALLBACK" and training.get("per_run_wall_time_seconds") == 7200 and training.get("total_new_disk_gib") == 10, "M2_LERT_RESOURCE_CONTRACT_INVALID", "resource limits changed")
    return {**frozen, "final": final}


def _download_snapshot(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    snapshot = cache_dir / "official-snapshot" / REVISION
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(snapshot), local_dir_use_symlinks=False, allow_patterns=list(DOWNLOAD_PATTERNS))
        except Exception as exc:
            raise ContractError("M2_LERT_DOWNLOAD_FAILED", "fixed official LERT revision could not be downloaded", exception_type=type(exc).__name__, detail=str(exc)) from exc
    _require(snapshot.is_dir(), "M2_LERT_CACHE_MISSING", "LERT fixed snapshot directory is missing")
    files = m1._hash_tree(snapshot)
    _require(any(item["path"] in {"pytorch_model.bin", "model.safetensors"} for item in files), "M2_LERT_MODEL_FILE_MISSING", "LERT weights are absent from fixed snapshot")
    identity = {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "required_relative_directory": "official-snapshot/" + REVISION, "files": files, "content_address": content_addressed_id({"model_id": MODEL_ID, "revision": REVISION, "files": files})}
    return snapshot, identity


def validate_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    frozen = _contract_requirements(contract_path)
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_LERT_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, identity = _download_snapshot(Path(cache_dir).resolve())
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": identity, "schema": schema, "train": train, "dev": dev, "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version, "data_scope": "Train_1822_and_Dev_448_only"}, "worktree": str(Path(worktree).resolve())}


def _config(contract: Mapping[str, Any], order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract["frozen_input_and_common_training_configuration"], "M2_LERT_CONTRACT_INVALID", "common configuration")
    opt = _mapping(common["optimizer"], "M2_LERT_CONTRACT_INVALID", "optimizer")
    return {"stage_id": STAGE_ID, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "local_files_only": True, "trust_remote_code": False, "initialization_source": "OFFICIAL_FIXED_LERT_ORIGINAL_WEIGHTS_NO_WARM_START", "shared_components": ["tokenizer", "input_builder", "embeddings", "transformer_blocks_0_through_10"], "per_head_components": "independent_transformer_block_11_and_output_head", "max_length": MAX_LENGTH, "truncation": "HEAD_TAIL", "padding": common["padding"], "token_type_ids": "NOT_EMITTED", "batch_size": BATCH_SIZE, "head_dropout": common["head_dropout"], "stock_code_token_cap": common["stock_code_token_cap"], "stock_name_token_cap": common["stock_name_token_cap"], "class_order": {head: list(order[head]) for head in V1_HEADS}, "optimizer": {"name": "AdamW", "block_learning_rate": BLOCK_LR, "head_learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY, "betas": opt["betas"], "epsilon": opt["epsilon"]}, "stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": 0.0}, "gradient_clipping_max_norm": 1.0, "reasoning_probability_threshold": 0.5, "fit_population": "Train_1822_only", "dev_role": "early_stopping_and_diagnostic_only", "sealed_roles_not_loaded": ["Test", "Anchor", "Gold", "OOD", "reference_predictions"], "selected_candidate": False}


def _locate_last_block(encoder: Any) -> tuple[str, Any]:
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    _require(layers is not None and len(layers) >= 12, "M2_LERT_ENCODER_ARCHITECTURE_INVALID", "LERT must expose twelve Transformer blocks")
    return "encoder.encoder.layer.11", layers[-1]


def _make_model(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, dropout: float) -> Any:
    nn = torch.nn

    class LertSpecialistModel(nn.Module):
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
            self.active_head: str | None = None
            for block in self.specialist_blocks.values():
                for parameter in block.parameters():
                    parameter.requires_grad = False
            for classifier in self.heads.values():
                for parameter in classifier.parameters():
                    parameter.requires_grad = False

        def set_active_head(self, head: str) -> None:
            _require(head in V1_HEADS, "M2_LERT_HEAD_INVALID", "unknown head", head=head)
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
            _require("args" in captured, "M2_LERT_ENCODER_FORWARD_INVALID", "LERT final-block input was not captured")
            result: dict[str, Any] = {}
            for head in V1_HEADS:
                output = self.specialist_blocks[head](*captured["args"], **captured["kwargs"])
                hidden = output[0] if isinstance(output, tuple) else output
                result[head] = self.heads[head](self.dropout(hidden[:, 0, :]))
            return result

    return LertSpecialistModel()


def validate_trainable_parameters(model: Any, head: str) -> dict[str, Any]:
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M2_LERT_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "only selected block and head may train", head=head)
    _require(not any(name.startswith("encoder.") for name in actual), "M2_LERT_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "shared LERT encoder is trainable")
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
            _finite(torch, logits, "M2_LERT_NONFINITE_LOGITS", "non-finite reasoning logits")
            labels.extend(batch["labels"][REASONING_HEAD].detach().cpu().int().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return labels, probabilities


def _target_metrics(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str) -> dict[str, Any]:
    return m1.diagnostic_metrics(torch, model, tokenizer, records, config, device)[head]


def _cpu_reload(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, tokenizer: Any, config: Mapping[str, Any], checkpoint: Mapping[str, Any], record: m1.M1Record, head: str) -> dict[str, Any]:
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model.set_active_head(head)
    model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    model = model.to(torch.device("cpu")); model.eval()
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    with torch.no_grad():
        output = model(batch["input_ids"], batch["attention_mask"])[head]
    finite = bool(torch.isfinite(output).all().item())
    _require(finite, "M2_LERT_CPU_RELOAD_FAILED", "LERT CPU reload emitted non-finite output", head=head, seed=checkpoint.get("seed"))
    return {"required": True, "device": "cpu", "head": head, "sample_id": record.sample_id, "finite": finite, "output_shape": list(output.shape)}


def _seed_head(*, runtime: tuple[Any, Any, Any, Any], head: str, seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / head / f"seed-{seed}"; folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); _limits(started, root, "before_model_load", seed, head); rbt3._set_seed(torch, np, seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])); model.set_active_head(head); model = model.to(device)
    parameter_identity = validate_trainable_parameters(model, head)
    trainable_block = list(model.specialist_blocks[head].parameters()); trainable_head = list(model.heads[head].parameters()); trainable = [*trainable_block, *trainable_head]
    optimizer = torch.optim.AdamW([{"params": trainable_block, "lr": BLOCK_LR}, {"params": trainable_head, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    best, best_epoch, stale = -math.inf, 0, 0; log = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(True); shuffled = list(train); random.Random(seed + epoch).shuffle(shuffled); losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            _limits(started, root, "during_epoch", seed, head); batch = m1._as_batch(torch, tokenizer, shuffled[offset:offset + BATCH_SIZE], config, device); optimizer.zero_grad(set_to_none=True); logits = model(batch["input_ids"], batch["attention_mask"]); _finite(torch, logits[head], "M2_LERT_NONFINITE_LOGITS", "non-finite selected-head logits"); loss = s3._single_head_loss(torch, logits, batch, head); _finite(torch, loss, "M2_LERT_NONFINITE_LOSS", "non-finite loss"); loss.backward()
            for parameter in trainable:
                if parameter.grad is not None: _finite(torch, parameter.grad, "M2_LERT_NONFINITE_GRADIENT", "non-finite gradient")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        score_metric = _target_metrics(torch, model, tokenizer, dev, config, device, head); score = float(score_metric["macro_f1"]); improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": seed, "frozen_config": config, "provenance": provenance, "parameter_identity": parameter_identity, "specialist_block_prefix": f"specialist_blocks.{head}", "specialist_block_state_dict": {k: v.detach().cpu() for k, v in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {k: v.detach().cpu() for k, v in model.heads[head].state_dict().items()}}, folder / "specialist-checkpoint.pt")
        else: stale += 1
        m1._jsonl_append(log, {"head": head, "seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_primary_macro_f1": score, "improved": improved, "stale_epochs": stale}); _limits(started, root, "after_epoch", seed, head)
        if stale >= PATIENCE: break
    checkpoint_path = folder / "specialist-checkpoint.pt"; _require(checkpoint_path.is_file(), "M2_LERT_CHECKPOINT_MISSING", "checkpoint missing", head=head, seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True); best_model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])); best_model.set_active_head(head); best_model = best_model.to(device); best_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); best_model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    train_metric = _target_metrics(torch, best_model, tokenizer, train, config, device, head); dev_metric = _target_metrics(torch, best_model, tokenizer, dev, config, device, head); thresholds = None; train_summary = None
    if head == REASONING_HEAD:
        train_labels, train_prob = _predict_reasoning(torch, best_model, tokenizer, train, config, device); dev_labels, dev_prob = _predict_reasoning(torch, best_model, tokenizer, dev, config, device); thresholds, train_summary = r2.choose_thresholds(train_labels, train_prob, config["class_order"][REASONING_HEAD]); dev_metric = r2.metrics_from_probabilities(dev_labels, dev_prob, config["class_order"][REASONING_HEAD], thresholds); m1._json_dump(folder / "reasoning-thresholds.json", {"selection_population": "Train_1822_only", "thresholds": thresholds, "train_summary": train_summary, "dev_metrics_after_frozen_thresholds": dev_metric})
    smoke = _cpu_reload(torch, AutoModel, snapshot, schema, tokenizer, config, checkpoint, dev[0], head); _limits(started, root, "after_cpu_reload", seed, head)
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "head": head, "seed": seed, "best_epoch": best_epoch, "early_stopping_metric": f"{head}.macro_f1", "sample_counts": {"train": len(train), "dev": len(dev)}, "train": {head: train_metric}, "dev": {head: dev_metric}, "thresholds": thresholds, "train_threshold_summary": train_summary, "cpu_reload_inference_smoke": smoke}; m1._json_dump(folder / "seed-metrics.json", metrics)
    resource = {"head": head, "seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}; m1._json_dump(folder / "resource-log.json", resource); _limits(started, root, "after_seed_evidence", seed, head)
    return {"head": head, "seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "thresholds_sha256": sha256_file(folder / "reasoning-thresholds.json") if head == REASONING_HEAD else None}


def _gates(contract: Mapping[str, Any], reasoning_results: Sequence[Mapping[str, Any]], all_results: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    view = {**contract, "final_specialist_candidate_contract": contract["lert_final_specialist_candidate_contract"]}
    reasoning_gate = rbt3._reasoning_gate(view, reasoning_results)
    final_gate = rbt3._classical_gate(view, all_results) if all_results is not None else None
    return reasoning_gate, final_gate


def _unified_bundle(runtime: tuple[Any, Any, Any, Any], root: Path, snapshot: Path, schema: Any, config: Mapping[str, Any], results: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any], record: m1.M1Record) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime; selected = [item for item in results if item["seed"] == 35]; _require(len(selected) == len(V1_HEADS), "M2_LERT_BUNDLE_INCOMPLETE", "seed-35 bundle requires seven heads")
    model = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])); bundle_dir = root / "unified-bundle"; bundle_dir.mkdir(parents=True, exist_ok=False); blocks: dict[str, Any] = {}; heads: dict[str, Any] = {}
    for item in selected:
        checkpoint = torch.load(root / item["head"] / "seed-35" / "specialist-checkpoint.pt", map_location="cpu", weights_only=True); blocks[item["head"]] = checkpoint["specialist_block_state_dict"]; heads[item["head"]] = checkpoint["head_state_dict"]; model.specialist_blocks[item["head"]].load_state_dict(blocks[item["head"]]); model.heads[item["head"]].load_state_dict(heads[item["head"]])
    path = bundle_dir / "seed-35-unified-bundle.pt"; torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "seed": 35, "provenance": provenance, "shared_bottom": "official_lert_embeddings_and_transformer_blocks_0_through_10", "specialist_block_state_dict": blocks, "heads_state_dict": heads}, path)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True); batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu")); model = model.to(torch.device("cpu")); model.eval()
    with torch.no_grad(): unified = model(batch["input_ids"], batch["attention_mask"])
    equivalence: dict[str, Any] = {}
    for head in V1_HEADS:
        checkpoint = torch.load(root / head / "seed-35" / "specialist-checkpoint.pt", map_location="cpu", weights_only=True); standalone = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])); standalone.set_active_head(head); standalone.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); standalone.heads[head].load_state_dict(checkpoint["head_state_dict"]); standalone.eval()
        with torch.no_grad(): solo = standalone(batch["input_ids"], batch["attention_mask"])[head]
        difference = float((unified[head] - solo).abs().max().item()); equivalence[head] = {"finite": bool(torch.isfinite(unified[head]).all().item() and torch.isfinite(solo).all().item()), "numerically_equivalent": difference <= 1e-6, "max_abs_diff": difference}
    m1._json_dump(bundle_dir / "cpu-equivalence.json", {"seed": 35, "per_head": equivalence, "all_finite": all(v["finite"] for v in equivalence.values()), "all_numerically_equivalent": all(v["numerically_equivalent"] for v in equivalence.values())})
    return {"bundle_path": str(path), "bundle_sha256": sha256_file(path), "equivalence_path": str(bundle_dir / "cpu-equivalence.json"), "equivalence_sha256": sha256_file(bundle_dir / "cpu-equivalence.json"), "equivalence": equivalence}


def _failure(root: Path | None, exc: ContractError, started_heads: Sequence[str], preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {"status": "LERT_SMALL_M2_CANDIDATE_REJECTED", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": bool(started_heads), "model_loaded": bool(started_heads), "cache_accessed": bool(preflight), "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False, "heads_started": list(started_heads)}
    if root and root.exists():
        m1._json_dump(root / "rejected-diagnostic-evidence.json", result); result["rejected_content_address"] = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-lert-final-specialist-rejected-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "selected_candidate": False, "m2_lineage": preflight["frozen_contract"]["final"]["lineage_id"] if preflight else None})["content_address"]
    return result


def run_lert(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path = DEFAULT_CACHE, *, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies, seed_executor: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]; contract_file = Path(contract_path).resolve() if contract_path else worktree_path / CONTRACT_RELATIVE_PATH; preflight: dict[str, Any] | None = None; started_heads: list[str] = []; root: Path | None = None
    try: preflight = validate_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
    except ContractError as exc: return _failure(None, exc, [], None)
    try:
        runtime = runtime_loader(); runtime_identity = _plain_json(s1.validate_runtime_identity(runtime, preflight["frozen_contract"])); root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"]); root.mkdir(parents=True, exist_ok=False); started = time.monotonic(); config = _config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order); m1._json_dump(root / "training-config.json", config); provenance = {**preflight["identity"], "runtime_identity": runtime_identity, "contract_sha256": preflight["frozen_contract"]["contract_sha256"], "cache_snapshot": preflight["snapshot_identity"]}; m1._json_dump(root / "provenance.json", provenance); execute = seed_executor or _seed_head
        reasoning: list[Mapping[str, Any]] = []
        for seed in SEEDS:
            started_heads.append(REASONING_HEAD); reasoning.append(execute(runtime=runtime, head=REASONING_HEAD, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, cache_dir=Path(cache_dir)))
        reasoning_gate, _ = _gates(preflight["frozen_contract"]["contract"], reasoning); m1._json_dump(root / "reasoning-fail-fast-gate.json", reasoning_gate)
        if not reasoning_gate["passed"]: return _failure(root, ContractError("M2_LERT_REASONING_CLASSICAL_GATE_FAILED", "LERT reasoning Classical gate failed", failures=[name for name, passed in reasoning_gate["checks"].items() if not passed]), started_heads, preflight)
        results = list(reasoning)
        for head in V1_HEADS:
            if head == REASONING_HEAD: continue
            for seed in SEEDS:
                started_heads.append(head); results.append(execute(runtime=runtime, head=head, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, cache_dir=Path(cache_dir)))
        gate = _gates(preflight["frozen_contract"]["contract"], reasoning, results)[1]; aggregate = {"stage_id": STAGE_ID, "seeds": list(SEEDS), "heads": list(V1_HEADS), "all_run_units_complete": True, "selected_candidate": False}; m1._json_dump(root / "stage-aggregate.json", aggregate); m1._json_dump(root / "final-classical-gate.json", gate); bundle = _unified_bundle(runtime, root, preflight["snapshot"], preflight["schema"], config, results, provenance, preflight["dev"][0]); m1._json_dump(root / "unified-bundle-evidence.json", bundle); _limits(started, root, "after_final_manifest")
        status = "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3" if gate["passed"] else "LERT_SMALL_M2_CANDIDATE_REJECTED"; manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-lert-final-specialist-artifact-manifest.v1", "stage_id": STAGE_ID, "status": status, "m2_lineage": preflight["frozen_contract"]["final"]["lineage_id"], "selected_candidate": gate["passed"], "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_policy": "MPS_FIRST_CPU_FALLBACK", "data_scope": "Train_1822_and_Dev_448_only", "seed35_predeclared_for_m3": True, "reasoning_fail_fast_gate_sha256": sha256_file(root / "reasoning-fail-fast-gate.json"), "final_classical_gate_sha256": sha256_file(root / "final-classical-gate.json"), "unified_bundle": bundle, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_inference_49054"]}); return {"status": status, "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": gate["passed"], "output_dir": str(root), "content_address": manifest["content_address"], "reasoning_gate": reasoning_gate, "final_gate": gate, "unified_bundle": bundle}
    except ContractError as exc: return _failure(root, exc, started_heads, preflight)
    except Exception as exc: return _failure(root, ContractError("M2_LERT_RUNTIME_EXCEPTION", "LERT final specialist runtime exception", exception_type=type(exc).__name__, detail=str(exc)), started_heads, preflight)


def _limits(start: float, root: Path, phase: str, seed: int | None = None, head: str | None = None) -> None:
    _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_LERT_WALL_TIME_LIMIT_EXCEEDED", "two-hour limit exceeded", phase=phase, seed=seed, head=head)
    _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_LERT_DISK_LIMIT_EXCEEDED", "ten GiB limit exceeded", phase=phase, seed=seed, head=head)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the final LERT-small specialist candidate")
    parser.add_argument("--config", required=True); parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE)); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(argv); result = run_lert(args.config, args.output_dir, args.cache_dir); (sys.stdout if result.get("status") in {"M2_SELECTED_CANDIDATE_FROZEN_FOR_M3", "LERT_SMALL_M2_CANDIDATE_REJECTED"} else sys.stderr).write(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"); return 0 if result.get("status") == "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3" else 2


if __name__ == "__main__":
    raise SystemExit(main())
