"""RBT3 reasoning-specific corrective diagnostic for M2.

This runner is intentionally narrower than S1/S2: it initializes the same
seven-head model as S3 from the fixed local RBT3 snapshot, freezes the
embeddings and first two Transformer blocks, and trains only the final block
and ``reasoning_tags`` head.  It emits reasoning evidence only; it can never
produce a seven-head candidate or a production decision.
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
from . import encoder_m2_s3 as s3
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS


STAGE_ID = "RBT3_REASONING_CORRECTIVE_V1"
SEEDS = (35, 71, 107)
MODEL_ID = "hfl/rbt3"
REVISION = "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"
LICENSE = "Apache-2.0"
BLOCK_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 16
MAX_LENGTH = 256
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_NEW_DISK_GIB = 10
TARGET_HEAD = "reasoning_tags"
EXPECTED_S1 = "04a23d76413049e57ff083655f80ad8c3dfc7ed90702a3c9ed66bcfd79f377f6"
EXPECTED_S2 = "3e452b6105df731abc869adebe73910bfa60d6abf196540765e72f8145932446"
EXPECTED_S3 = "7928614bdda834d0de6e3cc6b8d26bc02a10c821c4564dfa61e0ad419ac8899c"
DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[3] / "model-artifacts"
DEFAULT_S1_ARTIFACT = DEFAULT_ARTIFACT_ROOT / "m2-s1-first-three-seed-20260831"
DEFAULT_S2_ARTIFACT = DEFAULT_ARTIFACT_ROOT / "m2-s2-partial-last-one-three-seed-20260831"
DEFAULT_S3_ARTIFACT = DEFAULT_ARTIFACT_ROOT / "m2-s3-frozen-single-task-triggered-heads-20260831"


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


def validate_corrective_preflight(
    config_path: str | Path,
    cache_dir: str | Path,
    *,
    worktree: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Validate only the frozen Train/Dev technical scope before imports."""

    frozen = s1._contract_requirements(contract_path)
    contract = frozen["contract"]
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_CORRECTIVE_CONTRACT_INVALID", "common training configuration")
    _require(frozen.get("contract_sha256"), "M2_CORRECTIVE_CONTRACT_INVALID", "contract hash is missing")
    _require(str(contract.get("status", "")).startswith("M2_"), "M2_CORRECTIVE_CONTRACT_INVALID", "M2 contract is not frozen")
    _require(contract.get("prohibitions", {}).get("test_evaluation") is False, "M2_CORRECTIVE_CONTRACT_INVALID", "Test must remain prohibited")
    _require(common.get("max_length") == MAX_LENGTH and common.get("batch_size") == BATCH_SIZE and common.get("truncation") == "HEAD_TAIL", "M2_CORRECTIVE_TRAINING_CONTRACT_INVALID", "input configuration changed")
    early = _mapping(common.get("early_stopping"), "M2_CORRECTIVE_TRAINING_CONTRACT_INVALID", "early stopping")
    gradients = _mapping(common.get("gradient_controls"), "M2_CORRECTIVE_TRAINING_CONTRACT_INVALID", "gradient controls")
    _require(early.get("max_epochs") == MAX_EPOCHS and early.get("patience_epochs") == PATIENCE and gradients.get("gradient_clipping_max_norm") == 1.0, "M2_CORRECTIVE_TRAINING_CONTRACT_INVALID", "stopping or gradient controls changed")
    config = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_CORRECTIVE_DATA_CONTRACT_INVALID", "expected Train 1822 and Dev 448")
    snapshot, snapshot_identity = s1.validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {
        "frozen_contract": frozen,
        "snapshot": snapshot,
        "snapshot_identity": snapshot_identity,
        "schema": schema,
        "train": train,
        "dev": dev,
        "identity": {
            "contract_sha256": frozen["contract_sha256"],
            "model_id": MODEL_ID,
            "revision": REVISION,
            "license": LICENSE,
            "train_rows": len(train),
            "dev_rows": len(dev),
            "schema_version": schema.schema_version,
            "data_scope": "Train_1822_and_Dev_448_only",
        },
        "worktree": str(Path(worktree).resolve()),
    }


def _set_seed(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def _reasoning_head_state_digest(torch: Any, model: Any) -> str:
    """Hash the initial head tensors without serializing the full model."""

    import hashlib

    digest = hashlib.sha256()
    for name, tensor in model.heads[TARGET_HEAD].state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_reasoning_head_snapshot(
    torch: Any,
    np: Any,
    seed: int,
    model_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Create a deterministic S3-compatible initial head identity.

    Both S3 and corrective call the shared M1 constructor after this exact
    seed setup.  Keeping this helper injectable makes the parity test fully
    synthetic and prevents CI from loading a model.
    """

    _set_seed(torch, np, seed)
    model = model_factory()
    return {"seed": seed, "reasoning_head_sha256": _reasoning_head_state_digest(torch, model)}


def _make_model(torch: Any, AutoModel: Any, snapshot_path: Path, schema: Any, dropout: float) -> tuple[Any, Any, str]:
    """Match S3 construction order while retaining gradients for the last block."""

    nn = torch.nn

    class ReasoningCorrectiveModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Keep this order identical to m1._make_model/S3: base Encoder,
            # dropout, then the seven heads in schema order.
            self.encoder = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True, trust_remote_code=False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, len(schema.class_order[head])) for head in V1_HEADS})
            inner_prefix, self.last_block, _ = s2._locate_last_transformer_block(self.encoder)
            self.last_transformer_block_prefix = f"encoder.{inner_prefix}"
            for parameter in self.last_block.parameters():
                parameter.requires_grad = True
            for head_name, head in self.heads.items():
                for parameter in head.parameters():
                    parameter.requires_grad = head_name == TARGET_HEAD

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            self.last_block.train(mode)
            self.dropout.train(mode)
            self.heads.train(mode)
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            # Frozen parameters do not accumulate gradients, but the Encoder
            # forward must remain grad-enabled so the final block can adapt.
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            representation = self.dropout(encoded.last_hidden_state[:, 0, :])
            return {head: classifier(representation) for head, classifier in self.heads.items()}

    model = ReasoningCorrectiveModel()
    return model, model.last_block, model.last_transformer_block_prefix


def validate_trainable_parameters(model: Any, last_block: Any, block_prefix: str) -> dict[str, Any]:
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"{block_prefix}.") or name.startswith(f"heads.{TARGET_HEAD}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M2_CORRECTIVE_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "only the final Transformer block and reasoning head may require gradients", expected=sorted(expected), observed=sorted(actual))
    unexpected = sorted(name for name in actual if name.startswith("encoder.") and not name.startswith(f"{block_prefix}."))
    _require(not unexpected, "M2_CORRECTIVE_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "embeddings or earlier Transformer blocks are trainable", unexpected=unexpected)
    frozen_heads = sorted(head for head in model.heads if head != TARGET_HEAD and any(parameter.requires_grad for parameter in model.heads[head].parameters()))
    _require(not frozen_heads, "M2_CORRECTIVE_TRAINABLE_PARAMETER_CONTRACT_VIOLATION", "non-reasoning heads are trainable", heads=frozen_heads)
    return {"last_transformer_block_prefix": block_prefix, "trainable_parameters": sorted(actual), "frozen_heads": [head for head in V1_HEADS if head != TARGET_HEAD]}


def _config(contract: Mapping[str, Any], order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract["frozen_input_and_common_training_configuration"], "M2_CORRECTIVE_CONTRACT_INVALID", "common training configuration")
    optimizer = _mapping(common["optimizer"], "M2_CORRECTIVE_CONTRACT_INVALID", "optimizer")
    early = _mapping(common["early_stopping"], "M2_CORRECTIVE_CONTRACT_INVALID", "early stopping")
    return {
        "stage_id": STAGE_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": LICENSE,
        "trust_remote_code": False,
        "local_files_only": True,
        "initialization_source": "OFFICIAL_FIXED_RBT3_ORIGINAL_WEIGHTS_NO_WARM_START",
        "initialization_matching": "S3_MATCHING_SEED_MODEL_AND_REASONING_HEAD_ORDER",
        "input_builder_version": common["input_builder_version"],
        "stock_code_token_cap": common["stock_code_token_cap"],
        "stock_name_token_cap": common["stock_name_token_cap"],
        "max_length": MAX_LENGTH,
        "truncation": "HEAD_TAIL",
        "padding": common["padding"],
        "token_type_ids": "NOT_EMITTED",
        "batch_size": BATCH_SIZE,
        "head_dropout": common["head_dropout"],
        "class_order": {head: list(order[head]) for head in V1_HEADS},
        "optimizer": {"name": "AdamW", "block_learning_rate": BLOCK_LEARNING_RATE, "reasoning_head_learning_rate": HEAD_LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "betas": optimizer["betas"], "epsilon": optimizer["epsilon"], "parameter_groups": ["final_transformer_block", "reasoning_tags_head"]},
        "stopping": {"metric": "reasoning_tags.macro_f1", "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": early["minimum_delta"], "restore_best_checkpoint": True},
        "gradient_clipping_max_norm": 1.0,
        "reasoning_probability_threshold": 0.5,
        "encoder_state": "EMBEDDINGS_AND_FIRST_TWO_BLOCKS_FROZEN_FINAL_BLOCK_TRAINABLE",
        "trainable_heads": [TARGET_HEAD],
        "frozen_heads_not_reported": [head for head in V1_HEADS if head != TARGET_HEAD],
        "fit_population": "Train_1822_only",
        "dev_role": "reasoning_macro_f1_early_stopping_and_diagnostic_only",
        "test_role": "not_loaded_not_used",
        "comparison_roles": {"S1": "matching_seed_frozen_shared_control", "S2": "descriptive_only_head_lr_and_stopping_differ", "S3": "matching_seed_single_variable_last_block_adaptation"},
        "selected_candidate": False,
    }


def _reasoning_metrics(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    metrics = m1.diagnostic_metrics(torch, model, tokenizer, records, config, device)[TARGET_HEAD]
    return metrics


def _finite(torch: Any, value: Any, code: str, message: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), code, message)


def _cpu_reload_corrective(torch: Any, model_factory: Callable[[], tuple[Any, Any, str]], checkpoint: Mapping[str, Any], tokenizer: Any, record: m1.M1Record, config: Mapping[str, Any]) -> dict[str, Any]:
    model, block, _ = model_factory()
    model = model.to(torch.device("cpu"))
    block.load_state_dict(checkpoint["last_transformer_block_state_dict"])
    model.heads[TARGET_HEAD].load_state_dict(checkpoint["reasoning_head_state_dict"])
    model.eval()
    batch = m1._as_batch(torch, tokenizer, [record], config, torch.device("cpu"))
    with torch.no_grad():
        outputs = model(batch["input_ids"], batch["attention_mask"])
    reasoning = outputs[TARGET_HEAD]
    finite = bool(torch.isfinite(reasoning).all().item())
    _require(finite, "M2_CORRECTIVE_CPU_RELOAD_SMOKE_FAILED", "CPU reload produced non-finite reasoning logits")
    return {"required": True, "device": "cpu", "sample_id": record.sample_id, "reasoning_logits_finite": finite, "reasoning_output_shape": list(reasoning.shape)}


def _seed(*, runtime: tuple[Any, Any, Any, Any], seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime
    folder = root / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    s1._limits(started, root, "before_model_load", seed)
    _set_seed(torch, np, seed)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model, last_block, block_prefix = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    model = model.to(device)
    parameter_identity = validate_trainable_parameters(model, last_block, block_prefix)
    initial_head_sha256 = _reasoning_head_state_digest(torch, model)
    s1._limits(started, root, "after_model_load", seed)
    trainable_block = list(last_block.parameters())
    trainable_head = list(model.heads[TARGET_HEAD].parameters())
    trainable = [*trainable_block, *trainable_head]
    optimizer = torch.optim.AdamW([{"params": trainable_block, "lr": BLOCK_LEARNING_RATE}, {"params": trainable_head, "lr": HEAD_LEARNING_RATE}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    _require([group["lr"] for group in optimizer.param_groups] == [BLOCK_LEARNING_RATE, HEAD_LEARNING_RATE], "M2_CORRECTIVE_OPTIMIZER_CONTRACT_INVALID", "optimizer groups changed")
    best, epoch_best, stale = -1.0, 0, 0
    log = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        last_block.train(True)
        shuffled = list(train)
        random.Random(seed + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            s1._limits(started, root, "during_epoch", seed)
            batch = m1._as_batch(torch, tokenizer, shuffled[offset : offset + BATCH_SIZE], config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            for head_name in V1_HEADS:
                _finite(torch, logits[head_name], "M2_CORRECTIVE_NONFINITE_LOGITS", f"non-finite {head_name} logits")
            loss = s3._single_head_loss(torch, logits, batch, TARGET_HEAD)
            _finite(torch, loss, "M2_CORRECTIVE_NONFINITE_LOSS", "non-finite reasoning loss")
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    _finite(torch, parameter.grad, "M2_CORRECTIVE_NONFINITE_GRADIENT", "non-finite reasoning gradient")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metric = _reasoning_metrics(torch, model, tokenizer, dev, config, device)
        score = float(dev_metric["macro_f1"])
        improved = score > best + float(config["stopping"]["minimum_delta"])
        if improved:
            best, epoch_best, stale = score, epoch, 0
            torch.save({"run_schema_version": "myresearcher.encoder-m2-reasoning-corrective-run.v1", "stage_id": STAGE_ID, "seed": seed, "frozen_config": config, "provenance": provenance, "parameter_identity": parameter_identity, "initial_reasoning_head_sha256": initial_head_sha256, "last_transformer_block_prefix": block_prefix, "last_transformer_block_state_dict": {key: value.detach().cpu() for key, value in last_block.state_dict().items()}, "reasoning_head_state_dict": {key: value.detach().cpu() for key, value in model.heads[TARGET_HEAD].state_dict().items()}}, folder / "reasoning-corrective-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log, {"seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses) / len(losses), 6), "dev_reasoning_macro_f1": score, "dev_reasoning_micro_f1": dev_metric["micro_f1"], "dev_reasoning_exact_set_accuracy": dev_metric["exact_set_accuracy"], "improved": improved, "stale_epochs": stale})
        s1._limits(started, root, "after_epoch", seed)
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "reasoning-corrective-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_CORRECTIVE_CHECKPOINT_MISSING", "reasoning checkpoint missing", seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model, best_block, best_prefix = _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    best_model = best_model.to(device)
    best_block.load_state_dict(checkpoint["last_transformer_block_state_dict"])
    best_model.heads[TARGET_HEAD].load_state_dict(checkpoint["reasoning_head_state_dict"])
    validate_trainable_parameters(best_model, best_block, best_prefix)
    train_metric = _reasoning_metrics(torch, best_model, tokenizer, train, config, device)
    dev_metric = _reasoning_metrics(torch, best_model, tokenizer, dev, config, device)
    s1._limits(started, root, "after_final_metrics", seed)
    try:
        smoke = _cpu_reload_corrective(torch, lambda: _make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc:
        raise ContractError("M2_CORRECTIVE_CPU_RELOAD_SMOKE_FAILED", "corrective CPU reload failed", cause=exc.code) from exc
    metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "seed": seed, "best_epoch": epoch_best, "early_stopping_metric": "reasoning_tags.macro_f1", "early_stopping_score": best, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": {TARGET_HEAD: train_metric}, "dev": {TARGET_HEAD: dev_metric}, "initial_reasoning_head_sha256": initial_head_sha256, "cpu_reload_inference_smoke": smoke}
    m1._json_dump(folder / "seed-metrics.json", metrics)
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    m1._json_dump(folder / "resource-log.json", resource)
    s1._limits(started, root, "after_seed_evidence", seed)
    return {"seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path)}


def _load_comparator(root: str | Path, expected_address: str, *, stage: str) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest = _read_json(root_path / "content-addressed-manifest.json", "M2_CORRECTIVE_COMPARATOR_MANIFEST_INVALID")
    _require(manifest.get("content_address") == expected_address and manifest.get("content_address") == content_addressed_id(manifest, omit_keys={"content_address"}), "M2_CORRECTIVE_COMPARATOR_ID_MISMATCH", f"{stage} artifact content address mismatch")
    _require(manifest.get("files") == m1._hash_tree(root_path, exclude={"content-addressed-manifest.json"}), "M2_CORRECTIVE_COMPARATOR_HASH_MISMATCH", f"{stage} artifact files changed")
    metrics: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        path = root_path / (f"seed-{seed}" / "seed-metrics.json" if stage != "S3" else f"reasoning_tags/seed-{seed}/seed-metrics.json")
        value = _read_json(path, "M2_CORRECTIVE_COMPARATOR_METRICS_INVALID")
        _require(value.get("sample_counts") == {"train": 1822, "dev": 448}, "M2_CORRECTIVE_COMPARATOR_METRICS_INVALID", f"{stage} sample counts differ")
        _require(TARGET_HEAD in value.get("dev", {}), "M2_CORRECTIVE_COMPARATOR_METRICS_INVALID", f"{stage} reasoning metrics missing")
        metrics[seed] = value
    return {"root": root_path, "manifest": manifest, "seed_metrics": metrics}


def _comparison(s1_control: Mapping[str, Any], s2_control: Mapping[str, Any], s3_control: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for comparator_name, comparator in (("S1", s1_control), ("S2", s2_control), ("S3", s3_control)):
        per_seed: dict[str, Any] = {}
        for seed, item in zip(SEEDS, results, strict=True):
            candidate = item["metrics"]["dev"][TARGET_HEAD]
            baseline = comparator["seed_metrics"][seed]["dev"][TARGET_HEAD]
            labels: dict[str, Any] = {}
            for label, baseline_row in baseline["per_label"].items():
                candidate_row = candidate["per_label"][label]
                labels[label] = {"support": int(baseline_row["support"]), "candidate_f1": float(candidate_row["f1"]), "baseline_f1": float(baseline_row["f1"]), "delta": round(float(candidate_row["f1"]) - float(baseline_row["f1"]), 6)}
            per_seed[str(seed)] = {"candidate_macro_f1": float(candidate["macro_f1"]), "baseline_macro_f1": float(baseline["macro_f1"]), "macro_delta": round(float(candidate["macro_f1"]) - float(baseline["macro_f1"]), 6), "candidate_micro_f1": float(candidate["micro_f1"]), "baseline_micro_f1": float(baseline["micro_f1"]), "micro_delta": round(float(candidate["micro_f1"]) - float(baseline["micro_f1"]), 6), "candidate_exact_set_accuracy": float(candidate["exact_set_accuracy"]), "baseline_exact_set_accuracy": float(baseline["exact_set_accuracy"]), "exact_delta": round(float(candidate["exact_set_accuracy"]) - float(baseline["exact_set_accuracy"]), 6), "per_label": labels}
        result[comparator_name] = per_seed
    return result


def _gate(s1_control: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = [float(item["metrics"]["dev"][TARGET_HEAD]["macro_f1"]) for item in results]
    s1_values = [float(s1_control["seed_metrics"][seed]["dev"][TARGET_HEAD]["macro_f1"]) for seed in SEEDS]
    candidate_micro = [float(item["metrics"]["dev"][TARGET_HEAD]["micro_f1"]) for item in results]
    s1_micro = [float(s1_control["seed_metrics"][seed]["dev"][TARGET_HEAD]["micro_f1"]) for seed in SEEDS]
    candidate_exact = [float(item["metrics"]["dev"][TARGET_HEAD]["exact_set_accuracy"]) for item in results]
    s1_exact = [float(s1_control["seed_metrics"][seed]["dev"][TARGET_HEAD]["exact_set_accuracy"]) for seed in SEEDS]
    cand_labels = [[float(item["metrics"]["dev"][TARGET_HEAD]["per_label"]["NO_REASON_GIVEN"]["f1"]) for item in results]]
    no_reason_candidate = cand_labels[0]
    no_reason_s1 = [float(s1_control["seed_metrics"][seed]["dev"][TARGET_HEAD]["per_label"]["NO_REASON_GIVEN"]["f1"]) for seed in SEEDS]
    macro_mean, s1_macro_mean = statistics.mean(candidate), statistics.mean(s1_values)
    macro_std = statistics.stdev(candidate)
    no_reason_deltas = [right - left for left, right in zip(no_reason_s1, no_reason_candidate, strict=True)]
    checks = {"macro_mean_improvement_at_least_0_01": macro_mean >= s1_macro_mean + 0.01, "micro_mean_drop_at_most_0_01": statistics.mean(candidate_micro) - statistics.mean(s1_micro) >= -0.01, "exact_mean_drop_at_most_0_01": statistics.mean(candidate_exact) - statistics.mean(s1_exact) >= -0.01, "no_reason_mean_not_below_s1": statistics.mean(no_reason_candidate) >= statistics.mean(no_reason_s1), "no_reason_worst_seed_drop_at_most_0_03": min(no_reason_deltas) >= -0.03, "macro_std_at_most_0_05": macro_std <= 0.05}
    return {"comparator": "S1_MATCHING_SEED", "candidate_macro_f1": {"per_seed": candidate, "mean": macro_mean, "sample_standard_deviation": macro_std, "worst_seed": min(candidate)}, "s1_macro_f1": {"per_seed": s1_values, "mean": s1_macro_mean}, "candidate_micro_f1": {"per_seed": candidate_micro, "mean": statistics.mean(candidate_micro)}, "s1_micro_f1": {"per_seed": s1_micro, "mean": statistics.mean(s1_micro)}, "candidate_exact_set_accuracy": {"per_seed": candidate_exact, "mean": statistics.mean(candidate_exact)}, "s1_exact_set_accuracy": {"per_seed": s1_exact, "mean": statistics.mean(s1_exact)}, "no_reason_given": {"candidate_per_seed": no_reason_candidate, "s1_per_seed": no_reason_s1, "delta_per_seed": [round(value, 6) for value in no_reason_deltas], "candidate_mean": statistics.mean(no_reason_candidate), "s1_mean": statistics.mean(no_reason_s1), "worst_seed_delta": min(no_reason_deltas)}, "checks": checks, "passed": all(checks.values()), "selected_candidate": False, "diagnostic_only": True}


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require([item.get("seed") for item in results] == list(SEEDS), "M2_CORRECTIVE_INCOMPLETE_SEEDS", "all three corrective seeds are required")
    devices = [str(item["resource"]["actual_device"]) for item in results]
    _require(len(set(devices)) == 1, "M2_CORRECTIVE_MIXED_DEVICE", "corrective run cannot aggregate mixed-device seeds", devices={str(item["seed"]): item["resource"]["actual_device"] for item in results})
    values = [float(item["metrics"]["dev"][TARGET_HEAD]["macro_f1"]) for item in results]
    _require(all(math.isfinite(value) for value in values), "M2_CORRECTIVE_METRICS_NONFINITE", "corrective aggregate has non-finite metrics")
    return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "actual_device": devices[0], "all_seeds_complete": True, "reasoning_tags": {"primary_macro_f1": {"per_seed_values": values, "mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values), "worst_seed": min(values)}}, "selected_candidate": False, "allowed_output": "REASONING_DIAGNOSTIC_ONLY"}


def _failure(root: Path | None, exc: ContractError, preflight: Mapping[str, Any] | None, entered: bool) -> dict[str, Any]:
    result = {"status": "REJECTED_DIAGNOSTIC_ONLY", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": entered, "model_loaded": entered, "cache_accessed": entered, "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False}
    if root and root.exists():
        m1._json_dump(root / "rejected-diagnostic-evidence.json", result)
        payload = {"manifest_schema_version": "myresearcher.encoder-m2-reasoning-corrective-rejected-artifact-manifest.v1", "status": result["status"], "stage_id": STAGE_ID, "failure": result, "selected_candidate": False}
        if preflight:
            payload.update({"m2_lineage": preflight["frozen_contract"]["contract"].get("new_model_lineage"), "complete_cache_snapshot": preflight.get("snapshot_identity")})
        result["rejected_content_address"] = m1._write_content_manifest(root, payload)["content_address"]
    return result


def run_corrective(
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    s1_artifact: str | Path = DEFAULT_S1_ARTIFACT,
    s2_artifact: str | Path = DEFAULT_S2_ARTIFACT,
    s3_artifact: str | Path = DEFAULT_S3_ARTIFACT,
    worktree: str | Path | None = None,
    contract_path: str | Path | None = None,
    runtime_loader: Callable[[], tuple[Any, Any, Any, Any]] = m1._load_runtime_dependencies,
    seed_executor: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]
    contract_file = Path(contract_path).resolve() if contract_path else worktree_path / s1.CONTRACT_RELATIVE_PATH
    preflight: dict[str, Any] | None = None
    try:
        preflight = validate_corrective_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract_file)
        s1_control = _load_comparator(s1_artifact, EXPECTED_S1, stage="S1")
        s2_control = _load_comparator(s2_artifact, EXPECTED_S2, stage="S2")
        s3_control = _load_comparator(s3_artifact, EXPECTED_S3, stage="S3")
    except ContractError as exc:
        return _failure(None, exc, preflight, False)
    root: Path | None = None
    results: list[Mapping[str, Any]] = []
    entered = False
    started = time.monotonic()
    try:
        runtime = runtime_loader()
        runtime_identity = s1.validate_runtime_identity(runtime, preflight["frozen_contract"])
        root = s1.validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"])
        root.mkdir(parents=True, exist_ok=False)
        s1._limits(started, root, "before_fit")
        config = _config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order)
        m1._json_dump(root / "training-config.json", config)
        provenance = {**preflight["identity"], "contract_sha256": preflight["frozen_contract"]["contract_sha256"], "runtime_identity": runtime_identity, "s1_artifact_content_address": EXPECTED_S1, "s2_artifact_content_address": EXPECTED_S2, "s3_artifact_content_address": EXPECTED_S3}
        m1._json_dump(root / "provenance.json", provenance)
        executor = seed_executor or _seed
        for seed in SEEDS:
            entered = True
            results.append(executor(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=provenance, cache_dir=Path(cache_dir)))
        aggregate = aggregate_results(results)
        comparison = _comparison(s1_control, s2_control, s3_control, results)
        gate = _gate(s1_control, results)
        m1._json_dump(root / "stage-aggregate.json", aggregate)
        m1._json_dump(root / "corrective-vs-s1-s2-s3-report.json", {"stage_id": STAGE_ID, "comparison": comparison, "gate": gate, "selected_candidate": False, "diagnostic_only": True})
        s1._limits(started, root, "after_comparison_report")
        manifest = m1._write_content_manifest(root, {"manifest_schema_version": "myresearcher.encoder-m2-reasoning-corrective-artifact-manifest.v1", "stage_id": STAGE_ID, "status": "PASSED_DIAGNOSTIC_ONLY" if gate["passed"] else "REJECTED_DIAGNOSTIC_ONLY", "diagnostic_only": True, "selected_candidate": False, "m2_lineage": preflight["frozen_contract"]["contract"].get("new_model_lineage"), "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "complete_cache_snapshot": preflight["snapshot_identity"], "runtime_identity": runtime_identity, "device_identity": {"actual_device": aggregate["actual_device"], "policy": "MPS_FIRST_CPU_FALLBACK"}, "s1_control": {"content_address": EXPECTED_S1, "artifact_path": str(s1_control["root"])}, "s2_control": {"content_address": EXPECTED_S2, "artifact_path": str(s2_control["root"])}, "s3_control": {"content_address": EXPECTED_S3, "artifact_path": str(s3_control["root"])}, "comparison_report_sha256": sha256_file(root / "corrective-vs-s1-s2-s3-report.json"), "stage_aggregate_sha256": sha256_file(root / "stage-aggregate.json"), "training_config_sha256": sha256_file(root / "training-config.json"), "provenance_sha256": sha256_file(root / "provenance.json"), "seed_checkpoints": {str(item["seed"]): item["checkpoint_sha256"] for item in results}, "initial_reasoning_head_sha256": {str(item["seed"]): item["metrics"]["initial_reasoning_head_sha256"] for item in results}, "comparison_scope": "S1_matching_seed; S2_descriptive_only; S3_single_variable_last_block_adaptation", "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "reference_predictions", "LLM", "cloud", "production_inference"]})
        s1._limits(started, root, "after_final_manifest")
        return {"status": manifest["status"], "stage_id": STAGE_ID, "training_invoked": True, "model_loaded": True, "cache_accessed": True, "output_created": True, "aggregate_created": True, "selected_candidate": False, "output_dir": str(root), "content_address": manifest["content_address"], "aggregate": aggregate, "gate": gate, "comparison": comparison}
    except ContractError as exc:
        return _failure(root, exc, preflight, entered)
    except Exception as exc:
        return _failure(root, ContractError("M2_CORRECTIVE_RUNTIME_EXCEPTION", "corrective runtime exception", exception_type=type(exc).__name__, detail=str(exc)), preflight, entered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RBT3 reasoning corrective diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--s1-artifact", default=str(DEFAULT_S1_ARTIFACT))
    parser.add_argument("--s2-artifact", default=str(DEFAULT_S2_ARTIFACT))
    parser.add_argument("--s3-artifact", default=str(DEFAULT_S3_ARTIFACT))
    args = parser.parse_args(argv)
    result = run_corrective(args.config, args.output_dir, args.cache_dir, s1_artifact=args.s1_artifact, s2_artifact=args.s2_artifact, s3_artifact=args.s3_artifact)
    stream = sys.stdout if result.get("status") in {"PASSED_DIAGNOSTIC_ONLY", "REJECTED_DIAGNOSTIC_ONLY"} and result.get("training_invoked") else sys.stderr
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result.get("status") == "PASSED_DIAGNOSTIC_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
