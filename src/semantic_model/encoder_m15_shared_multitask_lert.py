"""M15: identity-stripped shared multi-task LERT fine-tuning.

Earlier human-first encoder runs trained separate specialist blocks per head.
M15 instead fine-tunes one shared top-four-layer encoder while every mini-batch
supervises all seven heads.  The representation is therefore updated by seven
human judgements per sample rather than by one isolated head at a time.

The run is bounded: one architecture, one optimiser configuration, one seed,
six OOF epochs and one globally selected epoch.  Human147 is the only fit and
selection population.  Recent Check19, Human Check20 and Human48 are loaded
only after the final checkpoint and reasoning thresholds have frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_m12_human_first_cv as m12
from . import encoder_m12_hybrid_runtime as hybrid
from . import encoder_m14_human_char_tfidf as m14
from . import encoder_terminal_macbert_base as terminal
from .errors import ContractError
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


STAGE_ID = "M15-SHARED-MULTITASK-LERT-HUMAN147"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m15-shared-multitask.v1"
SEED = 35
N_FOLDS = 5
MAX_EPOCHS = 6
BATCH_SIZE = 8
MAX_LENGTH = 192
ENCODER_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
TRAINABLE_TOP_LAYERS = 4
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_POOL = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
DEFAULT_RECENT_FEED = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "feed" / "feed-input.jsonl"
DEFAULT_RECENT_TRAIN = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "human" / "recent-train50-final.jsonl"
DEFAULT_RECENT_CHECK = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "human" / "recent-check19-valid-final.jsonl"
DEFAULT_HUMAN48 = Path("/tmp/human48.jsonl")
DEFAULT_HUMAN48_INPUT = REPO_ROOT / "runs" / "m3c-shadow500-20260907" / "shadow500-input.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m15-shared-multitask-lert-human147-20260916"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _set_seed(torch: Any, value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(value)
        except Exception:
            pass


def multitask_fold_indices(
    records: Sequence[m1.M1Record],
    recent_ids: set[str],
    *,
    folds: int = N_FOLDS,
    seed: int = SEED,
) -> list[tuple[list[int], list[int]]]:
    """Stratify jointly on cohort and stance for one shared five-fold split."""

    _require(len(records) >= folds, "M15_FOLD_COUNT_INVALID", "not enough records for M15 folds")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        cohort = "recent" if record.sample_id in recent_ids else "base"
        groups.setdefault((cohort, str(record.label["stance"])), []).append(index)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    cursor = 0
    for key in sorted(groups):
        values = groups[key]
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(values)
        for offset, index in enumerate(values):
            buckets[(cursor + offset) % folds].append(index)
        cursor = (cursor + len(values)) % folds
    all_indices = set(range(len(records)))
    result: list[tuple[list[int], list[int]]] = []
    for bucket in buckets:
        validation = sorted(bucket)
        training = sorted(all_indices - set(validation))
        _require(validation and training, "M15_EMPTY_FOLD", "M15 produced an empty fold")
        _require(any(records[index].sample_id in recent_ids for index in training), "M15_RECENT_SOURCE_MISSING", "M15 fold train has no recent samples")
        _require(any(records[index].sample_id not in recent_ids for index in training), "M15_BASE_SOURCE_MISSING", "M15 fold train has no base samples")
        result.append((training, validation))
    _require(sorted(index for _training, validation in result for index in validation) == list(range(len(records))), "M15_FOLD_COVERAGE_INVALID", "M15 validation folds must cover each sample once")
    return result


def make_model(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema) -> Any:
    nn = torch.nn

    class SharedMultiTaskLert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
            layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            _require(layers is not None and len(layers) == 12, "M15_LERT_ARCHITECTURE_INVALID", "M15 requires twelve LERT blocks")
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            for layer in layers[-TRAINABLE_TOP_LAYERS:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(0.1)
            self.heads = nn.ModuleDict({head: nn.Linear(hidden, len(schema.class_order[head])) for head in V1_HEADS})

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            hidden = self.dropout(hidden)
            return {head: classifier(hidden) for head, classifier in self.heads.items()}

    return SharedMultiTaskLert()


def trainable_parameter_identity(model: Any) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    head_names = [name for name in names if name.startswith("heads.")]
    encoder_names = [name for name in names if name.startswith("encoder.encoder.layer.")]
    _require(len(head_names) == len(V1_HEADS) * 2, "M15_HEAD_SCOPE_INVALID", "all and only M15 output heads must train", observed=len(head_names))
    _require(encoder_names and all(any(f"encoder.encoder.layer.{index}." in name for index in range(8, 12)) for name in encoder_names), "M15_ENCODER_SCOPE_INVALID", "only top-four encoder layers may train")
    unexpected = [name for name in names if name not in head_names and name not in encoder_names]
    _require(not unexpected, "M15_TRAINABLE_SCOPE_INVALID", "M15 has unexpected trainable parameters", names=unexpected)
    return {"trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)), "trainable_top_layers": [8, 9, 10, 11], "heads": list(V1_HEADS)}


def _token_batch(torch: Any, tokenizer: Any, records: Sequence[m1.M1Record], device: Any) -> Mapping[str, Any]:
    encoded = tokenizer(
        m14._texts(records),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"].to(device), "attention_mask": encoded["attention_mask"].to(device)}


def _class_balance(torch: Any, records: Sequence[m1.M1Record], schema: LabelSchema, device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        order = list(schema.class_order[head])
        counts = Counter(str(record.label[head]) for record in records)
        present = [count for count in counts.values() if count > 0]
        largest = max(present)
        values = [math.sqrt(largest / counts[label]) if counts.get(label, 0) else 1.0 for label in order]
        mean = sum(values[index] for index, label in enumerate(order) if counts.get(label, 0)) / len(present)
        result[head] = torch.tensor([value / mean for value in values], dtype=torch.float32, device=device)
    order = list(schema.class_order["reasoning_tags"])
    positives = np.asarray([[int(label in record.label["reasoning_tags"]) for label in order] for record in records], dtype=float).sum(axis=0)
    negatives = len(records) - positives
    values = np.sqrt(np.maximum(negatives, 1.0) / np.maximum(positives, 1.0))
    result["reasoning_tags"] = torch.tensor(np.clip(values, 1.0, 3.0), dtype=torch.float32, device=device)
    return result


def source_weight_map(records: Sequence[m1.M1Record], recent_ids: set[str]) -> dict[str, float]:
    values = m14.source_balanced_weights(records, recent_ids)
    return {record.sample_id: float(values[index]) for index, record in enumerate(records)}


def _source_weights(torch: Any, records: Sequence[m1.M1Record], weights_by_id: Mapping[str, float], device: Any) -> Any:
    _require(all(record.sample_id in weights_by_id for record in records), "M15_SOURCE_WEIGHT_MISSING", "batch record lacks a frozen cohort weight")
    return torch.tensor([weights_by_id[record.sample_id] for record in records], dtype=torch.float32, device=device)


def _batch_loss(
    torch: Any,
    outputs: Mapping[str, Any],
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
    balances: Mapping[str, Any],
    sample_weights: Any,
    device: Any,
) -> tuple[Any, dict[str, float]]:
    losses: list[Any] = []
    observed: dict[str, float] = {}
    for head in SINGLE_LABEL_HEADS:
        order = list(schema.class_order[head])
        target = torch.tensor([order.index(str(record.label[head])) for record in records], dtype=torch.long, device=device)
        raw = torch.nn.functional.cross_entropy(outputs[head], target, weight=balances[head], reduction="none")
        value = (raw * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
        losses.append(value)
        observed[head] = float(value.detach().cpu())
    order = list(schema.class_order["reasoning_tags"])
    target = torch.tensor(
        [[float(label in record.label["reasoning_tags"]) for label in order] for record in records],
        dtype=torch.float32,
        device=device,
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["reasoning_tags"], target, pos_weight=balances["reasoning_tags"], reduction="none"
    ).mean(dim=1)
    reasoning = (raw * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
    losses.append(reasoning)
    observed["reasoning_tags"] = float(reasoning.detach().cpu())
    total = torch.stack(losses).mean()
    _require(bool(torch.isfinite(total).item()), "M15_NONFINITE_LOSS", "M15 produced non-finite loss")
    return total, observed


def _predict(
    torch: Any,
    model: Any,
    tokenizer: Any,
    records: Sequence[m1.M1Record],
    device: Any,
) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            current = records[offset : offset + BATCH_SIZE]
            batch = _token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for head in SINGLE_LABEL_HEADS:
                values[head].append(torch.softmax(outputs[head], dim=-1).detach().cpu().numpy())
            values["reasoning_tags"].append(torch.sigmoid(outputs["reasoning_tags"]).detach().cpu().numpy())
    return {head: np.concatenate(chunks, axis=0) for head, chunks in values.items()}


def _train_curve(
    *,
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    schema: LabelSchema,
    train_records: Sequence[m1.M1Record],
    validation_records: Sequence[m1.M1Record],
    recent_ids: set[str],
    fold: int,
    device: Any,
) -> list[dict[str, np.ndarray]]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    _set_seed(torch, SEED + fold)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = make_model(torch, AutoModel, snapshot, schema).to(device)
    trainable_parameter_identity(model)
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("heads.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": ENCODER_LR}, {"params": head_parameters, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    balances = _class_balance(torch, train_records, schema, device)
    weights_by_id = source_weight_map(train_records, recent_ids)
    curve: list[dict[str, np.ndarray]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        ordered = list(train_records)
        random.Random(SEED + fold * 1000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), BATCH_SIZE):
            current = ordered[offset : offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch = _token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            weights = _source_weights(torch, current, weights_by_id, device)
            loss, _head_losses = _batch_loss(torch, outputs, current, schema, balances, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], GRADIENT_CLIP)
            optimizer.step()
        curve.append(_predict(torch, model, tokenizer, validation_records, device))
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return curve


def select_global_epoch(
    oof_by_epoch: Sequence[Mapping[str, np.ndarray]],
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
) -> tuple[int, dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    best_key: tuple[float, float, int] | None = None
    best_epoch = 1
    best_thresholds: dict[str, float] = {}
    best_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for index, arrays in enumerate(oof_by_epoch):
        thresholds = m12.calibrate_reasoning_thresholds(arrays["reasoning_tags"], records, schema.class_order["reasoning_tags"])
        metrics = {
            head: m12._metrics(arrays[head], records, schema, head, thresholds if head == "reasoning_tags" else None)
            for head in V1_HEADS
        }
        macro = float(np.mean([metrics[head]["macro_f1"] for head in V1_HEADS]))
        primary = float(np.mean([metrics[head].get("accuracy", metrics[head].get("micro_f1", 0.0)) for head in V1_HEADS]))
        key = (macro, primary, -index)
        rows.append({"epoch": index + 1, "mean_head_macro_f1": macro, "mean_head_primary": primary})
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = index + 1
            best_thresholds = thresholds
            best_metrics = metrics
    return best_epoch, best_thresholds, best_metrics, rows


def _train_final(
    *,
    runtime: tuple[Any, Any, Any, Any],
    snapshot: Path,
    schema: LabelSchema,
    records: Sequence[m1.M1Record],
    recent_ids: set[str],
    epochs: int,
    output: Path,
    device: Any,
) -> dict[str, Any]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    _set_seed(torch, SEED + 100)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = make_model(torch, AutoModel, snapshot, schema).to(device)
    identity = trainable_parameter_identity(model)
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("heads.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": ENCODER_LR}, {"params": head_parameters, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    balances = _class_balance(torch, records, schema, device)
    weights_by_id = source_weight_map(records, recent_ids)
    updates = 0
    epoch_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        ordered = list(records)
        random.Random(SEED + 10000 + epoch).shuffle(ordered)
        model.train(True)
        batch_losses: list[float] = []
        for offset in range(0, len(ordered), BATCH_SIZE):
            current = ordered[offset : offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch = _token_batch(torch, tokenizer, current, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            weights = _source_weights(torch, current, weights_by_id, device)
            loss, _head_losses = _batch_loss(torch, outputs, current, schema, balances, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], GRADIENT_CLIP)
            optimizer.step()
            updates += 1
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))
    checkpoint = output / "shared-multitask-checkpoint.pt"
    torch.save(
        {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "epochs": epochs,
            "optimizer_updates": updates,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
            "parameter_identity": identity,
            "identity_stripping": True,
        },
        checkpoint,
    )
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return {"checkpoint": str(checkpoint), "epochs": epochs, "optimizer_updates": updates, "epoch_mean_losses": epoch_losses, "parameter_identity": identity}


def _load_final(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, output: Path) -> Any:
    checkpoint = torch.load(output / "shared-multitask-checkpoint.pt", map_location="cpu", weights_only=True)
    _require(checkpoint.get("stage_id") == STAGE_ID, "M15_CHECKPOINT_INVALID", "M15 checkpoint stage mismatch")
    model = make_model(torch, AutoModel, snapshot, schema)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _metric_table(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[m1.M1Record],
    schema: LabelSchema,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    return {
        head: m12._metrics(arrays[head], records, schema, head, thresholds if head == "reasoning_tags" else None)
        for head in V1_HEADS
    }


def run_m15(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    cache_dir: str | Path = DEFAULT_CACHE,
    selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT,
    pool_dir: str | Path = DEFAULT_POOL,
    recent_feed_jsonl: str | Path = DEFAULT_RECENT_FEED,
    recent_train_labels: str | Path = DEFAULT_RECENT_TRAIN,
    recent_check_labels: str | Path = DEFAULT_RECENT_CHECK,
    human48_jsonl: str | Path = DEFAULT_HUMAN48,
    human48_input: str | Path = DEFAULT_HUMAN48_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M15_OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    project = m1.ProjectConfig.load(config_path)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    base_train, _ = m5.load_human_train_pool(pool_dir)
    base_train, base_audit = m12.exclude_dependency_conflicts(base_train, role="base_human_train")
    recent_train, recent_audit = m12.load_recent_feedback_records(
        recent_train_labels,
        recent_feed_jsonl,
        role="recent_train",
        exclude_dependency_conflicts=True,
    )
    train = [*base_train, *recent_train]
    recent_ids = {record.sample_id for record in recent_train}
    _require(len(train) == 147 and len({record.sample_id for record in train}) == 147, "M15_TRAIN_INVALID", "M15 requires unique Human147")

    runtime = m1._load_runtime_dependencies()
    _np, torch, AutoModel, AutoTokenizer = runtime
    snapshot, snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
    device, device_name, mps_available = m1._choose_device(torch)
    splits = multitask_fold_indices(train, recent_ids)
    oof_by_epoch = [{head: np.zeros((len(train), len(schema.class_order[head])), dtype=float) for head in V1_HEADS} for _ in range(MAX_EPOCHS)]
    fold_sizes: list[dict[str, int]] = []
    for fold, (training_indices, validation_indices) in enumerate(splits):
        fit = [train[index] for index in training_indices]
        validation = [train[index] for index in validation_indices]
        curves = _train_curve(
            runtime=runtime,
            snapshot=snapshot,
            schema=schema,
            train_records=fit,
            validation_records=validation,
            recent_ids=recent_ids,
            fold=fold,
            device=device,
        )
        for epoch in range(MAX_EPOCHS):
            for head in V1_HEADS:
                oof_by_epoch[epoch][head][validation_indices] = curves[epoch][head]
        fold_sizes.append({"fold": fold, "train": len(fit), "validation": len(validation)})
    selected_epoch, reasoning_thresholds, oof_metrics, epoch_curve = select_global_epoch(oof_by_epoch, train, schema)
    _dump(output / "oof-selection.json", {"selected_epoch": selected_epoch, "epoch_curve": epoch_curve, "selected_metrics": oof_metrics, "fold_sizes": fold_sizes})
    _dump(output / "reasoning-thresholds.json", reasoning_thresholds)
    final_training = _train_final(
        runtime=runtime,
        snapshot=snapshot,
        schema=schema,
        records=train,
        recent_ids=recent_ids,
        epochs=selected_epoch,
        output=output,
        device=device,
    )

    # All independent human checks are opened only after the final checkpoint freezes.
    recent_check, recent_check_audit = m12.load_recent_feedback_records(recent_check_labels, recent_feed_jsonl, role="cumulative_recent_check")
    human_check, _ = m5.load_human_check_pool(pool_dir)
    human48 = terminal._load_human48(Path(human48_jsonl), Path(human48_input), schema)
    _require(len(recent_check) == 19, "M15_RECENT_CHECK_INVALID", "M15 requires recent Check19")
    final_model = _load_final(torch, AutoModel, snapshot, schema, output).to(torch.device("cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    current_runtime = hybrid.M12HybridInferenceRuntime.from_artifacts(
        selected_artifact=selected_artifact,
        config_path=config_path,
        cache_dir=cache_dir,
    )
    parent_bundle = torch.load(Path(selected_artifact).resolve() / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    current_thresholds = {label: float(parent_bundle.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}
    candidate_tables: dict[str, Any] = {}
    current_tables: dict[str, Any] = {}
    for name, records in {"recent_check19": recent_check, "human_check20": human_check, "human48": human48}.items():
        candidate_all = _metric_table(_predict(torch, final_model, tokenizer, records, torch.device("cpu")), records, schema, reasoning_thresholds)
        current_all = _metric_table(m12._baseline_probabilities(current_runtime, records, schema), records, schema, current_thresholds)
        if name == "human48":
            candidate_tables[name] = {head: candidate_all[head] for head in terminal.HUMAN48_HEADS}
            current_tables[name] = {head: current_all[head] for head in terminal.HUMAN48_HEADS}
        else:
            candidate_tables[name] = candidate_all
            current_tables[name] = current_all
    _dump(output / "metrics.json", {"current": current_tables, "candidate": candidate_tables})
    summary = {
        "stage_id": STAGE_ID,
        "status": "EVALUATED_AWAITING_SEMANTIC_DECISION",
        "actual_device": device_name,
        "mps_available": mps_available,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "train_samples": len(train),
        "recent_check_samples": len(recent_check),
        "human_check_samples": len(human_check),
        "human48_samples": len(human48),
        "selected_epoch": selected_epoch,
        "snapshot_identity": snapshot_identity,
        "base_audit": base_audit,
        "recent_train_audit": recent_audit,
        "recent_check_audit": recent_check_audit,
        "identity_stripping": True,
        "shared_multitask_training": True,
        "trainable_top_layers": [8, 9, 10, 11],
        "new_manual_labels_required": False,
        "deployment_changed": False,
        "final_training": final_training,
    }
    _dump(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--recent-feed-jsonl", type=Path, default=DEFAULT_RECENT_FEED)
    parser.add_argument("--recent-train-labels", type=Path, default=DEFAULT_RECENT_TRAIN)
    parser.add_argument("--recent-check-labels", type=Path, default=DEFAULT_RECENT_CHECK)
    parser.add_argument("--human48-jsonl", type=Path, default=DEFAULT_HUMAN48)
    parser.add_argument("--human48-input", type=Path, default=DEFAULT_HUMAN48_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_m15(
            config_path=args.config,
            cache_dir=args.cache_dir,
            selected_artifact=args.selected_artifact,
            pool_dir=args.pool_dir,
            recent_feed_jsonl=args.recent_feed_jsonl,
            recent_train_labels=args.recent_train_labels,
            recent_check_labels=args.recent_check_labels,
            human48_jsonl=args.human48_jsonl,
            human48_input=args.human48_input,
            output_dir=args.output_dir,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "M15_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
