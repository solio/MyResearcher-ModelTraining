"""Human-first, per-head LERT fine-tuning with out-of-fold model selection.

The weak-label LERT specialists are used only as initialization.  Training and
epoch selection use HumanTrain100; weak Dev is deliberately not an early-stop
target.  Each mini-batch performs an optimizer update, unlike the older M5 and
terminal diagnostic runners that accumulated one full-dataset update per
epoch.
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from . import encoder_m2_selected_runtime as selected
from . import encoder_m5_human100_source_balanced as m5
from . import encoder_terminal_macbert_base as terminal
from .errors import ContractError
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


STAGE_ID = "M12-HUMAN-FIRST-LERT-OOF"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m12-human-first-lert-oof.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
SEED = 35
N_FOLDS = 5
MAX_EPOCHS = 12
BATCH_SIZE = 16
BLOCK_LR = 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
MAX_LENGTH = 256
WALL_TIME_SECONDS = 2 * 60 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_WEAK_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-lert-classical-hybrid-seven-head-20260902-retry"
DEFAULT_SELECTED_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_POOL = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
DEFAULT_HUMAN48 = Path("/tmp/human48.jsonl")
DEFAULT_HUMAN48_INPUT = REPO_ROOT / "runs" / "m3c-shadow500-20260907" / "shadow500-input.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m12-human-first-lert-oof-20260915"
DEFAULT_PRIOR_M12_ARTIFACT = PROJECT_ROOT / "model-artifacts" / "m12-human-first-lert-oof-20260915"
BASE_SEED_BY_HEAD = {
    "target_mode": 107,
    "stance": 107,
    "emotion_primary": 71,
    "emotion_target": 71,
    "action_tendency": 35,
    "reasoning_tags": 35,
    "context_dependency": 35,
}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def _config(schema: LabelSchema) -> dict[str, Any]:
    return {
        "stage_id": STAGE_ID,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": LICENSE,
        "initialization": "best prior weak-label LERT specialist per head",
        "fit_population": "HumanTrain100_only",
        "selection": "five-fold out-of-fold HumanTrain100 macro-F1",
        "optimizer_step": "once_per_mini_batch",
        "trainable_scope": "independent_transformer_blocks_10_11_and_output_head",
        "class_balance": "square_root_inverse_frequency",
        "reasoning_thresholds": "calibrated_from_best_epoch_out_of_fold_probabilities",
        "max_length": MAX_LENGTH,
        "truncation": "HEAD_TAIL",
        "stock_code_token_cap": 8,
        "stock_name_token_cap": 16,
        "batch_size": BATCH_SIZE,
        "folds": N_FOLDS,
        "max_epochs": MAX_EPOCHS,
        "block_learning_rate": BLOCK_LR,
        "head_learning_rate": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "base_seed_by_head": dict(BASE_SEED_BY_HEAD),
        "class_order": {head: list(schema.class_order[head]) for head in V1_HEADS},
        "not_used_for_fit_or_selection": ["weak Dev", "Human Check20", "Human48", "Test", "Anchor", "Gold", "OOD", "LLM"],
    }


def _recent_input_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = m5._read_jsonl(Path(path), "M12_RECENT_INPUT_MISSING")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id.strip(), "M12_RECENT_INPUT_ID_INVALID", "recent model-ready row has an invalid sample_id")
        _require(sample_id not in result, "M12_RECENT_INPUT_DUPLICATE", "recent model-ready rows contain a duplicate sample_id", sample_id=sample_id)
        for field in ("stock_code", "stock_name", "model_text"):
            _require(isinstance(row.get(field), str) and row[field].strip(), "M12_RECENT_INPUT_FIELD_INVALID", "recent model-ready row has a missing required field", sample_id=sample_id, field=field)
        result[sample_id] = row
    return result


def load_recent_feedback_records(
    labels_path: str | Path,
    feed_path: str | Path,
    *,
    expected_samples: int,
    role: str,
    exclude_dependency_conflicts: bool = False,
) -> tuple[list[m1.M1Record], dict[str, Any]]:
    """Join a finalized recent seven-head label export to its feed inputs.

    The recent label export intentionally remains the source of truth: a
    dependency conflict is recorded verbatim and, for the training batch only,
    excludes the entire sample rather than repairing an answer.  Check rows
    are never silently excluded by this helper.
    """

    labels = m5._read_jsonl(Path(labels_path), "M12_RECENT_LABELS_MISSING")
    expected_rows = expected_samples * len(V1_HEADS)
    _require(len(labels) == expected_rows, "M12_RECENT_LABEL_COUNT_INVALID", "recent labels must contain seven final rows per sample", role=role, expected=expected_rows, observed=len(labels))
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in labels:
        sample_id, head = row.get("sample_id"), row.get("head")
        _require(isinstance(sample_id, str) and sample_id.strip(), "M12_RECENT_LABEL_ID_INVALID", "recent label has an invalid sample_id", role=role)
        _require(head in V1_HEADS, "M12_RECENT_LABEL_HEAD_INVALID", "recent label has an unknown head", role=role, sample_id=sample_id, head=head)
        _require(row.get("is_final") is True, "M12_RECENT_LABEL_NOT_FINAL", "recent label is not final", role=role, sample_id=sample_id, head=head)
        _require(row.get("schema_version") == "semantic-schema-calibrated-v0.2.1", "M12_RECENT_LABEL_SCHEMA_INVALID", "recent label schema differs from the frozen v0.2.1 schema", role=role, sample_id=sample_id)
        _require(m5._forbidden_key(row) is None, "M12_RECENT_LABEL_CONTAINS_PREDICTION", "recent label contains prediction data", role=role, sample_id=sample_id)
        _require(sample_id not in grouped or head not in grouped[sample_id], "M12_RECENT_LABEL_DUPLICATE", "recent labels contain a duplicate sample/head", role=role, sample_id=sample_id, head=head)
        answer = row.get("answer")
        if head == "reasoning_tags":
            _require(isinstance(answer, list) and len(answer) == len(set(answer)) and all(isinstance(tag, str) and tag in m5.feedback_adapter.CLASS_ORDER[head] for tag in answer), "M12_RECENT_REASONING_INVALID", "recent reasoning_tags answer is invalid", role=role, sample_id=sample_id)
        else:
            _require(isinstance(answer, str) and answer in m5.feedback_adapter.CLASS_ORDER[head], "M12_RECENT_SCALAR_INVALID", "recent scalar answer is outside the frozen taxonomy", role=role, sample_id=sample_id, head=head, answer=answer)
        grouped.setdefault(sample_id, {})[str(head)] = row
    _require(len(grouped) == expected_samples and all(set(heads) == set(V1_HEADS) for heads in grouped.values()), "M12_RECENT_HEAD_COVERAGE_INVALID", "recent labels must have exactly seven heads per sample", role=role, expected_samples=expected_samples, observed_samples=len(grouped))

    inputs = _recent_input_rows(feed_path)
    missing_inputs = sorted(set(grouped) - set(inputs))
    _require(not missing_inputs, "M12_RECENT_INPUT_MISSING_FOR_LABEL", "recent labels have no matching feed input", role=role, sample_ids=missing_inputs)
    records: list[m1.M1Record] = []
    excluded: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        labels_by_head = grouped[sample_id]
        labels_by_value = {head: labels_by_head[head]["answer"] for head in V1_HEADS}
        conflict = labels_by_value.get("emotion_primary") == "NONE_EXPLICIT" and labels_by_value.get("emotion_target") != "NOT_APPLICABLE"
        if conflict:
            if exclude_dependency_conflicts:
                excluded.append({
                    "sample_id": sample_id,
                    "reason": "EMOTION_DEPENDENCY_CONFLICT",
                    "emotion_primary": labels_by_value.get("emotion_primary"),
                    "emotion_target": labels_by_value.get("emotion_target"),
                })
                continue
            _require(False, "M12_RECENT_DEPENDENCY_CONFLICT", "recent check label violates the frozen emotion dependency", role=role, sample_id=sample_id, emotion_primary=labels_by_value.get("emotion_primary"), emotion_target=labels_by_value.get("emotion_target"))
        source = inputs[sample_id]
        records.append(m1.M1Record(
            sample_id=sample_id,
            stock_code=str(source["stock_code"]),
            stock_name=str(source["stock_name"]),
            model_text=str(source["model_text"]),
            label=labels_by_value,
            weights={head: 1.0 for head in V1_HEADS},
        ))
    audit = {
        "role": role,
        "labels_path": str(Path(labels_path).resolve()),
        "feed_path": str(Path(feed_path).resolve()),
        "received_samples": len(grouped),
        "received_rows": len(labels),
        "used_samples": len(records),
        "excluded_samples": len(excluded),
        "excluded": excluded,
        "excluded_sample_ids": [item["sample_id"] for item in excluded],
    }
    return records, audit


def _locate_layers(encoder: Any) -> Any:
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    _require(layers is not None and len(layers) == 12, "M12_LERT_ARCHITECTURE_INVALID", "LERT must expose twelve Transformer blocks")
    return layers


def make_model(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, dropout: float = 0.1) -> Any:
    nn = torch.nn

    class HumanFirstLert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
            source_layers = _locate_layers(self.encoder)
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.specialist_blocks = nn.ModuleDict(
                {head: nn.ModuleList([copy.deepcopy(source_layers[10]), copy.deepcopy(source_layers[11])]) for head in V1_HEADS}
            )
            self.heads = nn.ModuleDict({head: nn.Linear(hidden, len(schema.class_order[head])) for head in V1_HEADS})
            self.active_head: str | None = None
            self.set_active_head(None)

        def set_active_head(self, head: str | None) -> None:
            _require(head is None or head in V1_HEADS, "M12_HEAD_INVALID", "unknown head", head=head)
            self.active_head = head
            for name, parameter in self.named_parameters():
                parameter.requires_grad = bool(
                    head and (name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}."))
                )

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            captured: dict[str, Any] = {}

            def capture(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], _output: Any) -> None:
                captured["args"] = args
                captured["kwargs"] = kwargs

            hook = _locate_layers(self.encoder)[10].register_forward_hook(capture, with_kwargs=True)
            try:
                self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                hook.remove()
            _require("args" in captured, "M12_FORWARD_INVALID", "layer-10 input was not captured")
            result: dict[str, Any] = {}
            heads = (self.active_head,) if self.active_head else V1_HEADS
            for head in heads:
                args = captured["args"]
                kwargs = captured["kwargs"]
                first = self.specialist_blocks[head][0](*args, **kwargs)
                hidden = first[0] if isinstance(first, tuple) else first
                second_args = (hidden, *args[1:])
                second = self.specialist_blocks[head][1](*second_args, **kwargs)
                hidden = second[0] if isinstance(second, tuple) else second
                result[head] = self.heads[head](self.dropout(hidden[:, 0, :]))
            return result

    return HumanFirstLert()


def validate_trainable_parameters(model: Any, head: str) -> dict[str, Any]:
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")
    }
    _require(actual == expected and actual, "M12_TRAINABLE_SCOPE_INVALID", "only two specialist blocks and one output head may train", head=head)
    return {"head": head, "trainable_parameters": sorted(actual), "shared_encoder_frozen": True}


def _load_base_checkpoint(torch: Any, root: Path, head: str) -> Mapping[str, Any]:
    seed = BASE_SEED_BY_HEAD[head]
    path = root / head / f"seed-{seed}" / "specialist-checkpoint.pt"
    _require(path.is_file(), "M12_BASE_CHECKPOINT_MISSING", "weak-label specialist checkpoint is missing", head=head, path=str(path))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("head") == head, "M12_BASE_CHECKPOINT_INVALID", "checkpoint head mismatch", head=head)
    return checkpoint


def _initialize_head(model: Any, checkpoint: Mapping[str, Any], head: str) -> None:
    # Block 10 remains the official pretrained layer.  Block 11 and the output
    # classifier inherit the best prior weak-only specialist for this head.
    model.specialist_blocks[head][1].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[head].load_state_dict(checkpoint["head_state_dict"])


def fold_indices(records: Sequence[m1.M1Record], head: str, folds: int = N_FOLDS, seed: int = SEED) -> list[tuple[list[int], list[int]]]:
    _require(len(records) >= folds, "M12_FOLD_COUNT_INVALID", "not enough records for cross-validation")
    buckets: list[list[int]] = [[] for _ in range(folds)]
    if head in SINGLE_LABEL_HEADS:
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            groups.setdefault(str(record.label[head]), []).append(index)
        cursor = 0
        for label in sorted(groups):
            values = groups[label]
            random.Random(f"{seed}:{head}:{label}").shuffle(values)
            for offset, index in enumerate(values):
                buckets[(cursor + offset) % folds].append(index)
            cursor = (cursor + len(values)) % folds
    else:
        values = list(range(len(records)))
        random.Random(f"{seed}:{head}").shuffle(values)
        values.sort(key=lambda index: len(records[index].label[head]), reverse=True)
        label_counts = [Counter() for _ in range(folds)]
        for index in values:
            labels = list(records[index].label[head])
            target = min(
                range(folds),
                key=lambda fold: (sum(label_counts[fold][label] for label in labels), len(buckets[fold]), fold),
            )
            buckets[target].append(index)
            label_counts[target].update(labels)
    all_indices = set(range(len(records)))
    result = []
    for bucket in buckets:
        validation = sorted(bucket)
        training = sorted(all_indices - set(validation))
        _require(validation and training, "M12_EMPTY_FOLD", "cross-validation produced an empty fold")
        result.append((training, validation))
    _require(sorted(index for _train, validation in result for index in validation) == list(range(len(records))), "M12_FOLD_COVERAGE_INVALID", "validation folds must cover every record once")
    return result


def _balance(torch: Any, records: Sequence[m1.M1Record], schema: LabelSchema, head: str, device: Any) -> Any:
    order = list(schema.class_order[head])
    if head in SINGLE_LABEL_HEADS:
        counts = Counter(str(record.label[head]) for record in records)
        largest = max(counts.values())
        values = [math.sqrt(largest / max(counts.get(label, 0), 1)) for label in order]
        mean = sum(values) / len(values)
        return torch.tensor([value / mean for value in values], dtype=torch.float32, device=device)
    positives = np.asarray([[int(label in record.label[head]) for label in order] for record in records], dtype=float).sum(axis=0)
    negatives = len(records) - positives
    values = np.sqrt(np.maximum(negatives, 1.0) / np.maximum(positives, 1.0))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _batch_loss(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, schema: LabelSchema, head: str, balance: Any) -> Any:
    batch = m1._as_batch(torch, tokenizer, records, config, device)
    logits = model(batch["input_ids"], batch["attention_mask"])[head]
    _require(bool(torch.isfinite(logits).all().item()), "M12_NONFINITE_LOGITS", "non-finite logits", head=head)
    if head in SINGLE_LABEL_HEADS:
        raw = torch.nn.functional.cross_entropy(logits, batch["labels"][head], weight=balance, reduction="none")
    else:
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["labels"][head], pos_weight=balance, reduction="none"
        ).mean(dim=1)
    weights = batch["weights"][head]
    loss = (raw * weights).sum() / weights.sum().clamp_min(1e-12)
    _require(bool(torch.isfinite(loss).item()), "M12_NONFINITE_LOSS", "non-finite loss", head=head)
    return loss


def _probabilities(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset : offset + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            probs = torch.softmax(logits, dim=-1) if head in SINGLE_LABEL_HEADS else torch.sigmoid(logits)
            values.append(probs.detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def calibrate_reasoning_thresholds(probabilities: np.ndarray, records: Sequence[m1.M1Record], order: Sequence[str]) -> dict[str, float]:
    truth = np.asarray([[int(label in record.label["reasoning_tags"]) for label in order] for record in records], dtype=int)
    thresholds: dict[str, float] = {}
    grid = np.arange(0.10, 0.901, 0.05)
    for column, label in enumerate(order):
        y = truth[:, column]
        if int(y.sum()) == 0:
            thresholds[label] = 0.5
            continue
        best = (-1.0, -1.0, 0.5)
        for threshold in grid:
            pred = probabilities[:, column] >= threshold
            tp = int(((y == 1) & pred).sum())
            fp = int(((y == 0) & pred).sum())
            fn = int(((y == 1) & ~pred).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidate = (f1, -abs(float(threshold) - 0.5), -float(threshold))
            if candidate > best:
                best = candidate
                thresholds[label] = round(float(threshold), 4)
    return thresholds


def _metrics(probabilities: np.ndarray, records: Sequence[m1.M1Record], schema: LabelSchema, head: str, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    order = list(schema.class_order[head])
    if head in SINGLE_LABEL_HEADS:
        truth = np.asarray([order.index(str(record.label[head])) for record in records], dtype=int)
        predicted = probabilities.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=list(range(len(order))), zero_division=0
        )
        return {
            "accuracy": float(np.mean(truth == predicted)),
            "macro_f1": float(np.mean(f1)),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)},
            "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(order)))).tolist(),
        }
    truth = np.asarray([[int(label in record.label[head]) for label in order] for record in records], dtype=int)
    threshold_values = np.asarray([float((thresholds or {}).get(label, 0.5)) for label in order])
    predicted = probabilities >= threshold_values
    empty = ~predicted.any(axis=1)
    if bool(empty.any()):
        predicted[empty, probabilities[empty].argmax(axis=1)] = True
    per_label: dict[str, Any] = {}
    tp_total = fp_total = fn_total = 0
    for column, label in enumerate(order):
        tp = int(((truth[:, column] == 1) & predicted[:, column]).sum())
        fp = int(((truth[:, column] == 0) & predicted[:, column]).sum())
        fn = int(((truth[:, column] == 1) & ~predicted[:, column]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(truth[:, column].sum())}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return {
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "micro_f1": float(micro_f1),
        "exact_set_accuracy": float(np.mean(np.all(truth == predicted, axis=1))),
        "empty_set_fallback_count": int(empty.sum()),
        "per_label": per_label,
    }


def _train_curve(*, runtime: tuple[Any, Any, Any, Any], snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], base_root: Path, head: str, train_records: Sequence[m1.M1Record], validation_records: Sequence[m1.M1Record], fold: int, device: Any) -> list[np.ndarray]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    _set_seed(torch, SEED + fold)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = make_model(torch, AutoModel, snapshot, schema).to(device)
    model.set_active_head(head)
    _initialize_head(model, _load_base_checkpoint(torch, base_root, head), head)
    validate_trainable_parameters(model, head)
    blocks = list(model.specialist_blocks[head].parameters())
    classifier = list(model.heads[head].parameters())
    optimizer = torch.optim.AdamW(
        [{"params": blocks, "lr": BLOCK_LR}, {"params": classifier, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    balance = _balance(torch, train_records, schema, head, device)
    curve: list[np.ndarray] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        ordered = list(train_records)
        random.Random(SEED + fold * 1000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(torch, model, tokenizer, ordered[offset : offset + BATCH_SIZE], config, device, schema, head, balance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*blocks, *classifier], GRADIENT_CLIP)
            optimizer.step()
        curve.append(_probabilities(torch, model, tokenizer, validation_records, config, device, head))
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return curve


def _select_epoch(oof: np.ndarray, records: Sequence[m1.M1Record], schema: LabelSchema, head: str) -> tuple[int, dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_epoch = 1
    best_metric: dict[str, Any] = {}
    best_thresholds: dict[str, float] = {}
    for index in range(oof.shape[0]):
        thresholds = calibrate_reasoning_thresholds(oof[index], records, schema.class_order[head]) if head == "reasoning_tags" else {}
        metric = _metrics(oof[index], records, schema, head, thresholds)
        secondary = float(metric.get("accuracy", metric.get("exact_set_accuracy", 0.0)))
        key = (float(metric["macro_f1"]), secondary, -index)
        rows.append({"epoch": index + 1, "macro_f1": float(metric["macro_f1"]), "accuracy_or_exact": secondary, "thresholds": thresholds})
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = index + 1
            best_metric = metric
            best_thresholds = thresholds
    return best_epoch, best_metric, best_thresholds, rows


def _train_final(*, runtime: tuple[Any, Any, Any, Any], snapshot: Path, schema: LabelSchema, config: Mapping[str, Any], base_root: Path, head: str, records: Sequence[m1.M1Record], epochs: int, output: Path, device: Any) -> dict[str, Any]:
    _np, torch, AutoModel, AutoTokenizer = runtime
    _set_seed(torch, SEED + 100)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = make_model(torch, AutoModel, snapshot, schema).to(device)
    model.set_active_head(head)
    _initialize_head(model, _load_base_checkpoint(torch, base_root, head), head)
    identity = validate_trainable_parameters(model, head)
    blocks = list(model.specialist_blocks[head].parameters())
    classifier = list(model.heads[head].parameters())
    optimizer = torch.optim.AdamW(
        [{"params": blocks, "lr": BLOCK_LR}, {"params": classifier, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    balance = _balance(torch, records, schema, head, device)
    updates = 0
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        ordered = list(records)
        random.Random(SEED + 10000 + epoch).shuffle(ordered)
        model.train(True)
        for offset in range(0, len(ordered), BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(torch, model, tokenizer, ordered[offset : offset + BATCH_SIZE], config, device, schema, head, balance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*blocks, *classifier], GRADIENT_CLIP)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
    folder = output / head
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint_path = folder / "human-first-checkpoint.pt"
    torch.save(
        {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "head": head,
            "epochs": epochs,
            "optimizer_updates": updates,
            "base_seed": BASE_SEED_BY_HEAD[head],
            "specialist_blocks_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[head].state_dict().items()},
            "head_state_dict": {key: value.detach().cpu() for key, value in model.heads[head].state_dict().items()},
            "parameter_identity": identity,
        },
        checkpoint_path,
    )
    result = {"head": head, "epochs": epochs, "optimizer_updates": updates, "mean_batch_loss": float(np.mean(losses)), "checkpoint_path": str(checkpoint_path)}
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return result


def _load_final_model(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, output: Path) -> Any:
    model = make_model(torch, AutoModel, snapshot, schema)
    for head in V1_HEADS:
        checkpoint = torch.load(output / head / "human-first-checkpoint.pt", map_location="cpu", weights_only=True)
        model.specialist_blocks[head].load_state_dict(checkpoint["specialist_blocks_state_dict"])
        model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    model.set_active_head(None)
    model.eval()
    return model


def _all_probabilities(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    device = torch.device("cpu")
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset : offset + BATCH_SIZE], config, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for head in SINGLE_LABEL_HEADS:
                result[head].append(torch.softmax(outputs[head], dim=-1).cpu().numpy())
            result["reasoning_tags"].append(torch.sigmoid(outputs["reasoning_tags"]).cpu().numpy())
    return {head: np.concatenate(values, axis=0) for head, values in result.items()}


def _baseline_probabilities(runtime: Any, records: Sequence[m1.M1Record], schema: LabelSchema) -> dict[str, np.ndarray]:
    inference = [selected.InferenceRecord(record.sample_id, record.stock_code, record.stock_name, record.model_text) for record in records]
    rows = runtime.predict_records(inference)
    return m5._baseline_arrays(rows, schema)


def _metric_table(arrays: Mapping[str, np.ndarray], records: Sequence[m1.M1Record], schema: LabelSchema, thresholds: Mapping[str, float]) -> dict[str, Any]:
    return {head: _metrics(np.asarray(arrays[head]), records, schema, head, thresholds if head == "reasoning_tags" else None) for head in V1_HEADS}


def _load_reasoning_thresholds(artifact: str | Path, schema: LabelSchema) -> dict[str, float]:
    path = Path(artifact).resolve() / "reasoning-thresholds.json"
    _require(path.is_file(), "M12_REASONING_THRESHOLDS_MISSING", "reasoning threshold file is missing", path=str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ContractError("M12_REASONING_THRESHOLDS_INVALID", "reasoning threshold file is invalid", path=str(path)) from exc
    if isinstance(value, Mapping) and isinstance(value.get("thresholds"), Mapping):
        value = value["thresholds"]
    _require(isinstance(value, Mapping), "M12_REASONING_THRESHOLDS_INVALID", "reasoning thresholds must be an object", path=str(path))
    order = list(schema.class_order["reasoning_tags"])
    _require(set(value) == set(order), "M12_REASONING_THRESHOLDS_INVALID", "reasoning threshold labels do not match the frozen taxonomy", path=str(path), expected=order, observed=sorted(str(key) for key in value))
    thresholds: dict[str, float] = {}
    for label in order:
        try:
            threshold = float(value[label])
        except (TypeError, ValueError) as exc:
            raise ContractError("M12_REASONING_THRESHOLDS_INVALID", "reasoning threshold is not numeric", path=str(path), label=label) from exc
        _require(math.isfinite(threshold) and 0.0 <= threshold <= 1.0, "M12_REASONING_THRESHOLDS_INVALID", "reasoning threshold is outside [0, 1]", path=str(path), label=label, threshold=threshold)
        thresholds[label] = threshold
    return thresholds


def run_m12(*, config_path: str | Path = DEFAULT_CONFIG, cache_dir: str | Path = DEFAULT_CACHE, weak_artifact: str | Path = DEFAULT_WEAK_ARTIFACT, selected_artifact: str | Path = DEFAULT_SELECTED_ARTIFACT, pool_dir: str | Path = DEFAULT_POOL, recent_feed_jsonl: str | Path | None = None, recent_train_labels: str | Path | None = None, recent_check_labels: str | Path | None = None, prior_m12_artifact: str | Path = DEFAULT_PRIOR_M12_ARTIFACT, human48_jsonl: str | Path = DEFAULT_HUMAN48, human48_input: str | Path = DEFAULT_HUMAN48_INPUT, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M12_OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    recent_paths = (recent_feed_jsonl, recent_train_labels, recent_check_labels)
    recent_enabled = any(path is not None for path in recent_paths)
    _require(not recent_enabled or all(path is not None for path in recent_paths), "M12_RECENT_INPUTS_INCOMPLETE", "recent feed, train labels, and check labels must be supplied together")
    project = m1.ProjectConfig.load(config_path)
    schema, _weak, _dev = m1.load_m1_partitions(project)
    human_train, _train_audit = m5.load_human_train_pool(pool_dir)
    _require(len(human_train) == 100, "M12_HUMAN_TRAIN_INVALID", "expected HumanTrain100")
    recent_train_audit: dict[str, Any] = {
        "role": "recent_train",
        "received_samples": 0,
        "used_samples": 0,
        "excluded_samples": 0,
        "excluded": [],
        "excluded_sample_ids": [],
    }
    recent_check_audit: dict[str, Any] = {
        "role": "recent_check",
        "received_samples": 0,
        "used_samples": 0,
        "excluded_samples": 0,
        "excluded": [],
        "excluded_sample_ids": [],
    }
    recent_check: list[m1.M1Record] = []
    recent_check_loaded_after_checkpoint_freeze = False
    if recent_enabled:
        _require(recent_feed_jsonl is not None and recent_train_labels is not None and recent_check_labels is not None, "M12_RECENT_INPUTS_INCOMPLETE", "recent feed, train labels, and check labels must be supplied together")
        recent_train, recent_train_audit = load_recent_feedback_records(
            recent_train_labels,
            recent_feed_jsonl,
            expected_samples=10,
            role="recent_train",
            exclude_dependency_conflicts=True,
        )
        existing_ids = {record.sample_id for record in human_train}
        recent_ids = {record.sample_id for record in recent_train}
        _require(not existing_ids & recent_ids, "M12_HUMAN_TRAIN_RECENT_OVERLAP", "HumanTrain100 and recent Train samples overlap", sample_ids=sorted(existing_ids & recent_ids))
        human_train = [*human_train, *recent_train]
        _require(len(human_train) == 109, "M12_HUMAN_TRAIN_RECENT_COUNT_INVALID", "expected HumanTrain109 after excluding the one conflicting recent sample", observed=len(human_train), audit=recent_train_audit)
    runtime = m1._load_runtime_dependencies()
    _np, torch, AutoModel, AutoTokenizer = runtime
    snapshot, _snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
    config = _config(schema)
    config.update({
        "fit_population": f"HumanTrain{len(human_train)}_only",
        "selection": f"five-fold out-of-fold HumanTrain{len(human_train)} macro-F1",
        "human_train_samples": len(human_train),
        "recent_train_received": recent_train_audit["received_samples"],
        "recent_train_used": recent_train_audit["used_samples"],
        "recent_train_excluded": recent_train_audit["excluded_samples"],
        "recent_train_excluded_sample_ids": recent_train_audit["excluded_sample_ids"],
        "recent_check_not_used_for_fit_or_selection": True,
    })
    output.joinpath("training-config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    device, device_name, mps_available = m1._choose_device(torch)
    weak_root = Path(weak_artifact).resolve()
    folds_by_head: dict[str, Any] = {}
    final_results: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for head in V1_HEADS:
        _require(time.monotonic() - started < WALL_TIME_SECONDS, "M12_WALL_TIME_EXCEEDED", "training exceeded two hours", head=head)
        splits = fold_indices(human_train, head)
        width = len(schema.class_order[head])
        oof = np.full((MAX_EPOCHS, len(human_train), width), np.nan, dtype=np.float32)
        for fold, (train_indices, validation_indices) in enumerate(splits):
            train_records = [human_train[index] for index in train_indices]
            validation_records = [human_train[index] for index in validation_indices]
            curve = _train_curve(runtime=runtime, snapshot=snapshot, schema=schema, config=config, base_root=weak_root, head=head, train_records=train_records, validation_records=validation_records, fold=fold, device=device)
            for epoch_index, probabilities in enumerate(curve):
                oof[epoch_index, validation_indices, :] = probabilities
        _require(bool(np.isfinite(oof).all()), "M12_OOF_INCOMPLETE", "out-of-fold probability matrix is incomplete", head=head)
        best_epoch, best_metric, head_thresholds, curve_summary = _select_epoch(oof, human_train, schema, head)
        if head == "reasoning_tags":
            thresholds = head_thresholds
        folds_by_head[head] = {"best_epoch": best_epoch, "oof_metric": best_metric, "curve": curve_summary, "validation_fold_sizes": [len(validation) for _train, validation in splits]}
        final_results.append(_train_final(runtime=runtime, snapshot=snapshot, schema=schema, config=config, base_root=weak_root, head=head, records=human_train, epochs=best_epoch, output=output, device=device))
        print(json.dumps({"event": "head_complete", "head": head, "best_epoch": best_epoch, "oof_macro_f1": best_metric["macro_f1"], "elapsed_seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    _dump(output / "cross-validation.json", folds_by_head)
    _dump(output / "reasoning-thresholds.json", thresholds)

    # Human Check20 and Human48 are opened only after all seven final models
    # have been trained and frozen.
    if recent_enabled:
        _require(recent_feed_jsonl is not None and recent_check_labels is not None, "M12_RECENT_INPUTS_INCOMPLETE", "recent feed and check labels are required")
        recent_check, recent_check_audit = load_recent_feedback_records(
            recent_check_labels,
            recent_feed_jsonl,
            expected_samples=4,
            role="recent_check",
        )
        train_ids = {record.sample_id for record in human_train}
        check_ids = {record.sample_id for record in recent_check}
        _require(not train_ids & check_ids, "M12_RECENT_TRAIN_CHECK_OVERLAP", "recent Check samples overlap the fit population", sample_ids=sorted(train_ids & check_ids))
        recent_check_loaded_after_checkpoint_freeze = True
    human_check, _check_audit = m5.load_human_check_pool(pool_dir)
    human48 = terminal._load_human48(Path(human48_jsonl), Path(human48_input), schema)
    final_model = _load_final_model(torch, AutoModel, snapshot, schema, output).to(torch.device("cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    selected_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(selected_artifact, config_path=config_path, cache_dir=cache_dir)
    parent_bundle = torch.load(Path(selected_artifact).resolve() / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    baseline_thresholds = {label: float(parent_bundle.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}

    candidate_check_arrays = _all_probabilities(torch, final_model, tokenizer, human_check, config)
    baseline_check_arrays = _baseline_probabilities(selected_runtime, human_check, schema)
    candidate_check = _metric_table(candidate_check_arrays, human_check, schema, thresholds)
    baseline_check = _metric_table(baseline_check_arrays, human_check, schema, baseline_thresholds)
    _dump(output / "human-check20.json", {"selected_baseline": baseline_check, "human_first_candidate": candidate_check})

    candidate_h48_arrays = _all_probabilities(torch, final_model, tokenizer, human48, config)
    baseline_h48_arrays = _baseline_probabilities(selected_runtime, human48, schema)
    candidate_h48_all = _metric_table(candidate_h48_arrays, human48, schema, thresholds)
    baseline_h48_all = _metric_table(baseline_h48_arrays, human48, schema, baseline_thresholds)
    candidate_h48 = {head: candidate_h48_all[head] for head in terminal.HUMAN48_HEADS}
    baseline_h48 = {head: baseline_h48_all[head] for head in terminal.HUMAN48_HEADS}
    _dump(output / "human48.json", {"selected_baseline": baseline_h48, "human_first_candidate": candidate_h48})

    recent_comparison: dict[str, Any] = {}
    if recent_enabled:
        prior_root = Path(prior_m12_artifact).resolve()
        _require(prior_root != output, "M12_PRIOR_ARTIFACT_INVALID", "prior M12 artifact must differ from the new output directory")
        prior_model = _load_final_model(torch, AutoModel, snapshot, schema, prior_root).to(torch.device("cpu"))
        prior_thresholds = _load_reasoning_thresholds(prior_root, schema)
        candidate_recent = _metric_table(_all_probabilities(torch, final_model, tokenizer, recent_check, config), recent_check, schema, thresholds)
        prior_recent = _metric_table(_all_probabilities(torch, prior_model, tokenizer, recent_check, config), recent_check, schema, prior_thresholds)
        delta: dict[str, Any] = {}
        for head in V1_HEADS:
            primary = "accuracy" if head in SINGLE_LABEL_HEADS else "micro_f1"
            delta[head] = {
                "macro_f1": float(candidate_recent[head]["macro_f1"] - prior_recent[head]["macro_f1"]),
                primary: float(candidate_recent[head][primary] - prior_recent[head][primary]),
            }
        recent_comparison = {
            "samples": len(recent_check),
            "loaded_after_all_checkpoints_frozen": recent_check_loaded_after_checkpoint_freeze,
            "used_for_fit_or_selection": False,
            "prior_m12_human100_artifact": str(prior_root),
            "prior_m12_human100": prior_recent,
            "candidate_m12_human109": candidate_recent,
            "delta_109_minus_100": delta,
            "prior_reasoning_thresholds": prior_thresholds,
            "candidate_reasoning_thresholds": thresholds,
        }
        _dump(output / "recent-check4.json", recent_comparison)
        del prior_model

    baseline_train = _metric_table(_baseline_probabilities(selected_runtime, human_train, schema), human_train, schema, baseline_thresholds)
    per_head: dict[str, Any] = {}
    for head in V1_HEADS:
        oof_metric = folds_by_head[head]["oof_metric"]
        check_delta = float(candidate_check[head]["macro_f1"] - baseline_check[head]["macro_f1"])
        oof_delta = float(oof_metric["macro_f1"] - baseline_train[head]["macro_f1"])
        item = {
            "baseline_human_train_macro_f1": baseline_train[head]["macro_f1"],
            "candidate_oof_macro_f1": oof_metric["macro_f1"],
            "oof_delta": oof_delta,
            "baseline_check20_macro_f1": baseline_check[head]["macro_f1"],
            "candidate_check20_macro_f1": candidate_check[head]["macro_f1"],
            "check20_delta": check_delta,
            "improves_both_human_sets": bool(oof_delta > 0 and check_delta > 0),
        }
        if head in terminal.HUMAN48_HEADS:
            item.update({
                "baseline_human48_macro_f1": baseline_h48[head]["macro_f1"],
                "candidate_human48_macro_f1": candidate_h48[head]["macro_f1"],
                "human48_delta": float(candidate_h48[head]["macro_f1"] - baseline_h48[head]["macro_f1"]),
            })
        per_head[head] = item
    summary = {
        "stage_id": STAGE_ID,
        "status": "TRAINING_COMPLETED",
        "actual_device": device_name,
        "mps_available": mps_available,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "human_train_samples": len(human_train),
        "human_check_samples": len(human_check),
        "human48_samples": len(human48),
        "recent_train_received": recent_train_audit["received_samples"],
        "recent_train_used": recent_train_audit["used_samples"],
        "recent_train_excluded": recent_train_audit["excluded_samples"],
        "recent_train_excluded_sample_ids": recent_train_audit["excluded_sample_ids"],
        "recent_check_samples": recent_check_audit["used_samples"],
        "recent_check_loaded_after_checkpoint_freeze": recent_check_loaded_after_checkpoint_freeze,
        "recent_check_used_for_fit_or_selection": False,
        "recent_check_comparison": recent_comparison.get("delta_109_minus_100", {}),
        "per_head": per_head,
        "final_training": final_results,
        "global_gate": "NONE_PER_HEAD_EVIDENCE_ONLY",
    }
    _dump(output / "summary.json", summary)
    lines = [
        "# M12 human-first LERT training",
        "",
        "Status: **TRAINING_COMPLETED**",
        "",
        f"Device: `{device_name}`; elapsed: {summary['elapsed_seconds']:.1f}s.",
        "",
        "| head | baseline HumanTrain | candidate OOF | baseline Check20 | candidate Check20 | improves both |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for head in V1_HEADS:
        item = per_head[head]
        lines.append(f"| {head} | {item['baseline_human_train_macro_f1']:.4f} | {item['candidate_oof_macro_f1']:.4f} | {item['baseline_check20_macro_f1']:.4f} | {item['candidate_check20_macro_f1']:.4f} | {item['improves_both_human_sets']} |")
    lines += ["", "Each head is judged independently.  Human48 is reported separately for its five labeled heads.  No weak Dev, Test, Anchor, Gold, OOD, or LLM output selected an epoch."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run human-first LERT out-of-fold training")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--weak-artifact", default=str(DEFAULT_WEAK_ARTIFACT))
    parser.add_argument("--selected-artifact", default=str(DEFAULT_SELECTED_ARTIFACT))
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL))
    parser.add_argument("--recent-feed-jsonl", default=None)
    parser.add_argument("--recent-train-labels", default=None)
    parser.add_argument("--recent-check-labels", default=None)
    parser.add_argument("--prior-m12-artifact", default=str(DEFAULT_PRIOR_M12_ARTIFACT))
    parser.add_argument("--human48-jsonl", default=str(DEFAULT_HUMAN48))
    parser.add_argument("--human48-input", default=str(DEFAULT_HUMAN48_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        result = run_m12(config_path=args.config, cache_dir=args.cache_dir, weak_artifact=args.weak_artifact, selected_artifact=args.selected_artifact, pool_dir=args.pool_dir, recent_feed_jsonl=args.recent_feed_jsonl, recent_train_labels=args.recent_train_labels, recent_check_labels=args.recent_check_labels, prior_m12_artifact=args.prior_m12_artifact, human48_jsonl=args.human48_jsonl, human48_input=args.human48_input, output_dir=args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "M12_BLOCKED", "code": exc.code, "details": exc.details}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
