"""M5 Phase A: one seed, source-balanced Human100 supervision diagnostic.

This module deliberately implements one treatment only.  It warm-starts the
selected LERT-small candidate, trains every specialist head with a 0.5/0.5
weak-Train/Human-Train objective, and then performs one frozen Human Check20
diagnostic.  The treatment is evidence for the feedback loop, not a candidate
promotion path.  The Human Check20 files are not opened until all seven
checkpoints have been written and frozen.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from . import encoder_m2_selected_runtime as selected
from .errors import ContractError
from .hashes import sha256_file
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema
from tools import feedback_adapter_v1 as feedback_adapter


STAGE_ID = "M5-HUMAN100-SOURCE-BALANCED-SEED35"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m5-human100-source-balanced-run.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
SEED = 35
MAX_LENGTH = 256
BATCH_SIZE = 16
MAX_EPOCHS = 24
PATIENCE = 3
BLOCK_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WALL_TIME_SECONDS = 2 * 60 * 60
MAX_DISK_GIB = 10
WEAK_SOURCE_MASS = 0.5
HUMAN_SOURCE_MASS = 0.5
BASELINE_CONTENT_ADDRESS = "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline_v0.3.5.yaml"
DEFAULT_CACHE = REPO_ROOT / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_PARENT = PROJECT_ROOT / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
DEFAULT_POOL = REPO_ROOT / "runs" / "feedback-loop-v3-finalized-20260910"
DEFAULT_SNAPSHOT = REPO_ROOT / "runs" / "feedback-loop-v1-20260909" / "prediction-snapshot.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "model-artifacts" / "m5-human100-seven-head-seed35"
V1_RUN = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
V2_RUN = REPO_ROOT / "runs" / "feedback-loop-v2-20260909"
V3_RUN = REPO_ROOT / "runs" / "feedback-loop-v3-20260910"
FORBIDDEN_PREDICTION_KEYS = {
    "prediction", "predictions", "probability", "probabilities", "model_prediction",
    "model_predictions", "model_output", "logits", "weights", "weight",
}


class M5Error(ContractError):
    """Fail-closed M5 validation/runtime error."""


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise M5Error(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise M5Error(code, "required JSON is missing or invalid", path=str(path)) from exc
    _require(isinstance(value, dict), code, "required JSON root must be an object", path=str(path))
    return value


def _read_jsonl(path: Path, code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise M5Error(code, "required JSONL is missing", path=str(path))
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise M5Error(code, "invalid JSONL row", path=str(path), line=line_number) from exc
        _require(isinstance(value, dict), code, "JSONL rows must be objects", path=str(path), line=line_number)
        rows.append(value)
    return rows


def _forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_PREDICTION_KEYS:
                return str(key)
            found = _forbidden_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _forbidden_key(nested)
            if found is not None:
                return found
    return None


def _selection_rows(path: Path, expected: int, role: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "M5_SELECTION_MISSING")
    _require(len(rows) == expected, "M5_SELECTION_COUNT_INVALID", f"{role} selection count is invalid", expected=expected, observed=len(rows))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        group = row.get("exact_content_group")
        text = row.get("model_text", row.get("text"))
        _require(isinstance(sample_id, str) and sample_id.strip(), "M5_SELECTION_ID_INVALID", f"{role} has invalid sample_id")
        _require(sample_id not in result, "M5_SELECTION_DUPLICATE", f"{role} selection has duplicate sample_id", sample_id=sample_id)
        _require(isinstance(group, str) and group.strip(), "M5_SELECTION_GROUP_INVALID", f"{role} has invalid exact_content_group", sample_id=sample_id)
        _require(isinstance(text, str) and text.strip(), "M5_SELECTION_TEXT_INVALID", f"{role} has invalid model_text", sample_id=sample_id)
        result[sample_id] = dict(row, model_text=text)
    return result


def _all_selection_maps(*, include_check: bool) -> dict[str, dict[str, Any]]:
    maps: list[dict[str, dict[str, Any]]] = []
    for run, name, train_count, check_count in ((V1_RUN, "V1", 10, 4), (V2_RUN, "V2", 10, 4), (V3_RUN, "FL3", 80, 14)):
        maps.append(_selection_rows(run / "selection-train.jsonl", train_count, f"{name} Train"))
        if include_check:
            maps.append(_selection_rows(run / "selection-check.jsonl", check_count, f"{name} Check"))
    combined: dict[str, dict[str, Any]] = {}
    for values in maps:
        overlap = set(combined) & set(values)
        _require(not overlap, "M5_SELECTION_OVERLAP", "human selections contain duplicate sample IDs", sample_id=sorted(overlap)[:1])
        combined.update(values)
    return combined


def _pool_rows(pool: Path, filename: str, expected_samples: int, role: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(pool / filename, "M5_POOL_FILE_MISSING")
    _require(len(rows) == expected_samples * len(V1_HEADS), f"M5_{role.upper()}_COUNT_INVALID", f"{role} must have seven labels per sample", expected=expected_samples * len(V1_HEADS), observed=len(rows))
    seen: set[tuple[str, str]] = set()
    by_sample: dict[str, set[str]] = {}
    for row in rows:
        sample_id, head = row.get("sample_id"), row.get("head")
        _require(isinstance(sample_id, str) and sample_id.strip(), "M5_HUMAN_LABEL_ID_INVALID", "human label has invalid sample_id")
        _require(head in V1_HEADS, "M5_HUMAN_LABEL_HEAD_INVALID", "human label has unknown head", sample_id=sample_id, head=head)
        key = (sample_id, str(head))
        _require(key not in seen, "M5_HUMAN_LABEL_DUPLICATE", "duplicate human sample/head", sample_id=sample_id, head=head)
        seen.add(key)
        _require(row.get("is_final") is True, "M5_HUMAN_LABEL_NOT_FINAL", "human label is not final", sample_id=sample_id, head=head)
        _require(row.get("schema_version") == "semantic-schema-calibrated-v0.2.1", "M5_HUMAN_LABEL_SCHEMA_INVALID", "human label schema version is not frozen", sample_id=sample_id, head=head)
        forbidden = _forbidden_key(row)
        _require(forbidden is None, "M5_HUMAN_LABEL_CONTAINS_PREDICTION", "human label file contains model prediction data", field=forbidden)
        answer = row.get("answer")
        if head == "reasoning_tags":
            _require(isinstance(answer, list) and len(answer) == len(set(answer)) and all(isinstance(tag, str) and tag in feedback_adapter.CLASS_ORDER[head] for tag in answer), "M5_HUMAN_REASONING_LABEL_INVALID", "reasoning_tags answer must be a unique taxonomy list", sample_id=sample_id)
        else:
            _require(isinstance(answer, str) and answer in feedback_adapter.CLASS_ORDER[head], "M5_HUMAN_SCALAR_LABEL_INVALID", "scalar answer is outside frozen taxonomy", sample_id=sample_id, head=head, answer=answer)
        by_sample.setdefault(sample_id, set()).add(str(head))
    _require(len(by_sample) == expected_samples and all(heads == set(V1_HEADS) for heads in by_sample.values()), f"M5_{role.upper()}_HEAD_COVERAGE_INVALID", f"{role} does not have exactly seven heads per sample")
    return rows


def _records_from_labels(rows: Sequence[Mapping[str, Any]], selections: Mapping[str, Mapping[str, Any]], role: str) -> list[m1.M1Record]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        _require(sample_id in selections, "M5_HUMAN_SELECTION_MISMATCH", f"{role} label has no matching selection", sample_id=sample_id)
        grouped.setdefault(sample_id, {})[str(row["head"])] = row
    records: list[m1.M1Record] = []
    for sample_id in sorted(grouped):
        source = selections[sample_id]
        labels = {head: grouped[sample_id][head]["answer"] for head in V1_HEADS}
        records.append(m1.M1Record(sample_id=sample_id, stock_code=str(source.get("stock_code", "")), stock_name=str(source.get("stock_name", "")), model_text=str(source["model_text"]), label=labels, weights={head: 1.0 for head in V1_HEADS}))
    return records


def load_human_train_pool(pool: str | Path = DEFAULT_POOL) -> tuple[list[m1.M1Record], dict[str, Any]]:
    """Load only Human Train100.  Check20 is intentionally not opened here."""

    root = Path(pool).resolve()
    selections = _all_selection_maps(include_check=False)
    rows = _pool_rows(root, "human-train-100-seven-head.jsonl", 100, "human_train")
    records = _records_from_labels(rows, selections, "Human Train")
    _require(len(records) == 100 and len({record.sample_id for record in records}) == 100, "M5_HUMAN_TRAIN_COUNT_INVALID", "Human Train must contain exactly 100 unique samples")
    return records, {"pool": str(root), "rows": len(rows), "samples": len(records), "heads": list(V1_HEADS), "check20_opened": False}


def load_human_check_pool(pool: str | Path = DEFAULT_POOL) -> tuple[list[m1.M1Record], dict[str, Any]]:
    """Load Check20 only after all treatment checkpoints are frozen."""

    root = Path(pool).resolve()
    train_selection_ids = set(_all_selection_maps(include_check=False))
    # The first 100 samples are Train and the remaining selection IDs are the
    # six historical plus fourteen FL3 semantic checks.  We derive the exact
    # Check IDs from the check selection files instead of relying on ordering.
    check_selections: dict[str, dict[str, Any]] = {}
    for run, name, count in ((V1_RUN, "V1", 4), (V2_RUN, "V2", 4), (V3_RUN, "FL3", 14)):
        check_selections.update(_selection_rows(run / "selection-check.jsonl", count, f"{name} Check"))
    rows = _pool_rows(root, "human-check-20-semantic-seven-head.jsonl", 20, "human_check")
    records = _records_from_labels(rows, check_selections, "Human Check")
    _require(len(records) == 20 and len({record.sample_id for record in records}) == 20, "M5_HUMAN_CHECK_COUNT_INVALID", "semantic Human Check must contain exactly 20 unique samples")
    _require(not ({record.sample_id for record in records} & train_selection_ids), "M5_HUMAN_TRAIN_CHECK_OVERLAP", "Human Train and Check sample IDs overlap")
    return records, {"pool": str(root), "rows": len(rows), "samples": len(records), "heads": list(V1_HEADS), "check20_opened": True}


def source_balanced_objective(weak_loss: float, human_loss: float) -> float:
    """The frozen objective; exposed for a small non-real-data unit test."""

    return WEAK_SOURCE_MASS * float(weak_loss) + HUMAN_SOURCE_MASS * float(human_loss)


def _set_seed(torch: Any, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def _limits(started: float, root: Path, phase: str) -> None:
    _require(time.monotonic() - started <= WALL_TIME_SECONDS, "M5_WALL_TIME_LIMIT_EXCEEDED", "M5 two-hour limit exceeded", phase=phase)
    _require(m1._directory_size(root) <= MAX_DISK_GIB * 1024**3, "M5_DISK_LIMIT_EXCEEDED", "M5 ten GiB output limit exceeded", phase=phase)


def _load_bundle(path: Path, torch: Any) -> Mapping[str, Any]:
    selected._verify_manifest(path, expected_content_address=BASELINE_CONTENT_ADDRESS)
    try:
        bundle = torch.load(path / "unified-runtime-bundle.pt", map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path / "unified-runtime-bundle.pt", map_location="cpu")
    _require(isinstance(bundle, Mapping), "M5_PARENT_BUNDLE_INVALID", "selected candidate bundle must be a mapping")
    identity = bundle.get("model_identity")
    _require(isinstance(identity, Mapping) and identity.get("model_id") == MODEL_ID and identity.get("revision") == REVISION and identity.get("license") == LICENSE, "M5_PARENT_MODEL_INVALID", "selected candidate model identity changed")
    _require(bundle.get("status") == selected.STAGE_ID and bundle.get("selected_candidate") is True, "M5_PARENT_STATUS_INVALID", "selected candidate is not frozen")
    states = bundle.get("lert_specialist_block_state_dict")
    heads = bundle.get("lert_heads_state_dict")
    _require(isinstance(states, Mapping) and isinstance(heads, Mapping), "M5_PARENT_STATE_INVALID", "selected candidate LERT states are missing")
    _require(set(states) == {"target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "reasoning_tags"} and set(heads) == set(states), "M5_PARENT_STATE_INVALID", "selected candidate must contain six warm-start LERT specialists")
    return bundle


def _make_warm_model(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, config: Mapping[str, Any], parent: Mapping[str, Any], head: str) -> Any:
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]))
    for warm_head in parent["lert_specialist_block_state_dict"]:
        model.specialist_blocks[warm_head].load_state_dict(parent["lert_specialist_block_state_dict"][warm_head])
        model.heads[warm_head].load_state_dict(parent["lert_heads_state_dict"][warm_head])
    # The selected candidate used a Classical context fallback and therefore
    # has no LERT context specialist state.  Context receives a deterministic
    # official-snapshot initialization and is still trained as a seventh head;
    # the other six heads are genuine selected-candidate warm starts.
    model.set_active_head(head)
    lert.validate_trainable_parameters(model, head)
    return model


def _batch_loss(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str) -> tuple[Any, float]:
    batch = m1._as_batch(torch, tokenizer, records, config, device)
    logits = model(batch["input_ids"], batch["attention_mask"])[head]
    _require(bool(torch.isfinite(logits).all().item()), "M5_NONFINITE_LOGITS", "non-finite training logits", head=head)
    if head in SINGLE_LABEL_HEADS:
        raw = torch.nn.functional.cross_entropy(logits, batch["labels"][head], reduction="none")
    else:
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"][head], reduction="none").mean(dim=1)
    weights = batch["weights"][head]
    loss = (raw * weights).sum() / weights.sum().clamp_min(1e-12)
    _require(bool(torch.isfinite(loss).item()), "M5_NONFINITE_LOSS", "non-finite source loss", head=head)
    return loss, float(weights.detach().sum().cpu())


def _metric_from_logits(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], config: Mapping[str, Any], device: Any, head: str, thresholds: Mapping[str, float]) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset:offset + BATCH_SIZE], config, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "M5_NONFINITE_EVAL_LOGITS", "non-finite evaluation logits", head=head)
            values = torch.softmax(logits, dim=-1) if head in SINGLE_LABEL_HEADS else torch.sigmoid(logits)
            probabilities.extend(values.detach().cpu().numpy())
    values = np.asarray(probabilities, dtype=float)
    if head in SINGLE_LABEL_HEADS:
        order = list(config["class_order"][head])
        truth = [order.index(str(record.label[head])) for record in records]
        predicted = values.argmax(axis=1).tolist()
        precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(len(order))), zero_division=0)
        return {"accuracy": float(np.mean(np.asarray(truth) == np.asarray(predicted))), "macro_f1": float(np.mean(f1)), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(order)))).tolist(), "probabilities": values}
    return _reasoning_metric(values, records, list(config["class_order"][head]), thresholds)


def _reasoning_metric(values: np.ndarray, records: Sequence[m1.M1Record], order: Sequence[str], thresholds: Mapping[str, float]) -> dict[str, Any]:
    truth = np.asarray([[int(label in record.label["reasoning_tags"]) for label in order] for record in records], dtype=int)
    predicted = np.asarray([[int(value >= float(thresholds.get(label, 0.5))) for value, label in zip(row, order, strict=True)] for row in values], dtype=int)
    per_label: dict[str, Any] = {}
    tp_total = fp_total = fn_total = 0
    for index, label in enumerate(order):
        tp = int(((truth[:, index] == 1) & (predicted[:, index] == 1)).sum())
        fp = int(((truth[:, index] == 0) & (predicted[:, index] == 1)).sum())
        fn = int(((truth[:, index] == 1) & (predicted[:, index] == 0)).sum())
        support = int((truth[:, index] == 1).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        tp_total += tp; fp_total += fp; fn_total += fn
    micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return {"macro_f1": float(np.mean([value["f1"] for value in per_label.values()])), "micro_f1": float(micro_f1), "exact_set_accuracy": float(np.mean(np.all(truth == predicted, axis=1))), "per_label": per_label, "probabilities": values}


def _snapshot_arrays(rows: Sequence[Mapping[str, Any]], schema: LabelSchema) -> tuple[dict[str, np.ndarray], list[selected.InferenceRecord]]:
    records: list[selected.InferenceRecord] = []
    arrays: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    for row in rows:
        predictions = row.get("predictions")
        _require(isinstance(predictions, Mapping), "M5_SNAPSHOT_INVALID", "snapshot row has no predictions")
        records.append(selected.InferenceRecord(sample_id=str(row["sample_id"]), stock_code=str(row.get("stock_code", "")), stock_name=str(row.get("stock_name", "")), model_text=str(row.get("model_text", row.get("text", "")))))
        for head in SINGLE_LABEL_HEADS:
            order = list(schema.class_order[head]); probs = predictions[head]["probabilities"]
            arrays[head].append(np.asarray([float(probs[label]) for label in order], dtype=float))
        order = list(schema.class_order["reasoning_tags"]); probs = predictions["reasoning_tags"]["probabilities"]
        arrays["reasoning_tags"].append(np.asarray([float(probs[label]) for label in order], dtype=float))
    _require(len(records) == 300 and len({record.sample_id for record in records}) == 300, "M5_SNAPSHOT_COUNT_INVALID", "the original prediction snapshot must contain exactly 300 unique rows")
    return {head: np.asarray(values, dtype=float) for head, values in arrays.items()}, records


def _model_arrays(torch: Any, model: Any, tokenizer: Any, records: Sequence[selected.InferenceRecord], config: Mapping[str, Any], device: Any) -> dict[str, np.ndarray]:
    model.eval(); values: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch_records = records[offset:offset + BATCH_SIZE]
            # The saved M1 builder accepts the same input-only record shape;
            # labels are not needed for snapshot drift measurement.
            rows = [m1.build_input_ids(tokenizer, m1.M1Record(r.sample_id, r.stock_code, r.stock_name, r.model_text, {}, {head: 1.0 for head in V1_HEADS}), config) for r in batch_records]
            pad_id = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
            max_len = max(len(row) for row in rows)
            input_ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
            attention = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
            outputs = model(input_ids, attention)
            for head in SINGLE_LABEL_HEADS:
                values[head].append(torch.softmax(outputs[head], dim=-1).detach().cpu().numpy())
            values["reasoning_tags"].append(torch.sigmoid(outputs["reasoning_tags"]).detach().cpu().numpy())
    return {head: np.concatenate(chunks, axis=0) for head, chunks in values.items()}


def _js_divergence(left: np.ndarray, right: np.ndarray, *, reasoning: bool = False) -> float:
    if reasoning:
        left = np.stack([left, 1.0 - left], axis=-1); right = np.stack([right, 1.0 - right], axis=-1)
    left = np.clip(left, 1e-12, 1.0); right = np.clip(right, 1e-12, 1.0)
    midpoint = 0.5 * (left + right)
    return float(np.mean(0.5 * np.sum(left * np.log(left / midpoint), axis=-1) + 0.5 * np.sum(right * np.log(right / midpoint), axis=-1)))


def snapshot_drift(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], schema: LabelSchema, thresholds: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in V1_HEADS:
        old, new = np.asarray(before[head]), np.asarray(after[head])
        if head in SINGLE_LABEL_HEADS:
            old_pred, new_pred = old.argmax(axis=1), new.argmax(axis=1)
            old_conf, new_conf = old.max(axis=1), new.max(axis=1)
            changed = old_pred != new_pred
            result[head] = {"prediction_change_rate": float(changed.mean()), "prediction_change_count": int(changed.sum()), "mean_js_divergence": _js_divergence(old, new), "high_confidence_flip_count": int((changed & (old_conf >= 0.8) & (new_conf >= 0.8)).sum()), "category_distribution_before": {label: int((old_pred == index).sum()) for index, label in enumerate(schema.class_order[head])}, "category_distribution_after": {label: int((new_pred == index).sum()) for index, label in enumerate(schema.class_order[head])}}
        else:
            old_pred = old >= np.asarray([float(thresholds.get(label, 0.5)) for label in schema.class_order[head]])
            new_pred = new >= np.asarray([float(thresholds.get(label, 0.5)) for label in schema.class_order[head]])
            changed = np.any(old_pred != new_pred, axis=1)
            result[head] = {"prediction_change_rate": float(changed.mean()), "prediction_change_count": int(changed.sum()), "mean_js_divergence": _js_divergence(old, new, reasoning=True), "high_confidence_flip_count": int((changed & (old.max(axis=1) >= 0.8) & (new.max(axis=1) >= 0.8)).sum()), "label_positive_counts_before": {label: int(old_pred[:, index].sum()) for index, label in enumerate(schema.class_order[head])}, "label_positive_counts_after": {label: int(new_pred[:, index].sum()) for index, label in enumerate(schema.class_order[head])}}
    result["snapshot_rows"] = int(next(iter(before.values())).shape[0])
    return result


def _baseline_arrays(predictions: Sequence[Mapping[str, Any]], schema: LabelSchema) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}
    for row in predictions:
        for head in SINGLE_LABEL_HEADS:
            arrays[head].append(np.asarray([float(row["predictions"][head]["probabilities"][label]) for label in schema.class_order[head]], dtype=float))
        arrays["reasoning_tags"].append(np.asarray([float(row["predictions"]["reasoning_tags"]["probabilities"][label]) for label in schema.class_order["reasoning_tags"]], dtype=float))
    return {head: np.asarray(values, dtype=float) for head, values in arrays.items()}


def _metrics_from_arrays(arrays: Mapping[str, np.ndarray], records: Sequence[m1.M1Record], schema: LabelSchema, thresholds: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        order = list(schema.class_order[head]); truth = [order.index(str(record.label[head])) for record in records]; predicted = np.asarray(arrays[head]).argmax(axis=1).tolist(); precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(len(order))), zero_division=0)
        result[head] = {"accuracy": float(np.mean(np.asarray(truth) == np.asarray(predicted))), "macro_f1": float(np.mean(f1)), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(order)))).tolist()}
    result["reasoning_tags"] = _reasoning_metric(np.asarray(arrays["reasoning_tags"]), records, list(schema.class_order["reasoning_tags"]), thresholds)
    for head in V1_HEADS:
        result[head].pop("probabilities", None)
    return result


def classify_trend(baseline: Mapping[str, Mapping[str, float]], treatment: Mapping[str, Mapping[str, float]]) -> str:
    improved = sum(float(treatment[head]["macro_f1"]) > float(baseline[head]["macro_f1"]) + 1e-12 for head in V1_HEADS)
    declined = sum(float(treatment[head]["macro_f1"]) < float(baseline[head]["macro_f1"]) - 1e-12 for head in V1_HEADS)
    # A majority of improving heads is not sufficient when another head has
    # an obvious Check20 regression.  A five-point Macro-F1 drop marks the
    # one-shot result as mixed rather than positive.
    severe_declines = sum(float(treatment[head]["macro_f1"]) - float(baseline[head]["macro_f1"]) <= -0.05 for head in V1_HEADS)
    if improved >= 4 and declined <= 2 and severe_declines == 0:
        return "POSITIVE"
    if improved == 0 or declined >= 4:
        return "NEGATIVE"
    return "MIXED"


def _train_head(*, runtime: tuple[Any, Any, Any, Any], root: Path, snapshot: Path, schema: Any, config: Mapping[str, Any], parent: Mapping[str, Any], weak: Sequence[m1.M1Record], human: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], head: str, started: float, cache_dir: Path, device_override: Any = None) -> dict[str, Any]:
    np_module, torch, AutoModel, AutoTokenizer = runtime
    _set_seed(torch, SEED)
    device, device_name, mps_available = m1._choose_device(torch)
    if device_override is not None:
        device = device_override; device_name = str(device); mps_available = str(device) == "mps"
    folder = root / head / f"seed-{SEED}"; folder.mkdir(parents=True, exist_ok=False)
    seed_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = _make_warm_model(torch, AutoModel, snapshot, schema, config, parent, head).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    block_params = list(model.specialist_blocks[head].parameters()); head_params = list(model.heads[head].parameters())
    optimizer = torch.optim.AdamW([{"params": block_params, "lr": BLOCK_LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"]))
    thresholds = {label: float(parent.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}
    best_score, best_epoch, stale = -math.inf, 0, 0
    log_path = folder / "training-log.jsonl"
    weak_total = sum(float(record.weights[head]) for record in weak); human_total = sum(float(record.weights[head]) for record in human)
    for epoch in range(1, MAX_EPOCHS + 1):
        _limits(started, root, f"before_epoch_{head}")
        model.train(True)
        optimizer.zero_grad(set_to_none=True); source_losses: dict[str, float] = {}
        sources = (("weak", weak, WEAK_SOURCE_MASS, weak_total), ("human", human, HUMAN_SOURCE_MASS, human_total))
        for source_name, source_records, source_mass, source_total in sources:
            ordered = list(source_records); random.Random(SEED + epoch + (0 if source_name == "weak" else 10000)).shuffle(ordered); weighted_sum = 0.0; loss_sum = 0.0
            for offset in range(0, len(ordered), BATCH_SIZE):
                _limits(started, root, f"during_{source_name}_{head}")
                loss, weight_sum = _batch_loss(torch, model, tokenizer, ordered[offset:offset + BATCH_SIZE], config, device, head)
                (loss * (source_mass * weight_sum / source_total)).backward()
                weighted_sum += float(loss.detach().cpu()) * weight_sum; loss_sum += weight_sum
            source_losses[source_name] = weighted_sum / max(loss_sum, 1e-12)
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        for parameter in trainable:
            if parameter.grad is not None:
                _require(bool(torch.isfinite(parameter.grad).all().item()), "M5_NONFINITE_GRADIENT", "non-finite gradient", head=head)
        optimizer.step()
        dev_metric = _metric_from_logits(torch, model, tokenizer, dev, config, device, head, thresholds)
        score = float(dev_metric["macro_f1"]); improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": SEED, "parent_content_address": BASELINE_CONTENT_ADDRESS, "frozen_config": dict(config), "specialist_block_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {key: value.detach().cpu() for key, value in model.heads[head].state_dict().items()}}, folder / "human100-source-balanced-checkpoint.pt")
        else:
            stale += 1
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"head": head, "seed": SEED, "epoch": epoch, "weak_mean_loss": source_losses["weak"], "human_mean_loss": source_losses["human"], "objective": source_balanced_objective(source_losses["weak"], source_losses["human"]), "dev_macro_f1": score, "improved": improved, "stale_epochs": stale}, sort_keys=True) + "\n")
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "human100-source-balanced-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M5_CHECKPOINT_MISSING", "best checkpoint was not created", head=head)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = _make_warm_model(torch, AutoModel, snapshot, schema, config, parent, head).to(device)
    best_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); best_model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    train_metric = _metric_from_logits(torch, best_model, tokenizer, [*weak, *human], config, device, head, thresholds)
    dev_metric = _metric_from_logits(torch, best_model, tokenizer, dev, config, device, head, thresholds)
    cpu_model = _make_warm_model(torch, AutoModel, snapshot, schema, config, parent, head).to(torch.device("cpu")); cpu_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); cpu_model.heads[head].load_state_dict(checkpoint["head_state_dict"]); cpu_model.eval()
    cpu_tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    cpu_record = dev[0]
    cpu_batch = m1._as_batch(torch, cpu_tokenizer, [cpu_record], config, torch.device("cpu"))
    with torch.no_grad():
        cpu_output = cpu_model(cpu_batch["input_ids"], cpu_batch["attention_mask"])[head]
    cpu_smoke = {"required": True, "device": "cpu", "finite": bool(torch.isfinite(cpu_output).all().item()), "output_shape": list(cpu_output.shape), "sample_id": cpu_record.sample_id}
    _require(cpu_smoke["finite"], "M5_CPU_RELOAD_FAILED", "CPU reload emitted non-finite logits", head=head)
    elapsed = time.monotonic() - seed_started
    resource = {"head": head, "seed": SEED, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(elapsed, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": cpu_smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "source_balance": {"weak_mass": WEAK_SOURCE_MASS, "human_mass": HUMAN_SOURCE_MASS}}
    _dump(folder / "seed-metrics.json", {"head": head, "seed": SEED, "best_epoch": best_epoch, "early_stopping_metric": f"{head}.macro_f1", "train_population": {"weak_train": 1822, "human_train": 100, "objective": "0.5*mean(weak_train_loss)+0.5*mean(human_train_loss)"}, "train": {key: value for key, value in train_metric.items() if key != "probabilities"}, "dev": {key: value for key, value in dev_metric.items() if key != "probabilities"}, "cpu_reload": cpu_smoke})
    _dump(folder / "resource-log.json", resource)
    del model, best_model, cpu_model
    gc.collect()
    return {"head": head, "seed": SEED, "best_epoch": best_epoch, "actual_device": device_name, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "train": {key: value for key, value in train_metric.items() if key != "probabilities"}, "dev": {key: value for key, value in dev_metric.items() if key != "probabilities"}, "cpu_reload": cpu_smoke, "resource": resource}


def _load_checkpoint_model(torch: Any, AutoModel: Any, snapshot: Path, schema: Any, config: Mapping[str, Any], parent: Mapping[str, Any], root: Path, head: str, device: Any) -> tuple[Any, Any]:
    checkpoint_path = root / head / f"seed-{SEED}" / "human100-source-balanced-checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _make_warm_model(torch, AutoModel, snapshot, schema, config, parent, head).to(device)
    model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    return model, checkpoint


def run_m5(*, config_path: str | Path = DEFAULT_CONFIG, cache_dir: str | Path = DEFAULT_CACHE, parent_artifact: str | Path = DEFAULT_PARENT, pool_dir: str | Path = DEFAULT_POOL, snapshot_path: str | Path = DEFAULT_SNAPSHOT, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve(); _require(not output.exists(), "M5_OUTPUT_ALREADY_EXISTS", "M5 output directory must be new", path=str(output))
    started = time.monotonic()
    project = m1.ProjectConfig.load(config_path); schema, weak_train, dev = m1.load_m1_partitions(project); _require(len(weak_train) == 1822 and len(dev) == 448, "M5_TRAIN_DEV_SCOPE_INVALID", "expected Train 1822 and Dev 448")
    human_train, train_audit = load_human_train_pool(pool_dir); _require(not ({record.sample_id for record in human_train} & {record.sample_id for record in weak_train + dev}), "M5_HUMAN_SPLIT_OVERLAP", "Human Train overlaps canonical Train/Dev")
    runtime = m1._load_runtime_dependencies(); np_module, torch, AutoModel, AutoTokenizer = runtime; del np_module
    parent = _load_bundle(Path(parent_artifact).resolve(), torch); snapshot, snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
    frozen = lert._contract_requirements(REPO_ROOT / "manifests" / "encoder-m2-experiment-contract-v1.json"); config = lert._config(frozen["contract"], schema.class_order); config.update({"stage_id": STAGE_ID, "run_schema_version": RUN_SCHEMA_VERSION, "seed": SEED, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_delta": 0.0}, "fit_population": "Train_1822_plus_HumanTrain100_source_balanced", "human_check_role": "post_checkpoint_freeze_diagnostic_only", "source_balanced_objective": "0.5*mean(weak_train_loss)+0.5*mean(human_train_loss)", "selected_candidate": False})
    parent_thresholds = {label: float(parent.get("reasoning_thresholds", {}).get(label, 0.5)) for label in schema.class_order["reasoning_tags"]}; config["reasoning_thresholds"] = parent_thresholds
    output.mkdir(parents=True, exist_ok=False); _dump(output / "training-config.json", config)
    provenance = {"stage_id": STAGE_ID, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "parent_selected_artifact": {"path": str(Path(parent_artifact).resolve()), "content_address": BASELINE_CONTENT_ADDRESS, "manifest_sha256": sha256_file(Path(parent_artifact).resolve() / "content-addressed-manifest.json")}, "cache_snapshot": snapshot_identity, "data_scope": {"weak_train": 1822, "dev": 448, "human_train": 100, "human_check": 20}, "human_train_pool": train_audit, "forbidden_inputs_not_read": ["Test", "Anchor", "Gold", "OOD", "Human48", "Human100_workbook", "LLM", "cloud", "production"]}
    _dump(output / "provenance.json", provenance)
    # Baseline Dev and original 300 snapshot are read before training; Check20
    # is intentionally deferred until every checkpoint is frozen.
    baseline_runtime = selected.UnifiedCPUInferenceRuntime.from_artifact(parent_artifact, config_path=config_path, cache_dir=cache_dir)
    dev_inference = [selected.InferenceRecord(record.sample_id, record.stock_code, record.stock_name, record.model_text) for record in dev]
    baseline_dev_predictions = baseline_runtime.predict_records(dev_inference); baseline_dev_arrays = _baseline_arrays(baseline_dev_predictions, schema); baseline_dev_metrics = _metrics_from_arrays(baseline_dev_arrays, dev, schema, parent_thresholds)
    snapshot_rows = _read_jsonl(Path(snapshot_path).resolve(), "M5_SNAPSHOT_MISSING"); baseline_snapshot_arrays, snapshot_records = _snapshot_arrays(snapshot_rows, schema)
    results: list[dict[str, Any]] = []
    device_override = None
    for head in V1_HEADS:
        _limits(started, output, f"before_head_{head}")
        try:
            result = _train_head(runtime=runtime, root=output, snapshot=snapshot, schema=schema, config=config, parent=parent, weak=weak_train, human=human_train, dev=dev, head=head, started=started, cache_dir=Path(cache_dir).resolve(), device_override=device_override)
        except RuntimeError as exc:
            # MPS is first; a device/runtime failure retries this same head on
            # CPU from the untouched parent state, with no setting changes.
            if device_override is None and bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                shutil.rmtree(output / head, ignore_errors=True); device_override = torch.device("cpu"); result = _train_head(runtime=runtime, root=output, snapshot=snapshot, schema=schema, config=config, parent=parent, weak=weak_train, human=human_train, dev=dev, head=head, started=started, cache_dir=Path(cache_dir).resolve(), device_override=device_override)
            else:
                raise M5Error("M5_RUNTIME_EXCEPTION", "training runtime failed", head=head, exception_type=type(exc).__name__, detail=str(exc)) from exc
        results.append(result)
    _require(len(results) == len(V1_HEADS), "M5_CHECKPOINT_COVERAGE_INVALID", "all seven head checkpoints must be frozen")
    _limits(started, output, "after_checkpoint_freeze")
    # Only now open the semantic Human Check20 labels and run the single
    # descriptive before/after comparison.
    human_check, check_audit = load_human_check_pool(pool_dir)
    check_inference = [selected.InferenceRecord(record.sample_id, record.stock_code, record.stock_name, record.model_text) for record in human_check]
    baseline_check_predictions = baseline_runtime.predict_records(check_inference); baseline_check_arrays = _baseline_arrays(baseline_check_predictions, schema); baseline_check_metrics = _metrics_from_arrays(baseline_check_arrays, human_check, schema, parent_thresholds)
    treatment_models: dict[str, Any] = {}; treatment_arrays: dict[str, list[np.ndarray]] = {head: [] for head in V1_HEADS}; treatment_cpu_tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    # Assemble all frozen specialist states in one seven-head model.  Context
    # and reasoning are LERT outputs here; no Classical/SARCASM override is
    # allowed to omit feedback-trained heads from the treatment.
    treatment_model = lert._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"]));
    for result in results:
        head = result["head"]; checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True); treatment_model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"]); treatment_model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    treatment_model = treatment_model.to(torch.device("cpu")); treatment_model.eval()
    treatment_dev_arrays = _model_arrays(torch, treatment_model, treatment_cpu_tokenizer, [selected.InferenceRecord(record.sample_id, record.stock_code, record.stock_name, record.model_text) for record in dev], config, torch.device("cpu")); treatment_dev_metrics = _metrics_from_arrays(treatment_dev_arrays, dev, schema, parent_thresholds)
    treatment_check_arrays = _model_arrays(torch, treatment_model, treatment_cpu_tokenizer, check_inference, config, torch.device("cpu")); treatment_check_metrics = _metrics_from_arrays(treatment_check_arrays, human_check, schema, parent_thresholds)
    treatment_snapshot_arrays = _model_arrays(torch, treatment_model, treatment_cpu_tokenizer, snapshot_records, config, torch.device("cpu")); drift = snapshot_drift(baseline_snapshot_arrays, treatment_snapshot_arrays, schema, parent_thresholds)
    check_comparison = {head: {"baseline": baseline_check_metrics[head], "treatment": treatment_check_metrics[head], "macro_f1_delta": float(treatment_check_metrics[head]["macro_f1"] - baseline_check_metrics[head]["macro_f1"])} for head in V1_HEADS}
    dev_comparison = {head: {"baseline": baseline_dev_metrics[head], "treatment": treatment_dev_metrics[head], "macro_f1_delta": float(treatment_dev_metrics[head]["macro_f1"] - baseline_dev_metrics[head]["macro_f1"])} for head in V1_HEADS}
    trend = classify_trend(baseline_check_metrics, treatment_check_metrics)
    threshold_observations = {"action_tendency_above_20_percent": treatment_check_metrics["action_tendency"]["accuracy"] > 0.20, "emotion_primary_above_25_percent": treatment_check_metrics["emotion_primary"]["accuracy"] > 0.25, "stance_above_30_percent": treatment_check_metrics["stance"]["accuracy"] > 0.30, "reasoning_exact_set_above_20_percent": treatment_check_metrics["reasoning_tags"]["exact_set_accuracy"] > 0.20}
    _dump(output / "human-check20-before-after.json", {"baseline": baseline_check_metrics, "treatment": treatment_check_metrics, "comparison": check_comparison, "scope": "descriptive_after_all_seven_checkpoints_frozen"})
    _dump(output / "dev448-before-after.json", {"baseline": baseline_dev_metrics, "treatment": treatment_dev_metrics, "comparison": dev_comparison, "scope": "weak_label_regression_observation"})
    _dump(output / "snapshot300-drift.json", drift)
    _dump(output / "stage-aggregate.json", {"stage_id": STAGE_ID, "seed": SEED, "heads": list(V1_HEADS), "all_seven_checkpoints_frozen": True, "trend": trend, "threshold_observations": threshold_observations, "selected_candidate": False, "production_approval": False, "human_check_audit": check_audit})
    report_lines = ["# M5 Human100 source-balanced seed-35 diagnostic", "", f"Status: **{trend}**", "", "This is a single seed-35 source-balanced treatment.  The objective is exactly `0.5 * mean(weak_train_loss) + 0.5 * mean(human_train_loss)`; Human Check20 was read only after all seven checkpoints were frozen.  The result is diagnostic evidence and never replaces the frozen selected candidate.", "", f"Artifact: `{output}`", "", "## Human Check20", "", "| head | baseline Macro-F1 | treatment Macro-F1 | delta |", "|---|---:|---:|---:|"]
    for head in V1_HEADS:
        report_lines.append(f"| {head} | {baseline_check_metrics[head]['macro_f1']:.6f} | {treatment_check_metrics[head]['macro_f1']:.6f} | {check_comparison[head]['macro_f1_delta']:+.6f} |")
    report_lines += ["", "## Dev448 regression observation", "", "| head | baseline Macro-F1 | treatment Macro-F1 | delta |", "|---|---:|---:|---:|"]
    for head in V1_HEADS:
        report_lines.append(f"| {head} | {baseline_dev_metrics[head]['macro_f1']:.6f} | {treatment_dev_metrics[head]['macro_f1']:.6f} | {dev_comparison[head]['macro_f1_delta']:+.6f} |")
    report_lines += ["", "## Fixed observations", "", f"- action_tendency accuracy > 20%: **{threshold_observations['action_tendency_above_20_percent']}**", f"- emotion_primary accuracy > 25%: **{threshold_observations['emotion_primary_above_25_percent']}**", f"- stance accuracy > 30%: **{threshold_observations['stance_above_30_percent']}**", f"- reasoning exact-set accuracy > 20%: **{threshold_observations['reasoning_exact_set_above_20_percent']}**", "- The 300-row snapshot drift measures influence, not quality; it is not a target or promotion gate.", "- Context and reasoning are trained in this treatment; no selected-runtime Classical/SARCASM override is used for treatment outputs.", "", "No Test, Anchor, Gold, OOD, Human48, LLM, cloud, or production data were read."]
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = m1._write_content_manifest(output, {"manifest_schema_version": "myresearcher.encoder-m5-human100-source-balanced-artifact-manifest.v1", "stage_id": STAGE_ID, "status": trend, "selected_candidate": False, "production_approval": False, "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "parent_selected_artifact_content_address": BASELINE_CONTENT_ADDRESS, "cache_snapshot": snapshot_identity, "seed": SEED, "train_rows": 1822, "human_train_rows": 100, "dev_rows": 448, "human_check_rows": 20, "source_balanced_objective": "0.5*mean(weak_train_loss)+0.5*mean(human_train_loss)", "checkpoint_sha256": {item["head"]: item["checkpoint_sha256"] for item in results}, "human_check_sha256": sha256_file(output / "human-check20-before-after.json"), "dev_sha256": sha256_file(output / "dev448-before-after.json"), "snapshot_drift_sha256": sha256_file(output / "snapshot300-drift.json"), "forbidden_inputs": provenance["forbidden_inputs_not_read"]})
    _limits(started, output, "after_final_manifest")
    return {"status": trend, "stage_id": STAGE_ID, "output_dir": str(output), "content_address": manifest["content_address"], "results": results, "human_check": check_comparison, "dev": dev_comparison, "snapshot_drift": drift, "threshold_observations": threshold_observations, "training_invoked": True, "selected_candidate": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the one-shot M5 Human100 source-balanced seed-35 diagnostic")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE)); parser.add_argument("--parent-artifact", default=str(DEFAULT_PARENT)); parser.add_argument("--pool-dir", default=str(DEFAULT_POOL)); parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT)); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        result = run_m5(config_path=args.config, cache_dir=args.cache_dir, parent_artifact=args.parent_artifact, pool_dir=args.pool_dir, snapshot_path=args.snapshot, output_dir=args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "M5_BLOCKED_OR_REJECTED", "blocker_code": exc.code, "details": exc.details, "training_invoked": False}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
