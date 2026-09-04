"""M2.1 semantic-soft-label experiment for two scalar heads.

Only ``emotion_primary`` and ``action_tendency`` are trained.  Each specialist
starts from the fixed local LERT snapshot, uses Train 1,822 for fitting, and
uses Dev 448 for fixed early stopping and the predeclared adoption decision.
The human-review workbook is opened only after that decision and is never
used for fitting, early stopping, or selection.
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
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import encoder_m1 as m1
from . import encoder_m2_lert_final as lert
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .preprocessing import PreprocessingContract, build_model_inputs
from .schema import LabelSchema


STAGE_ID = "M2.1-SEMANTIC-SOFTLABEL-TWO-HEAD"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-1-semantic-softlabel-run.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
SEED = 35
HEADS = ("emotion_primary", "action_tendency")
SINGLE_HEADS = HEADS
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
SOFT_TRUE_MASS = 0.80
SOFT_ADJACENT_MASS = 0.20
M3B_WORKBOOK = Path(__file__).resolve().parents[3] / "human-review-artifacts/m3b-gold100-20260902/m3b-gold100-review-pack.xlsx"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".encoder-artifacts/hf-cache-lert-small"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "model-artifacts/m2-1-semantic-softlabel-two-head-20260904"
CANDIDATE_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts/m2-head-selected-unified-candidate-20260902"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "manifests/encoder-m2-experiment-contract-v1.json"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/baseline_v0.3.5.yaml"
SNAPSHOT_RELATIVE = Path("official-snapshot") / REVISION
EXPECTED_SNAPSHOT_FILES = {
    "added_tokens.json": (2, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
    "config.json": (774, "9450c1909a1c93b5fccc26e317f9cdf1ca451c33123a97a19d312ab473975dd4"),
    "pytorch_model.bin": (60775606, "d837ff714afa3f7df547e5787fdc8e4de7fcbaf549c465e562845aa60fbe7666"),
    "special_tokens_map.json": (112, "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"),
    "tokenizer.json": (268961, "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6"),
    "tokenizer_config.json": (19, "61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874"),
    "vocab.txt": (109540, "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"),
}

SIMILARITY_RELATIONS: dict[tuple[str, str], float] = {
    ("FEAR", "ANXIETY"): 0.7,
    ("ANGER", "FRUSTRATION"): 0.7,
    ("EXCITEMENT", "FOMO"): 0.7,
    ("HOPE", "EXCITEMENT"): 0.5,
    ("BUY", "ADD"): 0.8,
    ("SELL", "REDUCE"): 0.8,
    ("HOLD", "WATCH"): 0.5,
}

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SIMILARITY_RELATIONS.get((left, right), SIMILARITY_RELATIONS.get((right, left), 0.0))


def soft_label_distribution(head: str, label: str, order: Sequence[str]) -> list[float]:
    """Return the frozen 0.80/0.20 soft target for one scalar label."""

    _require(head in HEADS, "M2_1_HEAD_INVALID", "soft labels are restricted to the two experiment heads", head=head)
    _require(label in order, "M2_1_LABEL_INVALID", "label is not in the frozen class order", head=head, label=label)
    neighbors: list[tuple[str, float]] = []
    for (left, right), score in SIMILARITY_RELATIONS.items():
        if head == "emotion_primary" and left in {"FEAR", "ANXIETY", "ANGER", "FRUSTRATION", "EXCITEMENT", "FOMO", "HOPE"} and right in {"FEAR", "ANXIETY", "ANGER", "FRUSTRATION", "EXCITEMENT", "FOMO", "HOPE"}:
            if left == label:
                neighbors.append((right, score))
            elif right == label:
                neighbors.append((left, score))
        if head == "action_tendency" and left in {"BUY", "ADD", "SELL", "REDUCE", "HOLD", "WATCH"} and right in {"BUY", "ADD", "SELL", "REDUCE", "HOLD", "WATCH"}:
            if left == label:
                neighbors.append((right, score))
            elif right == label:
                neighbors.append((left, score))
    result = [0.0] * len(order)
    result[order.index(label)] = 1.0
    if not neighbors:
        return result
    result[order.index(label)] = SOFT_TRUE_MASS
    total = sum(score for _, score in neighbors)
    for neighbor, score in neighbors:
        result[order.index(neighbor)] = SOFT_ADJACENT_MASS * score / total
    return result


def soft_targets(records: Sequence[m1.M1Record], head: str, order: Sequence[str]) -> dict[str, list[float]]:
    return {record.sample_id: soft_label_distribution(head, str(record.label[head]), order) for record in records}


def expected_semantic_score(probabilities: np.ndarray, truth: Sequence[str], order: Sequence[str]) -> float:
    matrix = np.asarray([[_similarity(label, actual) for label in order] for actual in truth], dtype=float)
    return float(np.mean(np.sum(probabilities * matrix, axis=1)))


def strict_and_semantic_metrics(probabilities: np.ndarray, truth: Sequence[str], order: Sequence[str]) -> dict[str, Any]:
    predictions = [order[int(index)] for index in np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predictions, labels=order, zero_division=0)
    true_probability = np.asarray([probabilities[index, order.index(value)] for index, value in enumerate(truth)], dtype=float)
    semantic_top1 = float(np.mean([_similarity(prediction, actual) for prediction, actual in zip(predictions, truth, strict=True)]))
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(truth))),
        "macro_f1": float(f1_score(truth, predictions, labels=order, average="macro", zero_division=0)),
        "per_class": {label: {"f1": float(f1[index]), "precision": float(precision[index]), "recall": float(recall[index]), "support": int(support[index])} for index, label in enumerate(order)},
        "confusion_matrix": confusion_matrix(truth, predictions, labels=order).tolist(),
        "class_order": list(order),
        "semantic_top1_score": semantic_top1,
        "expected_semantic_score": expected_semantic_score(probabilities, truth, order),
        "true_label_probability": float(true_probability.mean()),
        "raw_probability": {"top1_probability_mean": float(probabilities.max(axis=1).mean()), "true_label_probability_mean": float(true_probability.mean())},
    }


def selection_gate(soft: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    expected_delta = float(soft["expected_semantic_score"]) - float(current["expected_semantic_score"])
    macro_delta = float(soft["macro_f1"]) - float(current["macro_f1"])
    # Allow only floating-point representation noise at the exact boundary.
    checks = {"expected_semantic_improvement_at_least_0_02": expected_delta >= 0.02 - 1e-12, "macro_f1_drop_within_0_01": macro_delta >= -0.01 - 1e-12}
    return {"passed": all(checks.values()), "checks": checks, "expected_semantic_delta": expected_delta, "macro_f1_delta": macro_delta, "decision": "ADOPT_SOFT_LABEL_HEAD" if all(checks.values()) else "KEEP_CURRENT_SOURCE"}


def _snapshot(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    snapshot = (cache_dir / SNAPSHOT_RELATIVE).resolve()
    _require(snapshot.is_dir(), "M2_1_FIXED_LERT_CACHE_MISSING", "fixed local LERT snapshot is missing", path=str(snapshot))
    files = []
    for name, (expected_bytes, expected_sha) in EXPECTED_SNAPSHOT_FILES.items():
        path = snapshot / name
        _require(path.is_file(), "M2_1_FIXED_LERT_CACHE_FILE_MISSING", "fixed LERT file is missing", file=name)
        observed = {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        _require(observed["bytes"] == expected_bytes and observed["sha256"] == expected_sha, "M2_1_FIXED_LERT_CACHE_FILE_MISMATCH", "fixed LERT snapshot differs", file=name, observed=observed, expected={"bytes": expected_bytes, "sha256": expected_sha})
        files.append(observed)
    identity = {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "required_relative_directory": SNAPSHOT_RELATIVE.as_posix(), "files": files}
    identity["content_address"] = content_addressed_id({"files": files})
    return snapshot, identity


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    return m1._load_runtime_dependencies()


def _set_seed(torch: Any, np_module: Any, seed: int = SEED) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def _load_weights(path: Path, torch: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
        if version_type is None:
            raise
        torch.serialization.add_safe_globals([version_type])
        return torch.load(path, map_location="cpu", weights_only=True)


def _model_for_checkpoint(torch: Any, AutoModel: Any, snapshot: Path, schema: LabelSchema, cfg: Mapping[str, Any], head: str, checkpoint: Mapping[str, Any], device: Any) -> Any:
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(cfg["head_dropout"]))
    model.set_active_head(head)
    model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
    model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    return model.to(device)


def _validate_trainable(model: Any, head: str) -> dict[str, Any]:
    expected = {name for name, parameter in model.named_parameters() if name.startswith(f"specialist_blocks.{head}.") or name.startswith(f"heads.{head}.")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _require(actual == expected and expected, "M2_1_TRAINABLE_SCOPE_INVALID", "only selected specialist block and head may train", head=head)
    _require(not any(name.startswith("encoder.") for name in actual), "M2_1_ENCODER_UNFROZEN", "shared LERT encoder must remain frozen", head=head)
    return {"head": head, "trainable_parameters": sorted(actual), "shared_encoder_frozen": True, "other_experiment_head_not_trained": True}


def _predict(torch: Any, model: Any, tokenizer: Any, records: Sequence[m1.M1Record], cfg: Mapping[str, Any], head: str, device: Any) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), BATCH_SIZE):
            batch = m1._as_batch(torch, tokenizer, records[offset:offset + BATCH_SIZE], cfg, device)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "M2_1_NONFINITE_LOGITS", "non-finite logits", head=head)
            values.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, len(cfg["class_order"][head])))


def _limits(started: float, root: Path, phase: str, seed: int = SEED) -> None:
    _require(time.monotonic() - started <= WALL_TIME_SECONDS, "M2_1_WALL_TIME_LIMIT_EXCEEDED", "two-hour experiment limit exceeded", phase=phase, seed=seed)
    _require(m1._directory_size(root) <= MAX_DISK_GIB * 1024**3, "M2_1_DISK_LIMIT_EXCEEDED", "ten GiB output limit exceeded", phase=phase, seed=seed)


def train_soft_head(*, runtime: tuple[Any, Any, Any, Any], head: str, root: Path, snapshot: Path, schema: LabelSchema, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], cfg: Mapping[str, Any], started: float, cache_dir: Path) -> dict[str, Any]:
    np_module, torch, AutoModel, AutoTokenizer = runtime
    order = list(schema.class_order[head])
    folder = root / head / f"seed-{SEED}"
    folder.mkdir(parents=True, exist_ok=False)
    _set_seed(torch, np_module)
    device, device_name, mps_available = m1._choose_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(cfg["head_dropout"]))
    model.set_active_head(head)
    model = model.to(device)
    parameter_identity = _validate_trainable(model, head)
    target_lookup = soft_targets(train, head, order)
    trainable_block = list(model.specialist_blocks[head].parameters())
    trainable_head = list(model.heads[head].parameters())
    optimizer = torch.optim.AdamW([{"params": trainable_block, "lr": BLOCK_LR}, {"params": trainable_head, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY, betas=tuple(cfg["optimizer"]["betas"]), eps=float(cfg["optimizer"]["epsilon"]))
    best_score, best_epoch, stale = -math.inf, 0, 0
    log_path = folder / "training-log.jsonl"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(True)
        shuffled = list(train)
        random.Random(SEED + epoch).shuffle(shuffled)
        losses: list[float] = []
        for offset in range(0, len(shuffled), BATCH_SIZE):
            _limits(started, root, "during_fit")
            batch_records = shuffled[offset:offset + BATCH_SIZE]
            batch = m1._as_batch(torch, tokenizer, batch_records, cfg, device)
            target = torch.tensor([target_lookup[item.sample_id] for item in batch_records], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])[head]
            _require(bool(torch.isfinite(logits).all().item()), "M2_1_NONFINITE_LOGITS", "non-finite logits", head=head)
            loss_vector = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=1)
            weights = batch["weights"][head]
            loss = (loss_vector * weights).sum() / weights.sum().clamp_min(1e-12)
            _require(bool(torch.isfinite(loss).item()), "M2_1_NONFINITE_LOSS", "non-finite soft-label loss", head=head)
            loss.backward()
            for parameter in [*trainable_block, *trainable_head]:
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all().item()), "M2_1_NONFINITE_GRADIENT", "non-finite gradient", head=head)
            torch.nn.utils.clip_grad_norm_([*trainable_block, *trainable_head], GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_prob = _predict(torch, model, tokenizer, dev, cfg, head, device)
        dev_truth = [str(record.label[head]) for record in dev]
        dev_metrics = strict_and_semantic_metrics(dev_prob, dev_truth, order)
        score = float(dev_metrics["macro_f1"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "head": head, "seed": SEED, "model_id": MODEL_ID, "revision": REVISION, "frozen_config": cfg, "soft_label_rule": {"true_label_mass": SOFT_TRUE_MASS, "adjacent_mass": SOFT_ADJACENT_MASS, "relations": [{"left": left, "right": right, "similarity": value} for (left, right), value in SIMILARITY_RELATIONS.items()]}, "parameter_identity": parameter_identity, "specialist_block_state_dict": {key: value.detach().cpu() for key, value in model.specialist_blocks[head].state_dict().items()}, "head_state_dict": {key: value.detach().cpu() for key, value in model.heads[head].state_dict().items()}}, folder / "softlabel-checkpoint.pt")
        else:
            stale += 1
        m1._jsonl_append(log_path, {"head": head, "seed": SEED, "epoch": epoch, "train_soft_label_loss": round(sum(losses) / len(losses), 6), "dev": dev_metrics, "early_stopping_metric": "strict_macro_f1", "improved": improved, "stale_epochs": stale})
        if stale >= PATIENCE:
            break
    checkpoint_path = folder / "softlabel-checkpoint.pt"
    _require(checkpoint_path.is_file(), "M2_1_CHECKPOINT_MISSING", "soft-label checkpoint was not created", head=head)
    checkpoint = _load_weights(checkpoint_path, torch)
    best_model = _model_for_checkpoint(torch, AutoModel, snapshot, schema, cfg, head, checkpoint, device)
    _validate_trainable(best_model, head)
    train_prob = _predict(torch, best_model, tokenizer, train, cfg, head, device)
    dev_prob = _predict(torch, best_model, tokenizer, dev, cfg, head, device)
    train_metrics = strict_and_semantic_metrics(train_prob, [str(record.label[head]) for record in train], order)
    dev_metrics = strict_and_semantic_metrics(dev_prob, [str(record.label[head]) for record in dev], order)
    cpu_model = _model_for_checkpoint(torch, AutoModel, snapshot, schema, cfg, head, checkpoint, torch.device("cpu"))
    cpu_model.eval()
    cpu_batch = m1._as_batch(torch, tokenizer, [dev[0]], cfg, torch.device("cpu"))
    with torch.no_grad():
        cpu_logits = cpu_model(cpu_batch["input_ids"], cpu_batch["attention_mask"])[head]
    cpu_finite = bool(torch.isfinite(cpu_logits).all().item())
    _require(cpu_finite, "M2_1_CPU_RELOAD_FAILED", "CPU reload emitted non-finite logits", head=head)
    metrics = {"head": head, "seed": SEED, "best_epoch": best_epoch, "early_stopping_metric": "strict_macro_f1", "early_stopping_score": best_score, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "soft_label_targets": {"quality_mass": SOFT_TRUE_MASS, "adjacent_mass": SOFT_ADJACENT_MASS, "no_adjacent_relation_is_one_hot": True}, "cpu_reload": {"required": True, "device": "cpu", "finite": cpu_finite, "output_shape": list(cpu_logits.shape)}}
    _dump(folder / "seed-metrics.json", metrics)
    resource = {"head": head, "seed": SEED, "actual_device": device_name, "mps_available": mps_available, "elapsed_seconds": round(time.monotonic() - started, 3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": metrics["cpu_reload"], "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "parameter_identity": parameter_identity}
    _dump(folder / "resource-log.json", resource)
    _limits(started, root, "after_head")
    return {"head": head, "seed": SEED, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "metrics": metrics, "resource": resource}


def _classical_probabilities(records: Sequence[Mapping[str, Any]], schema: LabelSchema, prep: PreprocessingContract, model_path: Path) -> dict[str, np.ndarray]:
    legacy = joblib.load(model_path)
    for estimator in [*legacy["scalar_models"].values(), *legacy["reason_models"].values()]:
        if not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"
    texts = build_model_inputs(records, prep)
    matrix = hstack([legacy["char_vectorizer"].transform(texts), legacy["word_vectorizer"].transform(texts)], format="csr")
    result: dict[str, np.ndarray] = {}
    for head in HEADS:
        estimator = legacy["scalar_models"][head]
        classes = [str(value) for value in estimator.classes_]
        native = estimator.predict_proba(matrix)
        ordered = np.zeros((len(records), len(schema.class_order[head])), dtype=float)
        for source_index, label in enumerate(classes):
            if label in schema.class_order[head]:
                ordered[:, schema.class_order[head].index(label)] = native[:, source_index]
        result[head] = ordered
    return result


def _read_human_workbook(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(text.text or "" for text in item.findall(".//a:t", NS)) for item in shared_root.findall("a:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        table: list[dict[str, str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", NS):
            if int(row.attrib["r"]) < 3:
                continue
            values: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                column = "".join(character for character in cell.attrib["r"] if character.isalpha())
                value_node = cell.find("a:v", NS)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    index = int(value_node.text or "0")
                    value = shared[index] if index < len(shared) else ""
                else:
                    value = value_node.text or ""
                values[column] = value
            table.append(values)
    header_columns = sorted(table[0], key=lambda value: (len(value), value))
    headers = [table[0][column] for column in header_columns]
    records = [{headers[index]: row.get(column, "") for index, column in enumerate(header_columns)} for row in table[1:]]
    _require(len(records) == 100, "M2_1_HUMAN100_ROW_COUNT_INVALID", "human review workbook must contain exactly 100 rows", observed=len(records))
    required = {"sample_id", "stock_code", "stock_name", "model_text", "human_emotion_primary", "human_action_tendency", "human_context_dependency", "human_reasoning_tags"}
    _require(required <= set(headers), "M2_1_HUMAN100_COLUMNS_INVALID", "human workbook columns are incomplete")
    _require(all(not row["human_reasoning_tags"].strip() for row in records), "M2_1_HUMAN100_REASONING_TAGS_NONEMPTY", "human_reasoning_tags must remain ignored and empty")
    _require(all(row["sample_id"].strip() and row["model_text"].strip() for row in records), "M2_1_HUMAN100_INPUT_EMPTY", "human sample IDs and model_text must be non-empty")
    return records


def _human_metrics(rows: Sequence[Mapping[str, str]], candidate_probabilities: Mapping[str, np.ndarray], classical_probabilities: Mapping[str, np.ndarray], schema: LabelSchema, output: Path) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"rows": len(rows), "scope": "human_reviewed_Gold100_only", "context_dependency_scored": False, "heads": {}}
    for head in HEADS:
        order = list(schema.class_order[head])
        truth = [str(row[f"human_{head}"]) for row in rows]
        candidate = strict_and_semantic_metrics(candidate_probabilities[head], truth, order)
        classical = strict_and_semantic_metrics(classical_probabilities[head], truth, order)
        candidate_pred = [order[int(index)] for index in np.argmax(candidate_probabilities[head], axis=1)]
        classical_pred = [order[int(index)] for index in np.argmax(classical_probabilities[head], axis=1)]
        c_correct = np.asarray(candidate_pred) == np.asarray(truth)
        k_correct = np.asarray(classical_pred) == np.asarray(truth)
        for index, row in enumerate(rows):
            comparisons.append({"row": index + 1, "sample_id": row["sample_id"], "head": head, "human_label": truth[index], "candidate": {"prediction": candidate_pred[index], "top1_probability": float(candidate_probabilities[head][index].max()), "human_label_probability": float(candidate_probabilities[head][index, order.index(truth[index])]), "correct": bool(c_correct[index]), "semantic_top1_score": _similarity(candidate_pred[index], truth[index])}, "classical": {"prediction": classical_pred[index], "top1_probability": float(classical_probabilities[head][index].max()), "human_label_probability": float(classical_probabilities[head][index, order.index(truth[index])]), "correct": bool(k_correct[index]), "semantic_top1_score": _similarity(classical_pred[index], truth[index])}, "paired_correctness_delta": int(c_correct[index]) - int(k_correct[index])})
        metrics["heads"][head] = {"candidate": candidate, "classical": classical, "candidate_minus_classical": {"accuracy": candidate["accuracy"] - classical["accuracy"], "macro_f1": candidate["macro_f1"] - classical["macro_f1"], "semantic_top1_score": candidate["semantic_top1_score"] - classical["semantic_top1_score"], "expected_semantic_score": candidate["expected_semantic_score"] - classical["expected_semantic_score"], "true_label_probability": candidate["true_label_probability"] - classical["true_label_probability"]}, "paired_outcomes": {"candidate_only_correct": int(np.sum(c_correct & ~k_correct)), "classical_only_correct": int(np.sum(~c_correct & k_correct)), "both_correct": int(np.sum(c_correct & k_correct)), "both_wrong": int(np.sum(~c_correct & ~k_correct))}}
    with (output / "human100-comparison.jsonl").open("x", encoding="utf-8") as handle:
        for item in comparisons:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    _dump(output / "human100-metrics.json", metrics)
    return metrics


def _content_manifest(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "content-addressed-manifest.json"]
    manifest = {**payload, "files": files}
    manifest["content_address"] = content_addressed_id(manifest)
    _dump(root / "content-addressed-manifest.json", manifest)
    return manifest


def _report(selection: Mapping[str, Any], human: Mapping[str, Any], output: Path, status: str) -> str:
    lines = ["# M2.1 semantic soft-label two-head experiment", "", f"Status: **{status}**", "", "Only `emotion_primary` and `action_tendency` were trained from the fixed local LERT snapshot. Human-review100 was opened once after Dev selection and was not used for fitting, early stopping, or selection.", "", f"Artifact: `{output}`", "", "## Dev selection"]
    lines.append("| Head | Current source Macro-F1 / Expected semantic | Soft-label Macro-F1 / Expected semantic | Δ Macro-F1 | Δ Expected semantic | Decision |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for head in HEADS:
        item = selection[head]; current, soft, gate = item["current"], item["soft_label"], item["gate"]
        lines.append(f"| {head} | {current['macro_f1']:.3f} / {current['expected_semantic_score']:.3f} | {soft['macro_f1']:.3f} / {soft['expected_semantic_score']:.3f} | {gate['macro_f1_delta']:+.3f} | {gate['expected_semantic_delta']:+.3f} | {gate['decision']} |")
    lines.extend(["", "## Human-review100 diagnostic", "", "Strict metrics, per-class F1, confusion matrices, raw probabilities, semantic auxiliary scores, and paired row outcomes are in `human100-metrics.json`; each row is in `human100-comparison.jsonl`."])
    for head in HEADS:
        item = human["heads"][head]
        lines.append(f"- `{head}`: Candidate Accuracy/Macro-F1 {item['candidate']['accuracy']:.3f}/{item['candidate']['macro_f1']:.3f}; Classical {item['classical']['accuracy']:.3f}/{item['classical']['macro_f1']:.3f}; strict Δ {item['candidate_minus_classical']['accuracy']:+.3f}/{item['candidate_minus_classical']['macro_f1']:+.3f}; expected-semantic Δ {item['candidate_minus_classical']['expected_semantic_score']:+.3f}.")
    lines.extend(["", "No Test, OOD, Anchor, LLM, cloud, or production data was accessed. No model was downloaded and no dependency was installed or changed.", ""])
    return "\n".join(lines)


def run(config_path: str | Path = CONFIG_PATH, cache_dir: str | Path = DEFAULT_CACHE, output_dir: str | Path = DEFAULT_OUTPUT, workbook: str | Path = M3B_WORKBOOK) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    _require(not root.exists(), "M2_1_OUTPUT_ALREADY_EXISTS", "output directory is immutable and must not already exist", output_dir=str(root))
    started = time.monotonic()
    contract = lert._contract_requirements(CONTRACT_PATH)
    config_obj = m1.ProjectConfig.load(config_path)
    schema, train, dev = m1.load_m1_partitions(config_obj)
    _require(len(train) == 1822 and len(dev) == 448, "M2_1_DATA_SCOPE_INVALID", "M2.1 requires Train 1,822 and Dev 448")
    cfg = lert._config(contract["contract"], schema.class_order)
    snapshot, snapshot_identity = _snapshot(Path(cache_dir).resolve())
    root.mkdir(parents=True, exist_ok=False)
    _dump(root / "training-config.json", {"stage_id": STAGE_ID, "run_schema_version": RUN_SCHEMA_VERSION, "model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "seed": SEED, "heads": list(HEADS), "encoder_state": "frozen_shared_bottom_with_independent_last_block_and_head", "fit_population": "Train_1822_only", "dev_role": "strict_metrics_early_stopping_and_selection_only", "human100_role": "one_post_selection_diagnostic_only", "max_length": MAX_LENGTH, "truncation": "HEAD_TAIL", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "optimizer": {"name": "AdamW", "block_learning_rate": BLOCK_LR, "head_learning_rate": HEAD_LR, "weight_decay": WEIGHT_DECAY}, "gradient_clip_norm": GRADIENT_CLIP, "device_policy": "MPS_FIRST_CPU_FALLBACK", "wall_time_seconds": WALL_TIME_SECONDS, "disk_limit_gib": MAX_DISK_GIB, "soft_label_rule": {"true_label_mass": SOFT_TRUE_MASS, "adjacent_mass": SOFT_ADJACENT_MASS, "no_adjacent_relation_is_one_hot": True, "relations": [{"left": left, "right": right, "similarity": value} for (left, right), value in SIMILARITY_RELATIONS.items()]}, "selection_gate": {"expected_semantic_score_min_improvement": 0.02, "macro_f1_max_drop": 0.01}, "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "LLM", "cloud", "production"]})
    runtime = _load_runtime()
    runtime_identity = {"python": platform.python_version(), "torch": runtime[1].__version__, "transformers": __import__("transformers").__version__, "tokenizers": __import__("tokenizers").__version__, "numpy": runtime[0].__version__, "device_policy": "MPS_FIRST_CPU_FALLBACK"}
    _dump(root / "runtime.json", runtime_identity)
    np_module, torch, AutoModel, AutoTokenizer = runtime
    prep = PreprocessingContract.load(config_obj.data_path("preprocessing_contract"))
    classical_path = CANDIDATE_ARTIFACT / "classical-fallback/original_model_v0.3.5.joblib"
    classical_dev = _classical_probabilities([{"sample_id": record.sample_id, "stock_code": record.stock_code, "stock_name": record.stock_name, "model_text": record.model_text} for record in dev], schema, prep, classical_path)
    baseline_emotion_checkpoint_path = CANDIDATE_ARTIFACT / "checkpoints/emotion_primary/seed-35/specialist-checkpoint.pt"
    baseline_emotion_checkpoint = _load_weights(baseline_emotion_checkpoint_path, torch)
    baseline_tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    baseline_model = _model_for_checkpoint(torch, AutoModel, snapshot, schema, cfg, "emotion_primary", baseline_emotion_checkpoint, torch.device("cpu"))
    baseline_dev_emotion = _predict(torch, baseline_model, baseline_tokenizer, dev, cfg, "emotion_primary", torch.device("cpu"))
    baseline_current = {"emotion_primary": strict_and_semantic_metrics(baseline_dev_emotion, [str(record.label["emotion_primary"]) for record in dev], list(schema.class_order["emotion_primary"])), "action_tendency": strict_and_semantic_metrics(classical_dev["action_tendency"], [str(record.label["action_tendency"]) for record in dev], list(schema.class_order["action_tendency"]))}
    _limits(started, root, "before_fit")
    results: dict[str, Any] = {}
    for head in HEADS:
        results[head] = train_soft_head(runtime=runtime, head=head, root=root, snapshot=snapshot, schema=schema, train=train, dev=dev, cfg=cfg, started=started, cache_dir=Path(cache_dir).resolve())
    selection: dict[str, Any] = {}
    for head in HEADS:
        selection[head] = {"current_source": "LERT seed-35 checkpoint" if head == "emotion_primary" else "Classical frozen estimator", "current": baseline_current[head], "soft_label": results[head]["metrics"]["dev"], "gate": selection_gate(results[head]["metrics"]["dev"], baseline_current[head])}
    _dump(root / "dev-selection.json", selection)
    human_rows = _read_human_workbook(Path(workbook).resolve())
    human_input_records = [{"sample_id": row["sample_id"], "stock_code": row.get("stock_code", ""), "stock_name": row.get("stock_name", ""), "model_text": row["model_text"]} for row in human_rows]
    human_classical = _classical_probabilities(human_input_records, schema, prep, classical_path)
    selected_candidate: dict[str, np.ndarray] = {}
    for head in HEADS:
        gate = selection[head]["gate"]
        if gate["passed"]:
            checkpoint = _load_weights(Path(results[head]["checkpoint"]), torch)
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
            model = _model_for_checkpoint(torch, AutoModel, snapshot, schema, cfg, head, checkpoint, torch.device("cpu"))
            selected_candidate[head] = _predict(torch, model, tokenizer, [m1.M1Record(sample_id=row["sample_id"], stock_code=row.get("stock_code", ""), stock_name=row.get("stock_name", ""), model_text=row["model_text"], label={h: schema.class_order[h][0] for h in schema.class_order} | {"reasoning_tags": [schema.class_order["reasoning_tags"][0]]}, weights={h: 1.0 for h in schema.class_order}) for row in human_rows], cfg, head, torch.device("cpu"))
        elif head == "emotion_primary":
            selected_candidate[head] = _predict(torch, baseline_model, baseline_tokenizer, [m1.M1Record(sample_id=row["sample_id"], stock_code=row.get("stock_code", ""), stock_name=row.get("stock_name", ""), model_text=row["model_text"], label={h: schema.class_order[h][0] for h in schema.class_order} | {"reasoning_tags": [schema.class_order["reasoning_tags"][0]]}, weights={h: 1.0 for h in schema.class_order}) for row in human_rows], cfg, head, torch.device("cpu"))
        else:
            selected_candidate[head] = human_classical[head].copy()
    human_metrics = _human_metrics(human_rows, selected_candidate, human_classical, schema, root)
    _dump(root / "human100-audit.json", {"workbook_sha256": sha256_file(Path(workbook).resolve()), "rows": len(human_rows), "mapping": {"human_context_dependency": "primary_reasoning_tag_not_scored_as_context", "human_reasoning_tags": "ignored_empty", "true_context_dependency": "SELF_CONTAINED_default_not_scored"}, "selection_frozen_before_open": True, "forbidden_inputs_not_read": ["Test", "M3-A_Test_predictions", "Anchor", "Gold/OOD", "reference_predictions"]})
    final_status = "VNext_PROMISING" if any(item["gate"]["passed"] for item in selection.values()) else "NO_GAIN_KEEP_CURRENT"
    provisional = {"stage_id": STAGE_ID, "status": final_status, "selected_heads": [head for head in HEADS if selection[head]["gate"]["passed"]], "retained_current_heads": [head for head in HEADS if not selection[head]["gate"]["passed"]], "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE}, "seed": SEED, "cache_snapshot": snapshot_identity, "candidate_artifact_content_address": json.loads((CANDIDATE_ARTIFACT / "content-addressed-manifest.json").read_text(encoding="utf-8"))["content_address"], "runtime_identity": runtime_identity, "selection": selection, "human100_metrics_path": "human100-metrics.json", "training_performed": True, "model_downloaded": False, "dependency_install_performed": False, "forbidden_inputs_not_read": ["Test", "Anchor", "Gold", "OOD", "LLM", "cloud", "production"]}
    report = _report(selection, human_metrics, root, final_status)
    (root / "report.md").write_text(report, encoding="utf-8")
    manifest = _content_manifest(root, provisional)
    return {"status": final_status, "output_dir": str(root), "content_address": manifest["content_address"], "selection": selection, "human100": human_metrics, "runtime": runtime_identity, "snapshot": snapshot_identity}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=M3B_WORKBOOK)
    args = parser.parse_args(argv)
    result = run(args.config, args.cache_dir, args.output_dir, args.workbook)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
