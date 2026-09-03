#!/usr/bin/env python3
"""Evaluate the frozen M2 head-selected candidate on the owner-reviewed Gold100.

This script deliberately reads only the supplied human-review workbook and the
frozen candidate/classical artifacts.  It does not load any project split,
reference prediction, Test, Anchor, Gold, or OOD data package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_lert_final as lert
from semantic_model.hashes import content_addressed_id, sha256_file
from semantic_model.preprocessing import PreprocessingContract, build_model_inputs
from semantic_model.schema import LabelSchema, SINGLE_LABEL_HEADS, V1_HEADS


REPO = Path(__file__).resolve().parents[1]
WORKBOOK = REPO.parent / "human-review-artifacts/m3b-gold100-20260902/m3b-gold100-review-pack.xlsx"
CANDIDATE = REPO.parent / "model-artifacts/m2-head-selected-unified-candidate-20260902"
OUTPUT = REPO.parent / "model-artifacts/m3b-gold100-human-evaluation-20260903"
MODEL_ID = "hfl/chinese-lert-small"
MODEL_REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LERT_HEADS = ("target_mode", "stance", "emotion_primary", "emotion_target", "reasoning_tags")
GOLD_SINGLE_HEADS = ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency")
REASONING_HEAD = "reasoning_tags"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260903

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

SIMILARITY_RELATIONS: dict[tuple[str, str], float] = {
    ("BUY", "ADD"): 0.8,
    ("SELL", "REDUCE"): 0.8,
    ("HOLD", "WATCH"): 0.5,
    ("FEAR", "ANXIETY"): 0.7,
    ("ANGER", "FRUSTRATION"): 0.7,
    ("EXCITEMENT", "FOMO"): 0.7,
    ("HOPE", "EXCITEMENT"): 0.5,
    ("BULL", "MIXED"): 0.6,
    ("BEAR", "MIXED"): 0.6,
    ("FUNDAMENTAL", "VALUATION"): 0.7,
    ("RUMOR", "UNKNOWN"): 0.5,
}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SIMILARITY_RELATIONS.get((left, right), SIMILARITY_RELATIONS.get((right, left), 0.0))


def _column_letters(cell_ref: str) -> str:
    return "".join(character for character in cell_ref if character.isalpha())


def read_gold_workbook(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    """Read sheet2 XML directly so shared-string index 20 remains an empty cell."""

    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared: list[str] = ["".join(text.text or "" for text in item.findall(".//a:t", NS)) for item in shared_root.findall("a:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", NS):
            row_number = int(row.attrib["r"])
            values: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                value_node = cell.find("a:v", NS)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    index = int(value_node.text or "0")
                    # Index 20 is a genuine empty shared string in this workbook.
                    value = shared[index] if 0 <= index < len(shared) else ""
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//a:t", NS))
                else:
                    value = value_node.text or ""
                values[_column_letters(cell.attrib["r"])] = value
            if row_number >= 3:
                rows.append(values)
    if not rows or rows[0].get("A") != "sample_id":
        raise ValueError("M3B_WORKBOOK_HEADER_INVALID")
    headers = [value for _, value in sorted(rows[0].items(), key=lambda item: (len(item[0]), item[0]))]
    records = [{headers[index]: values.get(column, "") for index, column in enumerate(sorted(rows[0], key=lambda col: (len(col), col)))} for values in rows[1:]]
    if len(records) != 100:
        raise ValueError(f"M3B_GOLD_ROW_COUNT_INVALID:{len(records)}")
    required = {"sample_id", "stock_code", "stock_name", "model_text", *(f"human_{head}" for head in GOLD_SINGLE_HEADS), "human_context_dependency", "human_reasoning_tags"}
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(f"M3B_WORKBOOK_COLUMNS_MISSING:{missing}")
    if any(not record["sample_id"].strip() or not record["model_text"].strip() for record in records):
        raise ValueError("M3B_WORKBOOK_ID_OR_TEXT_EMPTY")
    if len({record["sample_id"] for record in records}) != 100:
        raise ValueError("M3B_WORKBOOK_DUPLICATE_SAMPLE_ID")
    # The owner mapping says the reasoning-tags column is intentionally ignored.
    if any(record["human_reasoning_tags"].strip() for record in records):
        raise ValueError("M3B_WORKBOOK_REASONING_TAGS_MUST_BE_EMPTY")
    audit = {
        "workbook_sha256": sha256_file(path),
        "sheet": "Adjudication_100",
        "rows": 100,
        "human_mapping": {
            "single_label_heads": list(GOLD_SINGLE_HEADS),
            "human_context_dependency_as_primary_reasoning_tag": True,
            "human_reasoning_tags": "ignored_and_empty",
            "true_context_dependency": "SELF_CONTAINED_default_not_scored_per_row",
        },
        "empty_shared_string_index_20": "normalized_to_empty",
        "forbidden_sources_not_read": ["Test_467", "M3-A_Test_predictions", "Anchor", "Gold/OOD packages", "reference_predictions"],
    }
    return headers, records, audit


def _load_checkpoint(path: Path, torch: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Candidate checkpoints are trusted project artifacts; only add the
        # harmless TorchVersion class needed by weights_only deserialization.
        version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
        if version_type is None:
            raise
        torch.serialization.add_safe_globals([version_type])
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_lert(rows: Sequence[Mapping[str, str]], schema: LabelSchema, cfg: Mapping[str, Any], snapshot: Path, torch: Any, AutoModel: Any, AutoTokenizer: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    model = lert._make_model(torch, AutoModel, snapshot, schema, float(cfg["head_dropout"]))
    for head in LERT_HEADS:
        checkpoint_path = CANDIDATE / "checkpoints" / head / "seed-35" / "specialist-checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _load_checkpoint(checkpoint_path, torch)
        if checkpoint.get("seed") != 35 or checkpoint.get("head") != head:
            raise ValueError(f"M3B_CANDIDATE_CHECKPOINT_IDENTITY_INVALID:{head}")
        model.specialist_blocks[head].load_state_dict(checkpoint["specialist_block_state_dict"])
        model.heads[head].load_state_dict(checkpoint["head_state_dict"])
    model = model.to(torch.device("cpu"))
    model.eval()
    labels: dict[str, list[str]] = {head: list(schema.class_order[head]) for head in V1_HEADS}
    # Labels/weights are required by the shared batch builder but never used
    # for inference. Human rows remain untouched in the comparison output.
    dummy_label = {head: schema.class_order[head][0] for head in V1_HEADS}
    dummy_label[REASONING_HEAD] = [schema.class_order[REASONING_HEAD][0]]
    dummy_weights = {head: 1.0 for head in V1_HEADS}
    records = [m1.M1Record(sample_id=str(row["sample_id"]), stock_code=str(row.get("stock_code", "")), stock_name=str(row.get("stock_name", "")), model_text=str(row["model_text"]), label=dummy_label, weights=dummy_weights) for row in rows]
    scalar: dict[str, list[np.ndarray]] = {head: [] for head in LERT_HEADS if head != REASONING_HEAD}
    reason: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(records), 16):
            batch = m1._as_batch(torch, tokenizer, records[offset:offset + 16], cfg, torch.device("cpu"))
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for head in scalar:
                values = torch.softmax(outputs[head], dim=-1)
                if not bool(torch.isfinite(values).all().item()):
                    raise ValueError(f"M3B_NONFINITE_LERT_PROBABILITY:{head}")
                scalar[head].append(values.detach().cpu().numpy())
            values = torch.sigmoid(outputs[REASONING_HEAD])
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError("M3B_NONFINITE_LERT_REASONING_PROBABILITY")
            reason.append(values.detach().cpu().numpy())
    return {head: np.concatenate(values, axis=0) for head, values in scalar.items()}, {REASONING_HEAD: np.concatenate(reason, axis=0)}, {"device": "cpu", "model_id": MODEL_ID, "revision": MODEL_REVISION, "heads_loaded": list(LERT_HEADS), "checkpoint_seed": 35}


def _compatibility_fix(legacy: Mapping[str, Any]) -> None:
    for estimator in [*legacy.get("scalar_models", {}).values(), *legacy.get("reason_models", {}).values()]:
        if not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"


def _reorder_probabilities(estimator: Any, probabilities: np.ndarray, order: Sequence[str]) -> np.ndarray:
    classes = [str(value) for value in estimator.classes_]
    result = np.zeros((probabilities.shape[0], len(order)), dtype=float)
    for source_index, label in enumerate(classes):
        if label in order:
            result[:, order.index(label)] = probabilities[:, source_index]
    return result


def _load_classical(rows: Sequence[Mapping[str, str]], schema: LabelSchema, prep: PreprocessingContract) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    model_path = CANDIDATE / "classical-fallback" / "original_model_v0.3.5.joblib"
    legacy = joblib.load(model_path)
    _compatibility_fix(legacy)
    records = [{"sample_id": str(row["sample_id"]), "stock_code": str(row.get("stock_code", "")), "stock_name": str(row.get("stock_name", "")), "model_text": str(row["model_text"])} for row in rows]
    texts = build_model_inputs(records, prep)
    matrix = hstack([legacy["char_vectorizer"].transform(texts), legacy["word_vectorizer"].transform(texts)], format="csr")
    scalar: dict[str, np.ndarray] = {}
    for head in GOLD_SINGLE_HEADS:
        estimator = legacy["scalar_models"][head]
        scalar[head] = _reorder_probabilities(estimator, estimator.predict_proba(matrix), schema.class_order[head])
    reason = np.zeros((len(rows), len(schema.class_order[REASONING_HEAD])), dtype=float)
    for index, tag in enumerate(schema.class_order[REASONING_HEAD]):
        estimator = legacy["reason_models"][tag]
        probabilities = estimator.predict_proba(matrix)
        classes = [int(value) for value in estimator.classes_]
        if 1 in classes:
            reason[:, index] = probabilities[:, classes.index(1)]
    if not np.isfinite(reason).all() or not all(np.isfinite(values).all() for values in scalar.values()):
        raise ValueError("M3B_NONFINITE_CLASSICAL_PROBABILITY")
    return scalar, {REASONING_HEAD: reason}, {"device": "cpu", "model_path": str(model_path), "model_sha256": sha256_file(model_path), "estimator_type": "legacy_v0.3.5_dict", "input_contract": "frozen_preprocessing_contract"}


def _predicted_reasoning(probabilities: np.ndarray, order: Sequence[str], thresholds: Mapping[str, float], *, ensure_at_least_one: bool) -> tuple[list[str], list[int]]:
    selected = [tag for index, tag in enumerate(order) if float(probabilities[index]) >= float(thresholds[tag])]
    if not selected and ensure_at_least_one:
        selected = [order[int(np.argmax(probabilities))]]
    ranked = sorted(range(len(order)), key=lambda index: (-float(probabilities[index]), index))
    return selected, ranked


def _single_metrics(truth: Sequence[str], probabilities: np.ndarray, order: Sequence[str]) -> dict[str, Any]:
    predicted = [order[int(index)] for index in np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=order, zero_division=0)
    confidence = np.max(probabilities, axis=1)
    true_prob = np.asarray([probabilities[row, order.index(value)] for row, value in enumerate(truth)], dtype=float)
    return {"accuracy": float(np.mean(np.asarray(predicted) == np.asarray(truth))), "macro_f1": float(f1_score(truth, predicted, labels=order, average="macro", zero_division=0)), "per_class_f1": {label: float(f1[index]) for index, label in enumerate(order)}, "per_class": {label: {"f1": float(f1[index]), "precision": float(precision[index]), "recall": float(recall[index]), "support": int(support[index])} for index, label in enumerate(order)}, "confusion_matrix": confusion_matrix(truth, predicted, labels=order).tolist(), "class_order": list(order), "raw_probability": {"top1_confidence_mean": float(confidence.mean()), "true_label_probability_mean": float(true_prob.mean())}}


def _bootstrap(values: Sequence[Mapping[str, Any]], head: str, candidate_correct: np.ndarray, classical_correct: np.ndarray, candidate_macro: np.ndarray | None = None, classical_macro: np.ndarray | None = None, candidate_aux: np.ndarray | None = None, classical_aux: np.ndarray | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    deltas = (candidate_correct[indices].mean(axis=1) - classical_correct[indices].mean(axis=1)).astype(float)
    result: dict[str, Any] = {"accuracy_delta": {"estimate": float(candidate_correct.mean() - classical_correct.mean()), "ci95_percentile": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]}}
    if candidate_macro is not None and classical_macro is not None:
        macro_deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for index, sample in enumerate(indices):
            values_truth = [values[i][f"human_{head}"] for i in sample]
            values_order = values[0][f"order_{head}"]
            candidate_sample = [values[i][f"candidate_{head}"] for i in sample]
            classical_sample = [values[i][f"classical_{head}"] for i in sample]
            macro_deltas[index] = f1_score(values_truth, candidate_sample, labels=values_order, average="macro", zero_division=0) - f1_score(values_truth, classical_sample, labels=values_order, average="macro", zero_division=0)
        result["macro_f1_delta"] = {"estimate": float(candidate_macro - classical_macro), "ci95_percentile": [float(np.quantile(macro_deltas, 0.025)), float(np.quantile(macro_deltas, 0.975))]}
    if candidate_aux is not None and classical_aux is not None:
        aux_deltas = candidate_aux[indices].mean(axis=1) - classical_aux[indices].mean(axis=1)
        result["auxiliary_delta"] = {"estimate": float(candidate_aux.mean() - classical_aux.mean()), "ci95_percentile": [float(np.quantile(aux_deltas, 0.025)), float(np.quantile(aux_deltas, 0.975))]}
    result["method"] = {"type": "paired_row_bootstrap", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "interval": "percentile_95"}
    return result


def evaluate(workbook: Path = WORKBOOK, candidate: Path = CANDIDATE, output: Path = OUTPUT) -> dict[str, Any]:
    global CANDIDATE
    if output.exists():
        raise FileExistsError(f"M3B_OUTPUT_ALREADY_EXISTS:{output}")
    CANDIDATE = candidate.resolve()
    headers, rows, workbook_audit = read_gold_workbook(workbook.resolve())
    schema = LabelSchema.load(REPO / "schema/semantic-schema-calibrated-v0.2.1.json")
    config_obj = m1.ProjectConfig.load(REPO / "configs/baseline_v0.3.5.yaml")
    prep = PreprocessingContract.load(config_obj.data_path("preprocessing_contract"))
    frozen_contract = lert._contract_requirements(REPO / "manifests/encoder-m2-experiment-contract-v1.json")
    config = lert._config(frozen_contract["contract"], schema.class_order)
    snapshot = REPO / ".encoder-artifacts/hf-cache-lert-small/official-snapshot/69e3e69ba258be5b301b26937e5b55a076c90460"
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    import torch
    from transformers import AutoModel, AutoTokenizer

    lert_scalar, lert_reason, lert_runtime = _load_lert(rows, schema, config, snapshot, torch, AutoModel, AutoTokenizer)
    classical_scalar, classical_reason, classical_runtime = _load_classical(rows, schema, prep)
    order = {head: list(schema.class_order[head]) for head in V1_HEADS}
    human_scalar = {head: [row[f"human_{head}"] for row in rows] for head in GOLD_SINGLE_HEADS}
    human_reason = [row["human_context_dependency"] for row in rows]
    thresholds_path = candidate / "thresholds/reasoning_tags/seed-35/reasoning-thresholds.json"
    threshold_data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    candidate_thresholds = {str(key): float(value) for key, value in threshold_data["thresholds"].items()}
    candidate_reason_order = order[REASONING_HEAD]
    classical_reason_order = order[REASONING_HEAD]
    legacy = joblib.load(candidate / "classical-fallback/original_model_v0.3.5.joblib")
    _compatibility_fix(legacy)
    classical_thresholds = {tag: float(legacy["reason_thresholds"][tag]) for tag in classical_reason_order}

    candidate_predictions: dict[str, list[str]] = {}
    classical_predictions: dict[str, list[str]] = {}
    comparison: list[dict[str, Any]] = []
    candidate_reason_sets: list[list[str]] = []
    classical_reason_sets: list[list[str]] = []
    candidate_reason_rank: list[int] = []
    classical_reason_rank: list[int] = []
    candidate_reason_top1_hit: list[bool] = []
    classical_reason_top1_hit: list[bool] = []
    candidate_reason_top3_hit: list[bool] = []
    classical_reason_top3_hit: list[bool] = []
    candidate_reason_contains: list[bool] = []
    classical_reason_contains: list[bool] = []
    candidate_reason_human_prob: list[float] = []
    classical_reason_human_prob: list[float] = []
    candidate_reason_semantic: list[float] = []
    classical_reason_semantic: list[float] = []
    candidate_reason_top3_semantic: list[float] = []
    classical_reason_top3_semantic: list[float] = []
    for index, row in enumerate(rows):
        entry: dict[str, Any] = {"row": index + 1, "sample_id": row["sample_id"], "human": {head: row[f"human_{head}"] for head in GOLD_SINGLE_HEADS} | {"primary_reasoning_tag": human_reason[index]}, "candidate": {}, "classical": {}}
        for head in GOLD_SINGLE_HEADS:
            c_prob = (lert_scalar[head] if head in lert_scalar else classical_scalar[head])[index] if head in LERT_HEADS else classical_scalar[head][index]
            k_prob = classical_scalar[head][index]
            c_pred = order[head][int(np.argmax(c_prob))]
            k_pred = order[head][int(np.argmax(k_prob))]
            candidate_predictions.setdefault(head, []).append(c_pred)
            classical_predictions.setdefault(head, []).append(k_pred)
            entry["candidate"][head] = {"prediction": c_pred, "top1_probability": float(np.max(c_prob)), "human_label_probability": float(c_prob[order[head].index(row[f"human_{head}"])])}
            entry["classical"][head] = {"prediction": k_pred, "top1_probability": float(np.max(k_prob)), "human_label_probability": float(k_prob[order[head].index(row[f"human_{head}"])])}
            entry.setdefault("paired", {})[head] = {"candidate_correct": c_pred == row[f"human_{head}"], "classical_correct": k_pred == row[f"human_{head}"], "correctness_delta": int(c_pred == row[f"human_{head}"]) - int(k_pred == row[f"human_{head}"])}
        c_reason = np.array([lert_reason[REASONING_HEAD][index, candidate_reason_order.index(tag)] if tag != "SARCASM_IRONY" else classical_reason[REASONING_HEAD][index, candidate_reason_order.index(tag)] for tag in candidate_reason_order])
        k_reason = classical_reason[REASONING_HEAD][index]
        c_set, c_rank = _predicted_reasoning(c_reason, candidate_reason_order, candidate_thresholds, ensure_at_least_one=False)
        k_set, k_rank = _predicted_reasoning(k_reason, classical_reason_order, classical_thresholds, ensure_at_least_one=True)
        human_tag = human_reason[index]
        c_top3 = [candidate_reason_order[i] for i in c_rank[:3]]
        k_top3 = [classical_reason_order[i] for i in k_rank[:3]]
        c_top1 = candidate_reason_order[c_rank[0]]
        k_top1 = classical_reason_order[k_rank[0]]
        candidate_reason_sets.append(c_set); classical_reason_sets.append(k_set)
        candidate_reason_rank.append(c_rank.index(candidate_reason_order.index(human_tag)) + 1); classical_reason_rank.append(k_rank.index(classical_reason_order.index(human_tag)) + 1)
        candidate_reason_top1_hit.append(c_top1 == human_tag); classical_reason_top1_hit.append(k_top1 == human_tag)
        candidate_reason_top3_hit.append(human_tag in c_top3); classical_reason_top3_hit.append(human_tag in k_top3)
        candidate_reason_contains.append(human_tag in c_set); classical_reason_contains.append(human_tag in k_set)
        candidate_reason_human_prob.append(float(c_reason[candidate_reason_order.index(human_tag)])); classical_reason_human_prob.append(float(k_reason[classical_reason_order.index(human_tag)]))
        c_sem = _similarity(c_top1, human_tag); k_sem = _similarity(k_top1, human_tag)
        c_top3_sem = max((_similarity(tag, human_tag) for tag in c_top3), default=0.0); k_top3_sem = max((_similarity(tag, human_tag) for tag in k_top3), default=0.0)
        candidate_reason_semantic.append(c_sem); classical_reason_semantic.append(k_sem); candidate_reason_top3_semantic.append(c_top3_sem); classical_reason_top3_semantic.append(k_top3_sem)
        entry["candidate"][REASONING_HEAD] = {"top1": c_top1, "top3": c_top3, "human_label_probability": candidate_reason_human_prob[-1], "human_label_rank": candidate_reason_rank[-1], "prediction_set": c_set, "prediction_set_contains_human": candidate_reason_contains[-1], "top1_semantic_similarity": c_sem, "top3_max_semantic_similarity": c_top3_sem}
        entry["classical"][REASONING_HEAD] = {"top1": k_top1, "top3": k_top3, "human_label_probability": classical_reason_human_prob[-1], "human_label_rank": classical_reason_rank[-1], "prediction_set": k_set, "prediction_set_contains_human": classical_reason_contains[-1], "top1_semantic_similarity": k_sem, "top3_max_semantic_similarity": k_top3_sem}
        comparison.append(entry)

    metrics: dict[str, Any] = {"schema_version": "myresearcher.m3b-gold100-evaluation.v1", "rows": 100, "evaluation_scope": "owner_reviewed_Gold100_only", "single_label_heads": list(GOLD_SINGLE_HEADS), "context_dependency_scored": False, "reasoning_scope": "primary_tag_only_no_full_set_exact_no_unselected_negatives", "semantic_similarity": {"same_class": 1.0, "undeclared": 0.0, "relations": [{"left": left, "right": right, "similarity": value} for (left, right), value in SIMILARITY_RELATIONS.items()]}, "heads": {}}
    for head in GOLD_SINGLE_HEADS:
        truth = human_scalar[head]
        c_prob = lert_scalar[head] if head in lert_scalar else classical_scalar[head]
        k_prob = classical_scalar[head]
        c_metrics = _single_metrics(truth, c_prob, order[head]); k_metrics = _single_metrics(truth, k_prob, order[head])
        c_correct = np.asarray(candidate_predictions[head]) == np.asarray(truth); k_correct = np.asarray(classical_predictions[head]) == np.asarray(truth)
        paired_counts = {"candidate_only_correct": int(np.sum(c_correct & ~k_correct)), "classical_only_correct": int(np.sum(~c_correct & k_correct)), "both_correct": int(np.sum(c_correct & k_correct)), "both_wrong": int(np.sum(~c_correct & ~k_correct))}
        rows_for_bootstrap = [{f"human_{head}": truth[i], f"candidate_{head}": candidate_predictions[head][i], f"classical_{head}": classical_predictions[head][i], f"order_{head}": list(order[head])} for i in range(100)]
        bootstrap = _bootstrap(rows_for_bootstrap, head, c_correct.astype(float), k_correct.astype(float), c_metrics["macro_f1"], k_metrics["macro_f1"])
        metrics["heads"][head] = {"candidate": c_metrics, "classical": k_metrics, "candidate_minus_classical": {"accuracy": c_metrics["accuracy"] - k_metrics["accuracy"], "macro_f1": c_metrics["macro_f1"] - k_metrics["macro_f1"]}, "paired_row_outcomes": paired_counts, "paired_bootstrap": bootstrap}
    reason_metrics = {"candidate": {"top1_accuracy": float(np.mean(candidate_reason_top1_hit)), "top3_accuracy": float(np.mean(candidate_reason_top3_hit)), "prediction_set_contains_human_rate": float(np.mean(candidate_reason_contains)), "human_label_probability_mean": float(np.mean(candidate_reason_human_prob)), "human_label_rank_mean": float(np.mean(candidate_reason_rank)), "semantic_similarity_top1_mean": float(np.mean(candidate_reason_semantic)), "semantic_similarity_top3_max_mean": float(np.mean(candidate_reason_top3_semantic))}, "classical": {"top1_accuracy": float(np.mean(classical_reason_top1_hit)), "top3_accuracy": float(np.mean(classical_reason_top3_hit)), "prediction_set_contains_human_rate": float(np.mean(classical_reason_contains)), "human_label_probability_mean": float(np.mean(classical_reason_human_prob)), "human_label_rank_mean": float(np.mean(classical_reason_rank)), "semantic_similarity_top1_mean": float(np.mean(classical_reason_semantic)), "semantic_similarity_top3_max_mean": float(np.mean(classical_reason_top3_semantic))}, "candidate_minus_classical": {"top1_accuracy": float(np.mean(candidate_reason_top1_hit) - np.mean(classical_reason_top1_hit)), "top3_accuracy": float(np.mean(candidate_reason_top3_hit) - np.mean(classical_reason_top3_hit)), "prediction_set_contains_human_rate": float(np.mean(candidate_reason_contains) - np.mean(classical_reason_contains)), "human_label_probability_mean": float(np.mean(candidate_reason_human_prob) - np.mean(classical_reason_human_prob)), "semantic_similarity_top1_mean": float(np.mean(candidate_reason_semantic) - np.mean(classical_reason_semantic)), "semantic_similarity_top3_max_mean": float(np.mean(candidate_reason_top3_semantic) - np.mean(classical_reason_top3_semantic))}, "raw_probability": {"candidate_human_label_probability_mean": float(np.mean(candidate_reason_human_prob)), "classical_human_label_probability_mean": float(np.mean(classical_reason_human_prob))}, "bootstrap": _bootstrap([{"human_reasoning_tags": human_reason[i], "candidate_reasoning_tags": candidate_reason_order[candidate_reason_rank[i] - 1], "classical_reasoning_tags": classical_reason_order[classical_reason_rank[i] - 1], "order_reasoning_tags": candidate_reason_order} for i in range(100)], REASONING_HEAD, np.asarray(candidate_reason_top1_hit, dtype=float), np.asarray(classical_reason_top1_hit, dtype=float), candidate_aux=np.asarray(candidate_reason_human_prob), classical_aux=np.asarray(classical_reason_human_prob))}
    metrics["heads"][REASONING_HEAD] = reason_metrics

    provenance = {"evaluation_id": "M3-B-GOLD100-HUMAN-EVALUATION-20260903", "workbook": workbook_audit, "candidate": {"path": str(candidate.resolve()), "content_address": json.loads((candidate / "content-addressed-manifest.json").read_text(encoding="utf-8")).get("content_address"), "model_identity": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0"}, "selected_seed": 35, "thresholds_sha256": sha256_file(thresholds_path), "thresholds": candidate_thresholds}, "classical": {**classical_runtime, "candidate_copy_used": True}, "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": __import__("transformers").__version__, "numpy": np.__version__}, "source_files": {"schema_sha256": sha256_file(REPO / "schema/semantic-schema-calibrated-v0.2.1.json"), "contract_sha256": sha256_file(REPO / "manifests/encoder-m2-experiment-contract-v1.json"), "preprocessing_contract_sha256": sha256_file(config_obj.data_path("preprocessing_contract")), "candidate_manifest_sha256": sha256_file(candidate / "content-addressed-manifest.json")}, "data_access_policy": {"test_read": False, "m3a_predictions_read": False, "anchor_read": False, "gold_ood_package_read": False, "reference_predictions_read": False, "training": False, "threshold_tuning": False}}
    output.mkdir(parents=True, exist_ok=False)
    _dump(output / "metrics.json", metrics)
    with (output / "comparison.jsonl").open("x", encoding="utf-8") as handle:
        for item in comparison:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    _dump(output / "workbook-audit.json", workbook_audit)
    _dump(output / "provenance.json", provenance)
    report = render_report(metrics, provenance, output)
    (output / "report.md").write_text(report, encoding="utf-8")
    files = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "content-addressed-manifest.json"]
    manifest = {"manifest_schema_version": "myresearcher.m3b-gold100-content-addressed-manifest.v1", "evaluation_id": provenance["evaluation_id"], "scope": "human_reviewed_Gold100_only", "candidate_content_address": provenance["candidate"]["content_address"], "workbook_sha256": workbook_audit["workbook_sha256"], "files": files, "forbidden_sources_not_read": workbook_audit["forbidden_sources_not_read"]}
    manifest["content_address"] = content_addressed_id(manifest)
    _dump(output / "content-addressed-manifest.json", manifest)
    return {"output": str(output), "content_address": manifest["content_address"], "metrics": metrics, "provenance": provenance}


def render_report(metrics: Mapping[str, Any], provenance: Mapping[str, Any], output: Path) -> str:
    lines = ["# M3-B Gold100 human evaluation", "", "Scope: one owner-reviewed 100-row workbook; this is trend evidence, not production acceptance.", "", f"- Artifact: `{output}`", f"- Candidate M2 content address: `{provenance['candidate']['content_address']}`", f"- Workbook SHA-256: `{provenance['workbook']['workbook_sha256']}`", "- Human mapping: five scalar labels as entered; `human_context_dependency` is the primary reasoning tag; `human_reasoning_tags` ignored; no context Macro-F1.", "- Forbidden sources read: none (Test/M3-A/Anchor/reference predictions/Gold/OOD not opened).", "", "## Single-label heads", "", "| Head | Candidate Acc / Macro-F1 | Classical Acc / Macro-F1 | Δ Acc | Paired outcomes (C-only / K-only / both wrong) |", "|---|---:|---:|---:|---:|"]
    for head in GOLD_SINGLE_HEADS:
        item = metrics["heads"][head]; c, k = item["candidate"], item["classical"]; p = item["paired_row_outcomes"]
        lines.append(f"| {head} | {c['accuracy']:.3f} / {c['macro_f1']:.3f} | {k['accuracy']:.3f} / {k['macro_f1']:.3f} | {item['candidate_minus_classical']['accuracy']:+.3f} | {p['candidate_only_correct']} / {p['classical_only_correct']} / {p['both_wrong']} |")
    r = metrics["heads"][REASONING_HEAD]
    lines.extend(["", "## Primary reasoning tag", "", "Only Top-1/Top-3, human-label probability/rank, prediction-set inclusion, raw probabilities, and semantic similarity are reported. Full multi-label exact-set accuracy and unselected negatives are intentionally not computed.", "", "| Model | Top-1 | Top-3 | Set contains human | Human-label probability | Mean rank | Top-1 semantic similarity |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name in ("candidate", "classical"):
        item = r[name]; lines.append(f"| {name.title()} | {item['top1_accuracy']:.3f} | {item['top3_accuracy']:.3f} | {item['prediction_set_contains_human_rate']:.3f} | {item['human_label_probability_mean']:.3f} | {item['human_label_rank_mean']:.2f} | {item['semantic_similarity_top1_mean']:.3f} |")
    lines.extend(["", "## Trend interpretation", "", "Head-level improvement/tie/regression is determined from the strict exact-label Candidate−Classical Macro-F1 and accuracy deltas; semantic similarity is auxiliary only. Paired bootstrap intervals (10,000 resamples, seed 20260903, percentile 95%) are in `metrics.json`.", "", "This 100-row evaluation does not tune the model, change the frozen head mapping, or claim production readiness.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    result = evaluate(args.workbook, args.candidate, args.output)
    print(json.dumps({"status": "M3B_GOLD100_EVALUATION_COMPLETE", "output": result["output"], "content_address": result["content_address"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
