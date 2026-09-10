#!/usr/bin/env python3
"""One-shot seven-head additive feedback adapter.

This diagnostic consumes only the frozen Phase-A prediction snapshot and the
final answers from the 10-row Human Train batch.  It performs one deterministic
convex optimization over per-class additive biases on top of existing
probabilities; no Encoder, tokenizer, checkpoint, Test, or review artifact is
loaded.  Human Check is descriptive only, and the original 300-row snapshot is
used solely to measure drift.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
DEFAULT_LABELS = RUN_DIR / "human-labels-revisable.jsonl"
DEFAULT_SNAPSHOT = RUN_DIR / "prediction-snapshot.jsonl"
DEFAULT_SELECTION_TRAIN = RUN_DIR / "selection-train.jsonl"
DEFAULT_SELECTION_CHECK = RUN_DIR / "selection-check.jsonl"
DEFAULT_OUTPUT = RUN_DIR / "feedback-training-human10"
L2 = 1.0
EPS = 1e-7
HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "context_dependency",
    "reasoning_tags",
)
SCALAR_HEADS = HEADS[:-1]
CLASS_ORDER = {
    "target_mode": ["ON_TARGET", "CROSS_TARGET", "MARKET_GENERAL", "UNKNOWN"],
    "stance": ["BULL", "BEAR", "NEUTRAL", "MIXED", "UNKNOWN"],
    "emotion_primary": ["FEAR", "ANXIETY", "ANGER", "FRUSTRATION", "REGRET", "HOPE", "EXCITEMENT", "FOMO", "CALM", "NONE_EXPLICIT", "UNKNOWN"],
    "emotion_target": ["PRICE", "POSITION", "COMPANY", "MARKET", "OTHER", "NOT_APPLICABLE", "UNKNOWN"],
    "action_tendency": ["BUY", "ADD", "HOLD", "DO_T", "REDUCE", "SELL", "WATCH", "NO_ACTION_SIGNAL", "UNKNOWN"],
    "context_dependency": ["SELF_CONTAINED", "PARTIAL_CONTEXT", "EXTERNAL_CONTEXT_REQUIRED", "UNKNOWN"],
    "reasoning_tags": ["FUNDAMENTAL", "VALUATION", "TECHNICAL_PRICE", "FLOW_POSITIONING", "NEWS_EVENT", "RUMOR", "SOCIAL_PROOF", "MACRO_POLICY", "THEME_NARRATIVE", "NO_REASON_GIVEN", "RELATIVE_PERFORMANCE", "CROSS_STOCK_REFERENCE", "SARCASM_IRONY", "WORDPLAY", "UNKNOWN"],
}


class FeedbackAdapterError(ValueError):
    pass


def _semantic_character_count(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text))


def _is_low_information_input(text: Any) -> bool:
    """Mirror the feedback selection filter without assuming a row count."""

    if not isinstance(text, str) or not text.strip():
        return True
    normalised = re.sub(r"\s+", "", text)
    if normalised in {"……", "...", "．．．", "又了", "无", "暂无", "略", "？", "?"}:
        return True
    return _semantic_character_count(text) == 0


def _check_population_info(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify Check rows from their text; no low-information count is fixed."""

    low_information = [
        str(row["sample_id"])
        for row in rows
        if _is_low_information_input(row.get("model_text", row.get("text", "")))
    ]
    return {
        "low_information_inputs": low_information,
        "semantic_subset_n": len(rows) - len(low_information),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FeedbackAdapterError(f"missing JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FeedbackAdapterError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise FeedbackAdapterError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "content_address"})).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scalar_vector(row: Mapping[str, Any], head: str) -> np.ndarray:
    try:
        values = [float(row["predictions"][head]["probabilities"][label]) for label in CLASS_ORDER[head]]
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedbackAdapterError(f"snapshot missing complete probabilities for {head}") from exc
    vector = np.asarray(values, dtype=float)
    if not np.isfinite(vector).all() or (vector < 0).any() or vector.sum() <= 0:
        raise FeedbackAdapterError(f"invalid probabilities for {head}")
    return vector / vector.sum()


def _reasoning_vector(row: Mapping[str, Any]) -> np.ndarray:
    try:
        values = [float(row["predictions"]["reasoning_tags"]["probabilities"][label]) for label in CLASS_ORDER["reasoning_tags"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedbackAdapterError("snapshot missing complete reasoning probabilities") from exc
    vector = np.asarray(values, dtype=float)
    if not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
        raise FeedbackAdapterError("invalid reasoning probabilities")
    return vector


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1.0 - EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _layout() -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for head in HEADS:
        width = len(CLASS_ORDER[head])
        result[head] = slice(offset, offset + width)
        offset += width
    return result


def _answers(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("head")))
        if key in result:
            raise FeedbackAdapterError(f"duplicate human answer: {key}")
        if key[0] in {"", "None"} or key[1] not in HEADS or not row.get("is_final"):
            raise FeedbackAdapterError(f"human answer is not final or has invalid identity: {key}")
        result[key] = row.get("answer")
    return result


def _targets(sample_id: str, answers: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for head in SCALAR_HEADS:
        value = answers.get((sample_id, head))
        if not isinstance(value, str) or value not in CLASS_ORDER[head]:
            raise FeedbackAdapterError(f"invalid {head} answer for {sample_id}: {value!r}")
        targets[head] = value
    value = answers.get((sample_id, "reasoning_tags"))
    if not isinstance(value, list) or not all(isinstance(tag, str) and tag in CLASS_ORDER["reasoning_tags"] for tag in value):
        raise FeedbackAdapterError(f"invalid reasoning_tags answer for {sample_id}: {value!r}")
    targets["reasoning_tags"] = sorted(set(value))
    return targets


def _objective_factory(train_rows: Sequence[Mapping[str, Any]], target_map: Mapping[str, Mapping[str, Any]]) -> Any:
    layout = _layout()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.5 * L2 * float(np.dot(theta, theta))
        gradient = L2 * theta.copy()
        for row in train_rows:
            sample_id = str(row["sample_id"])
            target = target_map[sample_id]
            for head in SCALAR_HEADS:
                probabilities = _scalar_vector(row, head)
                logits = np.log(np.clip(probabilities, EPS, 1.0)) + theta[layout[head]]
                adjusted = _softmax(logits)
                index = CLASS_ORDER[head].index(target[head])
                value -= math.log(max(float(adjusted[index]), EPS))
                gradient[layout[head]] += adjusted
                gradient[layout[head].start + index] -= 1.0
            probabilities = _reasoning_vector(row)
            adjusted = _sigmoid(_logit(probabilities) + theta[layout["reasoning_tags"]])
            labels = np.asarray([float(label in target["reasoning_tags"]) for label in CLASS_ORDER["reasoning_tags"]])
            value -= float(np.sum(labels * np.log(np.clip(adjusted, EPS, 1.0 - EPS)) + (1.0 - labels) * np.log(np.clip(1.0 - adjusted, EPS, 1.0))))
            gradient[layout["reasoning_tags"]] += adjusted - labels
        return value, gradient

    return objective


def fit_biases(train_rows: Sequence[Mapping[str, Any]], answers: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    if not train_rows:
        raise FeedbackAdapterError("Human Train is empty")
    target_map = {str(row["sample_id"]): _targets(str(row["sample_id"]), answers) for row in train_rows}
    if len(target_map) != len(train_rows):
        raise FeedbackAdapterError("Human Train contains duplicate sample IDs")
    layout = _layout()
    theta0 = np.zeros(max(item.stop for item in layout.values()), dtype=float)
    result = minimize(_objective_factory(train_rows, target_map), theta0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(result.x).all() or not result.success:
        raise FeedbackAdapterError(f"convex adapter optimization failed: {result.message}")
    return {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "single_convex_optimization": True,
        "l2": L2,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "converged": bool(result.success),
        "biases": {head: [float(value) for value in result.x[layout[head]]] for head in HEADS},
    }


def apply_adapter(row: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    biases = parameters["biases"]
    predictions = {}
    for head in SCALAR_HEADS:
        base = _scalar_vector(row, head)
        adjusted = _softmax(np.log(np.clip(base, EPS, 1.0)) + np.asarray(biases[head], dtype=float))
        predictions[head] = {"probabilities": {label: float(adjusted[index]) for index, label in enumerate(CLASS_ORDER[head])}, "prediction": CLASS_ORDER[head][int(np.argmax(adjusted))]}
    base_reasoning = _reasoning_vector(row)
    adjusted_reasoning = _sigmoid(_logit(base_reasoning) + np.asarray(biases["reasoning_tags"], dtype=float))
    original_reasoning = row["predictions"]["reasoning_tags"]
    thresholds = original_reasoning.get("thresholds", {}) if isinstance(original_reasoning, Mapping) else {}
    threshold_map = {label: float(thresholds.get(label, 0.5)) for label in CLASS_ORDER["reasoning_tags"]}
    predicted = [label for index, label in enumerate(CLASS_ORDER["reasoning_tags"]) if adjusted_reasoning[index] >= threshold_map[label]]
    predictions["reasoning_tags"] = {"probabilities": {label: float(adjusted_reasoning[index]) for index, label in enumerate(CLASS_ORDER["reasoning_tags"])}, "prediction": predicted, "predicted_labels": predicted, "thresholds": threshold_map}
    return {"sample_id": row["sample_id"], "model_text": row.get("model_text"), "predictions": predictions}


def _true_probability(prediction: Mapping[str, Any], head: str, target: Any) -> float:
    probs = prediction["predictions"][head]["probabilities"]
    if head == "reasoning_tags":
        values = [float(probs[label]) for label in target]
        return float(sum(values) / len(values)) if values else float(np.mean([1.0 - float(value) for value in probs.values()]))
    return float(probs[target])


def metric_summary(rows: Sequence[Mapping[str, Any]], answers: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in HEADS:
        nll_values: list[float] = []
        hits = 0
        target_probabilities: list[float] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            target = _targets(sample_id, answers)[head]
            prediction = row["predictions"][head]
            if head == "reasoning_tags":
                probs = np.asarray([float(prediction["probabilities"][label]) for label in CLASS_ORDER[head]])
                labels = np.asarray([float(label in target) for label in CLASS_ORDER[head]])
                nll_values.append(float(-np.sum(labels * np.log(np.clip(probs, EPS, 1.0 - EPS)) + (1.0 - labels) * np.log(np.clip(1.0 - probs, EPS, 1.0)))))
                hits += int(set(prediction.get("predicted_labels", prediction.get("prediction", []))) == set(target))
            else:
                probability = float(prediction["probabilities"][target])
                nll_values.append(-math.log(max(probability, EPS)))
                hits += int(prediction["prediction"] == target)
            target_probabilities.append(_true_probability(row, head, target))
        result[head] = {"n": len(rows), "loss": "BCE" if head == "reasoning_tags" else "NLL", "before_or_after_nll_or_bce": float(np.mean(nll_values)), "strict_hit_rate": hits / len(rows), "target_probability_mean": float(np.mean(target_probabilities)), "target_probability_min": float(np.min(target_probabilities)), "target_probability_max": float(np.max(target_probabilities))}
    return result


def _js_scalar(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(np.clip(left, EPS, 1.0) / np.clip(midpoint, EPS, 1.0))) + 0.5 * np.sum(right * np.log(np.clip(right, EPS, 1.0) / np.clip(midpoint, EPS, 1.0))))


def _js_bernoulli(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    midpoint = 0.5 * (left + right)
    return float(np.mean(0.5 * (left * np.log(np.clip(left, EPS, 1.0) / np.clip(midpoint, EPS, 1.0)) + (1.0 - left) * np.log(np.clip(1.0 - left, EPS, 1.0) / np.clip(1.0 - midpoint, EPS, 1.0))) + 0.5 * (right * np.log(np.clip(right, EPS, 1.0) / np.clip(midpoint, EPS, 1.0)) + (1.0 - right) * np.log(np.clip(1.0 - right, EPS, 1.0) / np.clip(1.0 - midpoint, EPS, 1.0)))))


def drift_summary(before_rows: Sequence[Mapping[str, Any]], after_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    after_by_id = {str(row["sample_id"]): row for row in after_rows}
    result: dict[str, Any] = {"rows": len(before_rows), "heads": {}}
    for head in HEADS:
        changed = 0
        high_confidence_flip = 0
        js_values: list[float] = []
        before_distribution: Counter[str] = Counter()
        after_distribution: Counter[str] = Counter()
        for row in before_rows:
            after = after_by_id[str(row["sample_id"])]
            before_output, after_output = row["predictions"][head], after["predictions"][head]
            before_prediction = before_output.get("predicted_labels", before_output.get("prediction")) if head == "reasoning_tags" else before_output.get("prediction")
            after_prediction = after_output.get("predicted_labels", after_output.get("prediction")) if head == "reasoning_tags" else after_output.get("prediction")
            before_key = tuple(before_prediction) if isinstance(before_prediction, list) else before_prediction
            after_key = tuple(after_prediction) if isinstance(after_prediction, list) else after_prediction
            changed += int(before_key != after_key)
            if head == "reasoning_tags":
                before_values = np.asarray([float(before_output["probabilities"][label]) for label in CLASS_ORDER[head]])
                after_values = np.asarray([float(after_output["probabilities"][label]) for label in CLASS_ORDER[head]])
                js_values.append(_js_bernoulli(before_values, after_values))
                before_distribution.update(before_prediction or [])
                after_distribution.update(after_prediction or [])
                before_conf = float(np.mean(np.maximum(before_values, 1.0 - before_values)))
                after_conf = float(np.mean(np.maximum(after_values, 1.0 - after_values)))
            else:
                before_values = np.asarray([float(before_output["probabilities"][label]) for label in CLASS_ORDER[head]])
                after_values = np.asarray([float(after_output["probabilities"][label]) for label in CLASS_ORDER[head]])
                js_values.append(_js_scalar(before_values, after_values))
                before_distribution[before_prediction] += 1
                after_distribution[after_prediction] += 1
                before_conf = float(np.max(before_values))
                after_conf = float(np.max(after_values))
            if before_key != after_key and max(before_conf, after_conf) >= 0.8:
                high_confidence_flip += 1
        result["heads"][head] = {"prediction_change_count": changed, "prediction_change_rate": changed / len(before_rows), "mean_js_divergence": float(np.mean(js_values)), "high_confidence_flip_count": high_confidence_flip, "class_or_label_distribution_before": dict(sorted(before_distribution.items(), key=lambda item: str(item[0]))), "class_or_label_distribution_after": dict(sorted(after_distribution.items(), key=lambda item: str(item[0])))}
    return result


def _row_with_answer(row: Mapping[str, Any], answers: Mapping[tuple[str, str], Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    return {"sample_id": sample_id, "model_text": row.get("model_text"), "human_answers": {head: answers[(sample_id, head)] for head in HEADS}, "before": before, "after": after}


def run(*, labels_path: Path = DEFAULT_LABELS, snapshot_path: Path = DEFAULT_SNAPSHOT, selection_train_path: Path = DEFAULT_SELECTION_TRAIN, selection_check_path: Path = DEFAULT_SELECTION_CHECK, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists():
        raise FeedbackAdapterError(f"refusing to overwrite existing adapter artifact: {output_dir}")
    labels = _answers(_read_jsonl(labels_path))
    snapshot = _read_jsonl(snapshot_path)
    selection_train = _read_jsonl(selection_train_path)
    selection_check = _read_jsonl(selection_check_path)
    if len(selection_train) != 10 or len(selection_check) != 4:
        raise FeedbackAdapterError("frozen selection must contain 10 Train and 4 Check rows")
    snapshot_by_id = {str(row["sample_id"]): row for row in snapshot}
    train_ids = [str(row["sample_id"]) for row in selection_train]
    check_ids = [str(row["sample_id"]) for row in selection_check]
    if len(set(train_ids)) != 10 or len(set(check_ids)) != 4 or set(train_ids) & set(check_ids):
        raise FeedbackAdapterError("selection identities are not disjoint")
    if any(sample_id not in snapshot_by_id for sample_id in [*train_ids, *check_ids]):
        raise FeedbackAdapterError("selection is not covered by prediction snapshot")
    if any((sample_id, head) not in labels for sample_id in [*train_ids, *check_ids] for head in HEADS):
        raise FeedbackAdapterError("not all 98 final human answers are present")
    parameters = fit_biases([snapshot_by_id[sample_id] for sample_id in train_ids], labels)
    before_train = [snapshot_by_id[sample_id] for sample_id in train_ids]
    before_check = [snapshot_by_id[sample_id] for sample_id in check_ids]
    after_train = [apply_adapter(row, parameters) for row in before_train]
    after_check = [apply_adapter(row, parameters) for row in before_check]
    before_all = snapshot
    after_all = [apply_adapter(row, parameters) for row in snapshot]
    train_before_metrics = metric_summary(before_train, labels)
    train_after_metrics = metric_summary(after_train, labels)
    check_before_metrics = metric_summary(before_check, labels)
    check_after_metrics = metric_summary(after_check, labels)
    check_population_info = _check_population_info(selection_check)
    diagnostic_version = "V2" if "v2" in output_dir.name.lower() else "V1"
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "adapter-parameters.json", parameters)
    _write_jsonl(output_dir / "human-train-before-after.jsonl", [_row_with_answer(row, labels, before, after) for row, before, after in zip(selection_train, before_train, after_train)])
    _write_jsonl(output_dir / "human-check-before-after.jsonl", [_row_with_answer(row, labels, before, after) for row, before, after in zip(selection_check, before_check, after_check)])
    aggregate = {
        "status": None,
        "training_population": "Human_Train_10_only",
        "check_population": "Human_Check_4_descriptive_only",
        "human_train": {"before": train_before_metrics, "after": train_after_metrics, "delta": {head: {"loss_delta": train_after_metrics[head]["before_or_after_nll_or_bce"] - train_before_metrics[head]["before_or_after_nll_or_bce"], "strict_hit_rate_delta": train_after_metrics[head]["strict_hit_rate"] - train_before_metrics[head]["strict_hit_rate"], "target_probability_delta": train_after_metrics[head]["target_probability_mean"] - train_before_metrics[head]["target_probability_mean"]} for head in HEADS}},
        "human_check": {"before": check_before_metrics, "after": check_after_metrics, "descriptive_only": True, **check_population_info},
        "prediction_snapshot_300_drift": drift_summary(before_all, after_all),
        "adapter_scope": "all_seven_heads; probabilities only; no Encoder representation learning; no candidate replacement",
        "forbidden_inputs_not_read": ["Test", "Anchor", "Gold", "OOD", "Human48", "Human100", "LLM", "cloud", "production"],
    }
    improved_heads = [head for head in HEADS if train_after_metrics[head]["before_or_after_nll_or_bce"] < train_before_metrics[head]["before_or_after_nll_or_bce"] - 1e-9 and train_after_metrics[head]["target_probability_mean"] >= train_before_metrics[head]["target_probability_mean"] - 1e-9]
    if len(improved_heads) == len(HEADS):
        aggregate["status"] = "FEEDBACK_ABSORBED"
    elif improved_heads:
        aggregate["status"] = "PARTIALLY_ABSORBED"
    else:
        aggregate["status"] = "NOT_ABSORBED"
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    report_lines = [
        f"# Feedback Loop {diagnostic_version} — Human10 Seven-Head Adapter",
        "",
        f"Status: `{aggregate['status']}` (diagnostic only; frozen candidate is unchanged).",
        "",
        "A single deterministic convex optimization (L2=1.0) learned additive per-class biases on existing probabilities. It cannot learn new semantic representations and is not a substitute for data-sufficient Encoder training.",
        "",
        "## Scope",
        "",
        "- Human Train: 10 samples × 7 heads; the only fitting population.",
        "- Human Check: 4 samples × 7 heads; descriptive only, never used for optimization or selection.",
        "- Original prediction snapshot: 300 rows; drift measurement only.",
        f"- Check text is classified dynamically: `{len(check_population_info['low_information_inputs'])}` LOW_INFORMATION_INPUT row(s), `{check_population_info['semantic_subset_n']}` semantic subset row(s). Check is descriptive only and cannot support a stable quality conclusion.",
        "- No Test, Anchor, Gold, OOD, Human48, Human100, LLM, cloud, or production data was read.",
        "",
        "## Per-head training result",
        "",
        "| head | before loss | after loss | loss delta | before strict hit | after strict hit | target probability delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for head in HEADS:
        before, after = train_before_metrics[head], train_after_metrics[head]
        delta = aggregate["human_train"]["delta"][head]
        report_lines.append(f"| {head} | {before['before_or_after_nll_or_bce']:.6f} | {after['before_or_after_nll_or_bce']:.6f} | {delta['loss_delta']:+.6f} | {before['strict_hit_rate']:.3f} | {after['strict_hit_rate']:.3f} | {delta['target_probability_delta']:+.6f} |")
    report_lines += [
        "",
        "## Drift on original 300-row prediction snapshot",
        "",
        "Per-head prediction changes, mean Jensen–Shannon divergence, high-confidence flips, and before/after class or label distributions are below. This is an influence measurement, not a truth comparison.",
        "",
        "| head | changed / 300 | change rate | mean JS | high-confidence flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for head in HEADS:
        drift = aggregate["prediction_snapshot_300_drift"]["heads"][head]
        report_lines.append(f"| {head} | {drift['prediction_change_count']} | {drift['prediction_change_rate']:.3f} | {drift['mean_js_divergence']:.6f} | {drift['high_confidence_flip_count']} |")
    report_lines += [
        "",
        "No adapter output replaces the selected candidate. Any future use would require a new, explicitly scoped experiment with fresh data and its own evaluation.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "myresearcher.feedback-adapter-v1-artifact.v1",
        "status": aggregate["status"],
        "diagnostic_only": True,
        "selected_candidate": False,
        "parent_candidate_content_address": "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf",
        "files": {},
        "forbidden_inputs_not_read": aggregate["forbidden_inputs_not_read"],
    }
    for filename in ("adapter-parameters.json", "human-train-before-after.jsonl", "human-check-before-after.jsonl", "aggregate_metrics.json", "REPORT.md"):
        manifest["files"][filename] = _sha256(output_dir / filename)
    manifest["content_address"] = _content_address(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    return {"status": aggregate["status"], "artifact_dir": str(output_dir), "content_address": manifest["content_address"], "improved_heads": improved_heads, "train_rows": 10, "check_rows": 4, "snapshot_rows": 300}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--selection-train", type=Path, default=DEFAULT_SELECTION_TRAIN)
    parser.add_argument("--selection-check", type=Path, default=DEFAULT_SELECTION_CHECK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run(labels_path=args.labels, snapshot_path=args.snapshot, selection_train_path=args.selection_train, selection_check_path=args.selection_check, output_dir=args.output_dir)
    except (FeedbackAdapterError, OSError, RuntimeError) as exc:
        print(f"[feedback-adapter-v1] FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
