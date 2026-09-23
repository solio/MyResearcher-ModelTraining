"""Read a bounded owner-label follow-up package; never fit or infer a model.

Only explicitly requested sample/head assignments are analyzed.  Review labels
describe definition changes; natural labels describe the frozen sampling frame.
Neither source is automatically promoted to Gold or used to change old labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema/semantic-schema-calibrated-v0.2.1.json"
TERMINAL_DISPOSITIONS = {"RESOLVED_WITHOUT_LABEL", "NEEDS_ADJUDICATION", "SKIPPED", "NOT_APPLICABLE", "跳过", "无法判断", "缺少上下文"}


class FollowupError(ValueError):
    """The supplied package cannot be interpreted without guessing."""


def read_jsonl(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    if optional and not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FollowupError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise FollowupError(f"{path.name}:{number}: expected an object")
        result.append(row)
    return result


def _answer(answer: Any, head: str, orders: Mapping[str, list[str]]) -> Any:
    if head == "reasoning_tags":
        if not isinstance(answer, list) or not answer or any(not isinstance(x, str) for x in answer):
            raise FollowupError("reasoning requires a nonempty list of explicit labels")
        if len(set(answer)) != len(answer) or any(x not in orders[head] for x in answer):
            raise FollowupError("invalid or duplicated reasoning label")
        return sorted(answer, key=orders[head].index)
    if not isinstance(answer, str) or answer not in orders[head]:
        raise FollowupError("expected an explicit Schema label")
    return answer


def _timestamp(row: Mapping[str, Any]) -> float | None:
    value = row.get("updated_at")
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.timestamp()
    except ValueError as exc:
        raise FollowupError("updated_at must be an ISO timestamp with timezone") from exc


def _disposition(row: Mapping[str, Any]) -> str:
    return str(row.get("disposition") or "").upper()


def _terminal(row: Mapping[str, Any]) -> bool:
    return row.get("is_final") is True or _disposition(row) in TERMINAL_DISPOSITIONS


def _latest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    """Use the newest explicitly timestamped revision, including a new draft.

    Conflicting submissions without an unambiguous revision time are reported,
    not resolved by arbitrary input order.
    """
    candidates = rows
    if not candidates:
        return None, False
    signatures = {json.dumps([r.get("answer"), _disposition(r), _terminal(r)], sort_keys=True) for r in candidates}
    if len(signatures) == 1:
        return candidates[-1], False
    times = [_timestamp(r) for r in candidates]
    if any(t is None for t in times):
        return None, True
    latest = max(times)
    winners = [r for r, t in zip(candidates, times) if t == latest]
    signatures = {json.dumps([r.get("answer"), _disposition(r), _terminal(r)], sort_keys=True) for r in winners}
    return (winners[-1], False) if len(signatures) == 1 else (None, True)


def _scalar_metrics(pairs: list[tuple[Any, Any]], order: list[str]) -> dict[str, Any]:
    matrix = [[0 for _ in order] for _ in order]
    for truth, prediction in pairs:
        matrix[order.index(truth)][order.index(prediction)] += 1
    per_class = {}
    for i, label in enumerate(order):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        per_class[label] = {
            "support": support, "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
        }
    n = len(pairs)
    majority = max((v["support"] for v in per_class.values()), default=0)
    return {"n": n, "class_order": order, "confusion_matrix": matrix, "per_class": per_class,
            "accuracy": sum(matrix[i][i] for i in range(len(order))) / n if n else None,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(order) if n else None,
            "empirical_majority_accuracy_descriptive_only": majority / n if n else None}


def _multilabel_metrics(pairs: list[tuple[Any, Any]], order: list[str]) -> dict[str, Any]:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in order}
    for truth, prediction in pairs:
        for label, c in counts.items():
            c["support"] += label in truth
            c["tp"] += label in truth and label in prediction
            c["fp"] += label not in truth and label in prediction
            c["fn"] += label in truth and label not in prediction
    for c in counts.values():
        c["recall"] = c["tp"] / c["support"] if c["support"] else 0.0
        c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        c["f1"] = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if 2 * c["tp"] + c["fp"] + c["fn"] else 0.0
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn"))
    return {"n": len(pairs), "per_label": counts,
            "exact_set_accuracy": sum(set(t) == set(p) for t, p in pairs) / len(pairs) if pairs else None,
            "macro_f1": sum(c["f1"] for c in counts.values()) / len(order) if pairs else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _probabilities(payload: Mapping[str, Any], order: list[str], *, multilabel: bool) -> list[float] | None:
    raw = payload.get("probabilities")
    if raw is None:
        return None
    if isinstance(raw, dict):
        if set(raw) != set(order):
            raise FollowupError("probabilities must include exactly the head class order")
        vector = [raw[label] for label in order]
    elif isinstance(raw, list):
        declared = payload.get("class_order")
        if not isinstance(declared, list) or len(declared) != len(order) or set(declared) != set(order) or len(raw) != len(order):
            raise FollowupError("probability arrays need an explicit matching class_order")
        vector = [raw[declared.index(label)] for label in order]
    else:
        raise FollowupError("invalid probability representation")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in vector):
        raise FollowupError("probabilities must be finite numbers in [0,1]")
    if not multilabel and not math.isclose(sum(vector), 1.0, abs_tol=1e-5):
        raise FollowupError("scalar probabilities must sum to one")
    return vector


def _prediction(payload: Mapping[str, Any], head: str, orders: Mapping[str, list[str]]) -> Any:
    keys = ("prediction", "predicted_labels", "labels") if head == "reasoning_tags" else ("prediction", "label")
    value = next((payload[key] for key in keys if key in payload), None)
    # An empty MODEL set is a real prediction; an empty HUMAN set is missing
    # supervision.  Do not silently add a sentinel or retune stored thresholds.
    if head == "reasoning_tags" and value == []:
        return []
    return _answer(value, head, orders)


def _probability_summary(rows: list[tuple[Any, list[float]]], order: list[str], *, multilabel: bool) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "calibration_conclusion": "EVIDENCE_INSUFFICIENT"}
    briers, true_probs, confidences, correctness = [], [], [], []
    for truth, vector in rows:
        targets = [int(label in truth) if multilabel else int(label == truth) for label in order]
        briers.append(sum((p - y) ** 2 for p, y in zip(vector, targets)) / (len(order) if multilabel else 1))
        if not multilabel:
            true_probs.append(vector[order.index(truth)])
            top = max(range(len(order)), key=vector.__getitem__)
            confidences.append(vector[top])
            correctness.append(order[top] == truth)
    result: dict[str, Any] = {"n": len(rows), "brier": sum(briers) / len(rows),
        "brier_definition": "mean_label_binary_squared_error" if multilabel else "sum_class_squared_error",
        "calibration_conclusion": "DESCRIPTIVE_SMALL_SAMPLE_ONLY_NOT_CALIBRATION_ACCEPTANCE"}
    if not multilabel:
        bins = []
        for i in range(5):
            indexes = [j for j, p in enumerate(confidences) if min(4, int(p * 5)) == i]
            bins.append({"lower": i / 5, "upper": (i + 1) / 5, "n": len(indexes),
                "mean_confidence": sum(confidences[j] for j in indexes) / len(indexes) if indexes else None,
                "accuracy": sum(correctness[j] for j in indexes) / len(indexes) if indexes else None})
        result.update({"mean_true_label_probability": sum(true_probs) / len(rows),
            "mean_top1_probability": sum(confidences) / len(rows), "reliability_bins": bins,
            "ece_5_equal_width_bins": sum(b["n"] * abs(b["mean_confidence"] - b["accuracy"]) for b in bins if b["n"]) / len(rows)})
    return result


def evaluate(package: Path, annotations: Path) -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    orders = {h: v["class_order"] for h, v in schema["v1_prediction_heads"].items()}
    plan = json.loads((package / "plan.json").read_text(encoding="utf-8"))
    if plan.get("schema_version", schema["schema_version"]) != schema["schema_version"]:
        raise FollowupError("package category Schema differs from the supported frozen Schema")
    tasks: dict[tuple[str, str], str] = {}
    sample_roles: dict[str, str] = {}
    for role in ("review", "natural"):
        for row in read_jsonl(package / f"{role}-tasks.jsonl"):
            sid, heads = row.get("sample_id"), row.get("heads")
            if not isinstance(sid, str) or not sid or sid in sample_roles:
                raise FollowupError("tasks need unique nonempty sample_id values across both roles")
            if not isinstance(heads, list) or not 1 <= len(heads) <= min(3, plan.get("max_heads_per_sample", 3)) or len(set(heads)) != len(heads) or any(h not in orders for h in heads):
                raise FollowupError("each task needs one to three distinct Schema heads")
            if row.get("metadata", {}).get("role", role) != role:
                raise FollowupError("task role conflicts with its file")
            sample_roles[sid] = role
            tasks.update({(sid, head): role for head in heads})
    if len(sample_roles) > min(40, plan.get("sample_budget", 40)):
        raise FollowupError("package exceeds its bounded sample budget")
    grouped = defaultdict(list)
    extra_rows = []
    for row in read_jsonl(annotations):
        key = (row.get("sample_id"), row.get("head"))
        if key not in tasks:
            extra_rows.append({"sample_id": key[0], "head": key[1], "reason": "NOT_REQUESTED"})
            continue
        expected_batch = plan.get("batches", {}).get(tasks[key])
        if expected_batch and row.get("batch") != expected_batch:
            raise FollowupError(f"annotation batch differs for {key[0]}/{key[1]}")
        if row.get("schema_version") not in (None, schema["schema_version"]):
            raise FollowupError("annotation Schema differs from the unchanged category Schema")
        grouped[key].append(row)
    states, labels, chosen_rows = {}, {}, {}
    for key in tasks:
        row, conflict = _latest(grouped[key])
        chosen_rows[key] = row
        if conflict:
            states[key] = "CONFLICTING_REVISIONS"
        elif row is None:
            states[key] = "MISSING"
        elif not _terminal(row):
            states[key] = "IN_PROGRESS"
        elif _disposition(row) in TERMINAL_DISPOSITIONS:
            states[key] = _disposition(row)
        else:
            try:
                labels[key] = _answer(row.get("answer"), key[1], orders)
                states[key] = "LABELED"
            except FollowupError:
                states[key] = "INVALID_LABEL"
    violations = []
    jointly_requested = jointly_labeled = 0
    for sid in sample_roles:
        primary, target = (sid, "emotion_primary"), (sid, "emotion_target")
        if primary not in tasks or target not in tasks:
            continue
        jointly_requested += 1
        if primary not in labels or target not in labels:
            continue
        jointly_labeled += 1
        if labels[primary] == "NONE_EXPLICIT" and labels[target] != "NOT_APPLICABLE":
            violations.append({"sample_id": sid, "emotion_primary": labels[primary], "emotion_target": labels[target]})
            states[primary] = states[target] = "DEPENDENCY_CONFLICT"
    eligible = {key: label for key, label in labels.items() if states[key] == "LABELED"}
    by_head = {}
    for head in orders:
        keys = [k for k in tasks if k[1] == head]
        if not keys:
            continue
        by_head[head] = {"requested": len(keys), "states": dict(Counter(states[k] for k in keys)),
            "missing": [k[0] for k in keys if states[k] in {"MISSING", "IN_PROGRESS"}],
            "eligible_for_analysis": sum(k in eligible for k in keys),
            "explicit_unknown": sum(k in eligible and (eligible[k] == "UNKNOWN" or eligible[k] == ["UNKNOWN"]) for k in keys)}
    old = defaultdict(list)
    for row in read_jsonl(package / "baseline-labels.jsonl", optional=True):
        old[(row.get("sample_id"), row.get("head"))].append(row)
    changes, unavailable_baselines = [], []
    for key, label in eligible.items():
        if tasks[key] != "review":
            continue
        baseline, conflict = _latest(old[key])
        try:
            previous = _answer(baseline.get("answer") if baseline else None, key[1], orders)
        except FollowupError:
            previous = None
        if conflict or previous is None:
            unavailable_baselines.append({"sample_id": key[0], "head": key[1]})
        else:
            changes.append({"sample_id": key[0], "head": key[1], "previous": previous, "current": label, "changed": previous != label})
    prediction_map = {}
    for row in read_jsonl(package / "predictions.jsonl", optional=True):
        sid = row.get("sample_id")
        if sid in prediction_map:
            raise FollowupError("duplicate prediction sample_id")
        if sample_roles.get(sid) == "natural":
            prediction_map[sid] = row.get("predictions", {})
    natural = {}
    for head, order in orders.items():
        requested = [k for k in tasks if tasks[k] == "natural" and k[1] == head]
        if not requested:
            continue
        keys = [k for k in requested if k in eligible]
        support = Counter()
        pairs, probabilities, missing_predictions = [], [], []
        for key in keys:
            truth = eligible[key]
            support.update(truth if head == "reasoning_tags" else [truth])
            payload = prediction_map.get(key[0], {}).get(head)
            if not isinstance(payload, dict):
                missing_predictions.append(key[0])
                continue
            prediction = _prediction(payload, head, orders)
            pairs.append((truth, prediction))
            vector = _probabilities(payload, order, multilabel=head == "reasoning_tags")
            if vector is not None:
                probabilities.append((truth, vector))
        natural[head] = {"requested": len(requested), "eligible_for_analysis": len(keys),
            "support": {label: support[label] for label in order}, "missing_predictions": missing_predictions,
            "metrics": (_multilabel_metrics(pairs, order) if head == "reasoning_tags" else _scalar_metrics(pairs, order)) if pairs else None,
            "probability_statistics": _probability_summary(probabilities, order, multilabel=head == "reasoning_tags")}
    waiting = any(v in {"MISSING", "IN_PROGRESS"} for v in states.values())
    needs_resolution = len(eligible) != len(tasks)
    return {"status": "WAITING_FOR_HUMAN" if waiting else ("HUMAN_REVIEW_COMPLETE_ANALYSIS_LIMITED" if needs_resolution else "READY_FOR_BOUNDED_DIAGNOSTIC"),
        "category_schema_version": schema["schema_version"], "definition_revision": plan.get("definition_revision"),
        "evidence_role": "REVISABLE_OWNER_LABELS_NOT_GOLD", "samples": len(sample_roles),
        "assignments": len(tasks), "tasks_processed": sum(_terminal(row) for row in chosen_rows.values() if row),
        "eligible_for_analysis": len(eligible), "by_head": by_head, "unrequested_annotation_rows": extra_rows,
        "assignments_needing_attention": [{"sample_id": key[0], "head": key[1], "role": tasks[key], "state": state}
            for key, state in states.items() if state != "LABELED"],
        "emotion_dependency": {"rule": "jointly labeled NONE_EXPLICIT implies NOT_APPLICABLE", "jointly_requested": jointly_requested,
            "jointly_labeled": jointly_labeled, "violations": violations, "labels_automatically_changed": False},
        "review": {"comparisons": changes, "changed": sum(x["changed"] for x in changes), "baseline_unavailable": unavailable_baselines,
            "interpretation": "Definition-review changes only; not natural accuracy or automatic corrections."},
        "natural": natural, "sampling_scope": plan.get("natural_scope", plan.get("evaluation", {}).get("scope", plan.get("sampling_rules"))),
        "sampling": plan.get("sampling"), "package_evaluation_limits": plan.get("evaluation", {}).get("limits", []),
        "promotion_decision": "NOT_ELIGIBLE_FOR_PROMOTION_OR_AUTOMATIC_RETRAINING",
        "limitations": ["At most 40 samples and three requested heads per sample; unrequested heads are not missing labels.",
            "Natural metrics describe only this frozen sampling frame and labeled subset; review samples are excluded.",
            "Small per-class support and reused prediction history cannot establish generalization or calibration acceptance.",
            "No model inference, training, label correction, Test, or market outcomes are used."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Create a new report only; never overwrite an existing file.")
    args = parser.parse_args(argv)
    try:
        source_paths = {SCHEMA_PATH.resolve(), args.annotations.resolve()}
        source_paths.update((args.package / name).resolve() for name in ("plan.json", "review-tasks.jsonl", "natural-tasks.jsonl", "baseline-labels.jsonl", "predictions.jsonl", "manifest.json"))
        if args.output and args.output.resolve() in source_paths:
            raise FollowupError("output must not name an input source file")
        result = evaluate(args.package, args.annotations)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        print(serialized, end="")
    except (FollowupError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FOLLOWUP_INPUT_ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
