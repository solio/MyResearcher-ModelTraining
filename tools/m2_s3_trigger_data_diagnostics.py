"""Read-only Train/Dev diagnostics for the S2 CALM and NO_REASON_GIVEN triggers.

This tool deliberately stays on the data/metrics side of M2.  It uses M1's
Train/Dev loader, reads only existing S1/S2 ``seed-metrics.json`` files, and
emits aggregate JSON/Markdown.  It never loads a model, checkpoint, tokenizer,
cache, or prediction file and never reads a protected data role.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_model.config import ProjectConfig
from semantic_model.encoder_m1 import M1Record, load_m1_partitions
from semantic_model.errors import ContractError
from semantic_model.schema import FROZEN_CLASS_ORDER, SINGLE_LABEL_HEADS, V1_HEADS


SCHEMA_VERSION = "myresearcher.m2-s3-trigger-data-diagnostics.v1"
CALM = "CALM"
NO_REASON_GIVEN = "NO_REASON_GIVEN"
SEEDS = (35, 71, 107)
LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0-19", 0, 20),
    ("20-49", 20, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    ("200-399", 200, 400),
    ("400+", 400, None),
)


def _require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        raise ContractError(code, message, **details)


def _metric_number(value: Any, *, code: str = "M2_S3_METRICS_INVALID", **details: Any) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        code,
        "metric value must be numeric",
        **details,
    )
    number = float(value)
    _require(math.isfinite(number), code, "metric value must be finite", **details)
    return number


def _validate_metric_scope(document: Mapping[str, Any], *, stage: str, seed: int) -> None:
    scope = document.get("metric_scope")
    _require(isinstance(scope, str), "M2_S3_METRICS_SCOPE_INVALID", "metric_scope must be a string", stage=stage, seed=seed)
    normalized = scope.upper().replace("-", "_")
    required_markers = ("WEAK_LABEL", "NOT_GOLD", "NOT_TEST", "NOT_PRODUCTION")
    _require(
        all(marker in normalized for marker in required_markers),
        "M2_S3_METRICS_SCOPE_INVALID",
        "metrics must explicitly declare WEAK_LABEL, NOT_GOLD, NOT_TEST, and NOT_PRODUCTION scope",
        stage=stage,
        seed=seed,
        metric_scope=scope,
    )
    # Remove the negative markers before looking for a positive protected scope.
    residual = normalized
    for marker in ("NOT_GOLD", "NOT_TEST", "NOT_PRODUCTION"):
        residual = residual.replace(marker, "")
    _require(
        not any(marker in residual for marker in ("GOLD", "TEST", "PRODUCTION")),
        "M2_S3_METRICS_SCOPE_INVALID",
        "metrics must not claim positive Gold, Test, or production scope",
        stage=stage,
        seed=seed,
        metric_scope=scope,
    )


def _record_parts(record: M1Record) -> tuple[str, Mapping[str, Any], Mapping[str, Any], str]:
    sample_id = record.sample_id
    label = record.label
    weights = record.weights
    text = record.model_text
    _require(isinstance(sample_id, str) and sample_id, "M2_S3_RECORD_INVALID", "sample_id must be a non-empty string")
    _require(isinstance(label, Mapping), "M2_S3_RECORD_INVALID", "label must be an object", sample_id=sample_id)
    _require(isinstance(weights, Mapping), "M2_S3_RECORD_INVALID", "weights must be an object", sample_id=sample_id)
    _require(isinstance(text, str), "M2_S3_RECORD_INVALID", "model_text must be a string", sample_id=sample_id)
    for head in V1_HEADS:
        value = weights.get(head)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0,
            "M2_S3_RECORD_INVALID",
            "every V1 head requires a finite non-negative weight",
            sample_id=sample_id,
            head=head,
        )
    emotion = label.get("emotion_primary")
    _require(isinstance(emotion, str), "M2_S3_RECORD_INVALID", "emotion_primary must be a string", sample_id=sample_id)
    for head in SINGLE_LABEL_HEADS:
        _require(isinstance(label.get(head), str), "M2_S3_RECORD_INVALID", "scalar label must be a string", sample_id=sample_id, head=head)
    tags = label.get("reasoning_tags")
    _require(isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)), "M2_S3_RECORD_INVALID", "reasoning_tags must be a sequence", sample_id=sample_id)
    normalized_tags = list(tags)
    _require(all(isinstance(tag, str) for tag in normalized_tags), "M2_S3_RECORD_INVALID", "reasoning tags must be strings", sample_id=sample_id)
    _require(len(set(normalized_tags)) == len(normalized_tags), "M2_S3_RECORD_INVALID", "reasoning tags must be unique", sample_id=sample_id)
    unknown_tags = sorted(set(normalized_tags) - set(FROZEN_CLASS_ORDER["reasoning_tags"]))
    _require(not unknown_tags, "M2_S3_RECORD_INVALID", "reasoning tag is outside the frozen schema", sample_id=sample_id, tags=unknown_tags)
    return sample_id, label, weights, text


def _validate_partitions(train: Sequence[M1Record], dev: Sequence[M1Record]) -> None:
    _require(train and dev, "M2_S3_DATA_INVALID", "Train and Dev partitions must be non-empty")
    seen: set[str] = set()
    for population, rows in (("Train", train), ("Dev", dev)):
        for row in rows:
            sample_id, _, _, _ = _record_parts(row)
            _require(sample_id not in seen, "M2_S3_DATA_INVALID", "Train/Dev sample_id must be unique", sample_id=sample_id)
            seen.add(sample_id)
        _require(len(rows) == len({row.sample_id for row in rows}), "M2_S3_DATA_INVALID", "partition sample_id must be unique", population=population)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "p95": None, "max": None, "zero_fraction": None, "value_counts": {}}
    ordered = sorted(float(value) for value in values)

    def percentile(percent: int) -> float:
        index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
        return ordered[index]

    counts = Counter(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "max": ordered[-1],
        "zero_fraction": round(sum(value == 0 for value in ordered) / len(ordered), 6),
        "value_counts": {str(value): count for value, count in sorted(counts.items())},
    }


def _proportion(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def _category_cooccurrence(rows: Sequence[M1Record], head: str) -> dict[str, Any]:
    counts = Counter(str(row.label[head]) for row in rows)
    total = len(rows)
    return {
        "rows": total,
        "categories": {
            category: {"count": count, "proportion": _proportion(count, total)}
            for category, count in sorted(counts.items())
        },
    }


def _tag_cooccurrence(rows: Sequence[M1Record], *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    total = len(rows)
    result: dict[str, Any] = {}
    for tag in FROZEN_CLASS_ORDER["reasoning_tags"]:
        if tag in excluded:
            continue
        count = sum(tag in row.label["reasoning_tags"] for row in rows)
        result[tag] = {"count": count, "proportion": _proportion(count, total)}
    return {"rows": total, "tags": result}


def _length_buckets(rows: Sequence[M1Record]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        length = len(row.model_text)
        for name, lower, upper in LENGTH_BUCKETS:
            if length >= lower and (upper is None or length < upper):
                counts[name] += 1
                break
        else:
            raise ContractError("M2_S3_LENGTH_BUCKET_INVALID", "text length did not fit a bucket", sample_id=row.sample_id, length=length)
    total = len(rows)
    return {
        name: {"count": counts[name], "proportion": _proportion(counts[name], total)}
        for name, _, _ in LENGTH_BUCKETS
    }


def _cross_distribution(rows: Sequence[M1Record]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter()
    for row in rows:
        calm = row.label["emotion_primary"] == CALM
        no_reason = NO_REASON_GIVEN in row.label["reasoning_tags"]
        counts[(calm, no_reason)] += 1
    names = (
        ((True, True), "CALM_and_NO_REASON_GIVEN"),
        ((True, False), "CALM_only"),
        ((False, True), "NO_REASON_GIVEN_only"),
        ((False, False), "neither"),
    )
    return {
        name: {"count": counts[key], "proportion": _proportion(counts[key], total)}
        for key, name in names
    }


def _population(rows: Sequence[M1Record]) -> dict[str, Any]:
    total = len(rows)
    calm_rows = [row for row in rows if row.label["emotion_primary"] == CALM]
    no_reason_rows = [row for row in rows if NO_REASON_GIVEN in row.label["reasoning_tags"]]
    affected_rows = [row for row in rows if row in calm_rows or row in no_reason_rows]
    unaffected_rows = [row for row in rows if row not in affected_rows]
    calm_other_heads: dict[str, Any] = {
        head: _category_cooccurrence(calm_rows, head)
        for head in SINGLE_LABEL_HEADS
        if head != "emotion_primary"
    }
    calm_other_heads["reasoning_tags"] = _tag_cooccurrence(calm_rows)
    return {
        "rows": total,
        "targets": {
            "emotion_primary:CALM": {
                "count": len(calm_rows),
                "proportion": _proportion(len(calm_rows), total),
                "affected_head": "emotion_primary",
                "affected_head_weight_distribution": _stats([float(row.weights["emotion_primary"]) for row in calm_rows]),
            },
            "reasoning_tags:NO_REASON_GIVEN": {
                "count": len(no_reason_rows),
                "proportion": _proportion(len(no_reason_rows), total),
                "affected_head": "reasoning_tags",
                "affected_head_weight_distribution": _stats([float(row.weights["reasoning_tags"]) for row in no_reason_rows]),
            },
        },
        "affected_rows": {
            "count": len(affected_rows),
            "proportion": _proportion(len(affected_rows), total),
            "overlap_count": len([row for row in calm_rows if NO_REASON_GIVEN in row.label["reasoning_tags"]]),
        },
        "calm_cooccurrence": {
            "other_six_heads": calm_other_heads,
        },
        "no_reason_given_cooccurrence": {
            "six_scalar_heads": {head: _category_cooccurrence(no_reason_rows, head) for head in SINGLE_LABEL_HEADS},
            "other_reasoning_tags": _tag_cooccurrence(no_reason_rows, exclude={NO_REASON_GIVEN}),
        },
        "cross_distribution": _cross_distribution(rows),
        "text_character_length_buckets": {
            "affected_rows": _length_buckets(affected_rows),
            "remaining_rows": _length_buckets(unaffected_rows),
        },
    }


def _read_seed_metric(root: Path, seed: int, *, stage: str, head: str | None = None) -> Mapping[str, Any]:
    path = root / head / f"seed-{seed}" / "seed-metrics.json" if head else root / f"seed-{seed}" / "seed-metrics.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("M2_S3_METRICS_MISSING", "matching-seed metric file is missing", stage=stage, seed=seed, path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("M2_S3_METRICS_INVALID", "matching-seed metric file is not valid JSON", stage=stage, seed=seed, path=str(path)) from exc
    _require(isinstance(document, Mapping), "M2_S3_METRICS_INVALID", "metric document must be an object", stage=stage, seed=seed)
    _require(document.get("sample_counts") == {"train": 1822, "dev": 448}, "M2_S3_METRICS_INVALID", "metric population must be Train 1822 / Dev 448", stage=stage, seed=seed)
    _validate_metric_scope(document, stage=stage, seed=seed)
    _require(isinstance(document.get("dev"), Mapping), "M2_S3_METRICS_INVALID", "dev metric object is required", stage=stage, seed=seed)
    return document


def _read_s3_matching_report(root: Path, s1_metrics: Mapping[int, Mapping[str, Any]], s3_metrics: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    path = root / "s3-vs-s1-matching-seed-report.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("M2_S3_METRICS_MISSING", "S3 matching-seed report is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("M2_S3_METRICS_INVALID", "S3 matching-seed report is not valid JSON", path=str(path)) from exc
    _require(isinstance(document, Mapping), "M2_S3_METRICS_INVALID", "S3 matching-seed report must be an object")
    _require(document.get("stage_id") == "M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL", "M2_S3_METRICS_INVALID", "unexpected S3 stage id")
    _require(document.get("comparator") == "S1_FROZEN_SHARED_MATCHING_SEED", "M2_S3_METRICS_INVALID", "S3 report must compare matching S1 seeds")
    _require(document.get("selected_candidate") is False, "M2_S3_METRICS_SCOPE_INVALID", "S3 report must remain diagnostic-only")
    critical = document.get("critical_labels")
    per_head = document.get("per_head")
    _require(isinstance(critical, Mapping) and isinstance(per_head, Mapping), "M2_S3_METRICS_INVALID", "S3 matching report is incomplete")
    for head, label in (("emotion_primary", CALM), ("reasoning_tags", NO_REASON_GIVEN)):
        entry = critical.get(head, {}).get(label) if isinstance(critical.get(head), Mapping) else None
        _require(isinstance(entry, Mapping), "M2_S3_METRICS_INVALID", "S3 critical label entry is missing", head=head, label=label)
        s1_f1 = entry.get("s1_f1_per_seed")
        s3_f1 = entry.get("s3_f1_per_seed")
        deltas = entry.get("delta_per_seed")
        supports = entry.get("support_per_seed")
        _require(all(isinstance(values, list) and len(values) == len(SEEDS) for values in (s1_f1, s3_f1, deltas, supports)), "M2_S3_METRICS_INVALID", "S3 critical label arrays must contain three seeds", head=head, label=label)
        metric_head = s3_metrics[head]
        for index, seed in enumerate(SEEDS):
            s1_row = s1_metrics[seed]["dev"][head]["per_class" if head != "reasoning_tags" else "per_label"][label]
            s3_row = metric_head[seed]["dev"][head]["per_class" if head != "reasoning_tags" else "per_label"][label]
            _require(abs(float(s1_f1[index]) - float(s1_row["f1"])) <= 1e-6 and abs(float(s3_f1[index]) - float(s3_row["f1"])) <= 1e-6, "M2_S3_METRICS_INVALID", "S3 matching report disagrees with seed metric F1", head=head, label=label, seed=seed)
            _require(int(supports[index]) == int(s3_row["support"]) == int(s1_row["support"]), "M2_S3_METRICS_INVALID", "S3 matching report disagrees with seed metric support", head=head, label=label, seed=seed)
            _require(abs(float(deltas[index]) - (float(s3_row["f1"]) - float(s1_row["f1"]))) <= 1e-6, "M2_S3_METRICS_INVALID", "S3 matching report delta disagrees with seed metrics", head=head, label=label, seed=seed)
    head_values: dict[str, Any] = {}
    for head in ("emotion_primary", "reasoning_tags"):
        entry = per_head.get(head)
        _require(isinstance(entry, Mapping), "M2_S3_METRICS_INVALID", "S3 head comparison is missing", head=head)
        for field in ("s1_per_seed", "s3_per_seed", "s1_mean", "s3_mean", "mean_delta", "delta_per_seed", "worst_seed_delta"):
            _require(field in entry, "M2_S3_METRICS_INVALID", "S3 head comparison field is missing", head=head, field=field)
        _require(
            all(isinstance(entry[field], list) and len(entry[field]) == len(SEEDS) for field in ("s1_per_seed", "s3_per_seed", "delta_per_seed")),
            "M2_S3_METRICS_INVALID",
            "S3 head per-seed arrays must contain three seeds",
            head=head,
        )
        recomputed_s1: list[float] = []
        recomputed_s3: list[float] = []
        recomputed_delta: list[float] = []
        for seed in SEEDS:
            s1_head = s1_metrics[seed]["dev"].get(head)
            s3_head = s3_metrics[head][seed]["dev"].get(head)
            _require(isinstance(s1_head, Mapping) and isinstance(s3_head, Mapping), "M2_S3_METRICS_INVALID", "head Macro-F1 metric object is missing", head=head, seed=seed)
            s1_macro = _metric_number(s1_head.get("macro_f1"), head=head, stage="S1", seed=seed)
            s3_macro = _metric_number(s3_head.get("macro_f1"), head=head, stage="S3", seed=seed)
            recomputed_s1.append(s1_macro)
            recomputed_s3.append(s3_macro)
            recomputed_delta.append(s3_macro - s1_macro)
        recomputed_s1_mean = sum(recomputed_s1) / len(recomputed_s1)
        recomputed_s3_mean = sum(recomputed_s3) / len(recomputed_s3)
        recomputed_mean_delta = sum(recomputed_delta) / len(recomputed_delta)
        for index, seed in enumerate(SEEDS):
            _require(
                abs(_metric_number(entry["s1_per_seed"][index], head=head, field="s1_per_seed", seed=seed) - recomputed_s1[index]) <= 1e-6
                and abs(_metric_number(entry["s3_per_seed"][index], head=head, field="s3_per_seed", seed=seed) - recomputed_s3[index]) <= 1e-6
                and abs(_metric_number(entry["delta_per_seed"][index], head=head, field="delta_per_seed", seed=seed) - recomputed_delta[index]) <= 1e-6,
                "M2_S3_METRICS_INVALID",
                "S3 head matching report disagrees with seed Macro-F1 metrics",
                head=head,
                seed=seed,
            )
        _require(
            abs(_metric_number(entry["s1_mean"], head=head, field="s1_mean") - recomputed_s1_mean) <= 1e-6
            and abs(_metric_number(entry["s3_mean"], head=head, field="s3_mean") - recomputed_s3_mean) <= 1e-6
            and abs(_metric_number(entry["mean_delta"], head=head, field="mean_delta") - recomputed_mean_delta) <= 1e-6
            and abs(_metric_number(entry["worst_seed_delta"], head=head, field="worst_seed_delta") - min(recomputed_delta)) <= 1e-6,
            "M2_S3_METRICS_INVALID",
            "S3 head matching report aggregate disagrees with seed Macro-F1 metrics",
            head=head,
        )
        head_values[head] = {
            "S1_per_seed_macro_f1": [round(value, 6) for value in recomputed_s1],
            "S3_per_seed_macro_f1": [round(value, 6) for value in recomputed_s3],
            "S1_mean_macro_f1": round(recomputed_s1_mean, 6),
            "S3_mean_macro_f1": round(recomputed_s3_mean, 6),
            "mean_delta_S3_minus_S1": round(recomputed_mean_delta, 6),
            "delta_per_seed": [round(value, 6) for value in recomputed_delta],
            "worst_seed_delta": round(min(recomputed_delta), 6),
        }
    return {
        "stage_id": document["stage_id"],
        "comparator": document["comparator"],
        "triggered_heads": list(document.get("triggered_heads", [])),
        "stability_gate_passed": document.get("stability_gate_passed"),
        "selected_candidate": document.get("selected_candidate"),
        "head_macro_f1": head_values,
    }


def load_matching_seed_metrics(s1_root: str | Path, s2_root: str | Path, s3_root: str | Path | None = None) -> dict[str, Any]:
    """Read only matching-seed Dev metrics for the two trigger labels."""

    s1_path, s2_path = Path(s1_root), Path(s2_root)
    s1_metrics = {seed: _read_seed_metric(s1_path, seed, stage="S1") for seed in SEEDS}
    s2_metrics = {seed: _read_seed_metric(s2_path, seed, stage="S2") for seed in SEEDS}
    s3_metrics: dict[str, dict[int, Mapping[str, Any]]] = {}
    if s3_root is not None:
        s3_path = Path(s3_root)
        s3_metrics = {head: {seed: _read_seed_metric(s3_path, seed, stage="S3", head=head) for seed in SEEDS} for head in ("emotion_primary", "reasoning_tags")}
    values: dict[str, Any] = {"targets": {}}
    for label_key, head, metric_key in (
        ("emotion_primary:CALM", "emotion_primary", "per_class"),
        ("reasoning_tags:NO_REASON_GIVEN", "reasoning_tags", "per_label"),
    ):
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            s1 = s1_metrics[seed]
            s2 = s2_metrics[seed]
            s1_head = s1["dev"].get(head)
            s2_head = s2["dev"].get(head)
            _require(isinstance(s1_head, Mapping) and isinstance(s2_head, Mapping), "M2_S3_METRICS_INVALID", "trigger head is missing", head=head, seed=seed)
            s1_values = s1_head.get(metric_key)
            s2_values = s2_head.get(metric_key)
            _require(isinstance(s1_values, Mapping) and isinstance(s2_values, Mapping), "M2_S3_METRICS_INVALID", "per-label metric object is missing", head=head, seed=seed)
            s1_row, s2_row = s1_values.get(CALM if head == "emotion_primary" else NO_REASON_GIVEN), s2_values.get(CALM if head == "emotion_primary" else NO_REASON_GIVEN)
            _require(isinstance(s1_row, Mapping) and isinstance(s2_row, Mapping), "M2_S3_METRICS_INVALID", "trigger label metric is missing", head=head, seed=seed)
            for stage_name, row in (("S1", s1_row), ("S2", s2_row)):
                _require(isinstance(row.get("f1"), (int, float)) and isinstance(row.get("support"), int), "M2_S3_METRICS_INVALID", "F1 and support are required", stage=stage_name, head=head, seed=seed)
            s1_f1, s2_f1 = float(s1_row["f1"]), float(s2_row["f1"])
            per_seed[str(seed)] = {
                "S1": {"f1": s1_f1, "support": int(s1_row["support"])},
                "S2": {"f1": s2_f1, "support": int(s2_row["support"])},
                "delta_S2_minus_S1": round(s2_f1 - s1_f1, 6),
            }
            if s3_root is not None:
                s3 = s3_metrics[head][seed]
                s3_head = s3["dev"].get(head)
                _require(isinstance(s3_head, Mapping), "M2_S3_METRICS_INVALID", "S3 trigger head is missing", head=head, seed=seed)
                s3_values = s3_head.get(metric_key)
                _require(isinstance(s3_values, Mapping), "M2_S3_METRICS_INVALID", "S3 per-label metric object is missing", head=head, seed=seed)
                s3_row = s3_values.get(CALM if head == "emotion_primary" else NO_REASON_GIVEN)
                _require(isinstance(s3_row, Mapping), "M2_S3_METRICS_INVALID", "S3 trigger label metric is missing", head=head, seed=seed)
                _require(isinstance(s3_row.get("f1"), (int, float)) and isinstance(s3_row.get("support"), int), "M2_S3_METRICS_INVALID", "S3 F1 and support are required", head=head, seed=seed)
                s3_f1 = float(s3_row["f1"])
                _require(int(s3_row["support"]) == int(s1_row["support"]), "M2_S3_METRICS_INVALID", "S3 support differs from S1 support", head=head, seed=seed)
                per_seed[str(seed)]["S3"] = {"f1": s3_f1, "support": int(s3_row["support"])}
                per_seed[str(seed)]["delta_S3_minus_S1"] = round(s3_f1 - s1_f1, 6)
        values["targets"][label_key] = {
            "per_seed": per_seed,
            "S1_mean_f1": round(sum(item["S1"]["f1"] for item in per_seed.values()) / len(per_seed), 6),
            "S2_mean_f1": round(sum(item["S2"]["f1"] for item in per_seed.values()) / len(per_seed), 6),
            "mean_delta_S2_minus_S1": round(sum(item["delta_S2_minus_S1"] for item in per_seed.values()) / len(per_seed), 6),
            "interpretation": "weak-label Dev metric only; not Gold, truth, or model-selection evidence",
        }
        if s3_root is not None:
            values["targets"][label_key]["S3_mean_f1"] = round(sum(item["S3"]["f1"] for item in per_seed.values()) / len(per_seed), 6)
            values["targets"][label_key]["mean_delta_S3_minus_S1"] = round(sum(item["delta_S3_minus_S1"] for item in per_seed.values()) / len(per_seed), 6)
    if s3_root is not None:
        values["s3_matching_report"] = _read_s3_matching_report(Path(s3_root), s1_metrics, s3_metrics)
        values["head_macro_f1"] = values["s3_matching_report"]["head_macro_f1"]
    return values


def _hypotheses(combined: Mapping[str, Any], matching_seed_metrics: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    calm = combined["targets"]["emotion_primary:CALM"]
    no_reason = combined["targets"]["reasoning_tags:NO_REASON_GIVEN"]
    affected = combined["affected_rows"]
    calm_weight = calm["affected_head_weight_distribution"]
    no_reason_weight = no_reason["affected_head_weight_distribution"]
    calm_zero_fraction = float(calm_weight["zero_fraction"] or 0.0)
    no_reason_zero_fraction = float(no_reason_weight["zero_fraction"] or 0.0)
    head_metrics = matching_seed_metrics.get("head_macro_f1", {}) if matching_seed_metrics else {}
    emotion_delta = round(float(head_metrics["emotion_primary"]["mean_delta_S3_minus_S1"]), 6) if "emotion_primary" in head_metrics else None
    reasoning_delta = round(float(head_metrics["reasoning_tags"]["mean_delta_S3_minus_S1"]), 6) if "reasoning_tags" in head_metrics else None
    if emotion_delta is not None and reasoning_delta is not None:
        if emotion_delta > 0 and reasoning_delta > 0:
            coupling_disposition = "BOTH_HEADS_IMPROVED_BOUNDED_NON_CAUSAL"
            coupling_evidence = f"S3 emotion_primary Macro-F1 mean delta versus S1 is {emotion_delta:+.6f}; S3 reasoning_tags mean delta is {reasoning_delta:+.6f}. Both head means increased in this comparison, but this bounded observation does not establish a shared cause."
        else:
            coupling_disposition = "NOT_SUPPORTED_AS_A_SHARED_EXPLANATION"
            coupling_evidence = f"S3 emotion_primary Macro-F1 mean delta versus S1 is {emotion_delta:+.6f}; S3 reasoning_tags mean delta is {reasoning_delta:+.6f}. The two single-task heads did not both improve."
    else:
        coupling_disposition = "PENDING_S3_METRICS"
        coupling_evidence = "S3 matching-seed head metrics were not supplied in this invocation."
    length_disposition = "UNRESOLVED" if matching_seed_metrics and matching_seed_metrics.get("s3_matching_report") else "PENDING_S3_METRICS"
    weight_disposition = "UNRESOLVED" if matching_seed_metrics and matching_seed_metrics.get("s3_matching_report") else "PENDING_S3_METRICS"
    return [
        {
            "id": "HYPOTHESIS_LABEL_COUPLING",
            "disposition": coupling_disposition,
            "statement": f"CALM and NO_REASON_GIVEN overlap on {affected['overlap_count']} rows ({_proportion(affected['overlap_count'], combined['rows']):.1%} of all rows).",
            "evidence": coupling_evidence,
        },
        {
            "id": "HYPOTHESIS_TEXT_LENGTH_SHIFT",
            "disposition": length_disposition,
            "statement": "Affected rows have a different text-length mix from the remaining rows, but the current S3 artifact has no per-sample character-length result.",
            "evidence": "UNRESOLVED: the available S3 metrics cannot confirm or deny a length-bucket explanation.",
        },
        {
            "id": "HYPOTHESIS_AFFECTED_HEAD_WEIGHT",
            "disposition": weight_disposition,
            "statement": f"The affected-head weight distributions have CALM zero-weight fraction {calm_zero_fraction:.1%} and NO_REASON_GIVEN zero-weight fraction {no_reason_zero_fraction:.1%}, but the current S3 artifact has no per-sample weight-bucket result.",
            "evidence": "UNRESOLVED: the available S3 metrics cannot confirm or deny a weight-bucket explanation.",
        },
    ]


def build_diagnostic(
    train: Sequence[M1Record],
    dev: Sequence[M1Record],
    matching_seed_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_partitions(train, dev)
    populations = {"Train": _population(train), "Dev": _population(dev), "TrainPlusDev": _population([*train, *dev])}
    result: dict[str, Any] = {
        "analysis_schema_version": SCHEMA_VERSION,
        "scope": "TRAIN_DEV_DATA_ONLY_WEAK_LABEL_CONTEXT_NOT_GOLD_NOT_TEST_NOT_ANCHOR_NOT_OOD_NOT_MODEL_SELECTION",
        "loader": "semantic_model.encoder_m1.load_m1_partitions",
        "target_labels": {"emotion_primary": CALM, "reasoning_tags": NO_REASON_GIVEN},
        "populations": populations,
        "hypotheses": _hypotheses(populations["TrainPlusDev"], matching_seed_metrics),
        "limitations": [
            "All labels and metrics are frozen weak labels; no Gold or truth claim is made.",
            "The report is a data-side diagnostic and cannot select a model, authorize S3, or establish production quality.",
            "Text is used only for character-length buckets; no original text is emitted.",
        ],
    }
    if matching_seed_metrics is not None:
        result["matching_seed_metrics"] = dict(matching_seed_metrics)
    return result


def _format_fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _markdown_summary(report: Mapping[str, Any]) -> str:
    combined = report["populations"]["TrainPlusDev"]
    lines = [
        "# M2-S3 trigger data diagnostics",
        "",
        "Scope: read-only Train/Dev data context for S2-triggered `emotion_primary:CALM` and `reasoning_tags:NO_REASON_GIVEN`.",
        "All labels and metric comparisons below are weak-label diagnostics only; they are not Gold, truth, model-selection, Test, Anchor, OOD, or production evidence.",
        "",
        "## Target prevalence and affected-head weights",
        "",
        "| Population | Rows | CALM | NO_REASON_GIVEN | Affected union | CALM emotion-head weight mean / zero | NO_REASON reasoning-head weight mean / zero |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("Train", "Dev", "TrainPlusDev"):
        population = report["populations"][name]
        calm = population["targets"]["emotion_primary:CALM"]
        no_reason = population["targets"]["reasoning_tags:NO_REASON_GIVEN"]
        calm_weight = calm["affected_head_weight_distribution"]
        no_reason_weight = no_reason["affected_head_weight_distribution"]
        lines.append(
            f"| {name} | {population['rows']} | {calm['count']} ({_format_fraction(calm['proportion'])}) | {no_reason['count']} ({_format_fraction(no_reason['proportion'])}) | {population['affected_rows']['count']} ({_format_fraction(population['affected_rows']['proportion'])}) | {calm_weight['mean']} / {_format_fraction(calm_weight['zero_fraction'])} | {no_reason_weight['mean']} / {_format_fraction(no_reason_weight['zero_fraction'])} |"
        )
    lines.extend(["", "Weight distributions include count, min/mean/percentiles/max, zero fraction, and exact value counts in `aggregate-report.json`.", ""])

    lines.extend(["## CALM co-occurrence", "", "| Other head | Co-occurring category counts |", "| --- | --- |"])
    for head, value in combined["calm_cooccurrence"]["other_six_heads"].items():
        if head == "reasoning_tags":
            categories = ", ".join(f"{key}={item['count']}" for key, item in value["tags"].items())
        else:
            categories = ", ".join(f"{key}={item['count']}" for key, item in value["categories"].items())
        lines.append(f"| `{head}` | {categories or 'none'} |")
    lines.extend(["", "## NO_REASON_GIVEN co-occurrence", "", "| Scalar head | Co-occurring category counts |", "| --- | --- |"])
    for head, value in combined["no_reason_given_cooccurrence"]["six_scalar_heads"].items():
        categories = ", ".join(f"{key}={item['count']}" for key, item in value["categories"].items())
        lines.append(f"| `{head}` | {categories or 'none'} |")
    lines.extend(["", "Other reasoning-tag counts:", ""])
    lines.append("; ".join(f"`{tag}`={value['count']}" for tag, value in combined["no_reason_given_cooccurrence"]["other_reasoning_tags"]["tags"].items()))

    lines.extend(["", "## CALM × NO_REASON_GIVEN and text length", "", "| Population | Both | CALM only | NO_REASON_GIVEN only | Neither |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in ("Train", "Dev", "TrainPlusDev"):
        cross = report["populations"][name]["cross_distribution"]
        lines.append(f"| {name} | {cross['CALM_and_NO_REASON_GIVEN']['count']} | {cross['CALM_only']['count']} | {cross['NO_REASON_GIVEN_only']['count']} | {cross['neither']['count']} |")
    lines.extend(["", "Affected rows are the union of either target label. Character length uses Python Unicode character count; no text is emitted.", ""])
    lines.extend(["| Population | Affected buckets | Remaining buckets |", "| --- | --- | --- |"])
    for name in ("Train", "Dev", "TrainPlusDev"):
        lengths = report["populations"][name]["text_character_length_buckets"]
        affected = ", ".join(f"{bucket}={value['count']}" for bucket, value in lengths["affected_rows"].items())
        remaining = ", ".join(f"{bucket}={value['count']}" for bucket, value in lengths["remaining_rows"].items())
        lines.append(f"| {name} | {affected} | {remaining} |")

    if "matching_seed_metrics" in report:
        matching = report["matching_seed_metrics"]
        targets = matching["targets"]
        has_s3 = any("S3" in value for value in next(iter(targets.values()))["per_seed"].values())
        if has_s3:
            lines.extend(["", "## Matching S1/S2/S3 seed metrics", "", "F1, support, and deltas are read from existing Dev metric JSON only; all comparisons remain weak-label diagnostics.", "", "| Trigger | Seed | S1 F1 / support | S2 F1 / support | S3 F1 / support | S3−S1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        else:
            lines.extend(["", "## Matching S1/S2 seed metrics", "", "F1, support, and deltas are read from existing Dev metric JSON only; all comparisons remain weak-label diagnostics.", "", "| Trigger | Seed | S1 F1 / support | S2 F1 / support | S2−S1 |", "| --- | ---: | ---: | ---: | ---: |"])
        for trigger, value in targets.items():
            for seed, seed_value in value["per_seed"].items():
                if has_s3:
                    lines.append(f"| `{trigger}` | {seed} | {seed_value['S1']['f1']:.6f} / {seed_value['S1']['support']} | {seed_value['S2']['f1']:.6f} / {seed_value['S2']['support']} | {seed_value['S3']['f1']:.6f} / {seed_value['S3']['support']} | {seed_value['delta_S3_minus_S1']:+.6f} |")
                else:
                    lines.append(f"| `{trigger}` | {seed} | {seed_value['S1']['f1']:.6f} / {seed_value['S1']['support']} | {seed_value['S2']['f1']:.6f} / {seed_value['S2']['support']} | {seed_value['delta_S2_minus_S1']:+.6f} |")
        if "head_macro_f1" in matching:
            lines.extend(["", "S3 head Macro-F1 matching-seed summary:", ""])
            for head, value in matching["head_macro_f1"].items():
                lines.append(f"- `{head}`: S1 mean `{value['S1_mean_macro_f1']:.6f}`, S3 mean `{value['S3_mean_macro_f1']:.6f}`, delta `{value['mean_delta_S3_minus_S1']:+.6f}`.")

    lines.extend(["", "## HYPOTHESIS (not conclusions)", ""])
    for hypothesis in report["hypotheses"]:
        lines.extend([f"### {hypothesis['id']} — `{hypothesis['disposition']}`", "", hypothesis["statement"], "", hypothesis["evidence"], ""])
    lines.extend(["## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    aggregate_path = root / "aggregate-report.json"
    summary_path = root / "summary.md"
    aggregate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(_markdown_summary(report), encoding="utf-8")
    return aggregate_path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="existing config that resolves the canonical Train/Dev package")
    parser.add_argument("--s1-metrics-root", required=True, help="existing S1 artifact root containing seed-*/seed-metrics.json")
    parser.add_argument("--s2-metrics-root", required=True, help="existing S2 artifact root containing seed-*/seed-metrics.json")
    parser.add_argument("--s3-metrics-root", help="optional S3 artifact root containing per-head seed metrics and s3-vs-s1-matching-seed-report.json")
    parser.add_argument("--output-dir", default="runs/m2-s3-trigger-data-diagnostics")
    args = parser.parse_args(argv)
    try:
        config = ProjectConfig.load(args.config)
        _, train, dev = load_m1_partitions(config)
        matching = load_matching_seed_metrics(args.s1_metrics_root, args.s2_metrics_root, args.s3_metrics_root)
        report = build_diagnostic(train, dev, matching)
        aggregate_path, summary_path = write_outputs(report, args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "M2_S3_TRIGGER_DATA_DIAGNOSTICS_BLOCKED", "error": exc.as_dict()}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"status": "M2_S3_TRIGGER_DATA_DIAGNOSTICS_COMPLETED", "rows": {"train": len(train), "dev": len(dev)}, "aggregate": str(aggregate_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
