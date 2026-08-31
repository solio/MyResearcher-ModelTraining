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


def _read_seed_metric(root: Path, seed: int, *, stage: str) -> Mapping[str, Any]:
    path = root / f"seed-{seed}" / "seed-metrics.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError("M2_S3_METRICS_MISSING", "matching-seed metric file is missing", stage=stage, seed=seed, path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError("M2_S3_METRICS_INVALID", "matching-seed metric file is not valid JSON", stage=stage, seed=seed, path=str(path)) from exc
    _require(isinstance(document, Mapping), "M2_S3_METRICS_INVALID", "metric document must be an object", stage=stage, seed=seed)
    _require(document.get("sample_counts") == {"train": 1822, "dev": 448}, "M2_S3_METRICS_INVALID", "metric population must be Train 1822 / Dev 448", stage=stage, seed=seed)
    scope = str(document.get("metric_scope", ""))
    _require("WEAK_LABEL" in scope and "TEST" in scope and "PRODUCTION" in scope, "M2_S3_METRICS_SCOPE_INVALID", "S1/S2 metrics must declare weak-label, non-Test, non-production scope", stage=stage, seed=seed)
    _require(isinstance(document.get("dev"), Mapping), "M2_S3_METRICS_INVALID", "dev metric object is required", stage=stage, seed=seed)
    return document


def load_matching_seed_metrics(s1_root: str | Path, s2_root: str | Path) -> dict[str, Any]:
    """Read only S1/S2 Dev metric JSON for the two trigger labels."""

    s1_path, s2_path = Path(s1_root), Path(s2_root)
    values: dict[str, Any] = {}
    for label_key, head, metric_key in (
        ("emotion_primary:CALM", "emotion_primary", "per_class"),
        ("reasoning_tags:NO_REASON_GIVEN", "reasoning_tags", "per_label"),
    ):
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            s1 = _read_seed_metric(s1_path, seed, stage="S1")
            s2 = _read_seed_metric(s2_path, seed, stage="S2")
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
        values[label_key] = {
            "per_seed": per_seed,
            "S1_mean_f1": round(sum(item["S1"]["f1"] for item in per_seed.values()) / len(per_seed), 6),
            "S2_mean_f1": round(sum(item["S2"]["f1"] for item in per_seed.values()) / len(per_seed), 6),
            "mean_delta_S2_minus_S1": round(sum(item["delta_S2_minus_S1"] for item in per_seed.values()) / len(per_seed), 6),
            "interpretation": "weak-label Dev metric only; not Gold, truth, or model-selection evidence",
        }
    return values


def _hypotheses(combined: Mapping[str, Any]) -> list[dict[str, Any]]:
    calm = combined["targets"]["emotion_primary:CALM"]
    no_reason = combined["targets"]["reasoning_tags:NO_REASON_GIVEN"]
    affected = combined["affected_rows"]
    calm_weight = calm["affected_head_weight_distribution"]
    no_reason_weight = no_reason["affected_head_weight_distribution"]
    calm_zero_fraction = float(calm_weight["zero_fraction"] or 0.0)
    no_reason_zero_fraction = float(no_reason_weight["zero_fraction"] or 0.0)
    return [
        {
            "id": "HYPOTHESIS_LABEL_COUPLING",
            "statement": f"CALM and NO_REASON_GIVEN may form a correlated weak-label bundle: {affected['overlap_count']} rows overlap ({_proportion(affected['overlap_count'], combined['rows']):.1%} of all rows).",
            "supporting_S3_result": "An S3 CALM single-task and an S3 reasoning single-task run both improve the corresponding Dev weak-label F1, especially without broad regressions in the other heads.",
            "falsifying_S3_result": "Neither single-task result improves the trigger label, or the apparent gain disappears when the overlap rows are separated.",
        },
        {
            "id": "HYPOTHESIS_TEXT_LENGTH_SHIFT",
            "statement": "Rows affected by either trigger may have a different text-length mix from the remaining rows; this could make the two heads sensitive to truncation or sparse lexical evidence.",
            "supporting_S3_result": "S3 single-task gains concentrate in the affected length buckets identified in this report, with stable gains across the remaining buckets.",
            "falsifying_S3_result": "The S3 per-bucket result is flat or the same length pattern appears in unaffected rows.",
        },
        {
            "id": "HYPOTHESIS_AFFECTED_HEAD_WEIGHT",
            "statement": f"The affected-head weight distributions may under-emphasize these labels: CALM emotion-head zero-weight fraction is {calm_zero_fraction:.1%}, and NO_REASON_GIVEN reasoning-head zero-weight fraction is {no_reason_zero_fraction:.1%}.",
            "supporting_S3_result": "An S3 single-task run raises trigger-label recall/F1 in proportion to the low-weight buckets while preserving the fixed evaluation boundary.",
            "falsifying_S3_result": "Single-task training shows no trigger-label gain despite the observed weight distribution, or gains are unrelated to low-weight buckets.",
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
        "hypotheses": _hypotheses(populations["TrainPlusDev"]),
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
        lines.extend(["", "## Matching S1/S2 seed metrics", "", "F1, support, and deltas are read from existing Dev metric JSON only; all comparisons remain weak-label diagnostics.", "", "| Trigger | Seed | S1 F1 / support | S2 F1 / support | Delta (S2−S1) |", "| --- | ---: | ---: | ---: | ---: |"])
        for trigger, value in report["matching_seed_metrics"].items():
            for seed, seed_value in value["per_seed"].items():
                lines.append(f"| `{trigger}` | {seed} | {seed_value['S1']['f1']:.6f} / {seed_value['S1']['support']} | {seed_value['S2']['f1']:.6f} / {seed_value['S2']['support']} | {seed_value['delta_S2_minus_S1']:+.6f} |")

    lines.extend(["", "## HYPOTHESIS (not conclusions)", ""])
    for hypothesis in report["hypotheses"]:
        lines.extend([f"### {hypothesis['id']}", "", hypothesis["statement"], "", f"Support if S3 shows: {hypothesis['supporting_S3_result']}", "", f"Weaken/deny if S3 shows: {hypothesis['falsifying_S3_result']}", ""])
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
    parser.add_argument("--output-dir", default="runs/m2-s3-trigger-data-diagnostics")
    args = parser.parse_args(argv)
    try:
        config = ProjectConfig.load(args.config)
        _, train, dev = load_m1_partitions(config)
        matching = load_matching_seed_metrics(args.s1_metrics_root, args.s2_metrics_root)
        report = build_diagnostic(train, dev, matching)
        aggregate_path, summary_path = write_outputs(report, args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "M2_S3_TRIGGER_DATA_DIAGNOSTICS_BLOCKED", "error": exc.as_dict()}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"status": "M2_S3_TRIGGER_DATA_DIAGNOSTICS_COMPLETED", "rows": {"train": len(train), "dev": len(dev)}, "aggregate": str(aggregate_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
