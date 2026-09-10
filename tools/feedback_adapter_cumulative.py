#!/usr/bin/env python3
"""One-shot cumulative V1+V2 Human Train feedback adapter diagnostic.

This entry point reuses the fixed seven-head additive-bias adapter from
``feedback_adapter_v1``.  It fits exactly once on the disjoint 20-row
V1+V2 Human Train population, evaluates V1 Check, V2 Check, and merged
Check-8 separately, and measures influence on the original 300-row snapshot.
It never reads Test/Gold/OOD or any model/runtime artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import feedback_adapter_v1 as adapter


V1_RUN = REPO_ROOT / "runs" / "feedback-loop-v1-20260909"
V2_RUN = REPO_ROOT / "runs" / "feedback-loop-v2-20260909"
V1_ADAPTER = V1_RUN / "feedback-training-human10"
V2_ADAPTER = REPO_ROOT / "runs" / "feedback-training-human10-v2"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "feedback-training-human20-cumulative"


class CumulativeAdapterError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return adapter._read_jsonl(path)


def _write_json(path: Path, value: Any) -> None:
    adapter._write_json(path, value)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    adapter._write_jsonl(path, rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(adapter._canonical({key: item for key, item in value.items() if key != "content_address"})).hexdigest()


def _selection(path: Path, expected_rows: int, round_name: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != expected_rows:
        raise CumulativeAdapterError(f"{round_name} selection expected {expected_rows} rows, got {len(rows)}")
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids) or len(set(ids)) != expected_rows:
        raise CumulativeAdapterError(f"{round_name} selection has duplicate or invalid sample IDs")
    return [dict(row, feedback_round=round_name) for row in rows]


def _load_snapshot(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != 300:
        raise CumulativeAdapterError(f"prediction snapshot expected 300 rows, got {len(rows)}")
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids) or len(set(ids)) != 300:
        raise CumulativeAdapterError("prediction snapshot has duplicate or invalid sample IDs")
    return rows


def _load_answers(path: Path, expected_rows: int, round_name: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Any]]:
    rows = _read_jsonl(path)
    if len(rows) != expected_rows * len(adapter.HEADS):
        raise CumulativeAdapterError(f"{round_name} labels expected {expected_rows * len(adapter.HEADS)} rows, got {len(rows)}")
    if any(row.get("is_final") is not True for row in rows):
        raise CumulativeAdapterError(f"{round_name} labels contain non-final answers")
    keys = [(str(row.get("sample_id")), str(row.get("head"))) for row in rows]
    if len(set(keys)) != len(keys):
        raise CumulativeAdapterError(f"{round_name} labels contain duplicate sample/head keys")
    return rows, adapter._answers(rows)


def _merge_answers(*answer_maps: Mapping[tuple[str, str], Any]) -> dict[tuple[str, str], Any]:
    merged: dict[tuple[str, str], Any] = {}
    for values in answer_maps:
        overlap = set(merged).intersection(values)
        if overlap:
            raise CumulativeAdapterError(f"V1/V2 human answers overlap: {sorted(overlap)[:3]}")
        merged.update(values)
    return merged


def _validate_layout(
    v1_train: Sequence[Mapping[str, Any]],
    v1_check: Sequence[Mapping[str, Any]],
    v2_train: Sequence[Mapping[str, Any]],
    v2_check: Sequence[Mapping[str, Any]],
    answers: Mapping[tuple[str, str], Any],
    snapshot: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_sets = [{str(row["sample_id"]) for row in v1_train}, {str(row["sample_id"]) for row in v2_train}]
    check_sets = [{str(row["sample_id"]) for row in v1_check}, {str(row["sample_id"]) for row in v2_check}]
    train_ids = set().union(*train_sets)
    check_ids = set().union(*check_sets)
    if len(train_ids) != 20 or len(check_ids) != 8:
        raise CumulativeAdapterError(f"expected 20 Train and 8 Check IDs, got {len(train_ids)} and {len(check_ids)}")
    if train_sets[0] & train_sets[1] or check_sets[0] & check_sets[1] or train_ids & check_ids:
        raise CumulativeAdapterError("V1/V2 Train/Check identities are not disjoint")
    snapshot_ids = {str(row["sample_id"]) for row in snapshot}
    if not (train_ids | check_ids) <= snapshot_ids:
        raise CumulativeAdapterError("human selection is not covered by the 300-row snapshot")
    missing = [(sample_id, head) for sample_id in sorted(train_ids | check_ids) for head in adapter.HEADS if (sample_id, head) not in answers]
    if missing:
        raise CumulativeAdapterError(f"human labels missing sample/head: {missing[0]}")
    return {
        "train_rows": 20,
        "check_rows": 8,
        "train_sample_ids": sorted(train_ids),
        "check_sample_ids": sorted(check_ids),
        "v1_train_sample_ids": sorted(train_sets[0]),
        "v2_train_sample_ids": sorted(train_sets[1]),
        "v1_check_sample_ids": sorted(check_sets[0]),
        "v2_check_sample_ids": sorted(check_sets[1]),
    }


def _before_after_rows(
    rows: Sequence[Mapping[str, Any]],
    snapshot_by_id: Mapping[str, Mapping[str, Any]],
    answers: Mapping[tuple[str, str], Any],
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for selection_row in rows:
        sample_id = str(selection_row["sample_id"])
        before = snapshot_by_id[sample_id]
        after = adapter.apply_adapter(before, parameters)
        result.append({
            "sample_id": sample_id,
            "feedback_round": selection_row.get("feedback_round"),
            "model_text": selection_row.get("model_text"),
            "human_answers": {head: answers[(sample_id, head)] for head in adapter.HEADS},
            "before": before,
            "after": after,
        })
    return result


def _metrics_for_rows(
    rows: Sequence[Mapping[str, Any]],
    snapshot_by_id: Mapping[str, Mapping[str, Any]],
    answers: Mapping[tuple[str, str], Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    before = [snapshot_by_id[str(row["sample_id"])] for row in rows]
    after = [adapter.apply_adapter(row, parameters) for row in before]
    return {
        "before": adapter.metric_summary(before, answers),
        "after": adapter.metric_summary(after, answers),
        "rows": len(rows),
        "semantic_subset": adapter._check_population_info(rows),
    }


def _delta_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        head: {
            "loss_delta": metrics["after"][head]["before_or_after_nll_or_bce"] - metrics["before"][head]["before_or_after_nll_or_bce"],
            "strict_hit_rate_delta": metrics["after"][head]["strict_hit_rate"] - metrics["before"][head]["strict_hit_rate"],
            "target_probability_delta": metrics["after"][head]["target_probability_mean"] - metrics["before"][head]["target_probability_mean"],
        }
        for head in adapter.HEADS
    }


def _loss_delta(summary: Mapping[str, Any], head: str) -> float:
    return float(summary["after"][head]["before_or_after_nll_or_bce"] - summary["before"][head]["before_or_after_nll_or_bce"])


def _prior_summary(artifact_dir: Path) -> dict[str, Any]:
    aggregate_path = artifact_dir / "aggregate_metrics.json"
    manifest_path = artifact_dir / "manifest.json"
    if not aggregate_path.is_file() or not manifest_path.is_file():
        raise CumulativeAdapterError(f"prior adapter artifact missing: {artifact_dir}")
    return {
        "artifact_dir": str(artifact_dir),
        "content_address": json.loads(manifest_path.read_text(encoding="utf-8")).get("content_address"),
        "aggregate": json.loads(aggregate_path.read_text(encoding="utf-8")),
    }


def _consistency(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    before = []
    after = []
    for head in adapter.HEADS:
        before.append(abs(v1["before"][head]["strict_hit_rate"] - v2["before"][head]["strict_hit_rate"]))
        after.append(abs(v1["after"][head]["strict_hit_rate"] - v2["after"][head]["strict_hit_rate"]))
    return {
        "mean_absolute_strict_gap_before": sum(before) / len(before),
        "mean_absolute_strict_gap_after": sum(after) / len(after),
        "per_head_absolute_strict_gap_after": {head: abs(v1["after"][head]["strict_hit_rate"] - v2["after"][head]["strict_hit_rate"]) for head in adapter.HEADS},
    }


def _bias_view(drift: Mapping[str, Any], head: str, label: str) -> dict[str, Any]:
    values = drift["heads"][head]
    before = values["class_or_label_distribution_before"]
    after = values["class_or_label_distribution_after"]
    return {
        "label": label,
        "before_count": int(before.get(label, 0)),
        "after_count": int(after.get(label, 0)),
        "count_delta": int(after.get(label, 0)) - int(before.get(label, 0)),
        "prediction_change_rate": values["prediction_change_rate"],
        "mean_js_divergence": values["mean_js_divergence"],
    }


def run(
    *,
    v1_run: Path = V1_RUN,
    v2_run: Path = V2_RUN,
    snapshot_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output_dir.exists():
        raise CumulativeAdapterError(f"refusing to overwrite cumulative artifact: {output_dir}")
    v1_train = _selection(v1_run / "selection-train.jsonl", 10, "V1")
    v1_check = _selection(v1_run / "selection-check.jsonl", 4, "V1")
    v2_train = _selection(v2_run / "selection-train.jsonl", 10, "V2")
    v2_check = _selection(v2_run / "selection-check.jsonl", 4, "V2")
    _, v1_answers = _load_answers(v1_run / "human-labels-revisable.jsonl", 14, "V1")
    _, v2_answers = _load_answers(v2_run / "human-labels-revisable.jsonl", 14, "V2")
    answers = _merge_answers(v1_answers, v2_answers)
    snapshot_path = snapshot_path or (v1_run / "prediction-snapshot.jsonl")
    snapshot = _load_snapshot(snapshot_path)
    snapshot_by_id = {str(row["sample_id"]): row for row in snapshot}
    layout = _validate_layout(v1_train, v1_check, v2_train, v2_check, answers, snapshot)
    train_rows = [snapshot_by_id[str(row["sample_id"])] for row in [*v1_train, *v2_train]]
    parameters = adapter.fit_biases(train_rows, answers)
    all_after = [adapter.apply_adapter(row, parameters) for row in snapshot]
    train_metrics = {
        "before": adapter.metric_summary(train_rows, answers),
        "after": adapter.metric_summary([adapter.apply_adapter(row, parameters) for row in train_rows], answers),
        "rows": 20,
    }
    v1_check_metrics = _metrics_for_rows(v1_check, snapshot_by_id, answers, parameters)
    v2_check_metrics = _metrics_for_rows(v2_check, snapshot_by_id, answers, parameters)
    merged_check = [*v1_check, *v2_check]
    merged_check_metrics = _metrics_for_rows(merged_check, snapshot_by_id, answers, parameters)
    drift = adapter.drift_summary(snapshot, all_after)
    prior_v1 = _prior_summary(V1_ADAPTER)
    prior_v2 = _prior_summary(V2_ADAPTER)
    aggregate: dict[str, Any] = {
        "status": None,
        "diagnostic_only": True,
        "selected_candidate": False,
        "training_population": "V1_Human_Train_10_plus_V2_Human_Train_10_only",
        "check_population": "V1_Check_4_and_V2_Check_4_descriptive_only",
        "input_validation": layout | {"combined_human_rows": len(answers)},
        "human_train_20": {**train_metrics, "delta": _delta_metrics(train_metrics)},
        "human_check_v1_4": {**v1_check_metrics, "delta": _delta_metrics(v1_check_metrics)},
        "human_check_v2_4": {**v2_check_metrics, "delta": _delta_metrics(v2_check_metrics)},
        "human_check_merged_8": {**merged_check_metrics, "delta": _delta_metrics(merged_check_metrics)},
        "prediction_snapshot_300_drift": drift,
        "prior_single_batch_results": {"V1_only": prior_v1, "V2_only": prior_v2},
        "cross_check_consistency": _consistency(v1_check_metrics, v2_check_metrics),
        "bias_assessment": {
            "v1_train_label_counts": {head: _label_counts(v1_answers, {str(row["sample_id"]) for row in v1_train}, head) for head in adapter.HEADS},
            "v2_train_label_counts": {head: _label_counts(v2_answers, {str(row["sample_id"]) for row in v2_train}, head) for head in adapter.HEADS},
            "cumulative_train_label_counts": {head: _label_counts(answers, {str(row["sample_id"]) for row in [*v1_train, *v2_train]}, head) for head in adapter.HEADS},
            "stance": _bias_view(drift, "stance", "BEAR"),
            "context_dependency": _bias_view(drift, "context_dependency", "SELF_CONTAINED"),
            "prior_stance": {name: _bias_view(value["aggregate"]["prediction_snapshot_300_drift"], "stance", "BEAR") for name, value in (("V1_only", prior_v1), ("V2_only", prior_v2))},
            "prior_context_dependency": {name: _bias_view(value["aggregate"]["prediction_snapshot_300_drift"], "context_dependency", "SELF_CONTAINED") for name, value in (("V1_only", prior_v1), ("V2_only", prior_v2))},
        },
        "action_tendency": {
            "train_before_strict": train_metrics["before"]["action_tendency"]["strict_hit_rate"],
            "train_after_strict": train_metrics["after"]["action_tendency"]["strict_hit_rate"],
            "check_v1_before_after": [v1_check_metrics["before"]["action_tendency"]["strict_hit_rate"], v1_check_metrics["after"]["action_tendency"]["strict_hit_rate"]],
            "check_v2_before_after": [v2_check_metrics["before"]["action_tendency"]["strict_hit_rate"], v2_check_metrics["after"]["action_tendency"]["strict_hit_rate"]],
            "worsened_on_cumulative_train_strict": train_metrics["after"]["action_tendency"]["strict_hit_rate"] < train_metrics["before"]["action_tendency"]["strict_hit_rate"],
            "worsened_on_merged_check_strict": merged_check_metrics["after"]["action_tendency"]["strict_hit_rate"] < merged_check_metrics["before"]["action_tendency"]["strict_hit_rate"],
        },
        "emotion_primary": {
            "train_before_strict": train_metrics["before"]["emotion_primary"]["strict_hit_rate"],
            "train_after_strict": train_metrics["after"]["emotion_primary"]["strict_hit_rate"],
            "check_v1_before_after": [v1_check_metrics["before"]["emotion_primary"]["strict_hit_rate"], v1_check_metrics["after"]["emotion_primary"]["strict_hit_rate"]],
            "check_v2_before_after": [v2_check_metrics["before"]["emotion_primary"]["strict_hit_rate"], v2_check_metrics["after"]["emotion_primary"]["strict_hit_rate"]],
            "worsened_on_cumulative_train_strict": train_metrics["after"]["emotion_primary"]["strict_hit_rate"] < train_metrics["before"]["emotion_primary"]["strict_hit_rate"],
            "worsened_on_merged_check_strict": merged_check_metrics["after"]["emotion_primary"]["strict_hit_rate"] < merged_check_metrics["before"]["emotion_primary"]["strict_hit_rate"],
        },
        "forbidden_inputs_not_read": ["Test", "Anchor", "Gold", "OOD", "Human48", "Human100", "LLM", "cloud", "production"],
    }
    improved_heads = [
        head
        for head in adapter.HEADS
        if train_metrics["after"][head]["before_or_after_nll_or_bce"] < train_metrics["before"][head]["before_or_after_nll_or_bce"] - 1e-9
        and train_metrics["after"][head]["target_probability_mean"] >= train_metrics["before"][head]["target_probability_mean"] - 1e-9
    ]
    aggregate["status"] = "FEEDBACK_ABSORBED" if len(improved_heads) == len(adapter.HEADS) else ("PARTIALLY_ABSORBED" if improved_heads else "NOT_ABSORBED")
    aggregate["improved_heads"] = improved_heads
    consistency = aggregate["cross_check_consistency"]
    aggregate["cumulative_bias_reduction"] = {
        "cross_check_strict_gap_reduced": consistency["mean_absolute_strict_gap_after"] < consistency["mean_absolute_strict_gap_before"],
        "stance_change_rate": drift["heads"]["stance"]["prediction_change_rate"],
        "context_change_rate": drift["heads"]["context_dependency"]["prediction_change_rate"],
        "interpretation": "Measured descriptively; lower drift than a single batch is required before calling bias weakened.",
    }
    prior_stance_rates = {name: value["aggregate"]["prediction_snapshot_300_drift"]["heads"]["stance"]["prediction_change_rate"] for name, value in (("V1_only", prior_v1), ("V2_only", prior_v2))}
    prior_context_rates = {name: value["aggregate"]["prediction_snapshot_300_drift"]["heads"]["context_dependency"]["prediction_change_rate"] for name, value in (("V1_only", prior_v1), ("V2_only", prior_v2))}
    bias_weakened = aggregate["cumulative_bias_reduction"]["stance_change_rate"] < min(prior_stance_rates.values()) and aggregate["cumulative_bias_reduction"]["context_change_rate"] < min(prior_context_rates.values())
    aggregate["cumulative_bias_reduction"].update({
        "prior_stance_change_rates": prior_stance_rates,
        "prior_context_change_rates": prior_context_rates,
        "global_bias_weakened_vs_both_single_batches": bias_weakened,
        "adapter_route_recommendation": "CONTINUE_ONLY_WITH_NEWLY_COLLECTED_BALANCED_DATA" if bias_weakened and aggregate["cumulative_bias_reduction"]["cross_check_strict_gap_reduced"] else "STOP_GLOBAL_BIAS_ADAPTER_ROUTE_UNTIL_MORE_HUMAN_DATA",
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "adapter-parameters.json", parameters)
    _write_jsonl(output_dir / "human-train-before-after.jsonl", _before_after_rows([*v1_train, *v2_train], snapshot_by_id, answers, parameters))
    _write_jsonl(output_dir / "human-check-v1-before-after.jsonl", _before_after_rows(v1_check, snapshot_by_id, answers, parameters))
    _write_jsonl(output_dir / "human-check-v2-before-after.jsonl", _before_after_rows(v2_check, snapshot_by_id, answers, parameters))
    _write_jsonl(output_dir / "human-check-merged-before-after.jsonl", _before_after_rows(merged_check, snapshot_by_id, answers, parameters))
    _write_json(output_dir / "aggregate_metrics.json", aggregate)
    report_lines = [
        "# Feedback Loop — Cumulative V1+V2 Human Train 20",
        "",
        f"Status: `{aggregate['status']}` (diagnostic only; selected candidate unchanged).",
        "",
        "A single L2=1.0 convex additive-bias adapter was fit on the disjoint V1+V2 Human Train 20 only. It does not learn new semantic representations and cannot replace Encoder training.",
        "",
        "## Population and checks",
        "",
        "- Human Train: V1 10 + V2 10 = 20 distinct samples; the only fitting population.",
        "- Human Check: V1 4, V2 4, and merged Check 8 are reported separately and are descriptive only.",
        f"- V1 Check semantic subset: {v1_check_metrics['semantic_subset']['semantic_subset_n']}; V2 Check semantic subset: {v2_check_metrics['semantic_subset']['semantic_subset_n']}; merged Check semantic subset: {merged_check_metrics['semantic_subset']['semantic_subset_n']}.",
        "- Original 300-row snapshot is used for drift measurement only.",
        "",
        "## Single-batch comparison",
        "",
        f"- V1-only status: `{prior_v1['aggregate']['status']}`; V2-only status: `{prior_v2['aggregate']['status']}`. Their artifacts are retained and not modified.",
        "",
        "| head | V1-only loss Δ | V2-only loss Δ | cumulative loss Δ | V1-only strict Δ | V2-only strict Δ | cumulative strict Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for head in adapter.HEADS:
        v1_delta = prior_v1["aggregate"]["human_train"]["delta"][head]
        v2_delta = prior_v2["aggregate"]["human_train"]["delta"][head]
        cumulative_delta = aggregate["human_train_20"]["delta"][head]
        report_lines.append(f"| {head} | {v1_delta['loss_delta']:+.6f} | {v2_delta['loss_delta']:+.6f} | {cumulative_delta['loss_delta']:+.6f} | {v1_delta['strict_hit_rate_delta']:+.3f} | {v2_delta['strict_hit_rate_delta']:+.3f} | {cumulative_delta['strict_hit_rate_delta']:+.3f} |")
    report_lines += [
        "",
        "## Cumulative Train result",
        "",
        "| head | loss before | loss after | strict before | strict after | target probability Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for head in adapter.HEADS:
        before, after = train_metrics["before"][head], train_metrics["after"][head]
        report_lines.append(f"| {head} | {before['before_or_after_nll_or_bce']:.6f} | {after['before_or_after_nll_or_bce']:.6f} | {before['strict_hit_rate']:.3f} | {after['strict_hit_rate']:.3f} | {aggregate['human_train_20']['delta'][head]['target_probability_delta']:+.6f} |")
    report_lines += [
        "",
        "## Check loss delta comparison",
        "",
        "| head | V1-only Check loss Δ | V2-only Check loss Δ | cumulative V1 Check loss Δ | cumulative V2 Check loss Δ | merged Check8 loss Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for head in adapter.HEADS:
        report_lines.append(f"| {head} | {_loss_delta(prior_v1['aggregate']['human_check'], head):+.6f} | {_loss_delta(prior_v2['aggregate']['human_check'], head):+.6f} | {_delta_metrics(v1_check_metrics)[head]['loss_delta']:+.6f} | {_delta_metrics(v2_check_metrics)[head]['loss_delta']:+.6f} | {_delta_metrics(merged_check_metrics)[head]['loss_delta']:+.6f} |")
    report_lines += [
        "",
        "## Check strict hit rates",
        "",
        "| head | V1 Check before→after | V2 Check before→after | merged Check8 before→after |",
        "|---|---:|---:|---:|",
    ]
    for head in adapter.HEADS:
        report_lines.append(f"| {head} | {v1_check_metrics['before'][head]['strict_hit_rate']:.3f} → {v1_check_metrics['after'][head]['strict_hit_rate']:.3f} | {v2_check_metrics['before'][head]['strict_hit_rate']:.3f} → {v2_check_metrics['after'][head]['strict_hit_rate']:.3f} | {merged_check_metrics['before'][head]['strict_hit_rate']:.3f} → {merged_check_metrics['after'][head]['strict_hit_rate']:.3f} |")
    report_lines += [
        "",
        f"- Mean absolute V1/V2 Check strict gap: {consistency['mean_absolute_strict_gap_before']:.3f} → {consistency['mean_absolute_strict_gap_after']:.3f}; this small-sample change is descriptive only.",
        "",
        "## 300-row drift and bias",
        "",
        "| head | changed / 300 | change rate | mean JS | high-confidence flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for head in adapter.HEADS:
        value = drift["heads"][head]
        report_lines.append(f"| {head} | {value['prediction_change_count']} | {value['prediction_change_rate']:.3f} | {value['mean_js_divergence']:.6f} | {value['high_confidence_flip_count']} |")
    stance = aggregate["bias_assessment"]["stance"]
    context = aggregate["bias_assessment"]["context_dependency"]
    report_lines += [
        "",
        f"- Cumulative Train contains BEAR {aggregate['bias_assessment']['cumulative_train_label_counts']['stance'].get('BEAR', 0)}/20 and SELF_CONTAINED {aggregate['bias_assessment']['cumulative_train_label_counts']['context_dependency'].get('SELF_CONTAINED', 0)}/20; V1/V2 batch counts remain available as source evidence.",
        f"- Cumulative stance BEAR predictions: {stance['before_count']} → {stance['after_count']}; context SELF_CONTAINED predictions: {context['before_count']} → {context['after_count']}.",
        f"- Stance drift rates: V1-only {prior_stance_rates['V1_only']:.3f}, V2-only {prior_stance_rates['V2_only']:.3f}, cumulative {aggregate['cumulative_bias_reduction']['stance_change_rate']:.3f}; context drift rates: V1-only {prior_context_rates['V1_only']:.3f}, V2-only {prior_context_rates['V2_only']:.3f}, cumulative {aggregate['cumulative_bias_reduction']['context_change_rate']:.3f}.",
        f"- Global bias weakened versus both single batches: `{aggregate['cumulative_bias_reduction']['global_bias_weakened_vs_both_single_batches']}`. Adapter route: `{aggregate['cumulative_bias_reduction']['adapter_route_recommendation']}`.",
        "- These shifts are reported as possible global-prior effects, not as truth or quality gains.",
        "",
        "## Action and emotion checks",
        "",
        f"- action_tendency cumulative Train strict: {aggregate['action_tendency']['train_before_strict']:.3f} → {aggregate['action_tendency']['train_after_strict']:.3f}; V1 Check {aggregate['action_tendency']['check_v1_before_after'][0]:.3f} → {aggregate['action_tendency']['check_v1_before_after'][1]:.3f}; V2 Check {aggregate['action_tendency']['check_v2_before_after'][0]:.3f} → {aggregate['action_tendency']['check_v2_before_after'][1]:.3f}.",
        f"- emotion_primary cumulative Train strict: {aggregate['emotion_primary']['train_before_strict']:.3f} → {aggregate['emotion_primary']['train_after_strict']:.3f}; V1 Check {aggregate['emotion_primary']['check_v1_before_after'][0]:.3f} → {aggregate['emotion_primary']['check_v1_before_after'][1]:.3f}; V2 Check {aggregate['emotion_primary']['check_v2_before_after'][0]:.3f} → {aggregate['emotion_primary']['check_v2_before_after'][1]:.3f}.",
        "",
        "Cumulative evidence is diagnostic only. It does not authorize candidate replacement, production use, Test evaluation, or further adapter tuning.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "myresearcher.feedback-adapter-cumulative-v1-artifact.v1",
        "status": aggregate["status"],
        "diagnostic_only": True,
        "selected_candidate": False,
        "fit_once": True,
        "train_rows": 20,
        "check_rows": 8,
        "source_v1_run": str(v1_run),
        "source_v2_run": str(v2_run),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _sha256(snapshot_path),
        "parent_candidate_content_address": "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf",
        "files": {},
        "forbidden_inputs_not_read": aggregate["forbidden_inputs_not_read"],
    }
    for filename in ("adapter-parameters.json", "human-train-before-after.jsonl", "human-check-v1-before-after.jsonl", "human-check-v2-before-after.jsonl", "human-check-merged-before-after.jsonl", "aggregate_metrics.json", "REPORT.md"):
        manifest["files"][filename] = _sha256(output_dir / filename)
    manifest["content_address"] = _content_address(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    return {"status": aggregate["status"], "artifact_dir": str(output_dir), "content_address": manifest["content_address"], "train_rows": 20, "check_rows": 8, "snapshot_rows": 300, "improved_heads": improved_heads}


def _label_counts(answers: Mapping[tuple[str, str], Any], sample_ids: set[str], head: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample_id in sample_ids:
        value = answers.get((sample_id, head))
        if head == "reasoning_tags":
            for label in value or []:
                counts[str(label)] = counts.get(str(label), 0) + 1
        else:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-run", type=Path, default=V1_RUN)
    parser.add_argument("--v2-run", type=Path, default=V2_RUN)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run(v1_run=args.v1_run, v2_run=args.v2_run, snapshot_path=args.snapshot, output_dir=args.output_dir)
    except (CumulativeAdapterError, OSError, RuntimeError, ValueError) as exc:
        print(f"[feedback-adapter-cumulative] FAIL-CLOSED: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
