from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_model.encoder_m1 import M1Record
from semantic_model.errors import ContractError
from tools.m2_s3_trigger_data_diagnostics import (
    NO_REASON_GIVEN,
    build_diagnostic,
    load_matching_seed_metrics,
    write_outputs,
)


SCALAR_DEFAULTS = {
    "target_mode": "ON_TARGET",
    "stance": "BULL",
    "emotion_target": "PRICE",
    "action_tendency": "HOLD",
    "context_dependency": "SELF_CONTAINED",
}
ALL_HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "reasoning_tags",
    "context_dependency",
)


def record(
    sample_id: str,
    *,
    emotion: str = "NONE_EXPLICIT",
    tags: list[str] | None = None,
    text_length: int = 10,
    weight: float = 1.0,
    overrides: dict[str, str] | None = None,
) -> M1Record:
    label = {**SCALAR_DEFAULTS, "emotion_primary": emotion, "reasoning_tags": tags or []}
    label.update(overrides or {})
    weights = {head: weight for head in ALL_HEADS}
    return M1Record(
        sample_id=sample_id,
        stock_code="601012",
        stock_name="隆基绿能",
        model_text="x" * text_length,
        label=label,
        weights=weights,
    )


def test_counts_weights_cross_and_lengths_are_aggregate_only():
    train = [
        record("T1", emotion="CALM", tags=[NO_REASON_GIVEN], text_length=10, weight=0.0),
        record("T2", emotion="CALM", tags=["FUNDAMENTAL"], text_length=60, weight=0.5),
        record("T3", tags=[NO_REASON_GIVEN, "RUMOR"], text_length=500, weight=1.0),
    ]
    dev = [
        record("D1", emotion="CALM", tags=[], text_length=25, weight=0.25),
        record("D2", tags=["VALUATION"], text_length=100, weight=1.0),
    ]
    report = build_diagnostic(train, dev)
    combined = report["populations"]["TrainPlusDev"]
    calm = combined["targets"]["emotion_primary:CALM"]
    no_reason = combined["targets"]["reasoning_tags:NO_REASON_GIVEN"]
    assert combined["rows"] == 5
    assert calm["count"] == 3 and calm["proportion"] == 0.6
    assert no_reason["count"] == 2 and no_reason["proportion"] == 0.4
    assert calm["affected_head_weight_distribution"]["value_counts"] == {"0.0": 1, "0.25": 1, "0.5": 1}
    assert no_reason["affected_head_weight_distribution"]["zero_fraction"] == 0.5
    assert combined["affected_rows"] == {"count": 4, "proportion": 0.8, "overlap_count": 1}
    assert combined["cross_distribution"]["CALM_and_NO_REASON_GIVEN"]["count"] == 1
    assert combined["cross_distribution"]["CALM_only"]["count"] == 2
    assert combined["cross_distribution"]["NO_REASON_GIVEN_only"]["count"] == 1
    assert combined["cross_distribution"]["neither"]["count"] == 1
    assert combined["text_character_length_buckets"]["affected_rows"]["0-19"]["count"] == 1
    assert combined["text_character_length_buckets"]["remaining_rows"]["100-199"]["count"] == 1


def test_cooccurrence_covers_six_other_heads_and_all_reasoning_tags():
    rows = [
        record("A", emotion="CALM", tags=[NO_REASON_GIVEN, "FUNDAMENTAL"], overrides={"stance": "BEAR"}),
        record("B", emotion="CALM", tags=["FUNDAMENTAL"], overrides={"stance": "BEAR"}),
    ]
    report = build_diagnostic(rows, [record("C", tags=[NO_REASON_GIVEN, "RUMOR"])])
    calm = report["populations"]["TrainPlusDev"]["calm_cooccurrence"]["other_six_heads"]
    assert set(calm) == {"target_mode", "stance", "emotion_target", "action_tendency", "context_dependency", "reasoning_tags"}
    assert calm["stance"]["categories"]["BEAR"]["count"] == 2
    assert calm["reasoning_tags"]["tags"]["FUNDAMENTAL"]["count"] == 2
    assert calm["reasoning_tags"]["tags"][NO_REASON_GIVEN]["count"] == 1
    no_reason = report["populations"]["TrainPlusDev"]["no_reason_given_cooccurrence"]
    assert set(no_reason["six_scalar_heads"]) == set(ALL_HEADS) - {"reasoning_tags"}
    assert no_reason["other_reasoning_tags"]["tags"]["RUMOR"]["count"] == 1
    assert NO_REASON_GIVEN not in no_reason["other_reasoning_tags"]["tags"]


def test_matching_seed_metrics_reports_f1_support_and_delta(tmp_path: Path):
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    for seed in (35, 71, 107):
        for root, value in ((s1, 0.2), (s2, 0.3)):
            payload = {
                "sample_counts": {"train": 1822, "dev": 448},
                "metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION",
                "dev": {
                    "emotion_primary": {"per_class": {"CALM": {"f1": value, "support": 48}}},
                    "reasoning_tags": {"per_label": {NO_REASON_GIVEN: {"f1": value, "support": 86}}},
                },
            }
            path = root / f"seed-{seed}" / "seed-metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_matching_seed_metrics(s1, s2)
    assert result["targets"]["emotion_primary:CALM"]["per_seed"]["35"] == {
        "S1": {"f1": 0.2, "support": 48},
        "S2": {"f1": 0.3, "support": 48},
        "delta_S2_minus_S1": 0.1,
    }
    assert result["targets"]["reasoning_tags:NO_REASON_GIVEN"]["per_seed"]["107"]["S1"]["support"] == 86


def test_optional_s3_metrics_read_seed_files_and_matching_report(tmp_path: Path):
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s3 = tmp_path / "s3"
    for seed in (35, 71, 107):
        for root, value in ((s1, 0.2), (s2, 0.3)):
            payload = {
                "sample_counts": {"train": 1822, "dev": 448},
                "metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION",
                "dev": {
                    "emotion_primary": {"per_class": {"CALM": {"f1": value, "support": 48}}},
                    "reasoning_tags": {"per_label": {NO_REASON_GIVEN: {"f1": value, "support": 86}}},
                },
            }
            path = root / f"seed-{seed}" / "seed-metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        for head, label, f1, support in (("emotion_primary", "CALM", 0.4, 48), ("reasoning_tags", NO_REASON_GIVEN, 0.5, 86)):
            metric_key = "per_class" if head == "emotion_primary" else "per_label"
            payload = {
                "sample_counts": {"train": 1822, "dev": 448},
                "metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION",
                "dev": {head: {metric_key: {label: {"f1": f1, "support": support}}}},
            }
            path = s3 / head / f"seed-{seed}" / "seed-metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
    matching_report = {
        "stage_id": "M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL",
        "comparator": "S1_FROZEN_SHARED_MATCHING_SEED",
        "triggered_heads": ["emotion_primary", "reasoning_tags"],
        "selected_candidate": False,
        "stability_gate_passed": True,
        "critical_labels": {
            "emotion_primary": {"CALM": {"s1_f1_per_seed": [0.2] * 3, "s3_f1_per_seed": [0.4] * 3, "delta_per_seed": [0.2] * 3, "support_per_seed": [48] * 3}},
            "reasoning_tags": {NO_REASON_GIVEN: {"s1_f1_per_seed": [0.2] * 3, "s3_f1_per_seed": [0.5] * 3, "delta_per_seed": [0.3] * 3, "support_per_seed": [86] * 3}},
        },
        "per_head": {
            "emotion_primary": {"s1_mean": 0.2, "s3_mean": 0.4, "mean_delta": 0.2, "delta_per_seed": [0.2] * 3, "worst_seed_delta": 0.2},
            "reasoning_tags": {"s1_mean": 0.2, "s3_mean": 0.5, "mean_delta": 0.3, "delta_per_seed": [0.3] * 3, "worst_seed_delta": 0.3},
        },
    }
    (s3 / "s3-vs-s1-matching-seed-report.json").write_text(json.dumps(matching_report), encoding="utf-8")
    result = load_matching_seed_metrics(s1, s2, s3)
    target = result["targets"]["emotion_primary:CALM"]["per_seed"]["35"]
    assert target["S3"] == {"f1": 0.4, "support": 48}
    assert target["delta_S3_minus_S1"] == 0.2
    assert result["head_macro_f1"]["emotion_primary"]["mean_delta_S3_minus_S1"] == 0.2


def test_s3_evidence_updates_hypothesis_dispositions():
    metrics = {
        "head_macro_f1": {
            "emotion_primary": {"mean_delta_S3_minus_S1": 0.021769},
            "reasoning_tags": {"mean_delta_S3_minus_S1": -0.004081},
        },
        "s3_matching_report": {"stage_id": "M2-S3-FROZEN-SINGLE-TASK-HEAD-CONTROL"},
    }
    report = build_diagnostic([record("T1", emotion="CALM", tags=[NO_REASON_GIVEN])], [record("D1")], metrics)
    dispositions = {item["id"]: item["disposition"] for item in report["hypotheses"]}
    assert dispositions == {
        "HYPOTHESIS_LABEL_COUPLING": "NOT_SUPPORTED_AS_A_SHARED_EXPLANATION",
        "HYPOTHESIS_TEXT_LENGTH_SHIFT": "UNRESOLVED",
        "HYPOTHESIS_AFFECTED_HEAD_WEIGHT": "UNRESOLVED",
    }


def test_missing_matching_metric_fails_closed(tmp_path: Path):
    with pytest.raises(ContractError) as error:
        load_matching_seed_metrics(tmp_path / "missing-s1", tmp_path / "missing-s2")
    assert error.value.code == "M2_S3_METRICS_MISSING"


def test_write_outputs_contains_no_original_text(tmp_path: Path):
    report = build_diagnostic([record("T1", emotion="CALM", text_length=999)], [record("D1")])
    aggregate, summary = write_outputs(report, tmp_path / "output")
    assert aggregate.name == "aggregate-report.json"
    assert summary.name == "summary.md"
    assert "x" * 100 not in aggregate.read_text(encoding="utf-8")
    assert "x" * 100 not in summary.read_text(encoding="utf-8")
    assert {path.name for path in aggregate.parent.iterdir()} == {"aggregate-report.json", "summary.md"}


def test_duplicate_sample_ids_fail_closed():
    with pytest.raises(ContractError) as error:
        build_diagnostic([record("same"), record("same")], [record("dev")])
    assert error.value.code == "M2_S3_DATA_INVALID"


def test_hypotheses_are_explicitly_bounded():
    report = build_diagnostic([record("T1", emotion="CALM", tags=[NO_REASON_GIVEN])], [record("D1")])
    assert len(report["hypotheses"]) <= 3
    assert all(item["id"].startswith("HYPOTHESIS_") for item in report["hypotheses"])
    assert all(item["disposition"] == "PENDING_S3_METRICS" for item in report["hypotheses"])
