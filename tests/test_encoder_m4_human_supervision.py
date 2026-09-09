from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_model import encoder_m4_human_supervision as m4
from semantic_model.errors import ContractError
from semantic_model.schema import LabelSchema


def test_m4_training_scope_excludes_context_and_reasoning():
    assert m4.TRAINED_HEADS == (
        "target_mode",
        "stance",
        "emotion_primary",
        "emotion_target",
        "action_tendency",
    )
    assert m4.REUSED_HEADS == ("context_dependency", "reasoning_tags")


def test_human_gate_requires_96_total_three_heads_and_no_large_drop():
    baseline = {head: {"correct_count": value} for head, value in zip(m4.TRAINED_HEADS, [20, 18, 17, 16, 15])}
    treatment = {head: {"correct_count": value} for head, value in zip(m4.TRAINED_HEADS, [22, 20, 19, 18, 17])}
    result = m4.human_gate(baseline, treatment)
    assert result["baseline_total"] == 86
    assert result["treatment_total"] == 96
    assert result["passed"] is True


@pytest.mark.parametrize(
    "counts",
    [
        [22, 20, 19, 18, 16],  # 95 total
        [22, 20, 17, 16, 17],  # only two heads improve
        [22, 20, 19, 12, 17],  # one head drops by four
    ],
)
def test_human_gate_rejects_negative_cases(counts):
    baseline = {head: {"correct_count": value} for head, value in zip(m4.TRAINED_HEADS, [20, 18, 17, 16, 15])}
    treatment = {head: {"correct_count": value} for head, value in zip(m4.TRAINED_HEADS, counts)}
    assert m4.human_gate(baseline, treatment)["passed"] is False


def test_strict_metrics_reports_supported_and_schema_all():
    probabilities = np.asarray([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1]])
    metrics = m4.strict_metrics(probabilities, ["A", "B"], ["A", "B", "C"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1_supported"] == 1.0
    assert metrics["macro_f1_schema_all"] < 1.0


def test_human_record_uses_only_five_labels_and_inert_placeholders():
    schema = LabelSchema.load("schema/semantic-schema-calibrated-v0.2.1.json")
    row = {"sample_id": "H1", "stock_code": "000001", "stock_name": "测试", "model_text": "文本"}
    for head in m4.TRAINED_HEADS:
        row[f"human_{head}"] = schema.class_order[head][0]
    row["human_context_dependency"] = "TECHNICAL_PRICE"
    row["human_reasoning_tags"] = ""
    record = m4.human_record(row, schema)
    assert record.label["context_dependency"] == "SELF_CONTAINED"
    assert record.label["reasoning_tags"] == ["FUNDAMENTAL"]
    assert all(record.weights[head] == 1.0 for head in m4.V1_HEADS)


def test_review_export_requires_full_final_350_rows(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text(json.dumps({"batch": "m3c-shadow500-review50-20260908", "sample_id": "x", "head": "stance", "is_final": True}) + "\n", encoding="utf-8")
    with pytest.raises(ContractError) as error:
        m4.read_review_export(path, tmp_path / "shadow.jsonl", LabelSchema.load("schema/semantic-schema-calibrated-v0.2.1.json"))
    assert error.value.code == "M4_REVIEW_EXPORT_INCOMPLETE"


def test_no_test_or_production_inputs_are_declared_for_fit():
    source = open(m4.__file__, encoding="utf-8").read()
    assert "human48_diagnostic" in source
    assert "Test" in source and "production" in source
    assert "reference_predictions" not in source
