from __future__ import annotations

import numpy as np

from semantic_model import encoder_m5_human100_source_balanced as m5
from semantic_model import encoder_m1 as m1
from tools import feedback_adapter_v1 as adapter


def _record(tags: list[str]) -> m1.M1Record:
    labels = {head: adapter.CLASS_ORDER[head][0] for head in m5.V1_HEADS}
    labels["reasoning_tags"] = tags
    return m1.M1Record("synthetic", "000001", "stock", "语义样本文本", labels, {head: 1.0 for head in m5.V1_HEADS})


def test_source_balanced_objective_uses_equal_source_mass():
    assert m5.source_balanced_objective(2.0, 4.0) == 3.0
    assert m5.source_balanced_objective(0.0, 2.0) == 1.0


def test_reasoning_metrics_use_parent_thresholds_and_report_exact():
    records = [_record(["FUNDAMENTAL"]), _record([])]
    order = ["FUNDAMENTAL", "NO_REASON_GIVEN"]
    values = np.asarray([[0.60, 0.10], [0.55, 0.10]], dtype=float)
    metrics = m5._reasoning_metric(values, records, order, {"FUNDAMENTAL": 0.50, "NO_REASON_GIVEN": 0.50})
    assert metrics["exact_set_accuracy"] == 0.5
    assert metrics["per_label"]["FUNDAMENTAL"]["support"] == 1


def test_trend_is_positive_only_when_most_heads_improve():
    baseline = {head: {"macro_f1": 0.2} for head in m5.V1_HEADS}
    treatment = {head: {"macro_f1": 0.3} for head in m5.V1_HEADS}
    assert m5.classify_trend(baseline, treatment) == "POSITIVE"
    treatment["stance"] = {"macro_f1": 0.1}
    treatment["emotion_primary"] = {"macro_f1": 0.1}
    treatment["emotion_target"] = {"macro_f1": 0.1}
    assert m5.classify_trend(baseline, treatment) == "MIXED"


def test_obvious_check_regression_blocks_positive_trend():
    baseline = {head: {"macro_f1": 0.2} for head in m5.V1_HEADS}
    treatment = {head: {"macro_f1": 0.3} for head in m5.V1_HEADS}
    treatment["action_tendency"] = {"macro_f1": 0.14}
    assert m5.classify_trend(baseline, treatment) == "MIXED"


def test_forbidden_prediction_key_is_rejected_recursively():
    assert m5._forbidden_key({"answer": "BULL", "metadata": {"probabilities": {"BULL": 1.0}}}) == "probabilities"
    assert m5._forbidden_key({"answer": "BULL"}) is None
