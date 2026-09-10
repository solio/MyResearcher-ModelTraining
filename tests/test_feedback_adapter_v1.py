from __future__ import annotations

import numpy as np
import pytest

from tools import feedback_adapter_v1 as adapter


def _prediction(sample_id: str = "s1") -> dict:
    outputs = {}
    for head, order in adapter.CLASS_ORDER.items():
        if head == "reasoning_tags":
            values = {label: 0.2 for label in order}
            values[order[0]] = 0.8
            outputs[head] = {"probabilities": values, "prediction": [order[0]], "predicted_labels": [order[0]], "thresholds": {label: 0.5 for label in order}}
        else:
            values = {label: (0.7 if index == 0 else 0.3 / (len(order) - 1)) for index, label in enumerate(order)}
            outputs[head] = {"probabilities": values, "prediction": order[0]}
    return {"sample_id": sample_id, "model_text": "synthetic semantic text", "predictions": outputs}


def _answers(sample_id: str = "s1") -> dict[tuple[str, str], object]:
    result = {(sample_id, head): adapter.CLASS_ORDER[head][1] for head in adapter.SCALAR_HEADS}
    result[(sample_id, "reasoning_tags")] = [adapter.CLASS_ORDER["reasoning_tags"][1]]
    return result


def test_single_convex_fit_touches_all_seven_heads_and_reduces_training_loss():
    row = _prediction()
    answers = _answers()
    params = adapter.fit_biases([row], answers)
    assert params["single_convex_optimization"] is True
    assert set(params["biases"]) == set(adapter.HEADS)
    assert all(np.asarray(params["biases"][head]).shape == (len(adapter.CLASS_ORDER[head]),) for head in adapter.HEADS)
    before = adapter.metric_summary([row], answers)
    after = adapter.metric_summary([adapter.apply_adapter(row, params)], answers)
    assert all(after[head]["before_or_after_nll_or_bce"] < before[head]["before_or_after_nll_or_bce"] for head in adapter.HEADS)


def test_apply_adapter_preserves_reasoning_thresholds_and_probability_shapes():
    row = _prediction()
    params = adapter.fit_biases([row], _answers())
    after = adapter.apply_adapter(row, params)
    for head in adapter.SCALAR_HEADS:
        probabilities = after["predictions"][head]["probabilities"]
        assert list(probabilities) == adapter.CLASS_ORDER[head]
        assert sum(probabilities.values()) == pytest.approx(1.0)
    reasoning = after["predictions"]["reasoning_tags"]
    assert list(reasoning["probabilities"]) == adapter.CLASS_ORDER["reasoning_tags"]
    assert reasoning["thresholds"]["FUNDAMENTAL"] == 0.5
    assert all(0.0 <= value <= 1.0 for value in reasoning["probabilities"].values())


def test_check_population_info_counts_semantic_rows_dynamically():
    rows = [
        {"sample_id": "low-1", "model_text": "……"},
        {"sample_id": "semantic-1", "model_text": "抄底"},
        {"sample_id": "semantic-2", "model_text": "看好"},
        {"sample_id": "semantic-3", "model_text": "明天涨停板"},
        {"sample_id": "semantic-4", "model_text": "已清仓，随便吧"},
    ]
    info = adapter._check_population_info(rows)
    assert info["low_information_inputs"] == ["low-1"]
    assert info["semantic_subset_n"] == 4
