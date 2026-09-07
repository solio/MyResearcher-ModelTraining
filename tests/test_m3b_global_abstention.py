from __future__ import annotations

import pytest

from tools.m3b_global_abstention import challenge_metrics, freeze_dev_threshold, six_head_mean_max_probability
from semantic_model.encoder_m2_selected_runtime import SCALAR_HEADS


def _row(scope: str, score: float, category: str | None = None) -> dict:
    category = category or scope
    probabilities = {head: {"top": score, "other": 1.0 - score} for head in SCALAR_HEADS}
    return {
        "sample_id": f"{scope}-{score}-{category}",
        "expected_scope": scope,
        "challenge_category": category,
        "predictions": {head: {"probabilities": probabilities[head], "prediction": "top"} for head in SCALAR_HEADS},
    }


def test_dev_threshold_uses_lower_percentile_and_strict_less_than_rule():
    result = freeze_dev_threshold([0.1, 0.2, 0.2, 0.4, 0.9])
    assert result["method"] == "lower"
    assert result["threshold"] == 0.1
    assert result["dev_abstain_count"] == 0
    assert six_head_mean_max_probability(_row("IN_DOMAIN_CONTROL", 0.7)) == pytest.approx(0.7)


def test_challenge_rates_and_highest_released_clear_ood_are_computed_without_labels():
    rows = [
        _row("IN_DOMAIN_CONTROL", 0.7),
        _row("IN_DOMAIN_CONTROL", 0.8),
        _row("CLEAR_OOD", 0.6, "CLEAR_OOD_NOISE"),
        _row("CLEAR_OOD", 0.8, "CLEAR_OOD_NOISE"),
        _row("SHIFT_CHALLENGE", 0.6, "NEW_ENTITY_INDUSTRY"),
        _row("SHIFT_CHALLENGE", 0.8, "NEW_ENTITY_INDUSTRY"),
    ]
    result = challenge_metrics(rows, 0.7)
    assert result["clear_ood"]["recall"] == 0.5
    assert result["in_domain_control"]["false_rejection_rate"] == 0.0
    assert result["shift_challenge"]["NEW_ENTITY_INDUSTRY"]["abstention_rate"] == 0.5
    assert result["highest_confidence_released_clear_ood"]["score"] == pytest.approx(0.8)
