from __future__ import annotations

import json

import pytest

from tools.m3e_shadow500_route import (
    EXPECTED_COUNTS,
    THRESHOLD,
    route,
    route_prediction_row,
    route_rows,
)


def _row(sample_id: str, eligibility: str, score: float) -> dict:
    return {
        "sample_id": sample_id,
        "stock_code": "000001",
        "stock_name": "Synthetic",
        "model_eligibility": eligibility,
        "text_quality_tier": "B_TITLE_RICH",
        "quality_flags": "",
        "six_head_mean_max_probability": score,
        "tokenizer_unk_ratio": 0.0,
        "abstain": score < THRESHOLD,
        "predictions": {"target_mode": {"prediction": "X", "probabilities": {"X": 1.0}}},
    }


def test_route_precedence_excludes_semantic_before_low_confidence():
    assert route("EXCLUDE_SEMANTIC", 0.1) == "INPUT_EXCLUDED"
    assert route("CHALLENGE_REVIEW", THRESHOLD) == "REVIEW_RECOMMENDED"
    assert route("ELIGIBLE", THRESHOLD) == "MODEL_OUTPUT"
    assert route("ELIGIBLE", THRESHOLD - 1e-12) == "LOW_CONFIDENCE_ABSTAIN"


def test_excluded_input_suppresses_normal_predictions_and_never_becomes_unknown():
    result = route_prediction_row(_row("excluded", "EXCLUDE_SEMANTIC", 0.1))
    assert result["route"] == "INPUT_EXCLUDED"
    assert result["prediction_available"] is False
    assert "predictions" not in result
    assert result["downstream_action"] == "drop_before_normal_seven_head_output"
    assert "UNKNOWN" not in json.dumps(result)


def test_review_recommended_retains_predictions_but_is_distinct_from_model_output():
    review = route_prediction_row(_row("review", "CHALLENGE_REVIEW", THRESHOLD))
    output = route_prediction_row(_row("output", "ELIGIBLE", THRESHOLD))
    assert review["route"] == "REVIEW_RECOMMENDED"
    assert output["route"] == "MODEL_OUTPUT"
    assert review["prediction_available"] is True
    assert output["prediction_available"] is True
    assert review["downstream_action"] != output["downstream_action"]
    assert "predictions" in review and "predictions" in output


def test_routes_are_mutually_exclusive_and_exhaustive():
    rows = [
        _row("a", "EXCLUDE_SEMANTIC", 0.9),
        _row("b", "ELIGIBLE", THRESHOLD - 0.01),
        _row("c", "CHALLENGE_REVIEW", THRESHOLD),
        _row("d", "ELIGIBLE", THRESHOLD),
    ]
    routed = route_rows(rows)
    assert [row["route"] for row in routed] == [
        "INPUT_EXCLUDED",
        "LOW_CONFIDENCE_ABSTAIN",
        "REVIEW_RECOMMENDED",
        "MODEL_OUTPUT",
    ]
    assert len({row["sample_id"] for row in routed}) == len(routed)


def test_unknown_eligibility_fails_closed():
    with pytest.raises(ValueError, match="unsupported model_eligibility"):
        route("UNEXPECTED", THRESHOLD)


def test_expected_counts_are_frozen():
    assert EXPECTED_COUNTS == {
        "INPUT_EXCLUDED": 8,
        "LOW_CONFIDENCE_ABSTAIN": 27,
        "REVIEW_RECOMMENDED": 158,
        "MODEL_OUTPUT": 307,
    }
