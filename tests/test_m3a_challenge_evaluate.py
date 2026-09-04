from __future__ import annotations

from tools.m3a_challenge_evaluate import clear_ood_confidence, ood_auc, summarize
from semantic_model.encoder_m2_selected_runtime import SCALAR_HEADS


def _row(scope: str, category: str, confidence: float, unk: float) -> dict:
    probabilities = {head: {"HIGH": confidence, "LOW": 1.0 - confidence} for head in SCALAR_HEADS}
    return {
        "sample_id": f"{scope}-{category}-{confidence}-{unk}",
        "expected_scope": scope,
        "challenge_category": category,
        "predictions": {head: {"probabilities": probabilities[head]} for head in SCALAR_HEADS},
        "diagnostics": {
            "scalar_head_max_probability": {head: confidence for head in SCALAR_HEADS},
            "six_head_mean_max_probability": confidence,
            "ood_score_one_minus_mean_max_probability": 1.0 - confidence,
            "tokenizer_unk_ratio": unk,
        },
    }


def test_clear_ood_confidence_reports_any_and_mean_thresholds():
    rows = [_row("CLEAR_OOD", "CLEAR_OOD_NOISE", 0.95, 0.2), _row("CLEAR_OOD", "CLEAR_OOD_NOISE", 0.75, 0.3)]
    result = clear_ood_confidence(rows)
    assert result["any_scalar_head_at_least_0_90_ratio"] == 0.5
    assert result["six_head_mean_at_least_0_80_ratio"] == 0.5
    assert result["both_conditions_ratio"] == 0.5


def test_auc_uses_only_in_domain_and_clear_ood():
    rows = [_row("IN_DOMAIN_CONTROL", "IN_DOMAIN_CONTROL", 0.95, 0.0), _row("IN_DOMAIN_CONTROL", "IN_DOMAIN_CONTROL", 0.90, 0.1), _row("CLEAR_OOD", "CLEAR_OOD_NOISE", 0.20, 0.8), _row("CLEAR_OOD", "CLEAR_OOD_NOISE", 0.10, 0.9), _row("SHIFT_CHALLENGE", "SHIFT", 0.01, 1.0)]
    auc = ood_auc(rows)
    assert auc["rows"] == 4
    assert auc["score_one_minus_six_head_mean_max_probability"] == 1.0
    assert auc["score_tokenizer_unk_ratio"] == 1.0


def test_summary_marks_separable_only_when_a_declared_auc_reaches_point_75():
    rows = [_row("IN_DOMAIN_CONTROL", "IN_DOMAIN_CONTROL", 0.95, 0.0), _row("CLEAR_OOD", "CLEAR_OOD_NOISE", 0.20, 0.8)]
    result = summarize(rows)
    assert result["status"] == "M3_CHALLENGE_SEPARABLE_TREND"
    assert result["shift_challenge_description"] == {}
    assert result["production_ood_claim"] is False

