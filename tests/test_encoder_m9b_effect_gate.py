from __future__ import annotations

from semantic_model.encoder_m9b_effect_gate import (
    M9A_COUNTS,
    active_shadow_rows,
    discourse_verdict,
    validate_m9a_assignments,
)


def _m9a_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head, count in M9A_COUNTS.items():
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"fresh-{head}-{index}",
                    "head": head,
                    "is_final": True,
                    "disposition": None,
                    "answer": "UNKNOWN" if head not in {"market_scope", "discourse_focus"} else ["UNKNOWN"],
                }
            )
    return rows


def test_m9a_pull_contract_is_exactly_50_final_unique_pairs():
    result = validate_m9a_assignments(_m9a_rows())
    assert result["assignment_count"] == 50
    assert result["sample_count"] == 50
    assert result["head_counts"] == M9A_COUNTS


def test_discourse_gate_uses_fixed_micro_and_empty_thresholds():
    assert discourse_verdict({"metrics": {"micro_f1": 0.151}, "prediction_diagnostics": {"empty_prediction_rate": 0.799}}) == "WEAK_SIGNAL_CONTINUE"
    assert discourse_verdict({"metrics": {"micro_f1": 0.15}, "prediction_diagnostics": {"empty_prediction_rate": 0.2}}) == "CURRENT_LINEAR_HEAD_FAILED_REDESIGN_REQUIRED"
    assert discourse_verdict({"metrics": {"micro_f1": 0.4}, "prediction_diagnostics": {"empty_prediction_rate": 0.8}}) == "CURRENT_LINEAR_HEAD_FAILED_REDESIGN_REQUIRED"


def test_active_shadow_removes_only_m9a_pairs():
    rows = [
        {
            "sample_id": "fresh-1",
            "applicable_factors": ["market_scope", "discourse_focus"],
            "auxiliary": {"market_scope": {"prediction": []}, "discourse_focus": {"prediction": []}},
        },
        {
            "sample_id": "fresh-2",
            "applicable_factors": ["expectation_phase"],
            "auxiliary": {"expectation_phase": {"prediction": ["ANTICIPATED"]}},
        },
    ]
    active = active_shadow_rows(rows, {("fresh-1", "market_scope")})
    assert active[0]["applicable_factors"] == ["discourse_focus"]
    assert "market_scope" not in active[0]["auxiliary"]
    assert active[1] == rows[1]

