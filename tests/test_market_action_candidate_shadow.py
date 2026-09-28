from tools.market_action_candidate_shadow import changed_extreme_dates, swap_action_prediction


def test_swap_action_only_preserves_other_heads_and_original() -> None:
    original = {"sample_id": "one", "predictions": {"stance": {"prediction": "BULL"},
                                                 "action_tendency": {"prediction": "SELL"}}}
    shadow = swap_action_prediction(original, ["BUY", "SELL"], [0.8, 0.2])
    assert shadow["predictions"]["action_tendency"]["prediction"] == "BUY"
    assert shadow["predictions"]["stance"] == original["predictions"]["stance"]
    assert original["predictions"]["action_tendency"]["prediction"] == "SELL"


def test_extreme_date_comparison_uses_only_action_values() -> None:
    def row(day: str, action: float) -> dict:
        return {"stock_code": "600000", "stock_name": "甲", "entry_date": day,
                "features": {"action_index": action}}

    original = [row("2026-09-01", -0.9), row("2026-09-02", -0.5), row("2026-09-03", 0.1)]
    candidate = [row("2026-09-01", -0.1), row("2026-09-02", -0.5), row("2026-09-03", -0.8)]
    result = changed_extreme_dates(original, candidate, count=1)
    assert result[0]["old_low_dates"] == ["2026-09-01"]
    assert result[0]["candidate_low_dates"] == ["2026-09-03"]
    assert result[0]["common_dates"] == []
