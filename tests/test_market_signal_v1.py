from __future__ import annotations

import pytest

from tools import market_signal_v1 as signal


def _prediction(sample_id: str, bull: float, bear: float):
    return {
        "sample_id": sample_id,
        "predictions": {
            "stance": {"probabilities": {"BULL": bull, "BEAR": bear}},
            "action_tendency": {
                "probabilities": {
                    "BUY": bull,
                    "ADD": 0.0,
                    "SELL": bear,
                    "REDUCE": 0.0,
                }
            },
            "target_mode": {"probabilities": {"ON_TARGET": 1.0}},
            "emotion_primary": {"probabilities": {"CALM": 1.0}},
            "emotion_target": {"probabilities": {"PRICE": 1.0}},
            "context_dependency": {"probabilities": {"SELF_CONTAINED": 1.0}},
            "reasoning_tags": {"probabilities": {"TECHNICAL_PRICE": 0.75}},
        },
    }


def _metadata(sample_id: str, published: str, entry: str):
    return {
        "sample_id": sample_id,
        "stock_code": "000001",
        "stock_name": "示例",
        "published_at": published,
        "signal_date": published[:10],
        "entry_date": entry,
        "model_text": sample_id,
        "unit_total_post_count": 10,
        "unit_unique_text_count": 8,
        "unit_sampled_post_count": 2,
    }


def test_next_entry_date_uses_actual_calendar_and_collapses_weekend():
    calendar = ["2026-09-04", "2026-09-07", "2026-09-08"]
    assert signal.next_entry_date("2026-09-04", calendar) == "2026-09-07"
    assert signal.next_entry_date("2026-09-05", calendar) == "2026-09-07"
    assert signal.next_entry_date("2026-09-06", calendar) == "2026-09-07"
    assert signal.next_entry_date("2026-09-08", calendar) is None


def test_parse_datetime_converts_utc_before_taking_shanghai_date():
    parsed = signal._parse_datetime("2026-09-04T16:30:00.000000Z")
    assert parsed.isoformat() == "2026-09-05T00:30:00+08:00"


def test_systematic_sample_is_deterministic_and_spans_full_period():
    rows = [{"index": index} for index in range(10)]
    first = signal.systematic_sample(rows, 4)
    second = signal.systematic_sample(rows, 4)
    assert first == second
    assert [row["index"] for row in first] == [0, 3, 6, 9]


def test_semantic_aggregation_uses_probabilities_not_hard_vote():
    metadata = [
        _metadata("a", "2026-09-05T09:00:00+08:00", "2026-09-07"),
        _metadata("b", "2026-09-06T18:00:00+08:00", "2026-09-07"),
    ]
    units = signal.aggregate_semantic_units(metadata, [_prediction("a", 0.9, 0.1), _prediction("b", 0.3, 0.7)])
    assert len(units) == 1
    unit = units[0]
    assert unit["entry_date_override"] == "2026-09-07"
    assert unit["semantic_features"]["mean::stance::BULL"] == pytest.approx(0.6)
    assert unit["semantic_features"]["mean::stance::BEAR"] == pytest.approx(0.4)
    assert unit["semantic_features"]["stance_score_mean"] == pytest.approx(0.2)
    assert unit["post_count"] == 10
    assert unit["sampled_post_count"] == 2


def test_purged_training_never_uses_outcome_available_on_or_after_eval_entry():
    rows = [
        {"entry_date": "2026-09-01", "outcome_date": "2026-09-03", "id": "ready"},
        {"entry_date": "2026-09-02", "outcome_date": "2026-09-04", "id": "same-day-outcome"},
        {"entry_date": "2026-09-03", "outcome_date": "2026-09-07", "id": "future-outcome"},
    ]
    selected = signal.purged_training_rows(rows, "2026-09-04")
    assert [row["id"] for row in selected] == ["ready"]


def test_walk_forward_has_one_prediction_per_evaluation_unit_and_model():
    rows = []
    dates = [f"2026-09-{day:02d}" for day in range(1, 7)]
    for date_index, entry_date in enumerate(dates):
        for stock_index, stock in enumerate(("000001", "000002")):
            positive = (date_index + stock_index) % 2 == 0
            rows.append(
                {
                    "market_unit_id": f"{stock}:{entry_date}",
                    "stock_code": stock,
                    "entry_date": entry_date,
                    "outcome_date": entry_date,
                    "horizon_trading_days": 1,
                    "outcome_status": "COMPLETE",
                    "excess_positive": positive,
                    "excess_return": 0.01 if positive else -0.01,
                    "semantic_features": {"signal": 1.0 if positive else -1.0},
                    "market_features": {"lag": float(stock_index)},
                }
            )
    predictions = signal.walk_forward_predictions(
        rows,
        feature_sets={
            "semantic": [("semantic_features", "signal")],
            "market": [("market_features", "lag")],
            "combined": [("semantic_features", "signal"), ("market_features", "lag")],
        },
        min_train_dates=2,
    )
    assert predictions
    assert all(set(row["probabilities"]) == {"null", "semantic", "market", "combined"} for row in predictions)
    assert all(row["training_dates"] >= 2 for row in predictions)


def test_direct_score_diagnostic_ranks_within_each_date():
    rows = []
    for entry_date in ("2026-09-01", "2026-09-02", "2026-09-03"):
        for index, score in enumerate((-1.0, -0.25, 0.25, 1.0)):
            rows.append(
                {
                    "stock_code": f"00000{index}",
                    "entry_date": entry_date,
                    "outcome_status": "COMPLETE",
                    "excess_return": score / 100.0,
                    "excess_positive": score > 0.0,
                    "semantic_features": {"stance_score_mean": score},
                }
            )
    result = signal.direct_score_diagnostics(rows, "stance_score_mean")
    assert result["n_entry_dates"] == 3
    assert result["mean_daily_spearman"] == pytest.approx(1.0)
    assert result["mean_daily_top_minus_bottom_excess"] == pytest.approx(0.02)
    assert result["top_quartile_positive_rate"] == 1.0
    assert result["bottom_quartile_positive_rate"] == 0.0
