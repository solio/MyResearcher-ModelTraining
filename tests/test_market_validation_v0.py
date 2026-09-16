from __future__ import annotations

import pytest

from tools import market_validation_v0 as validation


def _prediction(sample_id: str, bull: float, bear: float):
    return {
        "sample_id": sample_id,
        "stock_code": "000001",
        "stock_name": "示例",
        "published_at": "2026-09-01T10:00:00+08:00",
        "predictions": {
            "stance": {"prediction": "BULL" if bull >= bear else "BEAR", "probabilities": {"BULL": bull, "BEAR": bear}},
            "action_tendency": {"prediction": "BUY", "probabilities": {"BUY": bull, "ADD": 0.0, "SELL": bear, "REDUCE": 0.0}},
            "target_mode": {"prediction": "ON_TARGET", "probabilities": {"ON_TARGET": 1.0}},
            "emotion_primary": {"prediction": "CALM", "probabilities": {"CALM": 1.0}},
            "emotion_target": {"prediction": "PRICE", "probabilities": {"PRICE": 1.0}},
            "context_dependency": {"prediction": "SELF_CONTAINED", "probabilities": {"SELF_CONTAINED": 1.0}},
            "reasoning_tags": {
                "prediction": ["TECHNICAL_PRICE"],
                "probabilities": {"TECHNICAL_PRICE": 0.8},
                "thresholds": {"TECHNICAL_PRICE": 0.4},
            },
        },
    }


def _bar(code: str, day: str, opening: float, closing: float):
    return {"instrument_code": code, "trade_date": day, "open": opening, "close": closing}


def test_stock_day_aggregation_prevents_duplicate_market_units():
    rows = [_prediction("a", 0.8, 0.2), _prediction("b", 0.6, 0.4)]
    benchmark = [_bar("000300", "2026-09-02", 100.0, 101.0)]
    aggregate = validation.aggregate_market_units(rows, benchmark)
    assert len(aggregate) == 1
    assert aggregate[0]["post_count"] == 2
    assert aggregate[0]["entry_date_override"] == "2026-09-02"
    probabilities = aggregate[0]["predictions"]["stance"]["probabilities"]
    assert probabilities["BEAR"] == pytest.approx(0.3)
    assert probabilities["BULL"] == pytest.approx(0.7)
    assert aggregate[0]["predictions"]["reasoning_tags"]["prediction"] == ["TECHNICAL_PRICE"]


def test_join_uses_next_session_open_and_horizon_close():
    predictions = [_prediction("a", 0.8, 0.2)]
    qfq = [
        _bar("000001", "2026-09-01", 9.0, 9.5),
        _bar("000001", "2026-09-02", 10.0, 11.0),
        _bar("000001", "2026-09-03", 11.0, 12.0),
        _bar("000001", "2026-09-04", 12.0, 13.0),
    ]
    benchmark = [
        _bar("000300", "2026-09-02", 100.0, 101.0),
        _bar("000300", "2026-09-03", 101.0, 102.0),
        _bar("000300", "2026-09-04", 102.0, 103.0),
    ]
    joined = validation.join_outcomes(
        predictions,
        qfq_rows=qfq,
        raw_rows=qfq,
        benchmark_rows=benchmark,
        horizons=(1, 3, 5),
    )
    assert joined[0]["status"] == "COMPLETE"
    assert joined[0]["entry_date"] == "2026-09-02"
    assert joined[0]["outcome_date"] == "2026-09-02"
    assert joined[0]["stock_return"] == pytest.approx(0.1)
    assert joined[1]["outcome_date"] == "2026-09-04"
    assert joined[1]["stock_return"] == pytest.approx(0.3)
    assert joined[2]["status"] == "PENDING_OR_MISSING"


def test_suspended_stock_does_not_silently_move_entry_to_later_bar():
    predictions = [_prediction("a", 0.8, 0.2)]
    qfq = [_bar("000001", "2026-09-03", 11.0, 12.0)]
    benchmark = [
        _bar("000300", "2026-09-02", 100.0, 101.0),
        _bar("000300", "2026-09-03", 101.0, 102.0),
    ]
    joined = validation.join_outcomes(
        predictions,
        qfq_rows=qfq,
        raw_rows=qfq,
        benchmark_rows=benchmark,
        horizons=(1,),
    )
    assert joined[0]["status"] == "MISSING"
    assert joined[0]["entry_date"] == "2026-09-02"


def test_mixed_qfq_vintages_with_return_gap_are_excluded():
    predictions = [_prediction("a", 0.8, 0.2)]
    qfq = [
        dict(_bar("000001", "2026-09-02", 10.0, 10.0), fetched_at="old"),
        dict(_bar("000001", "2026-09-03", 20.0, 20.0), fetched_at="new"),
    ]
    raw = [
        dict(_bar("000001", "2026-09-02", 10.0, 10.0), fetched_at="old"),
        dict(_bar("000001", "2026-09-03", 10.0, 10.0), fetched_at="new"),
    ]
    benchmark = [
        _bar("000300", "2026-09-02", 100.0, 100.0),
        _bar("000300", "2026-09-03", 100.0, 100.0),
    ]
    joined = validation.join_outcomes(
        predictions,
        qfq_rows=qfq,
        raw_rows=raw,
        benchmark_rows=benchmark,
        horizons=(2,),
    )
    assert joined[0]["status"] == "ADJUSTMENT_VINTAGE_MISMATCH"


def test_spearman_detects_order_without_assuming_significance():
    assert validation.spearman([-1.0, 0.0, 1.0], [-2.0, 0.0, 3.0]) == 1.0
    assert validation.spearman([-1.0, 0.0, 1.0], [2.0, 0.0, -3.0]) == -1.0


def test_reasoning_tags_are_reported_as_overlapping_groups():
    rows = [
        {"reasoning_tags": ["FUNDAMENTAL", "NEWS_EVENT"], "stock_return": 0.01, "excess_return": 0.02},
        {"reasoning_tags": ["NEWS_EVENT"], "stock_return": -0.01, "excess_return": -0.02},
    ]
    result = validation._reasoning_tag_descriptive(rows)
    assert result["FUNDAMENTAL"]["excess_return"]["n"] == 1
    assert result["NEWS_EVENT"]["excess_return"]["n"] == 2


def test_direction_bands_are_fixed_before_market_results():
    rows = [
        {"score": -0.4, "stock_return": 0.0, "excess_return": 0.0},
        {"score": -0.1, "stock_return": 0.0, "excess_return": 0.0},
        {"score": 0.1, "stock_return": 0.0, "excess_return": 0.0},
        {"score": 0.4, "stock_return": 0.0, "excess_return": 0.0},
    ]
    result = validation._fixed_score_bands(rows, "score")
    assert [result[label]["score"]["n"] for label in result] == [1, 1, 1, 1]
