from __future__ import annotations

import copy
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tools import market_signal_v2_snapshot as snapshot


TZ = ZoneInfo("Asia/Shanghai")


def _prediction(sample_id: str, text: str, *, published: str = "2026-09-16T10:00:00+08:00"):
    return {
        "sample_id": sample_id,
        "stock_code": "600000",
        "stock_name": "浦发银行",
        "model_text": text,
        "published_at": published,
        "prediction_generated_at": "2026-09-16T14:00:00+08:00",
        "cutoff": "2026-09-16T14:00:00+08:00",
        "m12_artifact": "m12-test",
        "predictions": {
            "stance": {"prediction": "BULL", "probabilities": {"BULL": 0.8, "BEAR": 0.2}},
            "action_tendency": {"prediction": "BUY", "probabilities": {"BUY": 0.7, "SELL": 0.3}},
            "emotion_primary": {"prediction": "HOPE", "probabilities": {"HOPE": 0.6, "FEAR": 0.4}},
        },
    }


def test_snapshot_reuses_v2_features_and_marks_past_entry_as_backfill_without_outcomes():
    rows = [_prediction("p1", "同一段正文"), _prediction("p2", "另一段正文")]
    result = snapshot.build_community_snapshots(
        rows,
        ["2026-09-16", "2026-09-17"],
        generated_at=datetime(2026, 9, 18, 9, 0, tzinfo=TZ),
        source_sha256="a" * 64,
    )
    assert len(result) == 1
    unit = result[0]
    assert unit["snapshot_status"] == "BACKFILL_SMOKE"
    assert unit["next_trading_entry_date"] == "2026-09-17"
    assert unit["post_count"] == 2
    assert unit["deduplicated_count"] == 2
    assert "upstream population coverage is not asserted" in unit["sampling_coverage_scope"]
    assert set(unit["distributions"]) == {"stance", "action_tendency", "emotion_primary"}
    assert set(unit["distributions"]["stance"]) == {"BULL", "BEAR"}
    assert unit["direction_indices"]["stance_bull_minus_bear"] == pytest.approx(0.6)
    assert unit["model_uncertainty"]["mean"] >= 0
    assert "absolute_return" not in unit
    assert "verified_price_signal" not in unit
    assert unit["market_prices_read"] is False


def test_snapshot_deduplicates_normalized_text_and_only_forward_when_generated_before_entry():
    rows = [_prediction("p1", "相同  正文"), _prediction("p2", " 相同\n正文 ")]
    result = snapshot.build_community_snapshots(
        rows,
        ["2026-09-16", "2026-09-17"],
        generated_at=datetime(2026, 9, 16, 14, 30, tzinfo=TZ),
        source_sha256="b" * 64,
    )
    assert result[0]["snapshot_status"] == "FORWARD"
    assert result[0]["post_count"] == 2
    assert result[0]["deduplicated_count"] == 1
    assert result[0]["duplicate_post_count"] == 1
    assert result[0]["attention"]["sampling_coverage"] == 1.0


def test_snapshot_refuses_incomplete_calendar_and_duplicate_ids():
    with pytest.raises(ValueError, match="no next actual session"):
        snapshot.build_community_snapshots(
            [_prediction("p1", "正文")],
            ["2026-09-16"],
            generated_at=datetime(2026, 9, 16, 14, 30, tzinfo=TZ),
            source_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="unique non-empty sample_id"):
        snapshot.build_community_snapshots(
            [_prediction("p1", "正文"), _prediction("p1", "另一正文")],
            ["2026-09-17"],
            generated_at=datetime(2026, 9, 16, 14, 30, tzinfo=TZ),
            source_sha256="d" * 64,
        )


def test_backfill_writes_absolute_returns_separately_and_keeps_snapshot_unchanged():
    frozen = {
        "snapshot_id": "unit-1",
        "snapshot_status": "BACKFILL_SMOKE",
        "stock_code": "600000",
        "next_trading_entry_date": "2026-09-17",
        "distributions": {"stance": {"BULL": 1.0}},
    }
    before = copy.deepcopy(frozen)
    dates = ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]
    bars = {
        ("600000", "2026-09-17"): {"open": "10", "close": "11", "fetched_at": "v1"},
        ("600000", "2026-09-18"): {"open": "11", "close": "12", "fetched_at": "v1"},
        ("600000", "2026-09-21"): {"open": "12", "close": "13", "fetched_at": "v1"},
        ("600000", "2026-09-22"): {"open": "13", "close": "14", "fetched_at": "v1"},
        ("600000", "2026-09-23"): {"open": "14", "close": "15", "fetched_at": "v1"},
    }
    outcomes = snapshot.backfill_outcomes([frozen], dates, bars, as_of_date=date(2026, 9, 23))
    assert len(outcomes) == 3
    assert all(row["status"] == "COMPLETE" for row in outcomes)
    assert outcomes[0]["absolute_return"] == pytest.approx(0.1)
    assert outcomes[1]["absolute_return"] == pytest.approx(0.3)
    assert outcomes[2]["absolute_return"] == pytest.approx(0.5)
    assert outcomes[0]["evidence_type"] == "BACKFILL_SMOKE"
    assert frozen == before


def test_backfill_rejects_mixed_qfq_vintages():
    frozen = {
        "snapshot_id": "unit-1",
        "snapshot_status": "BACKFILL_SMOKE",
        "stock_code": "600000",
        "next_trading_entry_date": "2026-09-17",
    }
    dates = ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]
    bars = {
        ("600000", day): {"open": "10", "close": "11", "fetched_at": "v1" if day == dates[0] else "v2"}
        for day in dates
    }
    outcomes = snapshot.backfill_outcomes([frozen], dates, bars, as_of_date=date(2026, 9, 23))
    assert outcomes[0]["status"] == "COMPLETE"
    assert all(row["status"] == "ADJUSTMENT_VINTAGE_MISMATCH" for row in outcomes[1:])


def test_calendar_loader_rejects_price_columns(tmp_path):
    path = tmp_path / "calendar.csv"
    path.write_text("trade_date,close\n2026-09-17,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one trade_date"):
        snapshot.load_trading_dates(path)
