from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from tools import market_barrier_title_residual as candidate


def _bars(closes: list[float], *, entry_open: float = 100.0):
    dates = [f"2026-06-{day:02d}" for day in range(1, len(closes) + 1)]
    bars = {("000001", day): {"open": entry_open, "close": close}
            for day, close in zip(dates, closes)}
    return dates, bars


def test_barrier_uses_first_close_crossing_and_includes_entry_session():
    dates, bars = _bars([105, 94, *([100] * 8)])
    result = candidate.first_barrier_label("000001", 0, dates, bars)
    assert (result["label"], result["barrier_path"]) == (1, "UP_FIRST")
    assert result["first_up_date"] == dates[0]
    assert result["first_down_date"] == dates[1]
    assert result["outcome_date"] == dates[9]

    dates, bars = _bars([95, 105, *([100] * 8)])
    result = candidate.first_barrier_label("000001", 0, dates, bars)
    assert (result["label"], result["barrier_path"]) == (0, "DOWN_FIRST")


def test_barrier_requires_full_horizon_and_never_substitutes_intraday_high():
    dates, bars = _bars([104] * 10)
    bars[("000001", dates[0])]["high"] = 110
    result = candidate.first_barrier_label("000001", 0, dates, bars)
    assert (result["label"], result["barrier_path"]) == (0, "NEITHER")
    del bars[("000001", dates[-1])]
    with pytest.raises(ValueError, match="missing horizon bar"):
        candidate.first_barrier_label("000001", 0, dates, bars)


def test_title_bags_use_next_trading_day_and_text_dedup(tmp_path):
    db = tmp_path / "collector.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE posts(stock_code TEXT,title TEXT,published_at TEXT)")
        connection.executemany("INSERT INTO posts VALUES(?,?,?)", [
            ("000001", "看 空", "2026-05-06T09:30:00+08:00"),
            ("000001", "看  空", "2026-05-06T10:00:00+08:00"),
            ("000001", "回暖", "2026-05-07T15:05:00+08:00"),
            ("000001", "周末", "2026-05-09T11:00:00+08:00"),
            ("000002", "不相关", "2026-05-06T09:30:00+08:00"),
        ])
    calendar = ["2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11"]
    required = {("000001", day) for day in calendar[1:]}
    bags, counts = candidate.load_all_title_bags(db, calendar, required)
    assert bags[("000001", "2026-05-07")] == ["看 空"]
    assert bags[("000001", "2026-05-08")] == ["回暖"]
    assert bags[("000001", "2026-05-11")] == ["周末"]
    assert counts["raw_counts"][("000001", "2026-05-07")] == 2
    assert counts["unique_counts"][("000001", "2026-05-07")] == 1


def test_market_count_features_come_from_same_backup():
    calendar = [f"2026-06-{i:02d}" for i in range(1, 21)]
    row = {"stock_code": "000001", "entry_date": calendar[-1],
           "market": [1.0] * 7 + [999.0]}
    counts = {"raw_counts": {("000001", day): 3 for day in calendar},
              "unique_counts": {("000001", day): 2 for day in calendar}}
    candidate.add_backup_count_features([row], calendar, counts)
    assert row["market_same_backup"][:7] == [1.0] * 7
    assert row["market_same_backup"][7:] == [math.log1p(60), math.log1p(40),
                                               math.log1p(3), 0.0, math.log1p(2), 0.0]


def test_backup_count_timing_matches_text_recent_and_previous_windows():
    calendar = [f"2026-06-{i:02d}" for i in range(1, 21)]
    row = {"stock_code": "000001", "entry_date": calendar[-1], "market": [0.0] * 8}
    raw = {("000001", day): (1 if i < 15 else 3) for i, day in enumerate(calendar)}
    unique = {("000001", day): (1 if i < 15 else 2) for i, day in enumerate(calendar)}
    candidate.add_backup_count_features([row], calendar,
                                        {"raw_counts": raw, "unique_counts": unique})
    vector = row["market_same_backup"]
    assert vector[9] == math.log1p(3)
    assert vector[10] == pytest.approx(math.log1p(3) - math.log1p(1))
    assert vector[11] == math.log1p(2)
    assert vector[12] == pytest.approx(math.log1p(2) - math.log1p(1))


def test_price_path_controls_exclude_entry_day_bar():
    calendar = [f"2026-06-{i:02d}" for i in range(1, 22)]
    qfq = {("000001", day): {"pct_change": "1.0", "high": "102",
                              "low": "100", "open": "100"}
           for day in calendar[:-1]}
    qfq[("000001", calendar[-1])] = {"pct_change": "90", "high": "1000",
                                      "low": "1", "open": "100"}
    row = {"stock_code": "000001", "entry_date": calendar[-1],
           "market_same_backup": [0.0] * 13}
    candidate.add_known_price_path_features([row], calendar, qfq)
    expected = [(1.01 ** 20) - 1.0, 0.0, 0.02]
    np.testing.assert_allclose(row["market_same_backup"][-3:], expected, atol=1e-10)


def test_title_trajectory_weights_days_equally_and_tracks_cooling():
    calendar = [f"2026-06-{i:02d}" for i in range(1, 11)]
    daily = {("000001", day): np.asarray([1.0 if i < 5 else 3.0], dtype=np.float32)
             for i, day in enumerate(calendar)}
    rows = [{"stock_code": "000001", "entry_date": calendar[-1]}]
    vector = candidate.trajectory_vectors(rows, calendar, daily)
    np.testing.assert_allclose(vector, [[3.0, 2.0]])


def test_daily_vector_cache_requires_exact_units(tmp_path):
    unit = ("000001", "2026-06-01")
    path = tmp_path / "daily.npz"
    candidate.save_daily_mean_vectors(path, {unit: np.asarray([2.0, 3.0])})
    loaded = candidate.load_daily_mean_vectors(path, {unit})
    np.testing.assert_allclose(loaded[unit], [2.0, 3.0])
    with pytest.raises(ValueError, match="does not match"):
        candidate.load_daily_mean_vectors(path, {("000001", "2026-06-02")})


def test_shifted_placebo_never_moves_text_across_stocks():
    rows = [{"stock_code": code, "entry_date": f"2026-06-{day:02d}"}
            for code in ("000001", "000002") for day in range(1, 12)]
    values = np.arange(len(rows), dtype=np.float32)[:, None]
    shifted = candidate.stock_shifted_vectors(rows, values)
    assert shifted[0, 0] == 1
    assert shifted[11, 0] == 12
    assert set(shifted[:11, 0]) == set(values[:11, 0])
    assert set(shifted[11:, 0]) == set(values[11:, 0])
