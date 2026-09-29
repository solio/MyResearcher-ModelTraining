"""Boundary and leakage tests for the isolated shock-supervised reader."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from tools import market_reaction_reader as reader


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-05-08T14:59:59+08:00", None),
        ("2026-05-08T15:00:00+08:00", "2026-05-08"),
        ("2026-05-09T12:00:00+08:00", "2026-05-08"),
        ("2026-05-11T09:29:59+08:00", "2026-05-08"),
        ("2026-05-11T09:30:00+08:00", None),
        ("2026-05-11T15:00:00+08:00", "2026-05-11"),
    ],
)
def test_reaction_day_only_close_to_next_open(timestamp, expected):
    calendar = ["2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12"]
    assert reader.reaction_day(datetime.fromisoformat(timestamp), calendar) == expected


def test_title_bags_use_full_deduplicated_overnight_window(tmp_path):
    db = tmp_path / "collector.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE posts (stock_code TEXT,title TEXT,published_at TEXT)")
        connection.executemany("INSERT INTO posts VALUES (?,?,?)", [
            ("000001", "  A  title ", "2026-05-08T15:00:00+08:00"),
            ("000001", "A title", "2026-05-09T12:00:00+08:00"),
            ("000001", "B title", "2026-05-11T09:29:59+08:00"),
            ("000001", "EXCLUDED", "2026-05-11T09:30:00+08:00"),
            ("000001", "C title", "2026-05-11T15:00:00+08:00"),
        ])
    calendar = ["2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12"]
    units = {("000001", "2026-05-08"), ("000001", "2026-05-11")}
    bags, counts = reader.load_title_bags(db, calendar, units)
    assert bags[("000001", "2026-05-08")] == ["A title", "B title"]
    assert bags[("000001", "2026-05-11")] == ["C title"]
    assert counts["raw_posts"] == 4
    assert counts["unique_title_occurrences"] == 3


def test_weak_reaction_label_is_finished_day_not_future():
    calendar = ["2026-05-06", "2026-05-07", "2026-05-08"]
    qfq = {("000001", day): {"close": close} for day, close in
           zip(calendar, [100.0, 108.0, 108.0])}
    assert reader.reaction_label("000001", "2026-05-07", calendar, qfq) == (1, 1.0)
    assert reader.reaction_label("000001", "2026-05-08", calendar, qfq) == (None, 0.0)
    qfq[("000001", "2026-05-08")]["close"] = 80.0
    assert reader.reaction_label("000001", "2026-05-07", calendar, qfq) == (1, 1.0)


def test_h10_barrier_uses_entry_open_and_first_close_crossing():
    calendar = [f"d{i:02d}" for i in range(11)]
    qfq = {("000001", day): {"open": 100.0, "close": 100.0}
           for day in calendar}
    qfq[("000001", "d00")]["close"] = 94.0
    qfq[("000001", "d01")]["close"] = 106.0
    qfq[("000001", "d09")]["close"] = 110.0
    result = reader.future_labels("000001", "d00", calendar, qfq)
    assert result == {"barrier_up_first": 0, "h10_positive": 1, "outcome_date": "d09"}


def test_pre_entry_features_do_not_read_entry_or_future_bars():
    calendar = [f"d{i:02d}" for i in range(31)]
    code = "000001"
    def bar(i):
        return {"open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i,
                "close": 101.0 + i, "volume": 1000.0 + i, "amount": 10000.0 + i,
                "pct_change": 0.5, "turnover_rate": 2.0}
    qfq = {(code, day): bar(i) for i, day in enumerate(calendar)}
    raw = {(code, day): bar(i) for i, day in enumerate(calendar)}
    bench = {("000300", day): bar(i) for i, day in enumerate(calendar)}
    counts = {"raw_counts": {(code, day): 2 for day in calendar},
              "unique_counts": {(code, day): 1 for day in calendar}}
    row = {"code": code, "entry": "d20", "prior_units": reader.preceding_units(code, "d20", calendar)}
    before = reader.price_count_features(row, calendar, qfq, raw, bench, counts)
    qfq[(code, "d20")]["close"] = 10000.0
    qfq[(code, "d21")]["close"] = 1.0
    raw[(code, "d20")]["volume"] = 1e12
    assert reader.price_count_features(row, calendar, qfq, raw, bench, counts) == before
    assert all(day < row["entry"] for _, day in row["prior_units"])


def test_reader_trajectory_uses_only_prior_scores_and_explicit_missing():
    calendar = [f"d{i:02d}" for i in range(11)]
    units = reader.preceding_units("000001", "d10", calendar)
    scores = {key: score for key, score in zip(units, [-0.8, -0.4, 0.1, 0.4, 0.6])}
    counts = {"unique_counts": {key: count for key, count in zip(units, [100, 50, 25, 10, 2])}}
    row = {"prior_units": units}
    before = reader.reader_trajectory(row, scores, counts)
    assert before[0] == -0.8
    assert before[1] == pytest.approx(1.4)
    assert before[2] > 0
    scores[("000001", "d10")] = -1.0
    assert reader.reader_trajectory(row, scores, counts) == before
    del scores[units[-1]]
    assert reader.reader_trajectory(row, scores, counts)[1] == pytest.approx(1.2)
