from __future__ import annotations

import sqlite3

import pytest

from tools import market_event_title_coverage as coverage
from tools import market_signal_v1 as historical


SOURCE = "eastmoney_guba"
CALENDAR = ["2026-09-04", "2026-09-07", "2026-09-08"]
MONDAY = ("000001", "2026-09-07")
TUESDAY = ("000001", "2026-09-08")
POSTS = [
    # Thursday UTC is already Friday in Shanghai, so its next session is Monday.
    ("utc_boundary", "000001", "UTC边界", "2026-09-03T16:30:00Z"),
    ("first", "000001", "牛  市", "2026-09-04T09:00:00+08:00"),
    ("duplicate", "000001", "牛 市", "2026-09-05T10:00:00+08:00"),
    ("weekend", "000001", "周末", "2026-09-06T18:00:00+08:00"),
    # A Monday post cannot be used at Monday's open; it maps to Tuesday.
    ("monday", "000001", "周一", "2026-09-07T09:00:00+08:00"),
    ("after_calendar", "000001", "出界", "2026-09-08T09:00:00+08:00"),
    ("other_stock", "000002", "无关", "2026-09-05T10:00:00+08:00"),
    ("empty_title", "000001", "", "2026-09-06T12:00:00+08:00"),
]


def _id(item: str) -> str:
    published_at = next(row[3] for row in POSTS if row[0] == item)
    return historical._stable_sample_id(SOURCE, item, published_at)


@pytest.fixture
def title_inputs(tmp_path):
    db_path = tmp_path / "collector.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE posts (source TEXT, source_item_id TEXT, "
            "stock_code TEXT, title TEXT, published_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO posts VALUES (?, ?, ?, ?, ?)",
            [(SOURCE, *row) for row in POSTS],
        )
    sampled_bags = {
        MONDAY: [_id("utc_boundary"), _id("first")],
        TUESDAY: [_id("monday")],
    }
    frozen_titles = {
        _id("utc_boundary"): "UTC边界",
        _id("first"): "牛  市",
        _id("monday"): "周一",
    }
    return db_path, sampled_bags, frozen_titles


def test_title_universe_deduplicates_titles_and_maps_to_next_actual_session(title_inputs):
    db_path, sampled_bags, frozen_titles = title_inputs

    bags, titles_by_id, counts = coverage.title_universe(
        db_path, CALENDAR, {MONDAY, TUESDAY}, frozen_titles, sampled_bags
    )

    assert bags[MONDAY]["all_titles"] == [
        _id("utc_boundary"), _id("first"), _id("weekend")
    ]
    assert bags[TUESDAY]["all_titles"] == [_id("monday")]
    assert bags[MONDAY]["sample16"] == sampled_bags[MONDAY]
    assert bags[TUESDAY]["sample16"] == sampled_bags[TUESDAY]
    assert _id("duplicate") not in titles_by_id
    assert titles_by_id[_id("first")] == "牛  市"
    assert counts == {
        "raw_posts_in_required_units": 5,
        "unique_titles_in_required_units": 4,
        "sample16_titles_in_required_units": 3,
        "units": 2,
        "max_all_titles_per_unit": 3,
    }


def test_title_universe_rejects_old_sample_not_in_unique_title_universe(title_inputs):
    db_path, sampled_bags, frozen_titles = title_inputs
    sampled_bags[MONDAY].append(_id("duplicate"))
    frozen_titles[_id("duplicate")] = "牛 市"

    with pytest.raises(ValueError, match="old sampled ID not found"):
        coverage.title_universe(db_path, CALENDAR, {MONDAY, TUESDAY},
                                frozen_titles, sampled_bags)


def test_title_universe_rejects_changed_frozen_title(title_inputs):
    db_path, sampled_bags, frozen_titles = title_inputs
    frozen_titles[_id("first")] = "后来改过的标题"

    with pytest.raises(ValueError, match="frozen sampled title changed"):
        coverage.title_universe(db_path, CALENDAR, {MONDAY, TUESDAY},
                                frozen_titles, sampled_bags)


@pytest.mark.parametrize("sample", [[], ["repeated"] * 17])
def test_title_universe_rejects_missing_or_oversized_old_bag(title_inputs, sample):
    db_path, sampled_bags, frozen_titles = title_inputs
    sampled_bags[MONDAY] = sample

    with pytest.raises(ValueError, match="missing title unit or invalid old sample cap"):
        coverage.title_universe(db_path, CALENDAR, {MONDAY, TUESDAY},
                                frozen_titles, sampled_bags)
