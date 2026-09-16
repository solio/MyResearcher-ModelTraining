from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tools import market_data_bootstrap_v0 as bootstrap


def test_normalize_rows_preserves_provider_units_and_aliases():
    rows = bootstrap.normalize_provider_rows(
        [{"日期": "2026-08-25", "开盘": "10", "最高": 11, "最低": 9, "收盘": 10.5, "成交量": "12", "成交额": "1000", "涨跌幅": "1.2", "换手率": "0.4"}],
        instrument_code="000001",
        instrument_type="stock",
        adjustment="raw",
        provider="baostock",
        fetched_at="2026-09-16T00:00:00Z",
        volume_unit="shares",
    )
    assert rows[0]["trade_date"] == "2026-08-25"
    assert rows[0]["volume"] == 12.0
    assert rows[0]["volume_unit"] == "shares"
    assert rows[0]["amount_unit"] == "CNY"
    assert rows[0]["turnover_rate_unit"] == "percent"


def test_merge_csv_deduplicates_instrument_date_adjustment(tmp_path: Path):
    path = tmp_path / "bars.csv"
    row = {field: "" for field in bootstrap.MARKET_FIELDS}
    row.update({"instrument_code": "000001", "instrument_type": "stock", "trade_date": "2026-08-25", "adjustment": "raw", "close": 10.0})
    bootstrap._merge_csv(path, [row, row])
    bootstrap._merge_csv(path, [dict(row, close=11.0)])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    # Existing rows are immutable on rerun; a later duplicate cannot rewrite it.
    assert rows[0]["close"] == "10.0"


def test_select_latest_qualified_posts_is_cutoff_bounded_and_label_free(tmp_path: Path):
    db = tmp_path / "collector.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE posts (source TEXT, source_item_id TEXT, stock_code TEXT, title TEXT, content TEXT, author_id TEXT, author_name TEXT, published_at TEXT, url TEXT, created_at TEXT, updated_at TEXT)")
    con.executemany(
        "INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("eastmoney_guba", "old", "000001", "old", None, "secret", "secret", "2026-09-15T09:00:00+08:00", "https://private", "", ""),
            ("eastmoney_guba", "new", "000001", "new title", None, "secret", "secret", "2026-09-16T09:00:00+08:00", "https://private", "", ""),
            ("eastmoney_guba", "future", "000001", "future", None, "secret", "secret", "2026-09-17T09:00:00+08:00", "https://private", "", ""),
            ("eastmoney_guba", "blank", "000001", "", None, "secret", "secret", "2026-09-16T10:00:00+08:00", "https://private", "", ""),
        ],
    )
    con.commit()
    con.close()
    rows = bootstrap.select_latest_qualified_posts(db, cutoff=datetime(2026, 9, 16, 12, tzinfo=timezone.utc))
    assert [row["observation_id"] for row in rows] == ["new"]
    assert set(rows[0]) == {"sample_id", "observation_id", "source_item_id", "source", "stock_code", "stock_name", "published_at", "model_text"}
    assert "author_id" not in rows[0] and "url" not in rows[0]


def test_pending_horizon_business_day_estimate():
    assert bootstrap._next_weekdays(datetime(2026, 9, 16).date(), 1).isoformat() == "2026-09-17"
    assert bootstrap._next_weekdays(datetime(2026, 9, 16).date(), 3).isoformat() == "2026-09-21"
    assert bootstrap._next_weekdays(datetime(2026, 9, 16).date(), 5).isoformat() == "2026-09-23"
