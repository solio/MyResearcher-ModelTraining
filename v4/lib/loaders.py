"""Read-only input loaders and title aggregation for the v4 report."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .timebase import post_info_day


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_volume(value: str | None, unit: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    unit = (unit or "").strip().lower()
    if unit == "shares":
        return number
    if unit == "lots":
        return number * 100.0
    return None


def load_prices(config: dict) -> tuple[dict[str, dict[str, dict]], list[str], dict[str, dict[str, dict]]]:
    """Load per-stock bars and the CSI300 benchmark.

    The main CSV is authoritative.  The patch file only fills missing
    (stock, day) cells; it never overwrites a newer main-file row.
    """
    paths = config["paths"]
    bars: dict[str, dict[str, dict]] = defaultdict(dict)
    for path_key, patch in (("bars_main", False), ("bars_patch", True)):
        path = paths[path_key]
        if not Path(path).is_file():
            continue
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                code = row.get("instrument_code")
                day = row.get("trade_date")
                if not code or not day or row.get("instrument_type") != "stock":
                    continue
                if patch and day in bars.get(code, {}):
                    continue
                try:
                    o = float(row["open"])
                    h = float(row["high"])
                    l = float(row["low"])
                    c = float(row["close"])
                except (KeyError, TypeError, ValueError):
                    continue
                bars[code][day] = {
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": _norm_volume(row.get("volume"), row.get("volume_unit")),
                    "turnover": float(row.get("turnover_rate") or 0) or None,
                    "pct_change": float(row.get("pct_change") or 0) or None,
                }
    calendar = sorted({d for stock in bars.values() for d in stock})
    benchmark: dict[str, dict[str, dict]] = defaultdict(dict)
    path = paths.get("benchmark")
    if path and Path(path).is_file():
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                code = row.get("instrument_code")
                day = row.get("trade_date")
                if code != "000300" or not day:
                    continue
                try:
                    o = float(row["open"])
                    c = float(row["close"])
                except (KeyError, TypeError, ValueError):
                    continue
                benchmark[code][day] = {"open": o, "close": c}
    return dict(bars), calendar, dict(benchmark)


def compile_vocab(vocab_dir: str | Path) -> dict:
    vocab_dir = Path(vocab_dir)

    def pattern(name: str) -> re.Pattern:
        body = (vocab_dir / name).read_text(encoding="utf-8")
        source = "|".join(
            line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")
        )
        return re.compile(source)

    nicks = {}
    for path in sorted((vocab_dir / "nick").glob("*.v1.re")):
        nicks[path.name.split(".")[0]] = pattern(f"nick/{path.name}")
    return {
        "bear": pattern("bearw.v2.re"),
        "excite": pattern("excite.v1.re"),
        "declare": pattern("declare_sell.v1.re"),
        "nicks": nicks,
        "files": [
            {"file": str(p.relative_to(vocab_dir)), "sha256": sha256_file(p)}
            for p in sorted(vocab_dir.rglob("*.re"))
        ],
    }


def normalize_title(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"[\s~～·・、,，.。:：;；\"'\[\]()（）<>《》\-_=+*|\\/]+", "", text)
    return re.sub(r"(.)\1{2,}", lambda m: m.group(0) if m.group(1) in "!！哈" else m.group(1), text)


def aggregate_posts(
    config: dict,
    calendar: list[str],
    stocks: set[str],
    vocab: dict,
) -> dict:
    """One pass over the read-only collector database.

    No title, content, author id or URL leaves this function.
    """
    db_path = config["paths"]["collector"]
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    cells: dict[tuple[str, str], dict] = {}
    author_counter: Counter[str] = Counter()
    raw_days: dict[str, set[str]] = defaultdict(set)
    first_last: dict[str, list[str]] = {}
    content_nonempty = 0
    total = 0
    watermark = ""
    sql = "SELECT stock_code, title, content, author_id, published_at FROM posts"
    for stock, title, content, author_id, published in connection.execute(sql):
        if not published or stock not in stocks:
            continue
        info_day = post_info_day(published, calendar, config["protocol"]["info_cutoff"])
        if not info_day:
            continue
        total += 1
        watermark = max(watermark, published)
        if content and content.strip():
            content_nonempty += 1
        author = author_id or "UNKNOWN"
        author_counter[author] += 1
        raw_day = published[:10]
        raw_days[stock].add(raw_day)
        span = first_last.setdefault(stock, [raw_day, raw_day])
        span[0] = min(span[0], raw_day)
        span[1] = max(span[1], raw_day)
        cell = cells.setdefault(
            (stock, info_day),
            {"n": 0, "authors": Counter(), "bear": 0, "excite": 0, "declare": 0},
        )
        cell["n"] += 1
        cell["authors"][author] += 1
        normalized = normalize_title(title)
        nick = vocab["nicks"].get(stock)
        if vocab["bear"].search(normalized) or (nick is not None and nick.search(normalized)):
            cell["bear"] += 1
        if vocab["excite"].search(normalized):
            cell["excite"] += 1
        if vocab["declare"].search(normalized):
            cell["declare"] += 1
    connection.close()
    return {
        "cells": cells,
        "author_counter": author_counter,
        "raw_days": {k: sorted(v) for k, v in raw_days.items()},
        "first_last": first_last,
        "content_nonempty": content_nonempty,
        "total": total,
        "watermark": watermark,
    }


def author_share(author_counter: Counter[str], top_fraction: float = 0.01) -> dict:
    total = sum(author_counter.values())
    if not total:
        return {"total_authors": 0, "top1pct_post_share": None, "top_author_share": None, "n_authors": 0}
    n_top = max(1, int(len(author_counter) * top_fraction))
    top_posts = sum(count for _, count in author_counter.most_common(n_top))
    return {
        "total_authors": len(author_counter),
        "n_authors": sum(1 for _, count in author_counter.items() if count == 1),
        "top1pct_post_share": top_posts / total,
        "top_author_share": max(author_counter.values()) / total,
    }


def stocks_config(path: str | Path) -> list[dict]:
    return load_json(path)
