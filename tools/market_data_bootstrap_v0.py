"""Market Data Bootstrap V0 and a frozen M12 forward-shadow snapshot.

This module deliberately keeps the market-data layer small.  It uses AKShare's
daily endpoints first and falls back to BaoStock when the AKShare endpoint is
unavailable.  Market files are append-only by the
``(instrument_code, trade_date, adjustment)`` key.  The forward-shadow part
reads only the latest qualified publication day from the local collector DB,
runs the already selected M12 runtime, and writes a pending-outcome table; it
never reads labels or future bars to change a prediction.

The optional ``akshare`` and ``baostock`` packages are imported lazily.  This
keeps unit tests and environments that only inspect the contracts independent
of either provider.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEED = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "feed" / "prediction-snapshot.jsonl"
DEFAULT_COLLECTOR_DB = PROJECT_ROOT / "MyResearcher-DataCollector" / "data" / "collector.db"
DEFAULT_TARGETS = PROJECT_ROOT / "MyResearcher-DataCollector" / "config" / "targets.short-term.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916"
DEFAULT_START = date(2026, 8, 25)
BENCHMARK_CODE = "000300"
BENCHMARK_NAME = "沪深300"
MARKET_TZ = ZoneInfo("Asia/Shanghai")
MARKET_FIELDS = (
    "instrument_code",
    "instrument_type",
    "trade_date",
    "adjustment",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_unit",
    "amount",
    "amount_unit",
    "pct_change",
    "turnover_rate",
    "turnover_rate_unit",
    "provider",
    "fetched_at",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) >= 10 and text[4] in "-/" and text[7] in "-/":
        return text[:10].replace("/", "-")
    return text or None


def _column(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def normalize_provider_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    instrument_code: str,
    instrument_type: str,
    adjustment: str,
    provider: str,
    fetched_at: str,
    volume_unit: str,
) -> list[dict[str, Any]]:
    """Normalize AKShare/BaoStock rows while preserving explicit units."""

    output: list[dict[str, Any]] = []
    for source in rows:
        item = {
            "instrument_code": str(instrument_code),
            "instrument_type": instrument_type,
            "trade_date": _as_date(_column(source, "日期", "date", "Date", "trade_date")),
            "adjustment": adjustment,
            "open": _as_float(_column(source, "开盘", "open")),
            "high": _as_float(_column(source, "最高", "high")),
            "low": _as_float(_column(source, "最低", "low")),
            "close": _as_float(_column(source, "收盘", "close")),
            "volume": _as_float(_column(source, "成交量", "volume")),
            "volume_unit": volume_unit,
            "amount": _as_float(_column(source, "成交额", "amount")),
            "amount_unit": "CNY",
            "pct_change": _as_float(_column(source, "涨跌幅", "pctChg", "pct_change")),
            "turnover_rate": _as_float(_column(source, "换手率", "turn", "turnover_rate")),
            "turnover_rate_unit": "percent",
            "provider": provider,
            "fetched_at": fetched_at,
        }
        if item["trade_date"]:
            output.append(item)
    return output


def feed_stock_codes(feed_path: str | Path) -> tuple[list[str], dict[str, str]]:
    """Return unique stock codes and names from the existing prediction file."""

    rows = read_jsonl(Path(feed_path))
    names: dict[str, str] = {}
    codes: set[str] = set()
    for row in rows:
        code = str(row.get("stock_code") or "").strip()
        if not code:
            continue
        codes.add(code)
        name = str(row.get("stock_name") or "").strip()
        if name:
            names.setdefault(code, name)
    if not codes:
        raise ValueError(f"prediction snapshot contains no stock_code: {feed_path}")
    return sorted(codes), names


def _akshare_rows(code: str, start: str, end: str, adjustment: str, *, instrument_type: str) -> list[Mapping[str, Any]]:
    ak = importlib.import_module("akshare")
    if instrument_type == "index":
        frame = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end)
    else:
        frame = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="" if adjustment == "raw" else "qfq")
    # DataFrame is intentionally consumed through to_dict so pandas is an
    # implementation detail and never becomes part of the output contract.
    if hasattr(frame, "to_dict"):
        return list(frame.to_dict("records"))
    return list(frame)


def _baostock_rows(code: str, start: str, end: str, adjustment: str, *, instrument_type: str, bs: Any) -> list[Mapping[str, Any]]:
    if instrument_type == "index":
        symbol = f"sh.{code}"
    else:
        symbol = ("sh." if code.startswith(("6",)) else "sz.") + code
    adjustflag = "3" if adjustment == "raw" else "2"
    result = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,pctChg,turn",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjustflag,
    )
    if getattr(result, "error_code", "1") != "0":
        raise RuntimeError(f"BaoStock {symbol} {adjustment}: {getattr(result, 'error_msg', 'query failed')}")
    rows: list[Mapping[str, Any]] = []
    while result.next():
        values = result.get_row_data()
        rows.append(dict(zip(result.fields, values)))
    return rows


def _fetch_one(
    code: str,
    *,
    start: date,
    end: date,
    adjustment: str,
    instrument_type: str,
    baostock_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    errors: list[str] = []
    try:
        rows = _akshare_rows(code, start_text, end_text, adjustment, instrument_type=instrument_type)
        normalized = normalize_provider_rows(
            rows,
            instrument_code=code,
            instrument_type=instrument_type,
            adjustment=adjustment,
            provider="akshare",
            fetched_at=fetched_at,
            volume_unit="lots",
        )
        if normalized:
            return normalized, {"status": "ok", "provider": "akshare", "rows": len(normalized), "errors": []}
        errors.append("AKShare returned no rows")
    except Exception as exc:  # provider errors become evidence, not silent drops
        errors.append(f"AKShare {type(exc).__name__}: {exc}")
    try:
        if not baostock_state.get("attempted_login"):
            baostock_state["module"] = importlib.import_module("baostock")
            login = baostock_state["module"].login()
            if getattr(login, "error_code", "1") != "0":
                raise RuntimeError(getattr(login, "error_msg", "BaoStock login failed"))
            baostock_state["attempted_login"] = True
        rows = _baostock_rows(code, start_text[:4] + "-" + start_text[4:6] + "-" + start_text[6:], end_text[:4] + "-" + end_text[4:6] + "-" + end_text[6:], adjustment, instrument_type=instrument_type, bs=baostock_state["module"])
        normalized = normalize_provider_rows(
            rows,
            instrument_code=code,
            instrument_type=instrument_type,
            adjustment=adjustment,
            provider="baostock",
            fetched_at=fetched_at,
            # BaoStock documents volume as shares; it is not converted to lots.
            volume_unit="shares",
        )
        return normalized, {"status": "ok" if normalized else "empty", "provider": "baostock", "rows": len(normalized), "errors": errors + ([] if normalized else ["BaoStock returned no rows"])}
    except Exception as exc:
        errors.append(f"BaoStock {type(exc).__name__}: {exc}")
        return [], {"status": "error", "provider": None, "rows": 0, "errors": errors}


def _merge_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Append rows by immutable market key, retaining the first existing row."""

    existing: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            existing = [dict(row) for row in csv.DictReader(handle)]
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in existing:
        key = (str(row.get("instrument_code", "")), str(row.get("trade_date", "")), str(row.get("adjustment", "")))
        if all(key):
            by_key.setdefault(key, row)
    for row in rows:
        key = (str(row.get("instrument_code", "")), str(row.get("trade_date", "")), str(row.get("adjustment", "")))
        if all(key):
            by_key.setdefault(key, {field: row.get(field) for field in MARKET_FIELDS})
    merged = [by_key[key] for key in sorted(by_key)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MARKET_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in MARKET_FIELDS})
    return merged


def fetch_market_data(
    *,
    feed_path: str | Path = DEFAULT_FEED,
    output_dir: str | Path = DEFAULT_OUTPUT,
    start: date = DEFAULT_START,
    end: date | None = None,
    targets_path: str | Path = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """Fetch only feed instruments plus CSI300, with a per-request report."""

    if end is None:
        end = datetime.now(MARKET_TZ).date()
    if end < start:
        raise ValueError("end date precedes start date")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    codes, names = feed_stock_codes(feed_path)
    try:
        target_mapping = json.loads(Path(targets_path).read_text(encoding="utf-8"))
        names = {**{str(k): str(v) for k, v in target_mapping.get("stock_names", {}).items()}, **names}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    instruments = [
        {"instrument_code": code, "instrument_type": "stock", "stock_name": names.get(code, ""), "source": "prediction-snapshot"}
        for code in codes
    ] + [{"instrument_code": BENCHMARK_CODE, "instrument_type": "index", "stock_name": BENCHMARK_NAME, "source": "benchmark"}]
    _write_json(output / "instruments.json", {
        "source_prediction_snapshot": str(Path(feed_path).resolve()),
        "source_prediction_snapshot_sha256": sha256_file(Path(feed_path)),
        "start_date": start.isoformat(),
        "end_date_requested": end.isoformat(),
        "stock_count": len(codes),
        "instruments": instruments,
    })
    state: dict[str, Any] = {"attempted_login": False, "module": None}
    raw_rows: list[dict[str, Any]] = []
    qfq_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for code in codes:
        for adjustment in ("raw", "qfq"):
            rows, report = _fetch_one(code, start=start, end=end, adjustment=adjustment, instrument_type="stock", baostock_state=state)
            (raw_rows if adjustment == "raw" else qfq_rows).extend(rows)
            requests.append({"instrument_code": code, "instrument_type": "stock", "adjustment": adjustment, **report})
    rows, report = _fetch_one(BENCHMARK_CODE, start=start, end=end, adjustment="raw", instrument_type="index", baostock_state=state)
    benchmark_rows.extend(rows)
    requests.append({"instrument_code": BENCHMARK_CODE, "instrument_type": "index", "adjustment": "raw", **report})
    if state.get("attempted_login"):
        try:
            state["module"].logout()
        except Exception:
            pass
    merged_raw = _merge_csv(output / "daily-bars-raw.csv", raw_rows)
    merged_qfq = _merge_csv(output / "daily-bars-qfq.csv", qfq_rows)
    merged_benchmark = _merge_csv(output / "benchmark-000300.csv", benchmark_rows)
    calendar_dates = sorted({str(row.get("trade_date")) for row in merged_benchmark if row.get("trade_date")})
    if not calendar_dates:
        calendar_dates = sorted({str(row.get("trade_date")) for row in merged_raw + merged_qfq if row.get("trade_date")})
    for item in requests:
        code = item["instrument_code"]
        adjustment = item["adjustment"]
        source_rows = merged_benchmark if code == BENCHMARK_CODE else (merged_raw if adjustment == "raw" else merged_qfq)
        dates = {str(row.get("trade_date")) for row in source_rows if row.get("instrument_code") == code and row.get("trade_date")}
        item["missing_dates"] = [day for day in calendar_dates if day not in dates]
        item["latest_trade_date"] = max(dates) if dates else None
    report = {
        "status": "MARKET_DATA_FETCHED",
        "provider_precedence": ["akshare", "baostock"],
        "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
        "latest_complete_trading_date": max(calendar_dates) if calendar_dates else None,
        "stock_codes": codes,
        "stocks_with_raw_rows": sorted({str(row["instrument_code"]) for row in merged_raw if row.get("instrument_type") == "stock"}),
        "stocks_with_qfq_rows": sorted({str(row["instrument_code"]) for row in merged_qfq if row.get("instrument_type") == "stock"}),
        "stock_trading_days": sorted({str(row["trade_date"]) for row in merged_raw if row.get("trade_date")}),
        "raw_row_count": len(merged_raw),
        "qfq_row_count": len(merged_qfq),
        "benchmark_row_count": len(merged_benchmark),
        "volume_units": {"akshare": "lots (手)", "baostock": "shares (股)"},
        "amount_unit": "CNY",
        "turnover_rate_unit": "percent",
        "requests": requests,
        "files": {
            name: {"path": str(output / name), "sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in ("instruments.json", "daily-bars-raw.csv", "daily-bars-qfq.csv", "benchmark-000300.csv")
        },
    }
    _write_json(output / "fetch-report.json", report)
    return report


def _parse_datetime(text: str) -> datetime:
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=MARKET_TZ)
    return value


def _stable_forward_sample_id(source: str, source_item_id: str, published_at: str) -> str:
    payload = f"m12-forward-shadow-v0|{source}|{source_item_id}|{published_at}".encode("utf-8")
    return "forward-" + hashlib.sha256(payload).hexdigest()


def _load_stock_names(targets_path: Path) -> dict[str, str]:
    try:
        value = json.loads(targets_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in value.get("stock_names", {}).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def earliest_qualified_collector_date(db_path: str | Path, *, cutoff: datetime | None = None, targets_path: str | Path = DEFAULT_TARGETS) -> date | None:
    """Return the earliest local publication date in the read-only collector.

    This uses the same label-free qualification rule as the forward-shadow
    selector.  Database timestamps are normalized to Asia/Shanghai before the
    calendar date is returned, so a UTC ``2026-07-08T16:05:10Z`` row correctly
    becomes the user's remembered **2026-07-09** start date.
    """

    cutoff = (cutoff or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    earliest: date | None = None
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT stock_code,title,content,published_at FROM posts").fetchall()
    for stock_code, title, content, published_at in rows:
        code = str(stock_code or "").strip()
        if len(code) != 6 or not code.isdigit() or not (str(content or "").strip() or str(title or "").strip()):
            continue
        try:
            published = _parse_datetime(str(published_at)).astimezone(MARKET_TZ)
        except (TypeError, ValueError):
            continue
        if published > cutoff:
            continue
        calendar_date = published.date()
        if earliest is None or calendar_date < earliest:
            earliest = calendar_date
    return earliest


def select_latest_qualified_posts(db_path: str | Path, *, cutoff: datetime, targets_path: str | Path = DEFAULT_TARGETS) -> list[dict[str, Any]]:
    """Select the latest publication day, without labels or author/url fields.

    The collector's simple ``posts`` table has no separate qualification flag.
    Here *qualified* is intentionally observable and conservative: non-empty
    six-digit ``stock_code``, non-empty title or content, parseable
    ``published_at``, and publication at or before the fixed cutoff.  The
    latest calendar publication day is selected as the forward-shadow batch.
    """

    cutoff = cutoff.astimezone(MARKET_TZ)
    names = _load_stock_names(Path(targets_path))
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        columns = {row[1] for row in connection.execute("pragma table_info(posts)")}
        required = {"source", "source_item_id", "stock_code", "title", "content", "published_at"}
        if not required.issubset(columns):
            raise ValueError(f"collector posts table lacks required columns: {sorted(required - columns)}")
        raw = connection.execute("SELECT source,source_item_id,stock_code,title,content,published_at FROM posts").fetchall()
    candidates: list[tuple[datetime, sqlite3.Row, str]] = []
    for row in raw:
        code = str(row["stock_code"] or "").strip()
        if len(code) != 6 or not code.isdigit():
            continue
        text = str(row["content"] or "").strip() or str(row["title"] or "").strip()
        if not text:
            continue
        try:
            published = _parse_datetime(str(row["published_at"]))
        except (TypeError, ValueError):
            continue
        if published > cutoff:
            continue
        candidates.append((published, row, text))
    if not candidates:
        return []
    latest_day = max(value.astimezone(MARKET_TZ).date() for value, _, _ in candidates)
    selected: list[dict[str, Any]] = []
    for published, row, text in candidates:
        if published.astimezone(MARKET_TZ).date() != latest_day:
            continue
        source = str(row["source"])
        item_id = str(row["source_item_id"])
        published_text = str(row["published_at"])
        selected.append({
            "sample_id": _stable_forward_sample_id(source, item_id, published_text),
            "observation_id": item_id,
            "source_item_id": item_id,
            "source": source,
            "stock_code": str(row["stock_code"]),
            "stock_name": names.get(str(row["stock_code"]), ""),
            "published_at": published_text,
            "model_text": text,
        })
    selected.sort(key=lambda item: (item["published_at"], item["stock_code"], item["source_item_id"]))
    if len({item["sample_id"] for item in selected}) != len(selected):
        raise ValueError("collector latest-day selection produced duplicate stable sample_id")
    return selected


def _next_weekdays(start: date, count: int) -> date:
    current = start
    remaining = count
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _market_dates(path: Path, code: str) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted({str(row["trade_date"]) for row in csv.DictReader(handle) if row.get("instrument_code") == code and row.get("trade_date")})


def run_forward_shadow(
    *,
    collector_db: str | Path = DEFAULT_COLLECTOR_DB,
    output_dir: str | Path = DEFAULT_OUTPUT,
    cutoff: datetime | None = None,
    m12_artifact: str | Path | None = None,
    selected_artifact: str | Path | None = None,
    cache_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    targets_path: str | Path = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """Freeze M12 predictions for the latest qualified publication day."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "forward-shadow-predictions.jsonl"
    pending_path = output / "forward-outcomes-pending.jsonl"
    if prediction_path.exists() or pending_path.exists():
        raise FileExistsError("forward-shadow outputs already exist; frozen snapshot will not be overwritten")
    cutoff = cutoff or datetime.now(MARKET_TZ)
    records = select_latest_qualified_posts(collector_db, cutoff=cutoff, targets_path=targets_path)
    if not records:
        raise RuntimeError("no qualified collector posts at the fixed cutoff")
    # Import the runtime only here: market-data normalization and tests do not
    # import Torch/Transformers, while the real execution uses the M12 baseline.
    from semantic_model.encoder_m12_hybrid_runtime import M12HybridInferenceRuntime
    from semantic_model.encoder_m2_selected_runtime import InferenceRecord

    kwargs: dict[str, Any] = {}
    if m12_artifact is not None:
        kwargs["m12_artifact"] = m12_artifact
    if selected_artifact is not None:
        kwargs["selected_artifact"] = selected_artifact
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if config_path is not None:
        kwargs["config_path"] = config_path
    runtime = M12HybridInferenceRuntime.from_artifacts(**kwargs)
    inference_records = [InferenceRecord(item["sample_id"], item["stock_code"], item["stock_name"], item["model_text"]) for item in records]
    predictions = runtime.predict_records(inference_records)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    enriched: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    raw_path = output / "daily-bars-raw.csv"
    qfq_path = output / "daily-bars-qfq.csv"
    for source, prediction in zip(records, predictions):
        enriched.append({
            "sample_id": source["sample_id"],
            "observation_id": source["observation_id"],
            "source_item_id": source["source_item_id"],
            "source": source["source"],
            "stock_code": source["stock_code"],
            "stock_name": source["stock_name"],
            "published_at": source["published_at"],
            "model_text": source["model_text"],
            "cutoff": cutoff.isoformat(),
            "prediction_generated_at": generated_at,
            "m12_artifact": str(Path(getattr(runtime, "artifact", kwargs.get("m12_artifact", ""))).resolve()) if getattr(runtime, "artifact", None) else str(kwargs.get("m12_artifact", "default-M12-artifact")),
            "selected_artifact": str(getattr(runtime.selected_runtime, "artifact_dir", kwargs.get("selected_artifact", "default-selected-artifact"))),
            "predictions": prediction.get("predictions", {}),
        })
        published = _parse_datetime(source["published_at"]).astimezone(MARKET_TZ)
        available_dates = _market_dates(qfq_path, source["stock_code"]) or _market_dates(raw_path, source["stock_code"])
        entry_date = next((day for day in available_dates if day > published.date().isoformat()), None)
        for horizon in (1, 3, 5):
            pending.append({
                "sample_id": source["sample_id"],
                "stock_code": source["stock_code"],
                "published_at": source["published_at"],
                "horizon_trading_days": horizon,
                "entry_rule": "next_trading_day_open",
                "entry_date": entry_date,
                "entry_open": None,
                "outcome_date": None,
                "status": "PENDING",
                "reason": "future market bars are not used to alter the frozen prediction",
            })
    _write_jsonl(prediction_path, enriched)
    _write_jsonl(pending_path, pending)
    stock_codes = sorted({item["stock_code"] for item in records})
    # The entry is the next trading-day open.  A one-day outcome is therefore
    # observable at that entry day's close; horizon ``h`` maps to ``h`` trading
    # days after the publication cutoff (not h+1).
    earliest = {str(h): _next_weekdays(cutoff.astimezone(MARKET_TZ).date(), h).isoformat() for h in (1, 3, 5)}
    summary = {
        "status": "FORWARD_SHADOW_FROZEN",
        "cutoff": cutoff.isoformat(),
        "prediction_generated_at": generated_at,
        "selection_rule": "latest qualified publication calendar day at cutoff; non-empty title/content and six-digit stock_code",
        "records": len(enriched),
        "stock_codes": stock_codes,
        "m12_artifact": str(getattr(runtime, "artifact", kwargs.get("m12_artifact", "default-M12-artifact"))),
        "selected_artifact": str(getattr(runtime.selected_runtime, "artifact_dir", kwargs.get("selected_artifact", "default-selected-artifact"))),
        "prediction_file": str(prediction_path),
        "pending_outcome_file": str(pending_path),
        "horizons": [1, 3, 5],
        "entry_rule": "next trading day open",
        "earliest_expected_result_dates_business_day_estimate": earliest,
        "labels_read": False,
        "future_bars_used_to_modify_predictions": False,
        "m16b_used": False,
    }
    _write_json(output / "forward-shadow-summary.json", summary)
    return summary


def write_readme(output_dir: str | Path) -> None:
    output = Path(output_dir).resolve()
    (output / "README.md").write_text(
        "# Market Data Bootstrap V0 (2026-09-16)\n\n"
        "This ignored directory contains the first local daily-bar snapshot and a frozen M12 forward-shadow.\n\n"
        "* `daily-bars-raw.csv` keeps unadjusted history; `daily-bars-qfq.csv` keeps front-adjusted history.\n"
        "* `volume_unit=lots` means AKShare's 手 and `volume_unit=shares` means BaoStock's 股; values are never silently converted. `amount_unit` is CNY and `turnover_rate_unit` is percent.\n"
        "* Rows are append-only by `(instrument_code, trade_date, adjustment)`. Re-running the market command does not duplicate an existing key.\n"
        "* AKShare is attempted first. BaoStock is a free fallback when an AKShare request fails; no token or paid service is used.\n"
        "* `forward-shadow-predictions.jsonl` is frozen at the recorded cutoff. `forward-outcomes-pending.jsonl` waits for the next trading-day open and 1/3/5 trading-day closes.\n\n"
        "Repeat (with optional provider packages available on `PYTHONPATH`):\n\n"
        "```sh\n"
        "PYTHONPATH=src:/path/to/market-provider-packages .encoder-venv/bin/python -m tools.market_data_bootstrap_v0\n"
        "```\n\n"
        "The forward-shadow files are immutable on rerun; choose a new output directory for a new cutoff.\n",
        encoding="utf-8",
    )


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    end = date.fromisoformat(args.end_date) if args.end_date else datetime.now(MARKET_TZ).date()
    start = date.fromisoformat(args.start_date) if args.start_date else (earliest_qualified_collector_date(args.collector_db) or DEFAULT_START)
    result: dict[str, Any] = {}
    if not args.forward_only:
        result["market"] = fetch_market_data(feed_path=args.feed_predictions, output_dir=args.output_dir, start=start, end=end, targets_path=args.targets_config)
        write_readme(args.output_dir)
    if not args.market_only:
        result["forward"] = run_forward_shadow(
            collector_db=args.collector_db,
            output_dir=args.output_dir,
            m12_artifact=args.m12_artifact,
            selected_artifact=args.selected_artifact,
            cache_dir=args.cache_dir,
            config_path=args.config,
            targets_path=args.targets_config,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-predictions", type=Path, default=DEFAULT_FEED)
    parser.add_argument("--collector-db", type=Path, default=DEFAULT_COLLECTOR_DB)
    parser.add_argument("--targets-config", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default=None, help="inclusive start date; defaults to the earliest qualified collector publication date")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--m12-artifact", type=Path, default=None)
    parser.add_argument("--selected-artifact", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--market-only", action="store_true", help="fetch bars and stop before loading the M12 runtime")
    parser.add_argument("--forward-only", action="store_true", help="freeze the forward shadow using existing market files")
    args = parser.parse_args(argv)
    if args.market_only and args.forward_only:
        parser.error("--market-only and --forward-only are mutually exclusive")
    try:
        result = run_all(args)
    except (FileExistsError, OSError, RuntimeError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "MARKET_DATA_BOOTSTRAP_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the actual CLI
    raise SystemExit(main())
