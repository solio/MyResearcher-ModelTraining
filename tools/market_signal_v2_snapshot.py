"""Freeze forward community-state snapshots and independently backfill outcomes.

The snapshot path accepts only M12 prediction JSONL and a date-only CSI300
trading calendar. It deliberately has no market-price input. Outcome joining is
a separate command which writes a separate file and never edits snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from tools import market_signal_v1 as v1
from tools import market_signal_v2 as v2
from tools import market_validation_v0 as validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916" / "forward-shadow-predictions.jsonl"
DEFAULT_CALENDAR = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916" / "trading-calendar-dates.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "pipeline-runs" / "market-signal-v2-snapshot"
MARKET_TZ = ZoneInfo("Asia/Shanghai")
HORIZONS = (1, 3, 5)
SNAPSHOT_SCHEMA = "market-signal-v2-community-state-snapshot-v1"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_trading_dates(path: str | Path) -> list[str]:
    """Read a date-only calendar; reject market data columns by construction."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"trade_dates"}:
            raise ValueError("calendar JSON must contain only a trade_dates array")
        raw_dates = value["trade_dates"]
    elif source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["trade_date"]:
                raise ValueError("calendar CSV must have exactly one trade_date column")
            raw_dates = [row["trade_date"] for row in reader]
    else:
        raw_dates = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("trading calendar is empty")
    dates = [date.fromisoformat(str(value)).isoformat() for value in raw_dates]
    if len(dates) != len(set(dates)):
        raise ValueError("trading calendar contains duplicate dates")
    return sorted(dates)


def _normalize_text(value: Any) -> str:
    return v1._normalize_text(value)


def _validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("prediction input is empty")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prediction input requires unique non-empty sample_id values")
    for row in rows:
        code = str(row.get("stock_code") or "")
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"invalid stock_code for {row.get('sample_id')}: {code!r}")
        if not _normalize_text(row.get("model_text")):
            raise ValueError(f"missing model_text for {row.get('sample_id')}")
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError(f"missing predictions for {row.get('sample_id')}")
        for head in v2.GROUP_HEADS:
            probabilities = validation._probabilities(row, head)
            if len(probabilities) < 2:
                raise ValueError(f"missing full {head} probabilities for {row.get('sample_id')}")
            total = sum(probabilities.values())
            if not math.isfinite(total) or total <= 0:
                raise ValueError(f"invalid {head} probabilities for {row.get('sample_id')}")
        # Forward availability must include both inference time and cutoff.
        validation.signal_available_date(row, "forward")


def _snapshot_status(
    rows: Sequence[Mapping[str, Any]], *, entry_date: str, generated_at: datetime
) -> str:
    entry = date.fromisoformat(entry_date)
    if generated_at.date() >= entry:
        return "BACKFILL_SMOKE"
    for row in rows:
        available = validation.signal_available_date(row, "forward")
        if available >= entry:
            return "BACKFILL_SMOKE"
    return "FORWARD"


def build_community_snapshots(
    prediction_rows: Sequence[Mapping[str, Any]],
    trading_dates: Sequence[str],
    *,
    generated_at: datetime,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Aggregate unique normalized posts by stock and next actual session.

    ``trading_dates`` must be a trusted date-only CSI300 calendar. No OHLC or
    outcome values are accepted by this function.
    """

    _validate_prediction_rows(prediction_rows)
    calendar = sorted({date.fromisoformat(value).isoformat() for value in trading_dates})
    if not calendar:
        raise ValueError("trading calendar is empty")
    generated_at = generated_at.astimezone(MARKET_TZ)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        signal_available = validation.signal_available_date(row, "forward").isoformat()
        entry_date = v1.next_entry_date(signal_available, calendar)
        if entry_date is None:
            raise ValueError(f"calendar has no next actual session after {signal_available}")
        grouped[(str(row["stock_code"]), entry_date)].append(row)

    output: list[dict[str, Any]] = []
    for (stock_code, entry_date), posts in sorted(grouped.items()):
        # Keep the earliest sample for each exact normalized text; duplicates
        # affect attention counts but do not overweight the group distribution.
        unique_by_text: dict[str, Mapping[str, Any]] = {}
        for row in sorted(posts, key=lambda item: (str(item.get("published_at")), str(item["sample_id"]))):
            unique_by_text.setdefault(_normalize_text(row.get("model_text")), row)
        unique_posts = list(unique_by_text.values())
        metadata_rows = [
            {
                "sample_id": str(row["sample_id"]),
                "stock_code": stock_code,
                "entry_date": entry_date,
                "unit_total_post_count": len(posts),
                "unit_unique_text_count": len(unique_posts),
            }
            for row in unique_posts
        ]
        features_by_unit = v2.aggregate_crowd_state(metadata_rows, unique_posts)
        features = features_by_unit[f"market-signal-unit:{stock_code}:{entry_date}"]

        distributions: dict[str, dict[str, float]] = {}
        for head in v2.GROUP_HEADS:
            distributions[head] = {
                key.removeprefix(f"mean::{head}::"): value
                for key, value in features.items()
                if key.startswith(f"mean::{head}::")
            }
        post_dates = sorted({validation._publication_date(row.get("published_at")).isoformat() for row in posts})
        available_dates = sorted(
            {validation.signal_available_date(row, "forward").isoformat() for row in posts}
        )
        per_head_uncertainty = {
            head: features[f"model_uncertainty::{head}"] for head in v2.GROUP_HEADS
        }
        per_head_disagreement = {head: features[f"group_disagreement::{head}"] for head in v2.GROUP_HEADS}
        per_head_agreement = {head: features[f"cross_post_agreement::{head}"] for head in v2.GROUP_HEADS}
        per_head_concentration = {head: features[f"group_consensus::{head}"] for head in v2.GROUP_HEADS}
        model_artifacts = sorted(
            {
                str(row.get("m12_artifact") or row.get("selected_artifact") or "")
                for row in posts
                if row.get("m12_artifact") or row.get("selected_artifact")
            }
        )
        if len(model_artifacts) > 1:
            raise ValueError(f"mixed model artifacts in {stock_code} x {entry_date}: {model_artifacts}")
        snapshot_id = f"market-signal-v2-snapshot:{stock_code}:{entry_date}:{source_sha256[:12]}"
        output.append(
            {
                "schema": SNAPSHOT_SCHEMA,
                "snapshot_id": snapshot_id,
                "snapshot_status": _snapshot_status(posts, entry_date=entry_date, generated_at=generated_at),
                "stock_code": stock_code,
                "stock_name": next((str(row.get("stock_name")) for row in posts if row.get("stock_name")), ""),
                "signal_date_range": {"start": min(post_dates), "end": max(post_dates)},
                "signal_available_date_range": {"start": min(available_dates), "end": max(available_dates)},
                "next_trading_entry_date": entry_date,
                "generated_at": generated_at.isoformat(),
                "post_count": len(posts),
                "deduplicated_count": len(unique_posts),
                "duplicate_post_count": len(posts) - len(unique_posts),
                "sampling_coverage": 1.0 if unique_posts else 0.0,
                "sampling_coverage_scope": "deduplicated records present in this supplied prediction snapshot only; upstream population coverage is not asserted",
                "distributions": distributions,
                "direction_indices": {
                    "stance_bull_minus_bear": features["stance_index"],
                    "action_buy_add_minus_sell_reduce": features["action_index"],
                },
                "emotion_summary": {
                    "positive_mass": features["emotion_positive_mass"],
                    "negative_mass": features["emotion_negative_mass"],
                    "balance": features["emotion_balance"],
                    "fomo_mass": features["emotion_fomo_mass"],
                },
                "attention": {
                    "post_count": len(posts),
                    "deduplicated_count": len(unique_posts),
                    "duplicate_post_count": len(posts) - len(unique_posts),
                    "deduplicated_ratio": len(unique_posts) / len(posts) if posts else 0.0,
                    "log1p_total_post_count": features["log1p_total_post_count"],
                    "log1p_unique_text_count": features["log1p_unique_text_count"],
                    "unique_text_ratio": features["unique_text_ratio"],
                    "sampling_coverage": features["sampling_coverage"],
                    "sampling_coverage_scope": "deduplicated records present in this supplied prediction snapshot only; upstream population coverage is not asserted",
                },
                "model_uncertainty": {
                    "per_head": per_head_uncertainty,
                    "mean": statistics.fmean(per_head_uncertainty.values()),
                },
                "group_disagreement": {
                    "per_head": per_head_disagreement,
                    "mean": statistics.fmean(per_head_disagreement.values()),
                },
                "cross_post_agreement": {
                    "per_head": per_head_agreement,
                    "mean": statistics.fmean(per_head_agreement.values()),
                },
                "group_concentration": {
                    "per_head": per_head_concentration,
                    "mean": statistics.fmean(per_head_concentration.values()),
                },
                "m12_artifact": model_artifacts[0] if model_artifacts else None,
                "source_prediction_sha256": source_sha256,
                "market_prices_read": False,
                "outcomes_included": False,
            }
        )
    return output


def _read_qfq_bars(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"instrument_code", "trade_date", "adjustment", "open", "close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("qfq bars are missing instrument_code/trade_date/adjustment/open/close")
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in reader:
            if str(row.get("adjustment")) != "qfq":
                continue
            key = (str(row["instrument_code"]), str(row["trade_date"]))
            if key in rows:
                raise ValueError(f"duplicate qfq market bar: {key}")
            rows[key] = dict(row)
    if not rows:
        raise ValueError("qfq bars contain no adjustment=qfq records")
    return rows


def backfill_outcomes(
    snapshot_rows: Sequence[Mapping[str, Any]],
    trading_dates: Sequence[str],
    qfq_bars: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    as_of_date: date,
) -> list[dict[str, Any]]:
    """Create separate 1/3/5-session absolute-return rows; never mutate snapshots."""

    calendar = sorted({date.fromisoformat(value).isoformat() for value in trading_dates})
    calendar_index = {value: index for index, value in enumerate(calendar)}
    results: list[dict[str, Any]] = []
    for snapshot in snapshot_rows:
        entry_date = str(snapshot["next_trading_entry_date"])
        code = str(snapshot["stock_code"])
        entry_index = calendar_index.get(entry_date)
        for horizon in HORIZONS:
            base = {
                "snapshot_id": str(snapshot["snapshot_id"]),
                "stock_code": code,
                "entry_date": entry_date,
                "horizon_trading_days": int(horizon),
                "evidence_type": str(snapshot.get("snapshot_status") or "UNKNOWN"),
                "entry_rule": "next_actual_trading_session_open_to_horizon_session_close_qfq",
            }
            if entry_index is None:
                results.append(dict(base, status="PENDING_CALENDAR", reason="entry date absent from calendar"))
                continue
            exit_index = entry_index + int(horizon) - 1
            if exit_index >= len(calendar):
                results.append(dict(base, status="PENDING_HORIZON", reason="horizon trading session not yet listed"))
                continue
            exit_date = calendar[exit_index]
            if date.fromisoformat(exit_date) > as_of_date:
                results.append(
                    dict(base, status="PENDING_HORIZON", reason="horizon date is after as_of_date", outcome_date=exit_date)
                )
                continue
            entry_bar = qfq_bars.get((code, entry_date))
            exit_bar = qfq_bars.get((code, exit_date))
            if not entry_bar or not exit_bar:
                results.append(
                    dict(
                        base,
                        status="MISSING_BAR",
                        reason="matured entry or outcome qfq bar unavailable",
                        outcome_date=exit_date,
                    )
                )
                continue
            entry_vintage = str(entry_bar.get("fetched_at") or "")
            exit_vintage = str(exit_bar.get("fetched_at") or "")
            if not entry_vintage or not exit_vintage or entry_vintage != exit_vintage:
                results.append(
                    dict(
                        base,
                        status="ADJUSTMENT_VINTAGE_MISMATCH",
                        reason="entry and outcome qfq bars must share a fetch vintage",
                        outcome_date=exit_date,
                        entry_fetched_at=entry_vintage or None,
                        outcome_fetched_at=exit_vintage or None,
                    )
                )
                continue
            entry_open = validation._float(entry_bar.get("open"))
            exit_close = validation._float(exit_bar.get("close"))
            if entry_open in (None, 0.0) or exit_close is None:
                results.append(
                    dict(base, status="MISSING_PRICE", reason="entry open or outcome close unavailable", outcome_date=exit_date)
                )
                continue
            results.append(
                dict(
                    base,
                    status="COMPLETE",
                    reason=None,
                    outcome_date=exit_date,
                    entry_open_qfq=entry_open,
                    outcome_close_qfq=exit_close,
                    absolute_return=exit_close / entry_open - 1.0,
                    entry_fetched_at=entry_vintage,
                    outcome_fetched_at=exit_vintage,
                )
            )
    return results


def _snapshot_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    snapshot_path = output_dir / "community-state-snapshots.jsonl"
    summary_path = output_dir / "snapshot-summary.json"
    if snapshot_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output in {output_dir}")
    input_path = Path(args.input)
    raw_bytes = input_path.read_bytes()
    rows = _read_jsonl(input_path)
    generated_at = datetime.now(MARKET_TZ).replace(microsecond=0)
    snapshots = build_community_snapshots(
        rows,
        load_trading_dates(args.trading_calendar),
        generated_at=generated_at,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    _write_jsonl(snapshot_path, snapshots)
    _write_json(
        summary_path,
        {
            "status": "MARKET_SIGNAL_V2_COMMUNITY_SNAPSHOT_COMPLETE",
            "schema": SNAPSHOT_SCHEMA,
            "input_path": str(input_path.resolve()),
            "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "input_posts": len(rows),
            "snapshot_count": len(snapshots),
            "deduplicated_posts": sum(int(row["deduplicated_count"]) for row in snapshots),
            "duplicate_posts_removed": sum(int(row["duplicate_post_count"]) for row in snapshots),
            "sampling_coverage_scope": "deduplicated records present in this supplied prediction snapshot only; upstream population coverage is not asserted",
            "stock_codes": sorted({str(row["stock_code"]) for row in snapshots}),
            "entry_dates": sorted({str(row["next_trading_entry_date"]) for row in snapshots}),
            "snapshot_status_counts": {
                status: sum(row["snapshot_status"] == status for row in snapshots)
                for status in sorted({str(row["snapshot_status"]) for row in snapshots})
            },
            "generated_at": generated_at.isoformat(),
            "market_prices_read": False,
            "outcomes_included": False,
            "files": {
                "snapshots": snapshot_path.name,
                "summary": summary_path.name,
            },
        },
    )
    print(json.dumps({"snapshot_count": len(snapshots), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


def _backfill_command(args: argparse.Namespace) -> int:
    snapshot_rows = _read_jsonl(args.snapshot)
    dates = load_trading_dates(args.trading_calendar)
    qfq = _read_qfq_bars(args.qfq_bars)
    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else datetime.now(MARKET_TZ).date()
    rows = backfill_outcomes(snapshot_rows, dates, qfq, as_of_date=as_of)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing outcomes: {output_path}")
    _write_jsonl(output_path, rows)
    print(json.dumps({"outcome_rows": len(rows), "output": str(output_path)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="freeze community-state snapshot without market prices")
    snapshot.add_argument("--input", default=str(DEFAULT_INPUT))
    snapshot.add_argument("--trading-calendar", default=str(DEFAULT_CALENDAR))
    snapshot.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    snapshot.set_defaults(func=_snapshot_command)
    backfill = subparsers.add_parser("backfill-outcomes", help="write separate realized-return outcome rows")
    backfill.add_argument("--snapshot", required=True)
    backfill.add_argument("--trading-calendar", required=True)
    backfill.add_argument("--qfq-bars", required=True)
    backfill.add_argument("--output", required=True)
    backfill.add_argument("--as-of-date", help="ISO date; defaults to current Asia/Shanghai date")
    backfill.set_defaults(func=_backfill_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
