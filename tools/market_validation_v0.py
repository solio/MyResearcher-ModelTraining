"""Minimal market-outcome evaluation for frozen semantic predictions.

The primary descriptive unit is ``stock_code x entry_date``.  Individual
posts mapped to the same stock and entry session share the same outcome, so
treating them as independent observations would manufacture sample size.

Historical runs created after their market outcomes are explicitly marked as
pipeline diagnostics.  The same join code can later fill truly forward-frozen
predictions without changing the calculation contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "pipeline-runs" / "recent-chain-20260915" / "feed" / "prediction-snapshot.jsonl"
DEFAULT_MARKET_ROOT = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916"
DEFAULT_OUTPUT = PROJECT_ROOT / "pipeline-runs" / "market-validation-v0-20260916"
HORIZONS = (1, 3, 5)
MARKET_TZ = ZoneInfo("Asia/Shanghai")
SCALAR_HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "context_dependency",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            output.append(value)
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _publication_date(value: Any) -> date:
    text = str(value or "").strip()
    if len(text) < 10:
        raise ValueError(f"invalid published_at: {value!r}")
    return date.fromisoformat(text[:10])


def _datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MARKET_TZ)
    return parsed.astimezone(MARKET_TZ)


def signal_available_date(row: Mapping[str, Any], mode: str) -> date:
    if mode == "retrospective":
        return _publication_date(row.get("published_at"))
    if mode != "forward":
        raise ValueError(f"unsupported evaluation mode: {mode}")
    values = [row.get("published_at"), row.get("prediction_generated_at"), row.get("cutoff")]
    if any(not value for value in values):
        raise ValueError("forward prediction requires published_at, prediction_generated_at, and cutoff")
    return max(_datetime(value) for value in values).date()


def _bar_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["trade_date"]): row for row in rows}


def index_bars(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["instrument_code"])].append(row)
    for code in grouped:
        grouped[code].sort(key=lambda row: str(row["trade_date"]))
    return dict(grouped)


def _probabilities(row: Mapping[str, Any], head: str) -> dict[str, float]:
    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        return {}
    payload = predictions.get(head)
    if not isinstance(payload, Mapping):
        return {}
    values = payload.get("probabilities")
    if not isinstance(values, Mapping):
        return {}
    return {str(label): float(value) for label, value in values.items() if _float(value) is not None}


def _prediction(row: Mapping[str, Any], head: str) -> Any:
    predictions = row.get("predictions")
    payload = predictions.get(head) if isinstance(predictions, Mapping) else None
    return payload.get("prediction") if isinstance(payload, Mapping) else None


def stance_score(row: Mapping[str, Any]) -> float | None:
    probabilities = _probabilities(row, "stance")
    if not probabilities:
        return None
    return probabilities.get("BULL", 0.0) - probabilities.get("BEAR", 0.0)


def action_score(row: Mapping[str, Any]) -> float | None:
    probabilities = _probabilities(row, "action_tendency")
    if not probabilities:
        return None
    return (
        probabilities.get("BUY", 0.0)
        + probabilities.get("ADD", 0.0)
        - probabilities.get("SELL", 0.0)
        - probabilities.get("REDUCE", 0.0)
    )


def _mean_probabilities(rows: Sequence[Mapping[str, Any]], head: str) -> dict[str, float]:
    labels: set[str] = set()
    values = [_probabilities(row, head) for row in rows]
    for item in values:
        labels.update(item)
    if not labels:
        return {}
    return {label: statistics.fmean(item.get(label, 0.0) for item in values) for label in sorted(labels)}


def aggregate_market_units(
    predictions: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str = "retrospective",
) -> list[dict[str, Any]]:
    """Aggregate post probabilities by the shared ``stock x entry date`` outcome."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    trading_calendar = sorted(_bar_index(benchmark_rows))
    for row in predictions:
        available_date = signal_available_date(row, mode)
        entry_date = next(
            (trading_day for trading_day in trading_calendar if date.fromisoformat(trading_day) > available_date),
            None,
        )
        grouping_date = entry_date or f"PENDING_AFTER:{available_date.isoformat()}"
        grouped[(str(row.get("stock_code") or ""), grouping_date)].append(row)

    output: list[dict[str, Any]] = []
    for (stock_code, entry_date), rows in sorted(grouped.items()):
        predictions_payload: dict[str, Any] = {}
        for head in SCALAR_HEADS:
            probabilities = _mean_probabilities(rows, head)
            predictions_payload[head] = {
                "prediction": max(probabilities, key=probabilities.get) if probabilities else None,
                "probabilities": probabilities,
            }

        reasoning_probabilities = _mean_probabilities(rows, "reasoning_tags")
        threshold_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            reasoning = row.get("predictions", {}).get("reasoning_tags", {}) if isinstance(row.get("predictions"), Mapping) else {}
            thresholds = reasoning.get("thresholds", {}) if isinstance(reasoning, Mapping) else {}
            if isinstance(thresholds, Mapping):
                for label, value in thresholds.items():
                    number = _float(value)
                    if number is not None:
                        threshold_values[str(label)].append(number)
        thresholds = {label: statistics.fmean(values) for label, values in threshold_values.items()}
        predicted_tags = [
            label
            for label, probability in reasoning_probabilities.items()
            if probability >= thresholds.get(label, 0.5)
        ]
        predictions_payload["reasoning_tags"] = {
            "prediction": predicted_tags,
            "predicted_labels": predicted_tags,
            "probabilities": reasoning_probabilities,
            "thresholds": thresholds,
        }
        output.append(
            {
                "sample_id": f"market-unit:{stock_code}:{entry_date}",
                "stock_code": stock_code,
                "stock_name": next((str(row.get("stock_name")) for row in rows if row.get("stock_name")), ""),
                "published_at": max(str(row.get("published_at")) for row in rows),
                "published_dates": sorted({_publication_date(row.get("published_at")).isoformat() for row in rows}),
                "prediction_generated_at": max(
                    (str(row.get("prediction_generated_at")) for row in rows if row.get("prediction_generated_at")),
                    default=None,
                ),
                "cutoff": max((str(row.get("cutoff")) for row in rows if row.get("cutoff")), default=None),
                "entry_date_override": None if entry_date.startswith("PENDING_AFTER:") else entry_date,
                "post_count": len(rows),
                "source_sample_ids": [str(row.get("sample_id")) for row in rows],
                "predictions": predictions_payload,
            }
        )
    return output


def join_outcomes(
    predictions: Sequence[Mapping[str, Any]],
    *,
    qfq_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int] = HORIZONS,
    mode: str = "retrospective",
) -> list[dict[str, Any]]:
    """Join each frozen prediction to next-session-open forward outcomes."""

    qfq_by_code = index_bars(qfq_rows)
    raw_by_code = index_bars(raw_rows)
    benchmark_by_date = _bar_index(benchmark_rows)
    trading_calendar = sorted(benchmark_by_date)
    output: list[dict[str, Any]] = []

    for prediction in predictions:
        stock_code = str(prediction.get("stock_code") or "")
        published_date = _publication_date(prediction.get("published_at"))
        available_date = signal_available_date(prediction, mode)
        stock_qfq_by_date = _bar_index(qfq_by_code.get(stock_code, []))
        stock_raw_by_date = _bar_index(raw_by_code.get(stock_code, []))
        entry_override = prediction.get("entry_date_override")
        if entry_override:
            entry_calendar_index = next(
                (index for index, trading_day in enumerate(trading_calendar) if trading_day == str(entry_override)),
                None,
            )
        else:
            entry_calendar_index = next(
                (index for index, trading_day in enumerate(trading_calendar) if date.fromisoformat(trading_day) > available_date),
                None,
            )

        for horizon in horizons:
            base = {
                "sample_id": str(prediction.get("sample_id") or ""),
                "stock_code": stock_code,
                "stock_name": str(prediction.get("stock_name") or ""),
                "published_at": str(prediction.get("published_at") or ""),
                "published_date": published_date.isoformat(),
                "prediction_generated_at": prediction.get("prediction_generated_at"),
                "cutoff": prediction.get("cutoff"),
                "signal_available_date": available_date.isoformat(),
                "evaluation_mode": mode,
                "post_count": int(prediction.get("post_count") or 1),
                "horizon_trading_days": int(horizon),
                "entry_rule": (
                    "first_benchmark_trading_session_strictly_after_signal_available_date_open"
                    if mode == "forward"
                    else "first_benchmark_trading_session_after_publication_date_open"
                ),
                "stance_prediction": _prediction(prediction, "stance"),
                "stance_score": stance_score(prediction),
                "action_prediction": _prediction(prediction, "action_tendency"),
                "action_score": action_score(prediction),
                "head_predictions": {head: _prediction(prediction, head) for head in SCALAR_HEADS},
                "reasoning_tags": _prediction(prediction, "reasoning_tags") or [],
            }
            exit_calendar_index = None if entry_calendar_index is None else entry_calendar_index + int(horizon) - 1
            if entry_calendar_index is None or exit_calendar_index is None or exit_calendar_index >= len(trading_calendar):
                output.append(dict(base, status="PENDING_OR_MISSING", reason="insufficient future market-calendar bars"))
                continue

            entry_date = trading_calendar[entry_calendar_index]
            outcome_date = trading_calendar[exit_calendar_index]
            entry_qfq = stock_qfq_by_date.get(entry_date)
            exit_qfq = stock_qfq_by_date.get(outcome_date)
            if not entry_qfq or not exit_qfq:
                output.append(
                    dict(
                        base,
                        status="MISSING",
                        reason="stock bar missing on required market-calendar entry or outcome session",
                        entry_date=entry_date,
                        outcome_date=outcome_date,
                    )
                )
                continue
            entry_open_qfq = _float(entry_qfq.get("open"))
            outcome_close_qfq = _float(exit_qfq.get("close"))
            benchmark_entry = benchmark_by_date.get(entry_date)
            benchmark_exit = benchmark_by_date.get(outcome_date)
            if entry_open_qfq in (None, 0.0) or outcome_close_qfq is None:
                output.append(dict(base, status="MISSING", reason="stock open/close unavailable"))
                continue
            if not benchmark_entry or not benchmark_exit:
                output.append(dict(base, status="MISSING", reason="benchmark date unavailable"))
                continue
            benchmark_open = _float(benchmark_entry.get("open"))
            benchmark_close = _float(benchmark_exit.get("close"))
            if benchmark_open in (None, 0.0) or benchmark_close is None:
                output.append(dict(base, status="MISSING", reason="benchmark open/close unavailable"))
                continue

            stock_return = outcome_close_qfq / entry_open_qfq - 1.0
            benchmark_return = benchmark_close / benchmark_open - 1.0
            entry_raw = stock_raw_by_date.get(entry_date, {})
            exit_raw = stock_raw_by_date.get(outcome_date, {})
            entry_open_raw = _float(entry_raw.get("open"))
            outcome_close_raw = _float(exit_raw.get("close"))
            raw_return = (
                outcome_close_raw / entry_open_raw - 1.0
                if entry_open_raw not in (None, 0.0) and outcome_close_raw is not None
                else None
            )
            qfq_entry_vintage = str(entry_qfq.get("fetched_at") or "")
            qfq_outcome_vintage = str(exit_qfq.get("fetched_at") or "")
            qfq_vintage_match = qfq_entry_vintage == qfq_outcome_vintage
            raw_qfq_return_gap = abs(stock_return - raw_return) if raw_return is not None else None
            if not qfq_vintage_match and raw_qfq_return_gap is not None and raw_qfq_return_gap > 0.001:
                output.append(
                    dict(
                        base,
                        status="ADJUSTMENT_VINTAGE_MISMATCH",
                        reason="qfq entry/outcome came from different fetch vintages and diverge from raw return",
                        entry_date=entry_date,
                        outcome_date=outcome_date,
                        entry_open_qfq=entry_open_qfq,
                        outcome_close_qfq=outcome_close_qfq,
                        entry_open_raw=entry_open_raw,
                        outcome_close_raw=outcome_close_raw,
                        stock_return=stock_return,
                        stock_return_raw_unadjusted=raw_return,
                        qfq_entry_fetched_at=qfq_entry_vintage,
                        qfq_outcome_fetched_at=qfq_outcome_vintage,
                        qfq_vintage_match=False,
                        raw_qfq_return_gap=raw_qfq_return_gap,
                    )
                )
                continue
            output.append(
                dict(
                    base,
                    status="COMPLETE",
                    reason=None,
                    entry_date=entry_date,
                    outcome_date=outcome_date,
                    entry_open_qfq=entry_open_qfq,
                    outcome_close_qfq=outcome_close_qfq,
                    entry_open_raw=entry_open_raw,
                    outcome_close_raw=outcome_close_raw,
                    stock_return=stock_return,
                    stock_return_raw_unadjusted=raw_return,
                    qfq_entry_fetched_at=qfq_entry_vintage,
                    qfq_outcome_fetched_at=qfq_outcome_vintage,
                    qfq_vintage_match=qfq_vintage_match,
                    raw_qfq_return_gap=raw_qfq_return_gap,
                    benchmark_return=benchmark_return,
                    excess_return=stock_return - benchmark_return,
                    stock_up=stock_return > 0.0,
                    excess_positive=stock_return - benchmark_return > 0.0,
                )
            )
    return output


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return _pearson(_rank(xs), _rank(ys))


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "positive_rate": sum(value > 0.0 for value in clean) / len(clean),
    }


def _group_descriptive(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {
        label: {
            "stock_return": descriptive([float(row["stock_return"]) for row in items]),
            "excess_return": descriptive([float(row["excess_return"]) for row in items]),
        }
        for label, items in sorted(grouped.items())
    }


def _fixed_score_bands(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, Any]:
    """Apply outcome-independent direction bands frozen before forward results."""

    labels = ("NEGATIVE_STRONG", "NEGATIVE_WEAK", "POSITIVE_WEAK", "POSITIVE_STRONG")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        score = _float(row.get(score_key))
        if score is None:
            continue
        if score <= -0.25:
            label = "NEGATIVE_STRONG"
        elif score < 0.0:
            label = "NEGATIVE_WEAK"
        elif score < 0.25:
            label = "POSITIVE_WEAK"
        else:
            label = "POSITIVE_STRONG"
        grouped[label].append(row)
    return {
        label: {
            "score": descriptive([float(row[score_key]) for row in grouped.get(label, [])]),
            "stock_return": descriptive([float(row["stock_return"]) for row in grouped.get(label, [])]),
            "excess_return": descriptive([float(row["excess_return"]) for row in grouped.get(label, [])]),
        }
        for label in labels
    }


def _reasoning_tag_descriptive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        tags = row.get("reasoning_tags")
        if not isinstance(tags, list) or not tags:
            grouped["NO_SELECTED_TAGS"].append(row)
            continue
        for tag in tags:
            grouped[str(tag)].append(row)
    return {
        tag: {
            "stock_return": descriptive([float(row["stock_return"]) for row in items]),
            "excess_return": descriptive([float(row["excess_return"]) for row in items]),
        }
        for tag, items in sorted(grouped.items())
    }


def build_report(
    *,
    post_predictions: Sequence[Mapping[str, Any]],
    market_unit_predictions: Sequence[Mapping[str, Any]],
    post_outcomes: Sequence[Mapping[str, Any]],
    market_unit_outcomes: Sequence[Mapping[str, Any]],
    mode: str = "retrospective",
) -> dict[str, Any]:
    complete_units = sum(row.get("status") == "COMPLETE" for row in market_unit_outcomes)
    if mode == "forward":
        status = (
            "FORWARD_OUTCOMES_COMPLETE"
            if complete_units == len(market_unit_outcomes) and market_unit_outcomes
            else "FORWARD_OUTCOMES_PARTIAL"
            if complete_units
            else "FORWARD_OUTCOMES_PENDING"
        )
        warning = (
            "These predictions were frozen before their entry session. Treat completed outcomes as genuine "
            "out-of-time evidence, but do not tune on a single entry date or count posts as independent units."
        )
        entry_rule = "first CSI300 trading session strictly after signal availability date, at stock open"
    else:
        status = "RETROSPECTIVE_PIPELINE_DIAGNOSTIC_ONLY"
        warning = (
            "Predictions were generated after some or all historical outcomes and may overlap training data. "
            "Use this run to validate plumbing and form hypotheses, never as honest out-of-time performance."
        )
        entry_rule = "first CSI300 trading session strictly after publication date, at stock open"
    report: dict[str, Any] = {
        "status": status,
        "evaluation_mode": mode,
        "market_evidence": mode == "forward" and complete_units > 0,
        "warning": warning,
        "entry_rule": entry_rule,
        "horizons": list(HORIZONS),
        "post_predictions": len(post_predictions),
        "market_unit_predictions": len(market_unit_predictions),
        "primary_analysis_unit": "stock_code x entry_date",
        "same_stock_entry_posts_are_not_independent": True,
        "score_bands": {
            "NEGATIVE_STRONG": "score <= -0.25",
            "NEGATIVE_WEAK": "-0.25 < score < 0",
            "POSITIVE_WEAK": "0 <= score < 0.25",
            "POSITIVE_STRONG": "score >= 0.25",
        },
        "post_outcome_status": dict(sorted(_counts(post_outcomes, "status").items())),
        "market_unit_outcome_status": dict(sorted(_counts(market_unit_outcomes, "status").items())),
        "by_horizon": {},
    }
    for horizon in HORIZONS:
        rows = [
            row
            for row in market_unit_outcomes
            if row.get("status") == "COMPLETE" and int(row["horizon_trading_days"]) == horizon
        ]
        stance_x = [float(row["stance_score"]) for row in rows if _float(row.get("stance_score")) is not None]
        stance_y = [float(row["excess_return"]) for row in rows if _float(row.get("stance_score")) is not None]
        action_x = [float(row["action_score"]) for row in rows if _float(row.get("action_score")) is not None]
        action_y = [float(row["excess_return"]) for row in rows if _float(row.get("action_score")) is not None]
        head_groups: dict[str, Any] = {}
        for head in SCALAR_HEADS:
            head_rows = [dict(row, aggregate_label=(row.get("head_predictions") or {}).get(head)) for row in rows]
            head_groups[head] = _group_descriptive(head_rows, "aggregate_label")
        report["by_horizon"][str(horizon)] = {
            "complete_market_units": len(rows),
            "stocks": len({str(row["stock_code"]) for row in rows}),
            "entry_dates": len({str(row["entry_date"]) for row in rows}),
            "stock_return": descriptive([float(row["stock_return"]) for row in rows]),
            "excess_return": descriptive([float(row["excess_return"]) for row in rows]),
            "stance_score_spearman_vs_excess": spearman(stance_x, stance_y),
            "action_score_spearman_vs_excess": spearman(action_x, action_y),
            "stance_score_fixed_bands": _fixed_score_bands(rows, "stance_score"),
            "action_score_fixed_bands": _fixed_score_bands(rows, "action_score"),
            "stance_label": _group_descriptive(rows, "stance_prediction"),
            "action_label": _group_descriptive(rows, "action_prediction"),
            "all_scalar_heads_by_aggregate_label": head_groups,
            "reasoning_tags_by_aggregate_selection": _reasoning_tag_descriptive(rows),
        }
    return report


def _counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for row in rows:
        output[str(row.get(key))] += 1
    return dict(output)


def _percent(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number * 100:+.2f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    forward = report.get("evaluation_mode") == "forward"
    title = "Market Validation V0 — forward market evaluation" if forward else "Market Validation V0 — retrospective pipeline diagnostic"
    intro = (
        "These predictions were frozen before their entry session. Completed rows are genuine out-of-time observations; pending rows remain unresolved."
        if forward
        else "This run validates the prediction-to-market-outcome join and forms hypotheses. It is **not** honest market evidence: the predictions were generated after the historical outcomes and may overlap training data."
    )
    lines = [
        f"# {title}",
        "",
        f"Status: `{report['status']}`",
        "",
        intro,
        "",
        f"- Post predictions: {report['post_predictions']}",
        f"- Independent descriptive units: {report['market_unit_predictions']} stock × entry-date groups",
        f"- Entry: {report['entry_rule']}",
        "- Posts sharing the same stock and entry date are aggregated before analysis; post rows are not counted as independent market observations.",
        "",
        "## Directional diagnostics",
        "",
        "| Horizon | Market units | Entry dates | Stance score vs excess Spearman | Action score vs excess Spearman | Fixed-band stance excess means | Fixed-band action excess means |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for horizon in HORIZONS:
        item = report["by_horizon"][str(horizon)]
        stance_bins = item["stance_score_fixed_bands"]
        action_bins = item["action_score_fixed_bands"]
        band_order = ("NEGATIVE_STRONG", "NEGATIVE_WEAK", "POSITIVE_WEAK", "POSITIVE_STRONG")
        stance_means = " / ".join(_percent(stance_bins[label]["excess_return"].get("mean")) for label in band_order)
        action_means = " / ".join(_percent(action_bins[label]["excess_return"].get("mean")) for label in band_order)
        stance_corr = item["stance_score_spearman_vs_excess"]
        action_corr = item["action_score_spearman_vs_excess"]
        lines.append(
            f"| {horizon}d | {item['complete_market_units']} | {item['entry_dates']} | "
            f"{'n/a' if stance_corr is None else f'{stance_corr:+.3f}'} | "
            f"{'n/a' if action_corr is None else f'{action_corr:+.3f}'} | {stance_means} | {action_means} |"
        )
    lines.extend(
        [
            "",
            "The table is descriptive only. With few stock-entry units and repeated stocks across dates, no significance or trading claim is authorized.",
            "",
            "## Next evidence",
            "",
            (
                "Fetch the next complete market session and rerun this unchanged contract until 1d/3d/5d outcomes resolve. Do not alter the frozen predictions."
                if forward
                else "Use the already frozen 2026-09-16 M12 forward shadow. Fill its outcomes with this unchanged entry/horizon contract after the corresponding market bars exist. That future run—not this retrospective diagnostic—is the first honest market check."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    predictions_path: str | Path,
    qfq_path: str | Path,
    raw_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    mode: str = "retrospective",
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    qfq_rows = read_csv(qfq_path)
    raw_rows = read_csv(raw_path)
    benchmark_rows = read_csv(benchmark_path)
    market_units = aggregate_market_units(predictions, benchmark_rows, mode=mode)
    post_outcomes = join_outcomes(
        predictions,
        qfq_rows=qfq_rows,
        raw_rows=raw_rows,
        benchmark_rows=benchmark_rows,
        mode=mode,
    )
    market_unit_outcomes = join_outcomes(
        market_units,
        qfq_rows=qfq_rows,
        raw_rows=raw_rows,
        benchmark_rows=benchmark_rows,
        mode=mode,
    )
    report = build_report(
        post_predictions=predictions,
        market_unit_predictions=market_units,
        post_outcomes=post_outcomes,
        market_unit_outcomes=market_unit_outcomes,
        mode=mode,
    )
    output = Path(output_dir)
    prefix = "forward" if mode == "forward" else "retrospective"
    write_jsonl(output / f"{prefix}-post-outcomes.jsonl", post_outcomes)
    write_jsonl(output / f"{prefix}-market-unit-outcomes.jsonl", market_unit_outcomes)
    write_json(output / f"{prefix}-report.json", report)
    report_name = "FORWARD_REPORT.md" if mode == "forward" else "REPORT.md"
    (output / report_name).write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--qfq", type=Path, default=DEFAULT_MARKET_ROOT / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=DEFAULT_MARKET_ROOT / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_MARKET_ROOT / "benchmark-000300.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("retrospective", "forward"), default="retrospective")
    args = parser.parse_args(argv)
    report = run(
        predictions_path=args.predictions,
        qfq_path=args.qfq,
        raw_path=args.raw,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
