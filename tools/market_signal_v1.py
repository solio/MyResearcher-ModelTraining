"""Build and evaluate the first stock-session market-signal panel.

This module deliberately separates the two model layers:

* M12 converts sampled forum posts into seven-head semantic probabilities.
* A small, fixed Logistic Regression layer asks whether those probabilities
  add information about future excess-return probability beyond already-known
  market features.

The independent unit is ``stock_code x entry_date``.  Posts whose Shanghai
publication dates map to the same next CSI300 trading session are one signal
batch, including weekends and holidays.  Historical M12 inference is a
retrospective development feature: the purged walk-forward below is honest at
the market layer, but it is not full-system forward evidence because M12 was
frozen after part of the historical text period.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from tools import market_validation_v0 as validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECTOR_DB = PROJECT_ROOT / "MyResearcher-DataCollector" / "data" / "collector.db"
DEFAULT_TARGETS = PROJECT_ROOT / "MyResearcher-DataCollector" / "config" / "targets.short-term.json"
DEFAULT_MARKET_ROOT = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916"
DEFAULT_OUTPUT = PROJECT_ROOT / "pipeline-runs" / "market-signal-v1-20260917"
MARKET_TZ = ZoneInfo("Asia/Shanghai")
HORIZONS = (1, 3, 5)
HEADS = (*validation.SCALAR_HEADS, "reasoning_tags")
DEFAULT_POST_CAP = 16
DEFAULT_MIN_TRAIN_DATES = 15
LOGISTIC_C = 0.1


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return validation.read_jsonl(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MARKET_TZ)
    return parsed.astimezone(MARKET_TZ)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _stable_sample_id(source: str, source_item_id: str, published_at: str) -> str:
    payload = f"market-signal-v1|{source}|{source_item_id}|{published_at}".encode("utf-8")
    return "msv1-" + hashlib.sha256(payload).hexdigest()


def load_stock_names(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    names = value.get("stock_names", {})
    if not isinstance(names, Mapping):
        raise ValueError("targets stock_names must be an object")
    return {str(code): str(name) for code, name in names.items()}


def benchmark_calendar(path: Path) -> list[str]:
    rows = validation.read_csv(path)
    calendar = sorted({str(row.get("trade_date") or "") for row in rows if row.get("trade_date")})
    if not calendar:
        raise ValueError("benchmark calendar is empty")
    return calendar


def next_entry_date(publication_day: str, calendar: Sequence[str]) -> str | None:
    """Return the first actual benchmark session strictly after a signal day."""

    return next((trading_day for trading_day in calendar if trading_day > publication_day), None)


def systematic_sample(rows: Sequence[Mapping[str, Any]], cap: int) -> list[Mapping[str, Any]]:
    """Take deterministic time-spanning rows without consulting market targets."""

    if cap < 1:
        raise ValueError("post cap must be positive")
    if len(rows) <= cap:
        return list(rows)
    if cap == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (cap - 1)) for index in range(cap)]
    if len(set(indices)) != cap:
        raise AssertionError("systematic sample produced duplicate indices")
    return [rows[index] for index in indices]


def select_panel_posts(
    *,
    collector_db: Path,
    benchmark_path: Path,
    targets_path: Path,
    post_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select a fixed number of unique texts per stock-entry market unit."""

    calendar = benchmark_calendar(benchmark_path)
    names = load_stock_names(targets_path)
    uri = collector_db.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in connection.execute("pragma table_info(posts)")}
        required = {"source", "source_item_id", "stock_code", "title", "content", "published_at"}
        if not required.issubset(columns):
            raise ValueError(f"collector posts table lacks columns: {sorted(required - columns)}")
        source_rows = connection.execute(
            "SELECT source,source_item_id,stock_code,title,content,published_at FROM posts"
        ).fetchall()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pending_after_calendar = 0
    invalid_rows = 0
    for row in source_rows:
        code = str(row["stock_code"] or "").strip()
        text = _normalize_text(row["content"]) or _normalize_text(row["title"])
        if len(code) != 6 or not code.isdigit() or not text:
            invalid_rows += 1
            continue
        try:
            published = _parse_datetime(row["published_at"])
        except (TypeError, ValueError):
            invalid_rows += 1
            continue
        signal_date = published.date().isoformat()
        entry_date = next_entry_date(signal_date, calendar)
        if entry_date is None:
            pending_after_calendar += 1
            continue
        source = str(row["source"] or "")
        item_id = str(row["source_item_id"] or "")
        grouped[(code, entry_date)].append(
            {
                "sample_id": _stable_sample_id(source, item_id, str(row["published_at"])),
                "source": source,
                "source_item_id": item_id,
                "stock_code": code,
                "stock_name": names.get(code, ""),
                "published_at": str(row["published_at"]),
                "signal_date": signal_date,
                "entry_date": entry_date,
                "model_text": text,
                "title": str(row["title"] or ""),
                "content": str(row["content"] or ""),
                "text_source": "content" if _normalize_text(row["content"]) else "title",
            }
        )

    inference_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    unit_counts: list[int] = []
    unique_counts: list[int] = []
    for (code, entry_date), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: (item["published_at"], item["sample_id"]))
        unique_by_text: dict[str, dict[str, Any]] = {}
        for item in ordered:
            unique_by_text.setdefault(item["model_text"], item)
        unique_rows = list(unique_by_text.values())
        selected = systematic_sample(unique_rows, post_cap)
        unit_counts.append(len(ordered))
        unique_counts.append(len(unique_rows))
        for item in selected:
            inference_rows.append(
                {
                    "sample_id": item["sample_id"],
                    "stock_code": code,
                    "stock_name": item["stock_name"],
                    "model_text": item["model_text"],
                }
            )
            metadata_rows.append(
                {
                    **item,
                    "unit_total_post_count": len(ordered),
                    "unit_unique_text_count": len(unique_rows),
                    "unit_sampled_post_count": len(selected),
                }
            )

    sample_ids = [row["sample_id"] for row in inference_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("panel inference sample IDs are not unique")
    units = sorted(grouped)
    summary = {
        "status": "MARKET_SIGNAL_V1_INPUT_READY",
        "collector_db": str(collector_db.resolve()),
        "collector_rows": len(source_rows),
        "invalid_rows": invalid_rows,
        "pending_rows_after_market_calendar": pending_after_calendar,
        "market_units": len(units),
        "stocks": len({code for code, _ in units}),
        "entry_dates": len({entry for _, entry in units}),
        "entry_date_min": min((entry for _, entry in units), default=None),
        "entry_date_max": max((entry for _, entry in units), default=None),
        "eligible_posts": sum(unit_counts),
        "unique_texts": sum(unique_counts),
        "post_cap_per_market_unit": post_cap,
        "inference_posts": len(inference_rows),
        "selection_rule": "deduplicate exact normalized model_text within stock x entry_date, then deterministic time-spanning sample",
        "market_outcomes_read_for_selection": False,
    }
    return inference_rows, metadata_rows, summary


def prepare_inputs(
    *,
    collector_db: Path,
    benchmark_path: Path,
    targets_path: Path,
    output_dir: Path,
    post_cap: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inference": output_dir / "m12-inference-input.jsonl",
        "metadata": output_dir / "sampled-post-metadata.jsonl",
        "summary": output_dir / "input-summary.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Market Signal V1 input outputs already exist: {existing}")
    inference, metadata, summary = select_panel_posts(
        collector_db=collector_db,
        benchmark_path=benchmark_path,
        targets_path=targets_path,
        post_cap=post_cap,
    )
    _write_jsonl(paths["inference"], inference)
    _write_jsonl(paths["metadata"], metadata)
    summary = dict(
        summary,
        inference_path=str(paths["inference"]),
        metadata_path=str(paths["metadata"]),
        inference_sha256=_sha256(paths["inference"]),
        metadata_sha256=_sha256(paths["metadata"]),
    )
    _write_json(paths["summary"], summary)
    return summary


def _probabilities(row: Mapping[str, Any], head: str) -> dict[str, float]:
    predictions = row.get("predictions")
    payload = predictions.get(head) if isinstance(predictions, Mapping) else None
    probabilities = payload.get("probabilities") if isinstance(payload, Mapping) else None
    if not isinstance(probabilities, Mapping):
        raise ValueError(f"prediction row lacks {head} probabilities")
    result = {str(label): float(value) for label, value in probabilities.items()}
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"prediction row has invalid {head} probabilities")
    return result


def _entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in probabilities.values() if value > 0.0)


def aggregate_semantic_units(
    metadata_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    predictions_by_id = {str(row.get("sample_id")): row for row in prediction_rows}
    metadata_ids = [str(row.get("sample_id")) for row in metadata_rows]
    if len(predictions_by_id) != len(prediction_rows) or set(predictions_by_id) != set(metadata_ids):
        raise ValueError("metadata and M12 predictions differ in sample_id coverage")

    class_order: dict[str, list[str]] = {}
    for head in HEADS:
        labels: set[str] = set()
        for row in prediction_rows:
            labels.update(_probabilities(row, head))
        class_order[head] = sorted(labels)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for metadata in metadata_rows:
        grouped[(str(metadata["stock_code"]), str(metadata["entry_date"]))].append(metadata)

    units: list[dict[str, Any]] = []
    for (code, entry_date), items in sorted(grouped.items()):
        predictions = [predictions_by_id[str(item["sample_id"])] for item in items]
        semantic: dict[str, float] = {}
        mean_probabilities: dict[str, dict[str, float]] = {}
        for head in HEADS:
            row_probabilities = [_probabilities(row, head) for row in predictions]
            means = {
                label: statistics.fmean(probabilities.get(label, 0.0) for probabilities in row_probabilities)
                for label in class_order[head]
            }
            mean_probabilities[head] = means
            for label, value in means.items():
                semantic[f"mean::{head}::{label}"] = value
            semantic[f"entropy_mean::{head}"] = statistics.fmean(_entropy(value) for value in row_probabilities)

        stance_values = [
            probabilities.get("BULL", 0.0) - probabilities.get("BEAR", 0.0)
            for probabilities in (_probabilities(row, "stance") for row in predictions)
        ]
        action_values = [
            probabilities.get("BUY", 0.0)
            + probabilities.get("ADD", 0.0)
            - probabilities.get("SELL", 0.0)
            - probabilities.get("REDUCE", 0.0)
            for probabilities in (_probabilities(row, "action_tendency") for row in predictions)
        ]
        total_posts = int(items[0]["unit_total_post_count"])
        unique_texts = int(items[0]["unit_unique_text_count"])
        semantic.update(
            {
                "stance_score_mean": statistics.fmean(stance_values),
                "stance_score_std": statistics.pstdev(stance_values) if len(stance_values) > 1 else 0.0,
                "action_score_mean": statistics.fmean(action_values),
                "action_score_std": statistics.pstdev(action_values) if len(action_values) > 1 else 0.0,
                "log1p_total_post_count": math.log1p(total_posts),
                "log1p_unique_text_count": math.log1p(unique_texts),
                "unique_text_ratio": unique_texts / total_posts if total_posts else 0.0,
            }
        )
        aggregate_predictions: dict[str, Any] = {}
        for head in validation.SCALAR_HEADS:
            probabilities = mean_probabilities[head]
            aggregate_predictions[head] = {
                "prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
        reasoning = mean_probabilities["reasoning_tags"]
        aggregate_predictions["reasoning_tags"] = {
            "prediction": [label for label, probability in reasoning.items() if probability >= 0.5],
            "probabilities": reasoning,
            "thresholds": {label: 0.5 for label in reasoning},
        }
        units.append(
            {
                "sample_id": f"market-signal-unit:{code}:{entry_date}",
                "stock_code": code,
                "stock_name": str(items[0].get("stock_name") or ""),
                "published_at": max(str(item["published_at"]) for item in items),
                "published_dates": sorted({str(item["signal_date"]) for item in items}),
                "entry_date_override": entry_date,
                "post_count": total_posts,
                "unique_text_count": unique_texts,
                "sampled_post_count": len(items),
                "source_sample_ids": [str(item["sample_id"]) for item in items],
                "semantic_features": semantic,
                "predictions": aggregate_predictions,
            }
        )
    return units


def _number(row: Mapping[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_market_features(
    unit: Mapping[str, Any],
    *,
    qfq_by_code: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_by_code: Mapping[str, Sequence[Mapping[str, Any]]],
    benchmark_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    calendar = sorted({str(row["trade_date"]) for row in benchmark_rows})
    entry_date = str(unit["entry_date_override"])
    try:
        entry_index = calendar.index(entry_date)
    except ValueError:
        return {}
    previous_index = entry_index - 1
    if previous_index < 0:
        return {}
    code = str(unit["stock_code"])
    qfq = {str(row["trade_date"]): row for row in qfq_by_code.get(code, [])}
    raw = {str(row["trade_date"]): row for row in raw_by_code.get(code, [])}
    benchmark = {str(row["trade_date"]): row for row in benchmark_rows}
    previous_date = calendar[previous_index]

    def close_return(rows: Mapping[str, Mapping[str, Any]], start_index: int, end_index: int) -> float | None:
        if start_index < 0 or end_index < 0:
            return None
        start = _number(rows.get(calendar[start_index]), "close")
        end = _number(rows.get(calendar[end_index]), "close")
        return end / start - 1.0 if start not in (None, 0.0) and end is not None else None

    previous_qfq = qfq.get(previous_date)
    previous_raw = raw.get(previous_date)
    previous_benchmark = benchmark.get(previous_date)
    stock_open = _number(previous_qfq, "open")
    stock_close = _number(previous_qfq, "close")
    benchmark_open = _number(previous_benchmark, "open")
    benchmark_close = _number(previous_benchmark, "close")
    lag1_stock = stock_close / stock_open - 1.0 if stock_open not in (None, 0.0) and stock_close is not None else None
    lag1_benchmark = (
        benchmark_close / benchmark_open - 1.0
        if benchmark_open not in (None, 0.0) and benchmark_close is not None
        else None
    )
    lookback_dates = calendar[max(0, previous_index - 4) : previous_index + 1]
    volumes = [_number(raw.get(day), "volume") for day in lookback_dates]
    amounts = [_number(raw.get(day), "amount") for day in lookback_dates]
    volume_mean = statistics.fmean(value for value in volumes if value is not None) if any(value is not None for value in volumes) else None
    amount_mean = statistics.fmean(value for value in amounts if value is not None) if any(value is not None for value in amounts) else None
    previous_volume = _number(previous_raw, "volume")
    previous_amount = _number(previous_raw, "amount")
    turnover = _number(previous_raw, "turnover_rate")
    return {
        "lag1_stock_return": lag1_stock,
        "lag1_benchmark_return": lag1_benchmark,
        "lag1_excess_return": lag1_stock - lag1_benchmark if lag1_stock is not None and lag1_benchmark is not None else None,
        "trailing3_stock_return": close_return(qfq, previous_index - 2, previous_index),
        "trailing5_stock_return": close_return(qfq, previous_index - 4, previous_index),
        "trailing3_benchmark_return": close_return(benchmark, previous_index - 2, previous_index),
        "trailing5_benchmark_return": close_return(benchmark, previous_index - 4, previous_index),
        "previous_turnover_rate": turnover,
        "previous_volume_ratio5": previous_volume / volume_mean if previous_volume is not None and volume_mean not in (None, 0.0) else None,
        "previous_amount_ratio5": previous_amount / amount_mean if previous_amount is not None and amount_mean not in (None, 0.0) else None,
    }


def build_panel(
    *,
    metadata_path: Path,
    predictions_path: Path,
    qfq_path: Path,
    raw_path: Path,
    benchmark_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = _read_jsonl(metadata_path)
    predictions = _read_jsonl(predictions_path)
    units = aggregate_semantic_units(metadata, predictions)
    qfq_rows = validation.read_csv(qfq_path)
    raw_rows = validation.read_csv(raw_path)
    benchmark_rows = validation.read_csv(benchmark_path)
    qfq_by_code = validation.index_bars(qfq_rows)
    raw_by_code = validation.index_bars(raw_rows)
    outcomes = validation.join_outcomes(
        units,
        qfq_rows=qfq_rows,
        raw_rows=raw_rows,
        benchmark_rows=benchmark_rows,
        horizons=HORIZONS,
        mode="retrospective",
    )
    units_by_id = {str(unit["sample_id"]): unit for unit in units}
    panel: list[dict[str, Any]] = []
    for outcome in outcomes:
        unit = units_by_id[str(outcome["sample_id"])]
        panel.append(
            {
                "market_unit_id": unit["sample_id"],
                "stock_code": unit["stock_code"],
                "stock_name": unit["stock_name"],
                "entry_date": outcome.get("entry_date") or unit["entry_date_override"],
                "signal_dates": unit["published_dates"],
                "horizon_trading_days": outcome["horizon_trading_days"],
                "outcome_date": outcome.get("outcome_date"),
                "outcome_status": outcome["status"],
                "outcome_reason": outcome.get("reason"),
                "post_count": unit["post_count"],
                "unique_text_count": unit["unique_text_count"],
                "sampled_post_count": unit["sampled_post_count"],
                "semantic_features": unit["semantic_features"],
                "market_features": build_market_features(
                    unit,
                    qfq_by_code=qfq_by_code,
                    raw_by_code=raw_by_code,
                    benchmark_rows=benchmark_rows,
                ),
                "stock_return": outcome.get("stock_return"),
                "benchmark_return": outcome.get("benchmark_return"),
                "excess_return": outcome.get("excess_return"),
                "stock_up": outcome.get("stock_up"),
                "excess_positive": outcome.get("excess_positive"),
            }
        )
    summary = {
        "market_units": len(units),
        "panel_rows": len(panel),
        "stocks": len({str(unit["stock_code"]) for unit in units}),
        "entry_dates": len({str(unit["entry_date_override"]) for unit in units}),
        "outcome_status": _counts(panel, "outcome_status"),
        "rows_by_horizon": {
            str(horizon): sum(int(row["horizon_trading_days"]) == horizon for row in panel)
            for horizon in HORIZONS
        },
        "complete_by_horizon": {
            str(horizon): sum(
                int(row["horizon_trading_days"]) == horizon and row["outcome_status"] == "COMPLETE"
                for row in panel
            )
            for horizon in HORIZONS
        },
    }
    return panel, summary


def _counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for row in rows:
        output[str(row.get(key))] += 1
    return dict(sorted(output.items()))


def purged_training_rows(rows: Sequence[Mapping[str, Any]], evaluation_date: str) -> list[Mapping[str, Any]]:
    """Only use labels whose outcome existed before the evaluation entry."""

    return [
        row
        for row in rows
        if str(row.get("entry_date") or "") < evaluation_date
        and str(row.get("outcome_date") or "") < evaluation_date
    ]


def _feature_names(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(name) for row in rows for name in (row.get(key) or {})})


def _numeric_matrix(rows: Sequence[Mapping[str, Any]], feature_keys: Sequence[tuple[str, str]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for section, feature in feature_keys:
            raw = (row.get(section) or {}).get(feature)
            values.append(float(raw) if raw is not None and math.isfinite(float(raw)) else math.nan)
        matrix.append(values)
    return matrix


def _make_model(numeric_features: Sequence[tuple[str, str]]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_count = len(numeric_features)
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(range(numeric_count))),
            ("stock", OneHotEncoder(handle_unknown="ignore"), [numeric_count]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "logistic",
                LogisticRegression(C=LOGISTIC_C, solver="liblinear", max_iter=2000, random_state=20260917),
            ),
        ]
    )


def _model_matrix(rows: Sequence[Mapping[str, Any]], feature_keys: Sequence[tuple[str, str]]):
    import numpy as np

    numeric = _numeric_matrix(rows, feature_keys)
    return np.asarray(
        [values + [str(row["stock_code"])] for values, row in zip(numeric, rows)],
        dtype=object,
    )


def walk_forward_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_sets: Mapping[str, Sequence[tuple[str, str]]],
    min_train_dates: int,
) -> list[dict[str, Any]]:
    """Generate one probability per model with a purged expanding window."""

    complete = sorted(
        (row for row in rows if row.get("outcome_status") == "COMPLETE"),
        key=lambda row: (str(row["entry_date"]), str(row["stock_code"])),
    )
    dates = sorted({str(row["entry_date"]) for row in complete})
    output: list[dict[str, Any]] = []
    for date_index, evaluation_date in enumerate(dates):
        if date_index < min_train_dates:
            continue
        evaluation_rows = [row for row in complete if str(row["entry_date"]) == evaluation_date]
        training_rows = purged_training_rows(complete, evaluation_date)
        training_dates = sorted({str(row["entry_date"]) for row in training_rows})
        if len(training_dates) < min_train_dates:
            continue
        train_y = [int(bool(row["excess_positive"])) for row in training_rows]
        if len(set(train_y)) < 2:
            continue
        prior = sum(train_y) / len(train_y)
        predictions_by_model: dict[str, list[float]] = {"null": [prior] * len(evaluation_rows)}
        for model_name, feature_keys in feature_sets.items():
            model = _make_model(feature_keys)
            model.fit(_model_matrix(training_rows, feature_keys), train_y)
            probabilities = model.predict_proba(_model_matrix(evaluation_rows, feature_keys))[:, 1]
            predictions_by_model[model_name] = [float(value) for value in probabilities]
        for row_index, row in enumerate(evaluation_rows):
            output.append(
                {
                    "market_unit_id": row["market_unit_id"],
                    "stock_code": row["stock_code"],
                    "entry_date": row["entry_date"],
                    "outcome_date": row["outcome_date"],
                    "horizon_trading_days": row["horizon_trading_days"],
                    "excess_positive": bool(row["excess_positive"]),
                    "excess_return": float(row["excess_return"]),
                    "training_rows": len(training_rows),
                    "training_dates": len(training_dates),
                    "probabilities": {
                        model_name: values[row_index] for model_name, values in predictions_by_model.items()
                    },
                }
            )
    return output


def _model_metrics(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y = [int(bool(row["excess_positive"])) for row in rows]
    probabilities = [float(row["probabilities"][model_name]) for row in rows]
    predicted = [int(value >= 0.5) for value in probabilities]
    metrics: dict[str, Any] = {
        "n_market_units": len(rows),
        "n_entry_dates": len({str(row["entry_date"]) for row in rows}),
        "positive_rate": sum(y) / len(y) if y else None,
        "brier": float(brier_score_loss(y, probabilities)) if y else None,
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])) if y else None,
        "accuracy_at_0_5": float(accuracy_score(y, predicted)) if y else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
    }
    daily_spreads: list[float] = []
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    top_wins: list[bool] = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["entry_date"])].append(row)
    for items in grouped.values():
        ordered = sorted(items, key=lambda row: (float(row["probabilities"][model_name]), str(row["stock_code"])))
        bucket = max(1, len(ordered) // 4)
        bottom = ordered[:bucket]
        top = ordered[-bucket:]
        top_mean = statistics.fmean(float(row["excess_return"]) for row in top)
        bottom_mean = statistics.fmean(float(row["excess_return"]) for row in bottom)
        daily_spreads.append(top_mean - bottom_mean)
        top_returns.extend(float(row["excess_return"]) for row in top)
        bottom_returns.extend(float(row["excess_return"]) for row in bottom)
        top_wins.extend(bool(row["excess_positive"]) for row in top)
    metrics.update(
        {
            "top_quartile_positive_rate": sum(top_wins) / len(top_wins) if top_wins else None,
            "top_quartile_mean_excess_return": statistics.fmean(top_returns) if top_returns else None,
            "bottom_quartile_mean_excess_return": statistics.fmean(bottom_returns) if bottom_returns else None,
            "mean_daily_top_minus_bottom_excess": statistics.fmean(daily_spreads) if daily_spreads else None,
        }
    )
    return metrics


def _bootstrap_date_metric(
    rows: Sequence[Mapping[str, Any]],
    metric,
    *,
    seed: int = 20260917,
    draws: int = 2000,
) -> dict[str, float | int | None]:
    import numpy as np

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["entry_date"])].append(row)
    dates = sorted(grouped)
    if len(dates) < 2:
        return {"draws": 0, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = [row for day in sampled_dates for row in grouped[str(day)]]
        value = float(metric(sample))
        if math.isfinite(value):
            values.append(value)
    return {
        "draws": len(values),
        "lower_95": float(np.quantile(values, 0.025)) if values else None,
        "upper_95": float(np.quantile(values, 0.975)) if values else None,
    }


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_names = sorted(next(iter(rows), {}).get("probabilities", {}))
    models = {model_name: _model_metrics(rows, model_name) for model_name in model_names}

    def brier_gain(sample: Sequence[Mapping[str, Any]]) -> float:
        actual = [1.0 if row["excess_positive"] else 0.0 for row in sample]
        market = [float(row["probabilities"]["market"]) for row in sample]
        combined = [float(row["probabilities"]["combined"]) for row in sample]
        market_brier = statistics.fmean((a - p) ** 2 for a, p in zip(actual, market))
        combined_brier = statistics.fmean((a - p) ** 2 for a, p in zip(actual, combined))
        return market_brier - combined_brier

    brier_increment = (
        float(models["market"]["brier"] - models["combined"]["brier"])
        if "market" in models and "combined" in models
        else None
    )
    auc_increment = (
        float(models["combined"]["roc_auc"] - models["market"]["roc_auc"])
        if "market" in models
        and "combined" in models
        and models["market"]["roc_auc"] is not None
        and models["combined"]["roc_auc"] is not None
        else None
    )
    return {
        "n_market_units": len(rows),
        "n_entry_dates": len({str(row["entry_date"]) for row in rows}),
        "models": models,
        "semantic_increment_over_market": {
            "brier_gain_positive_is_better": brier_increment,
            "roc_auc_gain_positive_is_better": auc_increment,
            "brier_gain_date_block_bootstrap_95": _bootstrap_date_metric(rows, brier_gain),
        },
    }


def direct_score_diagnostics(rows: Sequence[Mapping[str, Any]], feature_name: str) -> dict[str, Any]:
    """Measure a semantic score as a cross-sectional ranking without fitting."""

    import numpy as np

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("outcome_status") == "COMPLETE":
            grouped[str(row["entry_date"])].append(row)
    daily: list[dict[str, float]] = []
    top_wins: list[bool] = []
    bottom_wins: list[bool] = []
    for entry_date, items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda row: (float((row.get("semantic_features") or {})[feature_name]), str(row["stock_code"])),
        )
        scores = [float((row.get("semantic_features") or {})[feature_name]) for row in ordered]
        returns = [float(row["excess_return"]) for row in ordered]
        correlation = validation.spearman(scores, returns)
        if correlation is None:
            continue
        bucket = max(1, len(ordered) // 4)
        bottom = ordered[:bucket]
        top = ordered[-bucket:]
        spread = statistics.fmean(float(row["excess_return"]) for row in top) - statistics.fmean(
            float(row["excess_return"]) for row in bottom
        )
        daily.append({"spearman": float(correlation), "top_minus_bottom_excess": spread})
        top_wins.extend(bool(row["excess_positive"]) for row in top)
        bottom_wins.extend(bool(row["excess_positive"]) for row in bottom)
    if not daily:
        return {"n_entry_dates": 0}
    values = np.asarray(
        [[item["spearman"], item["top_minus_bottom_excess"]] for item in daily],
        dtype=float,
    )
    rng = np.random.default_rng(20260917)
    draws = np.empty((2000, 2), dtype=float)
    for index in range(len(draws)):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    return {
        "feature": feature_name,
        "n_entry_dates": len(daily),
        "mean_daily_spearman": float(values[:, 0].mean()),
        "median_daily_spearman": float(np.median(values[:, 0])),
        "positive_daily_spearman_rate": float(np.mean(values[:, 0] > 0.0)),
        "mean_daily_top_minus_bottom_excess": float(values[:, 1].mean()),
        "top_quartile_positive_rate": sum(top_wins) / len(top_wins) if top_wins else None,
        "bottom_quartile_positive_rate": sum(bottom_wins) / len(bottom_wins) if bottom_wins else None,
        "date_block_bootstrap_95": {
            "draws": len(draws),
            "spearman_lower": float(np.quantile(draws[:, 0], 0.025)),
            "spearman_upper": float(np.quantile(draws[:, 0], 0.975)),
            "spread_lower": float(np.quantile(draws[:, 1], 0.025)),
            "spread_upper": float(np.quantile(draws[:, 1], 0.975)),
        },
    }


def evaluate_panel(panel: Sequence[Mapping[str, Any]], *, min_train_dates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = [row for row in panel if row.get("outcome_status") == "COMPLETE"]
    semantic_names = _feature_names(complete, "semantic_features")
    directional_semantic_names = [
        name
        for name in semantic_names
        if name.startswith("mean::stance::")
        or name.startswith("mean::action_tendency::")
        or name in {
            "stance_score_mean",
            "stance_score_std",
            "action_score_mean",
            "action_score_std",
            "entropy_mean::stance",
            "entropy_mean::action_tendency",
        }
    ]
    market_names = _feature_names(complete, "market_features")
    feature_sets = {
        # Only stance and action have an explicit directional interpretation.
        # Emotion, target, context, and reasoning remain available as the
        # predeclared full-feature diagnostic, but they do not silently become
        # direct price-direction features in the primary comparison.
        "semantic": [("semantic_features", name) for name in directional_semantic_names],
        "market": [("market_features", name) for name in market_names],
        "combined": [
            *[("market_features", name) for name in market_names],
            *[("semantic_features", name) for name in directional_semantic_names],
        ],
        "semantic_full": [("semantic_features", name) for name in semantic_names],
        "combined_full": [
            *[("market_features", name) for name in market_names],
            *[("semantic_features", name) for name in semantic_names],
        ],
    }
    predictions: list[dict[str, Any]] = []
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_rows = [row for row in complete if int(row["horizon_trading_days"]) == horizon]
        current = walk_forward_predictions(
            horizon_rows,
            feature_sets=feature_sets,
            min_train_dates=min_train_dates,
        )
        predictions.extend(current)
        horizon_report = summarize_predictions(current) if current else {
            "n_market_units": 0,
            "n_entry_dates": 0,
            "models": {},
            "semantic_increment_over_market": {},
        }
        horizon_report["direct_score_diagnostics"] = {
            "stance": direct_score_diagnostics(horizon_rows, "stance_score_mean"),
            "action": direct_score_diagnostics(horizon_rows, "action_score_mean"),
        }
        by_horizon[str(horizon)] = horizon_report

    usable = [item for item in by_horizon.values() if item.get("n_market_units")]
    brier_gains = [
        item["semantic_increment_over_market"].get("brier_gain_positive_is_better")
        for item in usable
        if item.get("semantic_increment_over_market")
    ]
    brier_gains = [float(value) for value in brier_gains if value is not None]
    direct_positive = []
    for item in usable:
        for diagnostic in item.get("direct_score_diagnostics", {}).values():
            interval = diagnostic.get("date_block_bootstrap_95", {})
            direct_positive.append(
                interval.get("spearman_lower") is not None
                and float(interval["spearman_lower"]) > 0.0
                and float(interval.get("spread_lower", 0.0)) > 0.0
            )
    if not usable:
        verdict = "NO_WALK_FORWARD_EVIDENCE"
    elif brier_gains and all(value > 0.0 for value in brier_gains) and any(direct_positive):
        verdict = "USABLE_SEMANTIC_INCREMENT_CANDIDATE_RETROSPECTIVE_ONLY"
    else:
        verdict = "NO_USABLE_MARKET_EDGE_RETROSPECTIVE__WEAK_HYPOTHESES_ONLY"
    report = {
        "status": "RETROSPECTIVE_MARKET_SIGNAL_DEVELOPMENT_ONLY",
        "verdict": verdict,
        "full_system_forward_evidence": False,
        "market_layer_walk_forward": True,
        "semantic_feature_model": "M12 fixed runtime, inferred retrospectively",
        "primary_unit": "stock_code x next_CSI300_entry_date",
        "target": "future excess return versus CSI300 is positive",
        "horizons": list(HORIZONS),
        "post_cap_per_market_unit": DEFAULT_POST_CAP,
        "min_train_dates": min_train_dates,
        "purge_rule": "training outcome_date must be strictly before evaluation entry_date",
        "models": {
            "null": "expanding-window training positive rate",
            "semantic": f"fixed L2 Logistic Regression C={LOGISTIC_C}; directional stance/action probabilities plus stock identity",
            "market": f"fixed L2 Logistic Regression C={LOGISTIC_C}; lagged market features plus stock identity",
            "combined": f"fixed L2 Logistic Regression C={LOGISTIC_C}; market plus directional stance/action probabilities plus stock identity",
            "semantic_full": "diagnostic only: all seven-head probability features plus stock identity",
            "combined_full": "diagnostic only: market plus all seven-head probability features plus stock identity",
        },
        "directional_semantic_feature_count": len(directional_semantic_names),
        "semantic_feature_count": len(semantic_names),
        "market_feature_count": len(market_names),
        "walk_forward_prediction_rows": len(predictions),
        "by_horizon": by_horizon,
        "interpretation": (
            "This run can select whether a market-signal layer is worth prospective testing. "
            "It cannot establish full-system out-of-time value because M12 semantic features were produced after part of the text period."
        ),
    }
    return predictions, report


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%" if percent else f"{float(value):.4f}"


def render_report(report: Mapping[str, Any], panel_summary: Mapping[str, Any]) -> str:
    lines = [
        "# Market Signal V1 — retrospective development result",
        "",
        f"Status: `{report['status']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "This is a market-layer development experiment, not full-system forward evidence. The M12 semantic model was fixed before this run but was trained after part of the historical text period.",
        "",
        "## Data",
        "",
        f"- Market units: {panel_summary['market_units']}",
        f"- Entry dates: {panel_summary['entry_dates']}",
        f"- Stocks: {panel_summary['stocks']}",
        f"- Complete outcomes by horizon: {panel_summary['complete_by_horizon']}",
        f"- Walk-forward prediction rows: {report['walk_forward_prediction_rows']}",
        "- Unit: one stock × next CSI300 entry session. Posts are never counted as independent market outcomes.",
        "- Purge: a training row is used only if its outcome date existed before the evaluation entry date.",
        "",
        "## Probability comparison",
        "",
        "| Horizon | Model | Units | Dates | Brier ↓ | ROC-AUC ↑ | Top quartile win rate | Daily top-bottom excess |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZONS:
        item = report["by_horizon"][str(horizon)]
        for model_name in ("null", "market", "semantic", "combined", "semantic_full", "combined_full"):
            metrics = item.get("models", {}).get(model_name)
            if not metrics:
                continue
            lines.append(
                f"| {horizon}d | {model_name} | {metrics['n_market_units']} | {metrics['n_entry_dates']} | "
                f"{_fmt(metrics['brier'])} | {_fmt(metrics['roc_auc'])} | "
                f"{_fmt(metrics['top_quartile_positive_rate'], percent=True)} | "
                f"{_fmt(metrics['mean_daily_top_minus_bottom_excess'], percent=True)} |"
            )
        increment = item.get("semantic_increment_over_market", {})
        if increment:
            interval = increment.get("brier_gain_date_block_bootstrap_95", {})
            lines.extend(
                [
                    "",
                    f"- {horizon}d combined-minus-market: Brier gain "
                    f"{_fmt(increment.get('brier_gain_positive_is_better'))}; ROC-AUC gain "
                    f"{_fmt(increment.get('roc_auc_gain_positive_is_better'))}; date-block bootstrap 95% "
                    f"[{_fmt(interval.get('lower_95'))}, {_fmt(interval.get('upper_95'))}].",
                    "",
                ]
            )
        direct = item.get("direct_score_diagnostics", {})
        for score_name in ("stance", "action"):
            diagnostic = direct.get(score_name, {})
            if not diagnostic:
                continue
            interval = diagnostic.get("date_block_bootstrap_95", {})
            lines.append(
                f"- {horizon}d direct `{score_name}` ranking: mean daily Spearman "
                f"{_fmt(diagnostic.get('mean_daily_spearman'))} "
                f"(95% [{_fmt(interval.get('spearman_lower'))}, {_fmt(interval.get('spearman_upper'))}]); "
                f"top-bottom excess {_fmt(diagnostic.get('mean_daily_top_minus_bottom_excess'), percent=True)} "
                f"(95% [{_fmt(interval.get('spread_lower'), percent=True)}, "
                f"{_fmt(interval.get('spread_upper'), percent=True)}]); top/bottom win rates "
                f"{_fmt(diagnostic.get('top_quartile_positive_rate'), percent=True)} / "
                f"{_fmt(diagnostic.get('bottom_quartile_positive_rate'), percent=True)}."
            )
        lines.append("")
    lines.extend(
        [
            "## Allowed conclusion",
            "",
            str(report["interpretation"]),
            "",
            "A positive retrospective increment is only a reason to freeze the market layer and test it on later dates. A mixed or negative result is a reason to inspect aggregation/target fit before changing the seven-head semantic model.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    *,
    metadata_path: Path,
    predictions_path: Path,
    qfq_path: Path,
    raw_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    min_train_dates: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "panel": output_dir / "market-panel.jsonl",
        "walk_forward": output_dir / "walk-forward-predictions.jsonl",
        "report": output_dir / "market-signal-report.json",
        "markdown": output_dir / "REPORT.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Market Signal V1 evaluation outputs already exist: {existing}")
    panel, panel_summary = build_panel(
        metadata_path=metadata_path,
        predictions_path=predictions_path,
        qfq_path=qfq_path,
        raw_path=raw_path,
        benchmark_path=benchmark_path,
    )
    predictions, report = evaluate_panel(panel, min_train_dates=min_train_dates)
    report = dict(
        report,
        panel_summary=panel_summary,
        inputs={
            "metadata": str(metadata_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "qfq": str(qfq_path.resolve()),
            "raw": str(raw_path.resolve()),
            "benchmark": str(benchmark_path.resolve()),
        },
    )
    _write_jsonl(outputs["panel"], panel)
    _write_jsonl(outputs["walk_forward"], predictions)
    report = dict(
        report,
        artifacts={
            "panel": {"path": str(outputs["panel"]), "sha256": _sha256(outputs["panel"])},
            "walk_forward": {"path": str(outputs["walk_forward"]), "sha256": _sha256(outputs["walk_forward"])},
        },
    )
    _write_json(outputs["report"], report)
    outputs["markdown"].write_text(render_report(report, panel_summary), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build deterministic M12 input without reading outcomes")
    prepare.add_argument("--collector-db", type=Path, default=DEFAULT_COLLECTOR_DB)
    prepare.add_argument("--benchmark", type=Path, default=DEFAULT_MARKET_ROOT / "benchmark-000300.csv")
    prepare.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--post-cap", type=int, default=DEFAULT_POST_CAP)

    evaluate = subparsers.add_parser("evaluate", help="join M12 predictions, construct panel, and walk forward")
    evaluate.add_argument("--metadata", type=Path, default=DEFAULT_OUTPUT / "sampled-post-metadata.jsonl")
    evaluate.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT / "m12-predictions.jsonl")
    evaluate.add_argument("--qfq", type=Path, default=DEFAULT_MARKET_ROOT / "daily-bars-qfq.csv")
    evaluate.add_argument("--raw", type=Path, default=DEFAULT_MARKET_ROOT / "daily-bars-raw.csv")
    evaluate.add_argument("--benchmark", type=Path, default=DEFAULT_MARKET_ROOT / "benchmark-000300.csv")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_inputs(
            collector_db=args.collector_db,
            benchmark_path=args.benchmark,
            targets_path=args.targets,
            output_dir=args.output_dir,
            post_cap=args.post_cap,
        )
    else:
        result = run_evaluation(
            metadata_path=args.metadata,
            predictions_path=args.predictions,
            qfq_path=args.qfq,
            raw_path=args.raw,
            benchmark_path=args.benchmark,
            output_dir=args.output_dir,
            min_train_dates=args.min_train_dates,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
