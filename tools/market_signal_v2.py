"""Evaluate crowd-state features against future absolute stock returns.

Market Signal V2 corrects the target mismatch in V1.  The tracked stocks are
evaluated on whether their own future return is positive; CSI300 remains an
already-known market-state input, not the outcome that every tracked stock
must beat.  The existing M12 post predictions are fixed inputs.  This module
does not train or modify the semantic model.

The market unit remains ``stock_code x next_CSI300_entry_date`` so posts that
map to the same tradable outcome are not counted as independent observations.
All market-layer fits use a purged expanding window: an outcome is eligible for
training only after its outcome date is strictly earlier than the evaluation
entry date.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools import market_signal_v1 as v1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_INPUT_ROOT = PROJECT_ROOT / "pipeline-runs" / "market-signal-v1-20260917"
MARKET_ROOT = PROJECT_ROOT / "pipeline-runs" / "market-data-v0-20260916"
DEFAULT_OUTPUT = PROJECT_ROOT / "pipeline-runs" / "market-signal-v2-absolute-20260918"
GROUP_HEADS = ("stance", "action_tendency", "emotion_primary")
POSITIVE_EMOTIONS = ("HOPE", "EXCITEMENT")
NEGATIVE_EMOTIONS = ("ANGER", "ANXIETY", "FEAR", "FRUSTRATION", "REGRET")
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
PRIMARY_HORIZON = 3
BOOTSTRAP_BLOCK_DATES = max(v1.HORIZONS)


def _entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in probabilities.values() if value > 0.0)


def _mean_distribution(rows: Sequence[Mapping[str, float]], labels: Sequence[str]) -> dict[str, float]:
    return {
        label: statistics.fmean(float(row.get(label, 0.0)) for row in rows)
        for label in labels
    }


def distribution_state_features(
    probabilities: Sequence[Mapping[str, float]],
    *,
    head: str,
) -> dict[str, float]:
    """Separate per-post model uncertainty from cross-post disagreement.

    ``H(mean(p)) - mean(H(p))`` is the equally weighted generalized
    Jensen-Shannon divergence.  It is zero for identical post distributions,
    including identically uncertain distributions, and grows when individually
    confident posts disagree with one another.
    """

    if not probabilities:
        raise ValueError(f"{head} requires at least one probability row")
    labels = sorted(str(label) for label in probabilities[0])
    if len(labels) < 2:
        raise ValueError(f"{head} requires at least two labels")
    normalized: list[dict[str, float]] = []
    for row in probabilities:
        if {str(label) for label in row} != set(labels):
            raise ValueError(f"{head} probability rows have inconsistent labels")
        values = {label: float(row.get(label, 0.0)) for label in labels}
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError(f"{head} has an invalid probability value")
        total = sum(values.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"{head} has an invalid probability row")
        normalized.append({label: value / total for label, value in values.items()})
    mean = _mean_distribution(normalized, labels)
    mean_post_entropy = statistics.fmean(_entropy(row) for row in normalized)
    entropy_of_mean = _entropy(mean)
    disagreement = max(0.0, entropy_of_mean - mean_post_entropy)
    maximum_entropy = math.log(len(labels))
    normalized_disagreement = min(1.0, disagreement / maximum_entropy)
    distribution_concentration = max(0.0, 1.0 - entropy_of_mean / maximum_entropy)
    output = {
        f"mean::{head}::{label}": value
        for label, value in mean.items()
    }
    output.update(
        {
            f"model_uncertainty::{head}": mean_post_entropy / maximum_entropy,
            f"group_disagreement::{head}": normalized_disagreement,
            f"cross_post_agreement::{head}": 1.0 - normalized_disagreement,
            f"group_consensus::{head}": distribution_concentration,
        }
    )
    return output


def aggregate_crowd_state(
    metadata_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    predictions_by_id = {str(row.get("sample_id")): row for row in prediction_rows}
    metadata_ids = [str(row.get("sample_id")) for row in metadata_rows]
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("metadata contains duplicate sample_id values")
    if len(predictions_by_id) != len(prediction_rows) or set(predictions_by_id) != set(metadata_ids):
        raise ValueError("metadata and predictions differ in sample_id coverage")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        grouped[(str(row["stock_code"]), str(row["entry_date"]))].append(row)

    result: dict[str, dict[str, float]] = {}
    for (stock_code, entry_date), items in sorted(grouped.items()):
        predictions = [predictions_by_id[str(item["sample_id"])] for item in items]
        features: dict[str, float] = {}
        for head in GROUP_HEADS:
            probabilities = [v1._probabilities(row, head) for row in predictions]
            features.update(distribution_state_features(probabilities, head=head))

        stance = [v1._probabilities(row, "stance") for row in predictions]
        action = [v1._probabilities(row, "action_tendency") for row in predictions]
        emotion = [v1._probabilities(row, "emotion_primary") for row in predictions]
        features["stance_index"] = statistics.fmean(
            row.get("BULL", 0.0) - row.get("BEAR", 0.0) for row in stance
        )
        features["action_index"] = statistics.fmean(
            row.get("BUY", 0.0)
            + row.get("ADD", 0.0)
            - row.get("SELL", 0.0)
            - row.get("REDUCE", 0.0)
            for row in action
        )
        emotion_mean = _mean_distribution(
            emotion,
            sorted({str(label) for row in emotion for label in row}),
        )
        positive_mass = sum(emotion_mean.get(label, 0.0) for label in POSITIVE_EMOTIONS)
        negative_mass = sum(emotion_mean.get(label, 0.0) for label in NEGATIVE_EMOTIONS)
        features.update(
            {
                "emotion_positive_mass": positive_mass,
                "emotion_negative_mass": negative_mass,
                "emotion_balance": positive_mass - negative_mass,
                "emotion_fomo_mass": emotion_mean.get("FOMO", 0.0),
            }
        )
        total_posts = int(items[0]["unit_total_post_count"])
        unique_texts = int(items[0]["unit_unique_text_count"])
        sampled_posts = len(items)
        features.update(
            {
                "log1p_total_post_count": math.log1p(total_posts),
                "log1p_unique_text_count": math.log1p(unique_texts),
                "unique_text_ratio": unique_texts / total_posts if total_posts else 0.0,
                "sampling_coverage": sampled_posts / unique_texts if unique_texts else 0.0,
            }
        )
        result[f"market-signal-unit:{stock_code}:{entry_date}"] = features
    return result


def build_absolute_panel(
    *,
    metadata_path: Path,
    predictions_path: Path,
    qfq_path: Path,
    raw_path: Path,
    benchmark_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = v1._read_jsonl(metadata_path)
    predictions = v1._read_jsonl(predictions_path)
    crowd_by_unit = aggregate_crowd_state(metadata, predictions)
    base_panel, summary = v1.build_panel(
        metadata_path=metadata_path,
        predictions_path=predictions_path,
        qfq_path=qfq_path,
        raw_path=raw_path,
        benchmark_path=benchmark_path,
    )
    panel: list[dict[str, Any]] = []
    for row in base_panel:
        unit_id = str(row["market_unit_id"])
        if unit_id not in crowd_by_unit:
            raise ValueError(f"crowd state missing for {unit_id}")
        panel.append(
            {
                **row,
                "crowd_features": crowd_by_unit[unit_id],
                "stock_positive": row.get("stock_up"),
                "realized_stock_return": row.get("stock_return"),
            }
        )
    complete_by_horizon = {
        str(horizon): [
            row
            for row in panel
            if row["outcome_status"] == "COMPLETE"
            and int(row["horizon_trading_days"]) == horizon
        ]
        for horizon in v1.HORIZONS
    }
    summary = dict(
        summary,
        target="future absolute stock return is positive",
        positive_rate_by_horizon={
            horizon: statistics.fmean(bool(row["stock_positive"]) for row in rows) if rows else None
            for horizon, rows in complete_by_horizon.items()
        },
        mean_stock_return_by_horizon={
            horizon: statistics.fmean(float(row["realized_stock_return"]) for row in rows) if rows else None
            for horizon, rows in complete_by_horizon.items()
        },
    )
    return panel, summary


def smoothed_prior(labels: Sequence[int]) -> float:
    return (sum(labels) + PRIOR_ALPHA) / (len(labels) + PRIOR_ALPHA + PRIOR_BETA)


def walk_forward_absolute_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_sets: Mapping[str, Sequence[tuple[str, str]]],
    min_train_dates: int,
) -> list[dict[str, Any]]:
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
        training_rows = v1.purged_training_rows(complete, evaluation_date)
        training_dates = sorted({str(row["entry_date"]) for row in training_rows})
        if len(training_dates) < min_train_dates:
            continue
        train_y = [int(bool(row["stock_positive"])) for row in training_rows]
        if len(set(train_y)) < 2:
            continue
        global_prior = smoothed_prior(train_y)
        labels_by_stock: dict[str, list[int]] = defaultdict(list)
        for row, label in zip(training_rows, train_y):
            labels_by_stock[str(row["stock_code"])].append(label)
        predictions_by_model: dict[str, list[float]] = {
            "global_prior": [global_prior] * len(evaluation_rows),
            "stock_prior": [
                smoothed_prior(labels_by_stock[str(row["stock_code"])])
                if labels_by_stock.get(str(row["stock_code"]))
                else global_prior
                for row in evaluation_rows
            ],
        }
        for model_name, feature_keys in feature_sets.items():
            model = v1._make_model(feature_keys)
            model.fit(v1._model_matrix(training_rows, feature_keys), train_y)
            probabilities = model.predict_proba(v1._model_matrix(evaluation_rows, feature_keys))[:, 1]
            predictions_by_model[model_name] = [float(value) for value in probabilities]
        for row_index, row in enumerate(evaluation_rows):
            output.append(
                {
                    "market_unit_id": row["market_unit_id"],
                    "stock_code": row["stock_code"],
                    "entry_date": row["entry_date"],
                    "outcome_date": row["outcome_date"],
                    "horizon_trading_days": row["horizon_trading_days"],
                    "stock_positive": bool(row["stock_positive"]),
                    "stock_return": float(row["realized_stock_return"]),
                    "training_rows": len(training_rows),
                    "training_dates": len(training_dates),
                    "probabilities": {
                        model_name: values[row_index]
                        for model_name, values in predictions_by_model.items()
                    },
                }
            )
    return output


def calibration_bins(rows: Sequence[Mapping[str, Any]], model_name: str) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:])):
        selected = [
            row
            for row in rows
            if lower <= float(row["probabilities"][model_name])
            and (float(row["probabilities"][model_name]) < upper or (index == len(CALIBRATION_EDGES) - 2 and float(row["probabilities"][model_name]) <= upper))
        ]
        if not selected:
            continue
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "n": len(selected),
                "mean_probability": statistics.fmean(float(row["probabilities"][model_name]) for row in selected),
                "positive_rate": statistics.fmean(bool(row["stock_positive"]) for row in selected),
                "mean_stock_return": statistics.fmean(float(row["stock_return"]) for row in selected),
            }
        )
    return bins


def macro_stock_brier(rows: Sequence[Mapping[str, Any]], model_name: str) -> float | None:
    errors_by_stock: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = 1.0 if row["stock_positive"] else 0.0
        probability = float(row["probabilities"][model_name])
        errors_by_stock[str(row["stock_code"])].append((target - probability) ** 2)
    return (
        statistics.fmean(statistics.fmean(errors) for errors in errors_by_stock.values())
        if errors_by_stock
        else None
    )


def _basic_metrics(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y = [int(bool(row["stock_positive"])) for row in rows]
    probabilities = [float(row["probabilities"][model_name]) for row in rows]
    predicted = [int(value >= 0.5) for value in probabilities]
    positive_prediction_count = sum(predicted)
    bins = calibration_bins(rows, model_name)
    expected_calibration_error = sum(
        item["n"] / len(rows) * abs(item["positive_rate"] - item["mean_probability"])
        for item in bins
    ) if rows else None
    macro_brier = macro_stock_brier(rows, model_name)
    return {
        "n_market_units": len(rows),
        "n_entry_dates": len({str(row["entry_date"]) for row in rows}),
        "positive_rate": sum(y) / len(y) if y else None,
        "mean_probability": statistics.fmean(probabilities) if probabilities else None,
        "brier": float(brier_score_loss(y, probabilities)) if y else None,
        "macro_stock_brier": macro_brier,
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])) if y else None,
        "accuracy_at_0_5": float(accuracy_score(y, predicted)) if y else None,
        "positive_prediction_coverage_at_0_5": positive_prediction_count / len(predicted) if predicted else None,
        "positive_predictive_value_at_0_5": (
            sum(target for target, decision in zip(y, predicted) if decision) / positive_prediction_count
            if positive_prediction_count
            else None
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
        "expected_calibration_error_5_bins": expected_calibration_error,
        "calibration_bins": bins,
    }


def per_stock_metrics(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stock_code"])].append(row)
    output: dict[str, Any] = {}
    for stock_code, items in sorted(grouped.items()):
        metrics = _basic_metrics(items, model_name)
        output[stock_code] = {
            "n_market_units": metrics["n_market_units"],
            "n_entry_dates": metrics["n_entry_dates"],
            "positive_rate": metrics["positive_rate"],
            "mean_probability": metrics["mean_probability"],
            "brier": metrics["brier"],
            "roc_auc": metrics["roc_auc"],
            "roc_auc_stable_min_10_each_class": (
                sum(bool(row["stock_positive"]) for row in items) >= 10
                and sum(not bool(row["stock_positive"]) for row in items) >= 10
            ),
            "calibration_bias_probability_minus_outcome": (
                metrics["mean_probability"] - metrics["positive_rate"]
            ),
            "mean_stock_return": statistics.fmean(float(row["stock_return"]) for row in items),
        }
    return output


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return statistics.fmean(defined) if defined else None


def _date_block_gain(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    candidate: str,
) -> dict[str, float | int | None]:
    import numpy as np

    def gain(sample: Sequence[Mapping[str, Any]]) -> float:
        baseline_brier = macro_stock_brier(sample, baseline)
        candidate_brier = macro_stock_brier(sample, candidate)
        if baseline_brier is None or candidate_brier is None:
            return math.nan
        return baseline_brier - candidate_brier

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["entry_date"])].append(row)
    dates = sorted(grouped)
    if len(dates) < BOOTSTRAP_BLOCK_DATES:
        return {
            "draws": 0,
            "block_entry_dates": BOOTSTRAP_BLOCK_DATES,
            "lower_95": None,
            "upper_95": None,
        }
    blocks = [dates[index : index + BOOTSTRAP_BLOCK_DATES] for index in range(len(dates) - BOOTSTRAP_BLOCK_DATES + 1)]
    rng = np.random.default_rng(20260918)
    values: list[float] = []
    for _ in range(2000):
        sampled_dates: list[str] = []
        while len(sampled_dates) < len(dates):
            sampled_dates.extend(blocks[int(rng.integers(0, len(blocks)))])
        sampled_dates = sampled_dates[: len(dates)]
        sample = [row for entry_date in sampled_dates for row in grouped[entry_date]]
        value = float(gain(sample))
        if math.isfinite(value):
            values.append(value)
    return {
        "draws": len(values),
        "block_entry_dates": BOOTSTRAP_BLOCK_DATES,
        "lower_95": float(np.quantile(values, 0.025)) if values else None,
        "upper_95": float(np.quantile(values, 0.975)) if values else None,
    }


def summarize_absolute_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_names = sorted(next(iter(rows), {}).get("probabilities", {}))
    models: dict[str, Any] = {}
    for model_name in model_names:
        stock_metrics = per_stock_metrics(rows, model_name)
        models[model_name] = {
            **_basic_metrics(rows, model_name),
            "macro_stock_roc_auc": _mean_defined(
                item["roc_auc"] for item in stock_metrics.values()
            ),
            "stable_macro_stock_roc_auc_min_10_each_class": _mean_defined(
                item["roc_auc"]
                for item in stock_metrics.values()
                if item["roc_auc_stable_min_10_each_class"]
            ),
            "per_stock": stock_metrics,
        }
    comparisons: dict[str, Any] = {}
    for baseline, candidate in (
        ("stock_prior", "crowd"),
        ("stock_prior", "combined"),
        ("market", "combined"),
    ):
        if baseline not in models or candidate not in models:
            continue
        key = f"{candidate}_over_{baseline}"
        comparisons[key] = {
            "macro_stock_brier_gain_positive_is_better": (
                models[baseline]["macro_stock_brier"] - models[candidate]["macro_stock_brier"]
            ),
            "roc_auc_gain_positive_is_better": (
                models[candidate]["roc_auc"] - models[baseline]["roc_auc"]
                if models[baseline]["roc_auc"] is not None and models[candidate]["roc_auc"] is not None
                else None
            ),
            "macro_stock_brier_skill_score_positive_is_better": (
                1.0 - models[candidate]["macro_stock_brier"] / models[baseline]["macro_stock_brier"]
                if models[baseline]["macro_stock_brier"] not in (None, 0.0)
                else None
            ),
            "date_block_bootstrap_95": _date_block_gain(rows, baseline, candidate),
        }
    return {
        "n_market_units": len(rows),
        "n_entry_dates": len({str(row["entry_date"]) for row in rows}),
        "models": models,
        "comparisons": comparisons,
    }


def evaluate_absolute_panel(
    panel: Sequence[Mapping[str, Any]],
    *,
    min_train_dates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = [row for row in panel if row.get("outcome_status") == "COMPLETE"]
    crowd_names = v1._feature_names(complete, "crowd_features")
    market_names = v1._feature_names(complete, "market_features")
    feature_sets = {
        "market": [("market_features", name) for name in market_names],
        "crowd": [("crowd_features", name) for name in crowd_names],
        "combined": [
            *[("market_features", name) for name in market_names],
            *[("crowd_features", name) for name in crowd_names],
        ],
    }
    predictions: list[dict[str, Any]] = []
    by_horizon: dict[str, Any] = {}
    for horizon in v1.HORIZONS:
        horizon_rows = [row for row in complete if int(row["horizon_trading_days"]) == horizon]
        current = walk_forward_absolute_predictions(
            horizon_rows,
            feature_sets=feature_sets,
            min_train_dates=min_train_dates,
        )
        predictions.extend(current)
        by_horizon[str(horizon)] = summarize_absolute_predictions(current) if current else {
            "n_market_units": 0,
            "n_entry_dates": 0,
            "models": {},
            "comparisons": {},
        }

    primary_comparison = by_horizon[str(PRIMARY_HORIZON)].get("comparisons", {}).get(
        "combined_over_stock_prior",
        {},
    )
    primary_gain = primary_comparison.get("macro_stock_brier_gain_positive_is_better")
    primary_interval = primary_comparison.get("date_block_bootstrap_95", {})
    primary_lower = primary_interval.get("lower_95")
    primary_upper = primary_interval.get("upper_95")
    if primary_gain is not None and primary_lower is not None and float(primary_gain) > 0.0 and float(primary_lower) > 0.0:
        verdict = "ABSOLUTE_RETURN_INCREMENT_CANDIDATE_RETROSPECTIVE_ONLY"
    elif primary_upper is not None and float(primary_upper) <= 0.0:
        verdict = "NO_ABSOLUTE_RETURN_IMPROVEMENT_OVER_STOCK_PRIOR"
    else:
        verdict = "ABSOLUTE_RETURN_RESULT_UNCERTAIN"
    report = {
        "status": "RETROSPECTIVE_ABSOLUTE_RETURN_DEVELOPMENT_ONLY",
        "verdict": verdict,
        "target": "future absolute stock return is positive",
        "benchmark_role": "CSI300 is a lagged market-state input; it is not the tracked-stock success target",
        "portfolio_relative_return_evaluated": False,
        "semantic_model_retrained": False,
        "semantic_feature_model": "existing fixed M12 predictions",
        "primary_unit": "stock_code x next_CSI300_entry_date",
        "horizons": list(v1.HORIZONS),
        "primary_horizon_trading_days": PRIMARY_HORIZON,
        "primary_comparison": "combined_over_stock_prior",
        "purge_rule": "training outcome_date must be strictly before evaluation entry_date",
        "prior_rule": "Beta(1,1)-smoothed expanding historical positive rate; stock_prior uses only the same stock",
        "models": {
            "global_prior": "Beta(1,1)-smoothed pooled training positive rate",
            "stock_prior": "Beta(1,1)-smoothed same-stock training positive rate",
            "market": f"fixed L2 Logistic Regression C={v1.LOGISTIC_C}; lagged market features plus stock identity",
            "crowd": f"fixed L2 Logistic Regression C={v1.LOGISTIC_C}; stance/action/emotion distributions, uncertainty, disagreement, concentration, attention plus stock identity",
            "combined": f"fixed L2 Logistic Regression C={v1.LOGISTIC_C}; market plus crowd-state features plus stock identity",
        },
        "crowd_feature_count": len(crowd_names),
        "market_feature_count": len(market_names),
        "min_train_dates": min_train_dates,
        "walk_forward_prediction_rows": len(predictions),
        "bootstrap": "paired moving blocks of 5 consecutive entry dates, carrying all tracked stocks together",
        "by_horizon": by_horizon,
        "stop_rule": (
            "Do not tune this retrospective panel. The predeclared primary result is 3-day combined versus the same-stock "
            "Beta(1,1) prior, using macro-stock Brier. A positive point gain with a paired 5-date-block bootstrap 95% "
            "lower bound above zero becomes a prospective candidate; an interval crossing zero is uncertain."
        ),
        "interpretation": (
            "This run tests the owner's tracked-stock objective: whether crowd state adds information about each stock's "
            "own future positive return. It does not evaluate an actual buy portfolio or investment skill versus CSI300."
        ),
    }
    return predictions, report


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%" if percent else f"{float(value):.4f}"


def render_report(report: Mapping[str, Any], panel_summary: Mapping[str, Any]) -> str:
    lines = [
        "# Market Signal V2 — absolute-return development result",
        "",
        f"Status: `{report['status']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "The current tracked-stock objective is absolute positive return, not excess return versus CSI300. This is retrospective development evidence, not a trading approval.",
        "",
        "## Data and target",
        "",
        f"- Market units: {panel_summary['market_units']}",
        f"- Entry dates: {panel_summary['entry_dates']}",
        f"- Stocks: {panel_summary['stocks']}",
        f"- Complete outcomes: {panel_summary['complete_by_horizon']}",
        f"- Positive rates by horizon: {panel_summary['positive_rate_by_horizon']}",
        "- Target: each tracked stock's own future return is positive.",
        "- CSI300: lagged market-state input only; portfolio-relative performance is not evaluated.",
        "",
        "## Walk-forward probability quality",
        "",
        "| Horizon | Model | Units | Dates | Positive rate | Macro-stock Brier ↓ | ECE ↓ | Pooled AUC ↑ | Macro-stock AUC ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in v1.HORIZONS:
        item = report["by_horizon"][str(horizon)]
        for model_name in ("global_prior", "stock_prior", "market", "crowd", "combined"):
            metrics = item.get("models", {}).get(model_name)
            if not metrics:
                continue
            lines.append(
                f"| {horizon}d | {model_name} | {metrics['n_market_units']} | {metrics['n_entry_dates']} | "
                f"{_fmt(metrics['positive_rate'], percent=True)} | {_fmt(metrics['macro_stock_brier'])} | "
                f"{_fmt(metrics['expected_calibration_error_5_bins'])} | {_fmt(metrics['roc_auc'])} | "
                f"{_fmt(metrics['macro_stock_roc_auc'])} |"
            )
        lines.append("")
        for comparison_name, comparison in item.get("comparisons", {}).items():
            interval = comparison["date_block_bootstrap_95"]
            lines.append(
                f"- {horizon}d `{comparison_name}`: Brier gain "
                f"{_fmt(comparison['macro_stock_brier_gain_positive_is_better'])}; Brier skill "
                f"{_fmt(comparison['macro_stock_brier_skill_score_positive_is_better'], percent=True)}; AUC gain "
                f"{_fmt(comparison['roc_auc_gain_positive_is_better'])}; date-block 95% "
                f"[{_fmt(interval.get('lower_95'))}, {_fmt(interval.get('upper_95'))}]."
            )
        lines.append("")
    lines.extend(
        [
            "## Allowed conclusion",
            "",
            str(report["interpretation"]),
            "",
            str(report["stop_rule"]),
            "",
        ]
    )
    return "\n".join(lines)


def run(
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
        "panel": output_dir / "absolute-return-panel.jsonl",
        "walk_forward": output_dir / "walk-forward-predictions.jsonl",
        "report": output_dir / "market-signal-v2-report.json",
        "markdown": output_dir / "REPORT.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Market Signal V2 outputs already exist: {existing}")
    panel, panel_summary = build_absolute_panel(
        metadata_path=metadata_path,
        predictions_path=predictions_path,
        qfq_path=qfq_path,
        raw_path=raw_path,
        benchmark_path=benchmark_path,
    )
    walk_forward, report = evaluate_absolute_panel(panel, min_train_dates=min_train_dates)
    report = {
        **report,
        "panel_summary": panel_summary,
        "inputs": {
            "metadata": str(metadata_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "qfq": str(qfq_path.resolve()),
            "raw": str(raw_path.resolve()),
            "benchmark": str(benchmark_path.resolve()),
        },
    }
    v1._write_jsonl(outputs["panel"], panel)
    v1._write_jsonl(outputs["walk_forward"], walk_forward)
    report = {
        **report,
        "artifacts": {
            "panel": {"path": str(outputs["panel"]), "sha256": v1._sha256(outputs["panel"])},
            "walk_forward": {
                "path": str(outputs["walk_forward"]),
                "sha256": v1._sha256(outputs["walk_forward"]),
            },
        },
    }
    v1._write_json(outputs["report"], report)
    outputs["markdown"].write_text(render_report(report, panel_summary), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=V1_INPUT_ROOT / "sampled-post-metadata.jsonl")
    parser.add_argument("--predictions", type=Path, default=V1_INPUT_ROOT / "m12-predictions.jsonl")
    parser.add_argument("--qfq", type=Path, default=MARKET_ROOT / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=MARKET_ROOT / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=MARKET_ROOT / "benchmark-000300.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-dates", type=int, default=v1.DEFAULT_MIN_TRAIN_DATES)
    args = parser.parse_args(argv)
    result = run(
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
