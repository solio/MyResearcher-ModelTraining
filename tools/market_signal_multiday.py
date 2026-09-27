"""Reusable multi-session crowd-state evaluation; no Encoder or network needed.

Run directly: .venv/bin/python tools/market_signal_multiday.py --help
The historical semantic inputs make this a retrospective market-layer study.
Expanding/rolling time folds only fit labels that matured before the test block.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import platform
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# A direct script invocation works from any directory, without PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools import market_signal_v1 as v1
from tools import market_signal_v2 as v2
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as validation

CORE_STATE = (
    "stance_index", "action_index", "emotion_balance", "emotion_fomo_mass",
    "group_disagreement", "model_uncertainty", "group_concentration",
    "log1p_total_post_count", "log1p_unique_text_count",
)
MODELS = ("global_prior", "stock_prior", "market_only", "crowd_only", "combined")
COMPARISONS = (("combined", "market_only"), ("combined", "global_prior"),
               ("combined", "stock_prior"), ("crowd_only", "stock_prior"))


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def prepare_daily_states(args: argparse.Namespace, calendar: Sequence[str]) -> tuple[list[dict], dict]:
    metadata = validation.read_jsonl(args.history_metadata)
    predictions = validation.read_jsonl(args.history_predictions)
    historical = v2.aggregate_crowd_state(metadata, predictions)
    items_by_unit: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metadata:
        items_by_unit[(str(row["stock_code"]), str(row["entry_date"]))].append(row)
    if not items_by_unit:
        raise ValueError("historical probability inputs are empty")
    historical_cap = max(len(items) for items in items_by_unit.values())
    if args.post_cap > historical_cap:
        raise ValueError(f"requested post cap {args.post_cap} exceeds the historical probability sample cap {historical_cap}")
    prediction_by_id = {row["sample_id"]: row for row in predictions}
    daily: dict[tuple[str, str], dict] = {}
    for key, items in items_by_unit.items():
        code, entry = key
        selected = v1.systematic_sample(sorted(items, key=lambda row: (row["published_at"], row["sample_id"])), args.post_cap)
        features = historical[f"market-signal-unit:{code}:{entry}"]
        if len(selected) < len(items):
            features = v2.aggregate_crowd_state(selected, [prediction_by_id[row["sample_id"]] for row in selected])[f"market-signal-unit:{code}:{entry}"]
        daily[key] = {
            "stock_code": code, "entry_date": entry,
            "stock_name": items[0].get("stock_name", ""),
            "features": features,
            "sampled_posts": len(selected),
            "observed_posts": int(items[0]["unit_total_post_count"]),
            "unique_texts": int(items[0]["unit_unique_text_count"]),
            "evidence_type": "RETROSPECTIVE_SEMANTIC_FEATURE",
            "source_file": str(Path(args.history_predictions).resolve()),
        }
    source_reports = [{
        "path": str(Path(args.history_predictions).resolve()), "prediction_rows": len(predictions),
        "daily_units": len(daily), "role": "historical sampled M12 probabilities",
    }]
    skipped = []
    paths: dict[Path, list[dict] | None] = {}
    if args.early_forward:
        path = Path(args.early_forward)
        if path.exists():
            paths[path.resolve()] = None
    directories = [Path(value) for value in (args.forward_dir or [])]
    if args.forward_glob:
        directories.extend(Path(value) for value in sorted(glob.glob(args.forward_glob)))
    for directory in sorted(set(directories)):
        state_file = directory / "community-state-snapshots.jsonl"
        prediction_file = directory / "forward-shadow-predictions.jsonl"
        if not state_file.exists() or not prediction_file.exists():
            skipped.append({"directory": str(directory), "reason": "no complete community snapshot and prediction pair"})
            continue
        states = validation.read_jsonl(state_file)
        if not states or any(row.get("snapshot_status") != "FORWARD" for row in states):
            skipped.append({"directory": str(directory), "reason": "snapshot is not a forward batch"})
            continue
        paths[prediction_file.resolve()] = states
    for path, frozen_states in sorted(paths.items()):
        rows = validation.read_jsonl(path)
        entry_by_stock = {str(row["stock_code"]): str(row["next_trading_entry_date"]) for row in frozen_states or []}
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            code = str(row["stock_code"])
            if frozen_states is not None:
                entry = entry_by_stock[code]
            else:
                available = snapshot._snapshot_signal_available_date(row).isoformat()
                entry = v1.next_entry_date(available, calendar)
            if entry is None:
                continue
            grouped[(code, entry)].append(row)
        added = 0
        for (code, entry), members in sorted(grouped.items()):
            # Match the original historical time-spanning sample policy.
            members = sorted(members, key=lambda row: (row["published_at"], row["sample_id"]))
            by_text: dict[str, dict] = {}
            for member in members:
                by_text.setdefault(v1._normalize_text(member.get("model_text")), member)
            unique = list(by_text.values())
            selected = v1.systematic_sample(unique, args.post_cap)
            meta = [{
                "sample_id": row["sample_id"], "stock_code": code, "entry_date": entry,
                "unit_total_post_count": len(members), "unit_unique_text_count": len(unique),
            } for row in selected]
            features = v2.aggregate_crowd_state(meta, selected)[f"market-signal-unit:{code}:{entry}"]
            key = (code, entry)
            if key in daily:
                raise ValueError(f"overlapping daily input batches for {code} x {entry}; supply one source per unit")
            daily[key] = {
                "stock_code": code, "entry_date": entry, "stock_name": members[0].get("stock_name", ""),
                "features": features, "sampled_posts": len(selected), "observed_posts": len(members),
                "unique_texts": len(unique), "evidence_type": "FORWARD_POST_PREDICTIONS",
                "source_file": str(path),
            }
            added += 1
        source_reports.append({"path": str(path), "prediction_rows": len(rows), "daily_units": added,
                               "role": "forward probabilities; resampled with the same post cap"})
    for row in daily.values():
        f = row["features"]
        for dest, prefix in (("group_disagreement", "group_disagreement"),
                             ("model_uncertainty", "model_uncertainty"),
                             ("group_concentration", "group_consensus")):
            f[dest] = statistics.fmean(f[f"{prefix}::{head}"] for head in v2.GROUP_HEADS)
    return list(sorted(daily.values(), key=lambda row: (row["entry_date"], row["stock_code"]))), {
        "sources": source_reports, "skipped_sources": skipped,
        "historical_posts": len(predictions), "all_input_prediction_rows": sum(s["prediction_rows"] for s in source_reports),
        "daily_units": len(daily), "daily_entry_dates": len({entry for _, entry in daily}),
        "stocks": len({code for code, _ in daily}),
        "historical_semantic_model_is_not_as_of_each_historical_date": True,
        "post_sampling_policy": f"deduplicated, time-spanning, at most {args.post_cap} probabilities per stock/session",
        "attention_scope": "counts observed in each original input; not guaranteed full forum population coverage",
    }


def rolling_features(rows: Sequence[Mapping[str, Any]], positions: Sequence[int], window: int) -> dict[str, float]:
    import numpy as np
    output: dict[str, float] = {}
    x = np.asarray(positions, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    for name in CORE_STATE:
        values = np.asarray([row["features"][name] for row in rows], dtype=float)
        mean = float(values.mean())
        output[f"mean::{name}"] = mean
        output[f"slope::{name}"] = float(np.dot(x, values - mean) / denominator) if denominator else 0.0
        recent = [float(value) for pos, value in zip(positions, values) if pos >= window - min(5, window)]
        output[f"recent5_minus_mean::{name}"] = statistics.fmean(recent) - mean if recent else 0.0
    for name in ("stance_index", "action_index", "emotion_balance"):
        values = [float(row["features"][name]) for row in rows]
        output[f"positive_fraction::{name}"] = statistics.fmean(value > 0.0 for value in values)
        output[f"negative_fraction::{name}"] = statistics.fmean(value < 0.0 for value in values)
    return output


def build_panel(daily: Sequence[Mapping[str, Any]], calendar: Sequence[str], qfq: Mapping, raw: Mapping,
                benchmark_rows: Sequence[Mapping], *, window: int, horizon: int,
                as_of: str, min_coverage: float, return_threshold: float) -> list[dict]:
    state = {(row["stock_code"], row["entry_date"]): row for row in daily}
    codes = sorted({row["stock_code"] for row in daily})
    qfq_by_code = validation.index_bars(qfq.values())
    raw_by_code = validation.index_bars(raw.values())
    rows = []
    for code in codes:
        for index, entry in enumerate(calendar):
            if index < window:
                continue
            interval = calendar[index - window + 1:index + 1]
            pairs = [(pos, state[(code, day)]) for pos, day in enumerate(interval) if (code, day) in state]
            base = {"stock_code": code, "entry_date": entry, "entry_index": index,
                    "window_trading_days": window, "horizon_trading_days": horizon,
                    "signal_window_first_entry": interval[0], "signal_window_last_entry": entry,
                    "signal_sessions_observed": len(pairs), "signal_coverage": len(pairs) / window}
            if len(pairs) / window < min_coverage:
                rows.append(dict(base, status="MISSING_SIGNAL_SESSIONS", missing_signal_sessions=[d for d in interval if (code, d) not in state]))
                continue
            if not pairs:
                rows.append(dict(base, status="MISSING_SIGNAL_SESSIONS"))
                continue
            selected = [item for _, item in pairs]
            crowd = rolling_features(selected, [pos for pos, _ in pairs], window)
            market = v1.build_market_features(
                {"stock_code": code, "entry_date_override": entry}, qfq_by_code=qfq_by_code,
                raw_by_code=raw_by_code, benchmark_rows=benchmark_rows)
            previous = qfq.get((code, calendar[index - 1]), {})
            older = qfq.get((code, calendar[index - window]), {})
            previous_close, older_close = finite_number(previous.get("close")), finite_number(older.get("close"))
            market["window_stock_close_return"] = previous_close / older_close - 1.0 if previous_close is not None and older_close else None
            prior_sessions = calendar[index - window:index]
            vols = [finite_number(raw.get((code, d), {}).get("volume")) for d in prior_sessions]
            valid_vols = [v for v in vols if v is not None]
            market["window_volume_mean"] = statistics.fmean(valid_vols) if valid_vols else None
            base.update({"crowd_features": crowd, "market_features": market,
                         "observed_posts_in_window": sum(r["observed_posts"] for r in selected),
                         "sampled_posts_in_window": sum(r["sampled_posts"] for r in selected),
                         "evidence_type": "RETROSPECTIVE_MARKET_LAYER_STUDY"})
            exit_index = index + horizon - 1
            if exit_index >= len(calendar) or calendar[exit_index] > as_of:
                rows.append(dict(base, status="PENDING_HORIZON"))
                continue
            exit_day = calendar[exit_index]
            checks = [(qfq, entry, "open"), (qfq, exit_day, "close"), (raw, entry, "open"), (raw, exit_day, "close")]
            values = [finite_number(bars.get((code, day), {}).get(field)) for bars, day, field in checks]
            base.update(outcome_date=exit_day, outcome_index=exit_index)
            if any(value is None or value <= 0 for value in values):
                rows.append(dict(base, status="MISSING_PRICE"))
                continue
            entry_q, exit_q, entry_raw, exit_raw = values
            ret, raw_ret = exit_q / entry_q - 1.0, exit_raw / entry_raw - 1.0
            same_vintage = qfq[(code, entry)].get("fetched_at") == qfq[(code, exit_day)].get("fetched_at")
            if not same_vintage and abs(ret - raw_ret) > 0.001:
                rows.append(dict(base, status="ADJUSTMENT_VINTAGE_MISMATCH"))
                continue
            rows.append(dict(base, status="COMPLETE", stock_return=ret, stock_positive=ret > return_threshold,
                             entry_open_qfq=entry_q, outcome_close_qfq=exit_q))
    return rows


def make_model(features: Sequence[tuple[str, str]], *, c: float, seed: int):
    model = v1._make_model(features)
    model.set_params(logistic__C=c, logistic__random_state=seed)
    return model


def time_cross_validation(panel: Sequence[Mapping[str, Any]], *, min_train_dates: int, test_block_dates: int,
                          train_window_dates: int, embargo_sessions: int, c: float, seed: int) -> tuple[list[dict], list[dict]]:
    complete = [row for row in panel if row["status"] == "COMPLETE"]
    by_date: dict[str, list[Mapping]] = defaultdict(list)
    for row in complete:
        by_date[row["entry_date"]].append(row)
    dates = sorted(by_date)
    crowd = [("crowd_features", key) for key in v1._feature_names(complete, "crowd_features")]
    market = [("market_features", key) for key in v1._feature_names(complete, "market_features")]
    feature_sets = {"market_only": market, "crowd_only": crowd, "combined": market + crowd}
    predictions, audit = [], []
    for start in range(0, len(dates), test_block_dates):
        test_dates = dates[start:start + test_block_dates]
        test_rows = [row for d in test_dates for row in by_date[d]]
        first_index = min(row["entry_index"] for row in test_rows)
        training = [row for row in complete if row["entry_index"] < first_index and row["outcome_index"] < first_index - embargo_sessions]
        eligible_dates = sorted({row["entry_date"] for row in training})
        if train_window_dates:
            retained = set(eligible_dates[-train_window_dates:])
            training = [row for row in training if row["entry_date"] in retained]
            eligible_dates = sorted(retained)
        fold = {"test_start": test_dates[0], "test_end": test_dates[-1], "test_dates": len(test_dates),
                "training_dates": len(eligible_dates), "training_rows": len(training),
                "train_outcome_max": max((row["outcome_date"] for row in training), default=None),
                "window_trading_days": panel[0]["window_trading_days"] if panel else None,
                "horizon_trading_days": panel[0]["horizon_trading_days"] if panel else None}
        if len(eligible_dates) < min_train_dates:
            audit.append(dict(fold, status="INSUFFICIENT_MATURED_TRAIN_DATES"))
            continue
        labels = [int(row["stock_positive"]) for row in training]
        # Both classes are necessary to fit a Logistic Regression.
        if len(set(labels)) < 2:
            audit.append(dict(fold, status="SINGLE_CLASS_TRAINING"))
            continue
        labels_by_stock: dict[str, list[int]] = defaultdict(list)
        for row, label in zip(training, labels):
            labels_by_stock[row["stock_code"]].append(label)
        probability = {"global_prior": [v2.smoothed_prior(labels)] * len(test_rows),
                       "stock_prior": [v2.smoothed_prior(labels_by_stock[row["stock_code"]]) for row in test_rows]}
        for name, keys in feature_sets.items():
            model = make_model(keys, c=c, seed=seed)
            model.fit(v1._model_matrix(training, keys), labels)
            probability[name] = model.predict_proba(v1._model_matrix(test_rows, keys))[:, 1].tolist()
        for i, row in enumerate(test_rows):
            predictions.append({
                "stock_code": row["stock_code"], "entry_date": row["entry_date"], "outcome_date": row["outcome_date"],
                "window_trading_days": row["window_trading_days"], "horizon_trading_days": row["horizon_trading_days"],
                "stock_positive": row["stock_positive"], "stock_return": row["stock_return"],
                "training_dates": len(eligible_dates), "training_rows": len(training),
                "train_outcome_max": fold["train_outcome_max"],
                "probabilities": {name: float(values[i]) for name, values in probability.items()},
            })
        audit.append(dict(fold, status="EVALUATED", numeric_features={key: len(values) for key, values in feature_sets.items()}))
    return predictions, audit


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr
    return finite_number(spearmanr(x, y).statistic)


def rank_metrics(rows: Sequence[Mapping], model: str) -> dict:
    def weighted_tail(items: Sequence[Mapping], count: int, high: bool) -> list[tuple[Mapping, float]]:
        ordered = sorted(items, key=lambda r: r["probabilities"][model], reverse=high)
        cutoff = ordered[count - 1]["probabilities"][model]
        better = [r for r in ordered if (r["probabilities"][model] > cutoff if high else r["probabilities"][model] < cutoff)]
        tied = [r for r in ordered if r["probabilities"][model] == cutoff]
        weight = (count - len(better)) / len(tied)
        return [(r, 1.0) for r in better] + [(r, weight) for r in tied]

    def weighted_mean(items: Sequence[tuple[Mapping, float]], field: str) -> float | None:
        total = sum(weight for _, weight in items)
        return sum(float(r[field]) * weight for r, weight in items) / total if total else None

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[row["entry_date"]].append(row)
    daily, top, bottom = [], [], []
    for day, items in sorted(grouped.items()):
        if len(items) < 4 or len({r["probabilities"][model] for r in items}) < 2:
            continue
        count = max(1, len(items) // 4)
        low, high = weighted_tail(items, count, False), weighted_tail(items, count, True)
        ic = spearman([row["probabilities"][model] for row in items], [row["stock_return"] for row in items])
        spread = weighted_mean(high, "stock_return") - weighted_mean(low, "stock_return")
        daily.append({"entry_date": day, "spearman": ic, "top_minus_bottom_return": spread})
        top.extend(high); bottom.extend(low)
    ics = [row["spearman"] for row in daily if row["spearman"] is not None]
    return {"mean_daily_spearman": statistics.fmean(ics) if ics else None,
            "top_quartile_positive_rate": weighted_mean(top, "stock_positive"),
            "top_quartile_mean_return": weighted_mean(top, "stock_return"),
            "bottom_quartile_mean_return": weighted_mean(bottom, "stock_return"),
            "mean_daily_top_minus_bottom_return": statistics.fmean(row["top_minus_bottom_return"] for row in daily) if daily else None,
            "daily_ranks": daily}


def paired_gain(rows: Sequence[Mapping], candidate: str, baseline: str, *, block_dates: int, repeats: int, seed: int) -> dict:
    import numpy as np
    point = v2.macro_stock_brier(rows, baseline) - v2.macro_stock_brier(rows, candidate) if rows else None
    dates = sorted({row["entry_date"] for row in rows})
    result = {"brier_gain": point, "positive_means_candidate_better": True,
              "block_dates": block_dates, "nonoverlapping_block_equivalent": len(dates) // block_dates,
              "ci95": None, "bootstrap_status": "TOO_FEW_DATES_FOR_TWO_BLOCKS"}
    if not rows or len(dates) < 2 * block_dates or repeats == 0:
        if repeats == 0:
            result["bootstrap_status"] = "DISABLED"
        return result
    by_date = {day: [row for row in rows if row["entry_date"] == day] for day in dates}
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(repeats):
        selected: list[str] = []
        while len(selected) < len(dates):
            start = int(rng.integers(0, len(dates) - block_dates + 1))
            selected.extend(dates[start:start + block_dates])
        sample = [row for day in selected[:len(dates)] for row in by_date[day]]
        gains.append(v2.macro_stock_brier(sample, baseline) - v2.macro_stock_brier(sample, candidate))
    result.update(ci95=np.quantile(gains, [0.025, 0.975]).tolist(), bootstrap_status="EXPLORATORY_DATE_BLOCK_BOOTSTRAP")
    return result


def describe_combo(panel: Sequence[Mapping], predictions: Sequence[Mapping], audit: Sequence[Mapping], *,
                   window: int, horizon: int, args: argparse.Namespace, calendar_size: int) -> dict:
    complete = [row for row in panel if row["status"] == "COMPLETE"]
    evaluated_dates = sorted({row["entry_date"] for row in predictions})
    models = {}
    for model in MODELS:
        models[model] = {**v2._basic_metrics(predictions, model), **rank_metrics(predictions, model)}
    block = args.bootstrap_block_dates or max(window, horizon)
    comparisons = {f"{candidate}_vs_{baseline}": paired_gain(predictions, candidate, baseline,
                   block_dates=block, repeats=args.bootstrap_repeats, seed=args.seed) for candidate, baseline in COMPARISONS}
    per_stock = []
    for code in sorted({row["stock_code"] for row in predictions}):
        selected = [row for row in predictions if row["stock_code"] == code]
        per_stock.append({"stock_code": code, "n": len(selected), "positive_rate": statistics.fmean(r["stock_positive"] for r in selected),
                          "brier": {name: v2.macro_stock_brier(selected, name) for name in MODELS}})
    # Fixed direct state ranks are descriptive, including combinations with no fit.
    diagnostics = {}
    for feature in ("stance_index", "action_index", "emotion_balance"):
        by_day: dict[str, list[Mapping]] = defaultdict(list)
        for row in complete:
            by_day[row["entry_date"]].append(row)
        values = [spearman([r["crowd_features"][f"mean::{feature}"] for r in day], [r["stock_return"] for r in day])
                  for day in by_day.values()]
        values = [value for value in values if value is not None]
        diagnostics[feature] = {"mean_daily_spearman": statistics.fmean(values) if values else None,
                                "dates": len(values), "role": "DESCRIPTIVE_NOT_CROSS_VALIDATED_PROBABILITY"}
    # Optimistic minimum: continuous signal coverage, two-class labels, no gaps.
    minimum_sessions = window + args.min_train_dates + 2 * horizon - 1 + args.embargo_sessions
    return {
        "window_trading_days": window, "horizon_trading_days": horizon,
        "status": "EXPLORATORY_TIME_CV_COMPLETE" if predictions else "INSUFFICIENT_TIME_HISTORY_FOR_CV",
        "panel_status_counts": dict(Counter(row["status"] for row in panel)),
        "matured_feature_rows": len(complete), "matured_feature_dates": len({r["entry_date"] for r in complete}),
        "cv_rows": len(predictions), "cv_entry_dates": len(evaluated_dates),
        "cv_date_start": evaluated_dates[0] if evaluated_dates else None,
        "cv_date_end": evaluated_dates[-1] if evaluated_dates else None,
        "fold_status_counts": dict(Counter(row["status"] for row in audit)),
        "optimistic_minimum_calendar_sessions_for_first_cv_result": minimum_sessions,
        "additional_calendar_sessions_lower_bound": max(0, minimum_sessions - calendar_size),
        "availability_estimate_assumptions": "complete signal coverage, daily test blocks, and both training classes; gaps can delay evaluation further",
        "models": models, "comparisons": comparisons, "per_stock": per_stock,
        "direct_state_rank_diagnostics": diagnostics,
    }


def common_sample_control(panels: Mapping[tuple[int, int], Sequence[Mapping]], args: argparse.Namespace,
                          calendar_size: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Compare N values on identical training/test identities and market inputs."""
    reports, predictions, folds = [], [], []
    if not args.common_sample_control or len(args.windows) < 2:
        return reports, predictions, folds
    for horizon in args.horizons:
        by_window = {window: [r for r in panels[(window, horizon)] if r["status"] == "COMPLETE"] for window in args.windows}
        identities = [set((r["stock_code"], r["entry_date"]) for r in rows) for rows in by_window.values()]
        common = set.intersection(*identities)
        candidates = []
        for window, rows in by_window.items():
            controlled = [{**r, "market_features": {key: value for key, value in r["market_features"].items() if not key.startswith("window_")}}
                          for r in rows if (r["stock_code"], r["entry_date"]) in common]
            pred, audit = time_cross_validation(controlled, min_train_dates=args.min_train_dates,
                test_block_dates=args.test_block_dates, train_window_dates=args.train_window_dates,
                embargo_sessions=args.embargo_sessions, c=args.logistic_c, seed=args.seed)
            candidates.append(describe_combo(controlled, pred, audit, window=window, horizon=horizon,
                                              args=args, calendar_size=calendar_size))
            predictions.extend(pred); folds.extend(audit)
        reports.append({"horizon_trading_days": horizon, "common_matured_stock_dates": len(common),
                        "market_features_policy": "identical fixed lag1/3/5 market covariates; N-specific market additions removed",
                        "training_and_test_identities_identical_across_N": True, "candidates": candidates})
    return reports, predictions, folds


def metric(value: Any, percent: bool = False) -> str:
    return "—" if value is None else (f"{100 * value:+.2f}%" if percent else f"{value:.4f}")


def markdown_report(report: Mapping) -> str:
    c = report["config"]
    evaluated = [combo for combo in report["combinations"] if combo["cv_rows"]]
    improved = sum(combo["comparisons"]["combined_vs_market_only"]["brier_gain"] > 0 for combo in evaluated)
    lines = ["# 多日群体状态与中期收益：时间交叉验证报告", "",
             f"生成时间：{report['generated_at']}；行情截至 {report['data']['as_of']}。", "",
             "## 本轮结果", "",
             f"{len(report['combinations'])} 个组合中，{len(evaluated)} 个产生样本外概率，{len(report['combinations']) - len(evaluated)} 个时间历史不足；可评测组合中，{improved} 个组合模型的 Brier 点估计优于纯行情。当前结果只支持进一步验证，没有自动晋级或选型。", "",
             "## 目标与方法", "",
             f"过去 N={c['windows']} 个交易日的群体状态 → 未来 H={c['horizons']} 个交易日，复权开盘到期限收盘的收益是否超过 {c['return_threshold']:.4%}。",
             "N 个状态按交易日等权；包括均值、变化斜率、最近 5 日相对窗口的变化及多空状态持续比例。周末/假期帖子归入下一实际入场交易日。",
             f"每折测试 {c['test_block_dates']} 个入场日；训练至少 {c['min_train_dates']} 个已经成熟的日期；仅使用 outcome_date < test_start 的标签。训练中的缺失值填补、标准化和模型全部仅在训练集拟合。",
             f"固定逻辑回归 C={c['logistic_c']}；比较同股历史胜率、纯行情、纯群体状态、行情加群体状态。没有调参搜索或挑选最佳组合。", "",
             "历史 M12 概率在历史市场日期之后生成，因此本报告是市场层的回顾性时间交叉验证，不是整个系统当时真实部署的前向成绩。", "",
             "## 数据覆盖", "",
             f"行情 {report['data']['calendar_start']}～{report['data']['calendar_end']}，{report['data']['calendar_sessions']} 个实际交易日；{report['inputs']['stocks']} 只股票；复用 {report['inputs']['all_input_prediction_rows']:,} 条已存在的帖子预测。",
             f"已构建 {report['inputs']['daily_units']} 个股票×入场日状态，{report['inputs']['daily_entry_dates']} 个不同入场日期。历史语义样本为确定性时间覆盖抽样，最近批次按同一 cap={c['post_cap']} 重新抽取既有概率。",
             f"信号窗口最低覆盖率 {c['min_signal_coverage']:.0%}；缺失日不会填成零情绪。注意力数量仅代表原输入中观察到的帖子，并非保证全论坛覆盖。", "",
             "## 六个窗口组合的结果", "",
             "Brier 越小越好；增益 = 基线 Brier − 候选 Brier，正值表示候选更好。收益窗口重叠，股票和日期均可能相关。", "",
             "| N | H | 成熟日期 | CV 日期/股票日 | 全体先验 Brier | 同股先验 Brier | 行情 Brier | 群体 Brier | 组合 Brier | 组合对行情增益 | 状态 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for combo in report["combinations"]:
        models = combo["models"]
        values = [metric(models[name]["macro_stock_brier"]) for name in ("global_prior", "stock_prior", "market_only", "crowd_only", "combined")]
        gain = metric(combo["comparisons"]["combined_vs_market_only"]["brier_gain"])
        status = "已完成，探索性" if combo["cv_rows"] else "时间历史不足"
        lines.append(f"| {combo['window_trading_days']} | {combo['horizon_trading_days']} | {combo['matured_feature_dates']} | {combo['cv_entry_dates']}/{combo['cv_rows']} | {' | '.join(values)} | {gain} | {status} |")
    lines.extend(["", "## 同训练与测试样本的窗口对照", "",
                  "上表尽量使用各组合的全部可用历史。下面只使用同一期限下各 N 都完整的股票×日期；训练和测试样本、同股先验及固定行情特征一致，单独对照群体状态窗口。", "",
                  "| H | N | 共同 CV 日期/股票日 | 全体先验 Brier | 行情 Brier | 群体 Brier | 组合 Brier | 组合对行情增益 | 组合前 25% 胜率 | 前 25% 平均收益 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for control in report.get("common_sample_controls", []):
        for combo in control["candidates"]:
            m = combo["models"]
            lines.append(f"| {combo['horizon_trading_days']} | {combo['window_trading_days']} | {combo['cv_entry_dates']}/{combo['cv_rows']} | {metric(m['global_prior']['macro_stock_brier'])} | {metric(m['market_only']['macro_stock_brier'])} | {metric(m['crowd_only']['macro_stock_brier'])} | {metric(m['combined']['macro_stock_brier'])} | {metric(combo['comparisons']['combined_vs_market_only']['brier_gain'])} | {metric(m['combined']['top_quartile_positive_rate'], True)} | {metric(m['combined']['top_quartile_mean_return'], True)} |")
    lines.extend(["", "共同样本对照没有扩大独立时间历史；它用于解释窗口差异，不用作额外调参或独立复验。",
                  "全体先验只使用该折已成熟的历史上涨率，不做股票区分。优于纯行情但未优于全体先验时，不能据此认定状态已有稳定预测价值。", "",
                  "## 增益、胜率与排序（各组合最大可用样本）", ""])
    for combo in report["combinations"]:
        n, h = combo["window_trading_days"], combo["horizon_trading_days"]
        lines += [f"### N={n}，H={h}", ""]
        lines += ["面板状态：" + "；".join(f"{key}={value}" for key, value in combo["panel_status_counts"].items()) + "。缺失信号日与复权批次不一致的价格区间未进入 CV。", ""]
        if not combo["cv_rows"]:
            lines += [f"已有 {combo['matured_feature_dates']} 个成熟特征日期，但尚无满足时间顺序和训练要求的测试折。至少还需 {combo['additional_calendar_sessions_lower_bound']} 个行情交易日（连续信号覆盖、两类标签等条件满足时的乐观下限）。", ""]
        else:
            lines += [f"测试入场范围 {combo['cv_date_start']}～{combo['cv_date_end']}，{combo['cv_entry_dates']} 个日期；以下为样本外预测。", "",
                      "| 模型 | 0.5 阈值覆盖率 | 预测正收益的实际胜率 | 排名前 25% 胜率 | 前 25% 平均收益 | 日均排序相关 |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name in MODELS:
                m = combo["models"][name]
                lines.append(f"| {name} | {metric(m['positive_prediction_coverage_at_0_5'], True)} | {metric(m['positive_predictive_value_at_0_5'], True)} | {metric(m['top_quartile_positive_rate'], True)} | {metric(m['top_quartile_mean_return'], True)} | {metric(m['mean_daily_spearman'])} |")
            lines.append("")
            for name, comparison in combo["comparisons"].items():
                ci = comparison["ci95"]
                text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "未计算：日期不足以形成两个相关性区块"
                lines += [f"- {name}：Brier 增益 {metric(comparison['brier_gain'])}；95% 日期区块区间 {text}，区块长度 {comparison['block_dates']}。"]
            lines.append("")
        direct = combo["direct_state_rank_diagnostics"]
        lines += ["固定状态均值与未来收益的描述性日均 Spearman：" + "；".join(f"{key}={metric(value['mean_daily_spearman'])}" for key, value in direct.items()) + "。这些相关性不是概率模型交叉验证成绩。", ""]
    lines += ["## 使用边界与下一步", "",
              "本轮只判断现有多日状态是否提供市场增量信息。点估计改善不等于稳定有效，尤其 20 日期限可能只有少量测试日期。没有交叉验证成绩的组合保留为数据不足，不判为模型失败。",
              "日期区块长度默认 max(N,H)，不足两个区块时不输出置信区间。输出区间也只是探索性估计，没有对多组合比较作显著性校正。",
              "结果是扣费前的期限末正收益概率；没有把期间曾经上涨、现金仓位、成交限制或组合复利算成策略收益。排序无概率差异时不报告前四分位成绩；边界处并列概率按等额权重分摊，不按股票代码强行选股。",
              "已有输入和历史预测保持原样；继续增加行情、每日状态后用同一配置重跑。若根据这份报告选定组合，后续新日期需要另做验证。", "",
              "运行配置见 run-config.json；逐条样本外概率见 cv-predictions.jsonl；切分见 fold-audit.jsonl；所有指标及每股结果见 report.json。", ""]
    return "\n".join(lines)


def heatmap_svg(report: Mapping) -> str:
    combos = report["combinations"]
    windows, horizons = report["config"]["windows"], report["config"]["horizons"]
    width, height = 230 + 200 * len(horizons), 115 + 85 * len(windows)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#fff"/>',
              '<g font-family="sans-serif" fill="#17233a">',
              '<text x="20" y="28" font-size="17">Brier gain: combined vs market-only</text>',
              '<text x="20" y="49" font-size="12">Positive = improvement; n = time-CV entry dates; gray = insufficient history</text>']
    for col, h in enumerate(horizons):
        pieces.append(f'<text x="{240 + col * 200}" y="85" font-size="14">H={h} sessions</text>')
    for row, n in enumerate(windows):
        y = 100 + row * 85
        pieces.append(f'<text x="20" y="{y + 36}" font-size="14">N={n} sessions</text>')
        for col, h in enumerate(horizons):
            combo = next(item for item in combos if item["window_trading_days"] == n and item["horizon_trading_days"] == h)
            gain = combo["comparisons"]["combined_vs_market_only"]["brier_gain"]
            fill = "#edf0f4" if gain is None else ("#d7efe7" if gain >= 0 else "#f7dedd")
            x = 215 + col * 200
            pieces += [f'<rect x="{x}" y="{y}" width="190" height="72" rx="6" fill="{fill}"/>',
                       f'<text x="{x + 14}" y="{y + 30}" font-size="18">{metric(gain)}</text>',
                       f'<text x="{x + 14}" y="{y + 53}" font-size="12">n={combo["cv_entry_dates"]}; stock-days={combo["cv_rows"]}</text>']
    return "".join(pieces) + "</g></svg>"


def html_report(report: Mapping, md: str, svg: str) -> str:
    sections = []
    for combo in report["combinations"]:
        if not combo["cv_rows"]:
            continue
        calibration = []
        for model in ("stock_prior", "market_only", "crowd_only", "combined"):
            for bin_row in combo["models"][model]["calibration_bins"]:
                calibration.append(f'<tr><td>{model}</td><td>{bin_row["n"]}</td><td>{bin_row["mean_probability"]:.3f}</td><td>{bin_row["positive_rate"]:.3f}</td></tr>')
        sections.append(f'<h2>N={combo["window_trading_days"]}, H={combo["horizon_trading_days"]}: probability calibration</h2>'
                        '<table><tr><th>Model</th><th>Count</th><th>Predicted win probability</th><th>Observed win rate</th></tr>' + ''.join(calibration) + '</table>')
    parts, table_open = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            if re.fullmatch(r"[| :\-]+", line):
                continue
            if not table_open:
                parts.append('<div class="table-scroll"><table>'); table_open = True
                tag = "th"
            else:
                tag = "td"
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            parts.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
            continue
        if table_open:
            parts.append("</table></div>"); table_open = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            parts.append(f"<h{level}>{html.escape(line[level:].strip())}</h{level}>")
        elif line:
            parts.append("<p>" + html.escape(line) + "</p>")
    if table_open:
        parts.append("</table></div>")
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多日群体状态评测</title><style>body{font:16px/1.7 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1100px;color:#17233a}svg{max-width:100%;height:auto}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}h1,h2,h3{line-height:1.3}h2{margin-top:36px}</style><h1>多日群体状态与中期收益</h1>' + svg + ''.join(parts) + ''.join(sections) + '</html>'


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", type=Path)
    pre, _ = early.parse_known_args(argv)
    defaults = json.loads(pre.config.read_text(encoding="utf-8")) if pre.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[early], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    history = PROJECT_ROOT / "pipeline-runs/market-signal-v1-20260917"
    market = PROJECT_ROOT / "pipeline-runs/market-data-v0-20260916"
    parser.add_argument("--history-metadata", type=Path, default=history / "sampled-post-metadata.jsonl")
    parser.add_argument("--history-predictions", type=Path, default=history / "m12-predictions.jsonl")
    parser.add_argument("--early-forward", default=str(market / "forward-shadow-predictions.jsonl"), help="early forward prediction file; empty string disables it")
    parser.add_argument("--forward-glob", default=str(PROJECT_ROOT / "pipeline-runs/market-signal-v2-forward-*"), help="directories containing completed FORWARD snapshots; empty string disables discovery")
    parser.add_argument("--forward-dir", action="append", default=[], help="additional frozen forward directory; repeatable")
    parser.add_argument("--qfq-bars", type=Path, default=market / "daily-bars-qfq.csv")
    parser.add_argument("--raw-bars", type=Path, default=market / "daily-bars-raw.csv")
    parser.add_argument("--benchmark-bars", type=Path, default=market / "benchmark-000300.csv")
    parser.add_argument("--as-of", default="latest", help="YYYY-MM-DD or latest; later prices are excluded")
    parser.add_argument("--windows", type=positive_int, nargs="+", default=[5, 20])
    parser.add_argument("--horizons", type=positive_int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--min-train-dates", type=positive_int, default=5, help="exploratory minimum matured training dates, shared by all models")
    parser.add_argument("--test-block-dates", type=positive_int, default=1, help="nonoverlapping test blocks; daily by default")
    parser.add_argument("--train-window-dates", type=int, default=0, help="0 = expanding; otherwise retain last K eligible training dates")
    parser.add_argument("--embargo-sessions", type=int, default=0, help="extra actual-session gap after training outcomes")
    parser.add_argument("--min-signal-coverage", type=float, default=1.0, help="observed sessions / N; default requires the complete window")
    parser.add_argument("--post-cap", type=positive_int, default=16, help="may reduce sampling; cannot exceed the available historical cap")
    parser.add_argument("--return-threshold", type=float, default=0.0, help="return fraction: 0.01 means +1%%; no fees inferred")
    parser.add_argument("--logistic-c", type=float, default=0.1, help="fixed regularization, no parameter search")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--bootstrap-block-dates", type=int, default=0, help="0 = max(N,H); no CI with fewer than two blocks")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--common-sample-control", action=argparse.BooleanOptionalAction, default=True,
                        help="also compare lookback windows using identical train/test identities and fixed market covariates")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "pipeline-runs" / ("market-signal-multiday-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    known = {action.dest for action in parser._actions}
    unknown = set(defaults) - known
    if unknown:
        parser.error(f"unknown configuration keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.windows = sorted(set(args.windows)); args.horizons = sorted(set(args.horizons))
    if any(value <= 0 for value in args.windows + args.horizons):
        parser.error("windows and horizons must be positive")
    if args.min_train_dates < 1 or args.test_block_dates < 1 or args.post_cap < 1:
        parser.error("training/test counts and post cap must be positive")
    if any(getattr(args, name) < 0 for name in ("train_window_dates", "embargo_sessions", "bootstrap_repeats", "bootstrap_block_dates")):
        parser.error("window/gap/bootstrap counts must be nonnegative")
    if args.train_window_dates and args.train_window_dates < args.min_train_dates:
        parser.error("train-window-dates must be >= min-train-dates")
    if not 0 < args.min_signal_coverage <= 1 or not math.isfinite(args.logistic_c) or args.logistic_c <= 0 or not math.isfinite(args.return_threshold):
        parser.error("invalid coverage, logistic C, or return threshold")
    return args


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}; use a new --output-dir")
    qfq = snapshot._read_stock_bars(Path(args.qfq_bars), adjustment="qfq")
    raw = snapshot._read_stock_bars(Path(args.raw_bars), adjustment="raw")
    benchmark = validation.read_csv(args.benchmark_bars)
    observed = sorted({row["trade_date"] for row in benchmark})
    as_of = observed[-1] if args.as_of == "latest" else date.fromisoformat(args.as_of).isoformat()
    calendar = [day for day in observed if day <= as_of]
    if not calendar:
        raise ValueError("no actual benchmark sessions at as-of date")
    # No future bars enter even features or calendar length estimates.
    qfq = {key: value for key, value in qfq.items() if key[1] <= as_of}
    raw = {key: value for key, value in raw.items() if key[1] <= as_of}
    benchmark = [row for row in benchmark if row["trade_date"] <= as_of]
    daily, inputs = prepare_daily_states(args, calendar)
    all_panel, all_predictions, all_audit, combinations, panels = [], [], [], [], {}
    for window in args.windows:
        for horizon in args.horizons:
            panel = build_panel(daily, calendar, qfq, raw, benchmark, window=window, horizon=horizon,
                                as_of=as_of, min_coverage=args.min_signal_coverage, return_threshold=args.return_threshold)
            panels[(window, horizon)] = panel
            predictions, audit = time_cross_validation(panel, min_train_dates=args.min_train_dates,
                test_block_dates=args.test_block_dates, train_window_dates=args.train_window_dates,
                embargo_sessions=args.embargo_sessions, c=args.logistic_c, seed=args.seed)
            combo = describe_combo(panel, predictions, audit, window=window, horizon=horizon,
                                   args=args, calendar_size=len(calendar))
            combinations.append(combo)
            all_panel.extend(panel); all_predictions.extend(predictions); all_audit.extend(audit)
            print(json.dumps({"N": window, "H": horizon, "cv_dates": combo["cv_entry_dates"],
                              "cv_rows": combo["cv_rows"], "status": combo["status"]}), flush=True)
    controls, control_predictions, control_folds = common_sample_control(panels, args, len(calendar))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ("config", "output_dir")}
    report = {"status": "RETROSPECTIVE_MULTIDAY_TIME_CV_COMPLETE", "generated_at": datetime.now(timezone.utc).isoformat(),
              "config": config, "inputs": inputs, "data": {"as_of": as_of, "calendar_start": calendar[0],
              "calendar_sessions": len(calendar), "calendar_end": calendar[-1], "qfq_rows": len(qfq), "raw_rows": len(raw)},
              "combinations": combinations, "common_sample_controls": controls, "no_automatic_model_promotion": True,
              "no_semantic_training_or_network_calls": True}
    from importlib.metadata import version
    report["runtime"] = {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")}}
    output.mkdir(parents=True, exist_ok=True)
    v1._write_json(output / "run-config.json", config)
    v1._write_json(output / "report.json", report)
    v1._write_jsonl(output / "daily-states.jsonl", daily)
    v1._write_jsonl(output / "rolling-panel.jsonl", all_panel)
    v1._write_jsonl(output / "cv-predictions.jsonl", all_predictions)
    v1._write_jsonl(output / "fold-audit.jsonl", all_audit)
    v1._write_jsonl(output / "common-sample-cv-predictions.jsonl", control_predictions)
    v1._write_jsonl(output / "common-sample-folds.jsonl", control_folds)
    v1._write_jsonl(output / "daily-cv-metrics.jsonl", [
        {"window_trading_days": row["window_trading_days"], "horizon_trading_days": row["horizon_trading_days"],
         "model": name, **day}
        for row in combinations for name in MODELS for day in row["models"][name]["daily_ranks"]
    ])
    md = markdown_report(report)
    svg = heatmap_svg(report)
    (output / "REPORT.md").write_text(md, encoding="utf-8")
    (output / "brier-gain.svg").write_text(svg, encoding="utf-8")
    (output / "REPORT.html").write_text(html_report(report, md, svg), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["N", "H", "status", "matured_dates", "cv_dates", "cv_rows", "stock_prior_brier", "market_brier", "crowd_brier", "combined_brier", "combined_vs_market_gain"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for combo in combinations:
            writer.writerow(dict(N=combo["window_trading_days"], H=combo["horizon_trading_days"], status=combo["status"],
                matured_dates=combo["matured_feature_dates"], cv_dates=combo["cv_entry_dates"], cv_rows=combo["cv_rows"],
                stock_prior_brier=combo["models"]["stock_prior"]["macro_stock_brier"],
                market_brier=combo["models"]["market_only"]["macro_stock_brier"],
                crowd_brier=combo["models"]["crowd_only"]["macro_stock_brier"],
                combined_brier=combo["models"]["combined"]["macro_stock_brier"],
                combined_vs_market_gain=combo["comparisons"]["combined_vs_market_only"]["brier_gain"]))
    print(json.dumps({"output_dir": str(output.resolve()), "cv_predictions": len(all_predictions),
                      "status": report["status"]}), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
