"""Causal market-layer replay of selective extreme/cooling opportunities.

Saved semantic states are historical, so this is retrospective research rather
than a frozen forward trading result. No semantic inference or downloads occur.
An entry-t state is already available before t; market features end at t-1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT / "pipeline-runs/action-contrast-shadow-20260929/source"
MARKET = PROJECT / "pipeline-runs/market-data-backfill-20260928"
SEEDS = (35, 71, 107)
MIN_RETURN = 0.001
REFERENCE = 20
SHRINK = 10.0
EPOCHS = 200
HIDDEN = 16
LR = 0.003
L2 = 0.01
MAX_SECONDS = 7200
MAX_BYTES = 10 * 1024**3
TITLE_COMPONENTS = 8
CALIBRATION_ALPHA = 10.0
CALIBRATION_MIN_DATES = 5
MARKET_NAMES = (
    "stock_return_n", "benchmark_return_n", "relative_return_n",
    "stock_return_1", "benchmark_return_1", "stock_volatility_n",
    "benchmark_volatility_n", "volume_recent5_previous5_log_ratio",
    "turnover_mean_n", "close_range_position_n",
)
CROWD_KEYS = ("stance_index", "action_index", "emotion_balance", "group_disagreement",
              "model_uncertainty", "group_concentration")
CROWD_NAMES = tuple(f"mean_n::{key}" for key in CROWD_KEYS) + (
    "log1p_mean_observed_posts_n", "stance_recent5_previous5_change",
    "emotion_recent5_previous5_change", "posts_recent5_previous5_log_ratio",
    "stance_current_minus_prior20_median", "emotion_current_minus_prior20_median",
)
VARIANTS = ("all_rows", "other_candidates", "unfiltered_candidates", "market_gate",
            "market_count_ensemble_gate",
            *(f"residual_gate_seed_{seed}" for seed in SEEDS), "residual_ensemble_gate",
            *(f"title_gate_seed_{seed}" for seed in SEEDS), "title_ensemble_gate",
            "market_context_calibrated_gate", "market_count_calibrated_gate",
            "residual_calibrated_gate", "title_calibrated_gate")


def read_title_vectors(path):
    """Read original frozen-encoder vectors, not previously fitted price heads."""
    with np.load(path, allow_pickle=False) as cache:
        units = cache["units"].tolist()
        vectors = cache["vectors"].astype(float)
    if vectors.ndim != 2 or len(units) != len(vectors) or not np.all(np.isfinite(vectors)):
        raise ValueError("invalid original daily title-vector cache")
    if len(set(units)) != len(units):
        raise ValueError("duplicate daily title-vector units")
    return {tuple(unit.split("|")): vector for unit, vector in zip(units, vectors)}


def attach_title_windows(panel, vectors, calendar, n):
    for row in panel:
        t = row["entry_index"]
        days = calendar[t - n + 1:t + 1]
        keys = [(row["stock_code"], day) for day in days]
        if any(key not in vectors for key in keys):
            if row["status"] != "COMPLETE":
                # Historical fit/evaluation never consumes pending outcome rows.
                row["title_history_available"] = False
                continue
            raise ValueError(f"missing title history for {row['stock_code']} {row['entry_date']}")
        row["title_history_available"] = True
        row["title_vector"] = np.mean([vectors[key] for key in keys], axis=0).tolist()
        row["title_window_start"] = days[0]
        row["title_window_end"] = days[-1]


def title_projection_fit(x, components=TITLE_COMPONENTS):
    mean = np.mean(x, axis=0)
    _, _, axes = np.linalg.svd(x - mean, full_matrices=False)
    axes = axes[:min(components, len(x) - 1)]
    return {"mean": mean, "axes": axes}


def title_project(projection, x):
    return (x - projection["mean"]) @ projection["axes"].T


def read_states(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = (row["stock_code"], row["entry_date"])
        if key in seen:
            raise ValueError(f"duplicate state {key}")
        seen.add(key)
        values = [row["features"][key] for key in CROWD_KEYS]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"nonfinite state {key}")
        if float(row["observed_posts"]) < 0:
            raise ValueError(f"negative observed posts {key}")
    return rows


def read_bars(path, *, benchmark=False):
    indexed = {}
    with Path(path).open(newline="") as handle:
        for source in csv.DictReader(handle):
            code = source["instrument_code"]
            key = source["trade_date"] if benchmark else (code, source["trade_date"])
            if key in indexed:
                raise ValueError(f"duplicate bar {key}")
            row = {**source}
            for field in ("open", "high", "low", "close", "volume", "turnover_rate"):
                row[field] = float(source[field])
                if not math.isfinite(row[field]):
                    raise ValueError(f"nonfinite bar {key} {field}")
            if min(row[field] for field in ("open", "high", "low", "close")) <= 0:
                raise ValueError(f"nonpositive price {key}")
            if row["volume"] < 0:
                raise ValueError(f"negative volume {key}")
            indexed[key] = row
    return indexed


def outcome(code, entry_index, horizon, calendar, qfq, raw):
    if entry_index + horizon > len(calendar):
        return {"status": "PENDING_HORIZON", "outcome_date": None, "return": None}
    days = calendar[entry_index:entry_index + horizon]
    entry = qfq.get((code, days[0]))
    if entry is None:
        return {"status": "MISSING_PRICE", "outcome_date": days[-1], "return": None}
    for day in days:
        bar, raw_bar = qfq.get((code, day)), raw.get((code, day))
        if bar is None or raw_bar is None or (code, days[0]) not in raw:
            return {"status": "MISSING_PRICE", "outcome_date": days[-1], "return": None}
        if entry.get("fetched_at") != bar.get("fetched_at"):
            adjusted = bar["close"] / entry["open"] - 1
            unadjusted = raw_bar["close"] / raw[(code, days[0])]["open"] - 1
            if abs(adjusted - unadjusted) > 0.001:
                return {"status": "ADJUSTMENT_VINTAGE_MISMATCH", "outcome_date": days[-1], "return": None}
    return {"status": "COMPLETE", "outcome_date": days[-1],
            "return": qfq[(code, days[-1])]["close"] / entry["open"] - 1}


def log_ratio(recent, previous):
    return math.log1p(recent) - math.log1p(previous)


def build_panel(states, calendar, qfq, raw, benchmark, n, horizon):
    """All feature/phase accesses are before entry; only outcomes access t onward.

    State windows include the entry-t keyed state, whose posts precede t.
    Prior20 reference excludes that state. State recent5 [t-4,t] is matched to
    market recent5 [t-5,t-1], never to entry-day price/volume.
    """
    by_stock = defaultdict(dict)
    for row in states:
        by_stock[row["stock_code"]][row["entry_date"]] = row
    panel, skipped = [], Counter()
    start = max(REFERENCE, n + 1)
    for code, series in sorted(by_stock.items()):
        for t in range(start, len(calendar)):
            day = calendar[t]
            if day not in series:
                skipped["missing_current_state"] += 1
                continue
            state_days = calendar[t - REFERENCE:t + 1]
            market_days = calendar[t - max(n + 1, 10):t]
            if any(date not in series for date in state_days):
                skipped["incomplete_state_history"] += 1
                continue
            if any((code, date) not in qfq or (code, date) not in raw or date not in benchmark
                   for date in market_days):
                skipped["incomplete_market_history"] += 1
                continue
            reference = [series[date] for date in calendar[t - REFERENCE:t]]
            recent = [series[date] for date in calendar[t - 4:t + 1]]
            previous = [series[date] for date in calendar[t - 9:t - 4]]
            current = series[day]
            stances = [row["features"]["stance_index"] for row in reference]
            emotions = [row["features"]["emotion_balance"] for row in reference]
            q10, q25 = np.quantile(stances, (0.10, 0.25))
            emotion_q25, emotion_q75 = np.quantile(emotions, (0.25, 0.75))
            stance_recent = float(np.mean([row["features"]["stance_index"] for row in recent]))
            stance_previous = float(np.mean([row["features"]["stance_index"] for row in previous]))
            posts_recent = sum(row["observed_posts"] for row in recent)
            posts_previous = sum(row["observed_posts"] for row in previous)
            volume_recent = sum(raw[(code, date)]["volume"] for date in calendar[t - 5:t])
            volume_previous = sum(raw[(code, date)]["volume"] for date in calendar[t - 10:t - 5])
            stance_reference_varies = max(stances) - min(stances) > 1e-12
            emotion_reference_varies = max(emotions) - min(emotions) > 1e-12
            extreme = stance_reference_varies and current["features"]["stance_index"] <= q10
            cooling = bool(stance_reference_varies and stance_previous <= q25 and stance_recent > stance_previous and
                           posts_recent < posts_previous and volume_recent < volume_previous)
            flat = emotion_reference_varies and emotion_q25 <= current["features"]["emotion_balance"] <= emotion_q75
            closes = np.array([qfq[(code, date)]["close"] for date in calendar[t - n - 1:t]])
            benchmark_closes = np.array([benchmark[date]["close"] for date in calendar[t - n - 1:t]])
            returns = closes[1:] / closes[:-1] - 1
            benchmark_returns = benchmark_closes[1:] / benchmark_closes[:-1] - 1
            return_n = float(closes[-1] / closes[0] - 1)
            benchmark_n = float(benchmark_closes[-1] / benchmark_closes[0] - 1)
            close_range = float(np.max(closes[1:]) - np.min(closes[1:]))
            range_position = float((closes[-1] - np.min(closes[1:])) / close_range) if close_range else 0.5
            market_features = [return_n, benchmark_n, return_n - benchmark_n,
                               float(returns[-1]), float(benchmark_returns[-1]),
                               float(np.std(returns)), float(np.std(benchmark_returns)),
                               log_ratio(volume_recent, volume_previous),
                               float(np.mean([raw[(code, date)]["turnover_rate"] for date in calendar[t - n:t]])),
                               range_position]
            crowd_window = [series[date] for date in calendar[t - n + 1:t + 1]]
            crowd_features = [float(np.mean([row["features"][key] for row in crowd_window])) for key in CROWD_KEYS]
            crowd_features += [math.log1p(float(np.mean([row["observed_posts"] for row in crowd_window]))),
                               stance_recent - stance_previous,
                               float(np.mean([row["features"]["emotion_balance"] for row in recent]) -
                                     np.mean([row["features"]["emotion_balance"] for row in previous])),
                               log_ratio(posts_recent, posts_previous),
                               float(current["features"]["stance_index"] - np.median(stances)),
                               float(current["features"]["emotion_balance"] - np.median(emotions))]
            result = outcome(code, t, horizon, calendar, qfq, raw)
            panel.append({"stock_code": code, "stock_name": current.get("stock_name", ""),
                          "entry_date": day, "entry_index": t, "n": n, "horizon": horizon,
                          "market_feature_last_date": calendar[t - 1],
                          "state_reference_start": state_days[0], "state_reference_end": calendar[t - 1],
                          "market_features": market_features, "crowd_features": crowd_features,
                          "extreme": bool(extreme), "cooling": cooling, "candidate": bool(extreme or cooling),
                          "emotion_flat": bool(flat), "stance_prior20_q10": float(q10),
                          "constant_stance_reference": not stance_reference_varies,
                          "constant_emotion_reference": not emotion_reference_varies,
                          "stance_prior20_q25": float(q25), "emotion_prior20_q25": float(emotion_q25),
                          "emotion_prior20_q75": float(emotion_q75),
                          "posts_recent5": posts_recent, "posts_previous5": posts_previous,
                          "volume_recent5": volume_recent, "volume_previous5": volume_previous,
                          **result})
    return panel, dict(skipped)


def make_folds(panel, min_fit_dates=15, test_block=5):
    mature = [row for row in panel if row["status"] == "COMPLETE"]
    dates = sorted({row["entry_date"] for row in mature})
    folds, skipped_dates = [], []
    cursor = 0
    while cursor < len(dates):
        first = dates[cursor]
        fit = [row for row in mature if row["outcome_date"] < first]
        if len({row["entry_date"] for row in fit}) < min_fit_dates:
            skipped_dates.append(first)
            cursor += 1
            continue
        test_dates = dates[cursor:cursor + test_block]
        test = [row for row in mature if row["entry_date"] in test_dates]
        folds.append({"fold": len(folds), "first_entry_date": first,
                      "test_dates": test_dates, "fit": fit, "test": test})
        cursor += len(test_dates)
    return folds, skipped_dates


def scaler_fit(x):
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    return mean, np.where(std > 1e-10, std, 1.0)


def fit_market_ridge(x, y, alpha=1.0):
    mean, std = scaler_fit(x)
    z = (x - mean) / std
    target_mean = float(np.mean(y))
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - target_mean))
    return {"mean": mean, "std": std, "beta": beta, "intercept": target_mean}


def market_predict(model, x):
    return (x - model["mean"]) / model["std"] @ model["beta"] + model["intercept"]


def fit_flat_reference(fit):
    flat = [row for row in fit if row["emotion_flat"]]
    if not flat:
        return {"global_mean": None, "global_count": 0, "stocks": {}}
    global_mean = float(np.mean([row["return"] for row in flat]))
    by_stock = defaultdict(list)
    for row in flat:
        by_stock[row["stock_code"]].append(row["return"])
    stocks = {code: {"count": len(values), "mean": float(np.mean(values)),
                     "shrunk_mean": (sum(values) + SHRINK * global_mean) / (len(values) + SHRINK)}
              for code, values in by_stock.items()}
    return {"global_mean": global_mean, "global_count": len(flat), "stocks": stocks}


def flat_reference_for(reference, code):
    stock = reference["stocks"].get(code, {})
    return stock.get("shrunk_mean") if stock.get("count", 0) >= 2 else None


def entry_gate(candidate, predicted_return, fit_flat_mean):
    return bool(candidate and fit_flat_mean is not None and
                predicted_return > MIN_RETURN and predicted_return > fit_flat_mean)


def check_budget(started, max_seconds):
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("fixed conditional replay time budget exceeded")
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        resident *= 1024
    if resident > MAX_BYTES:
        raise MemoryError("fixed conditional replay 10 GiB budget exceeded")


def fit_residual(torch, x, residual, seed, started, max_seconds):
    # This small model runs CPU, avoiding device-dependent checkpoint behavior.
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.Tanh(),
                                torch.nn.Linear(HIDDEN, 1))
    inputs = torch.tensor(x, dtype=torch.float32)
    targets = torch.tensor(residual, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2)
    for _ in range(EPOCHS):
        check_budget(started, max_seconds)
        optimizer.zero_grad()
        loss = torch.mean((model(inputs) - targets) ** 2)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite residual MLP loss")
        loss.backward()
        optimizer.step()
    model.eval()
    return model, float(loss.detach())


def evaluate_fold(fold, *, torch, output, n, horizon, started, max_seconds):
    fit, test = fold["fit"], fold["test"]
    x_fit = np.array([row["market_features"] for row in fit], dtype=float)
    x_test = np.array([row["market_features"] for row in test], dtype=float)
    y_fit = np.array([row["return"] for row in fit], dtype=float)
    market_model = fit_market_ridge(x_fit, y_fit)
    market_fit = market_predict(market_model, x_fit)
    market_test = market_predict(market_model, x_test)
    combined_fit = np.array([row["market_features"] + row["crowd_features"] for row in fit])
    combined_test = np.array([row["market_features"] + row["crowd_features"] for row in test])
    mean, std = scaler_fit(combined_fit)
    residual = y_fit - market_fit
    reference = fit_flat_reference(fit)
    # Identical capacity, optimizer, target and seeds separate semantic increment
    # from the effect of allowing a nonlinear price/observed-post mapping.
    count_indices = [len(MARKET_NAMES) + CROWD_NAMES.index(key) for key in (
        "log1p_mean_observed_posts_n", "posts_recent5_previous5_log_ratio")]
    count_columns = list(range(len(MARKET_NAMES))) + count_indices
    count_fit, count_test = combined_fit[:, count_columns], combined_test[:, count_columns]
    count_mean, count_std = scaler_fit(count_fit)
    predicted, count_predicted, title_predicted, logs = {}, {}, {}, []
    branches = [
            ("semantic", combined_fit, combined_test, mean, std, MARKET_NAMES + CROWD_NAMES, predicted),
            ("market-count", count_fit, count_test, count_mean, count_std,
             MARKET_NAMES + ("log1p_mean_observed_posts_n", "posts_recent5_previous5_log_ratio"), count_predicted)]
    projection = None
    if "title_vector" in fit[0]:
        raw_fit = np.array([row["title_vector"] for row in fit])
        raw_test = np.array([row["title_vector"] for row in test])
        projection = title_projection_fit(raw_fit)
        title_fit = np.column_stack((count_fit, title_project(projection, raw_fit)))
        title_test = np.column_stack((count_test, title_project(projection, raw_test)))
        title_mean, title_std = scaler_fit(title_fit)
        names = MARKET_NAMES + ("log1p_mean_observed_posts_n", "posts_recent5_previous5_log_ratio") + tuple(
            f"fit_only_title_pc_{i}" for i in range(len(projection["axes"])))
        branches.append(("all-title", title_fit, title_test, title_mean, title_std, names, title_predicted))
    for kind, fit_features, test_features, input_mean, input_std, feature_names, destination in branches:
        for seed in SEEDS:
            model, final_loss = fit_residual(torch, (fit_features - input_mean) / input_std, residual,
                                           seed, started, max_seconds)
            inputs = torch.tensor((test_features - input_mean) / input_std, dtype=torch.float32)
            with torch.no_grad():
                correction = model(inputs).flatten().numpy().astype(float)
            destination[seed] = market_test + correction
            checkpoint = output / f"n{n}-h{horizon}-fold{fold['fold']:02d}-{kind}-seed{seed}.pt"
            state = {"model_state": model.state_dict(), "input_mean": input_mean.tolist(),
                     "input_std": input_std.tolist(), "market_model": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                                  for key, value in market_model.items()},
                     "kind": kind, "seed": seed, "n": n, "horizon": horizon, "first_entry_date": fold["first_entry_date"],
                     "fit_flat_reference": reference, "market_feature_names": list(MARKET_NAMES),
                     "residual_feature_names": list(feature_names)}
            if kind == "all-title":
                state["title_projection"] = {key: value.tolist() for key, value in projection.items()}
            torch.save(state, checkpoint)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            reload = torch.nn.Sequential(torch.nn.Linear(fit_features.shape[1], HIDDEN), torch.nn.Tanh(),
                                         torch.nn.Linear(HIDDEN, 1))
            reload.load_state_dict(restored["model_state"])
            reload.eval()
            with torch.no_grad():
                restored_prediction = reload(inputs).flatten().numpy()
            reload_delta = float(np.max(np.abs(restored_prediction - correction)))
            if reload_delta > 1e-7:
                raise ValueError("CPU checkpoint reload prediction changed")
            logs.append({"kind": kind, "seed": seed, "final_fit_loss": final_loss,
                         "checkpoint": checkpoint.name, "cpu_reload_max_abs_delta": reload_delta})
    ensemble = np.mean([predicted[seed] for seed in SEEDS], axis=0)
    count_ensemble = np.mean([count_predicted[seed] for seed in SEEDS], axis=0)
    title_ensemble = np.mean([title_predicted[seed] for seed in SEEDS], axis=0) if title_predicted else None
    records = []
    for i, row in enumerate(test):
        quiet_mean = flat_reference_for(reference, row["stock_code"])
        seed_predictions = {str(seed): float(predicted[seed][i]) for seed in SEEDS}
        gates = {"all_rows": True, "other_candidates": not row["candidate"],
                 "unfiltered_candidates": row["candidate"],
                 "market_gate": entry_gate(row["candidate"], float(market_test[i]), quiet_mean),
                 "market_count_ensemble_gate": entry_gate(row["candidate"], float(count_ensemble[i]), quiet_mean),
                 **{f"residual_gate_seed_{seed}": entry_gate(row["candidate"], float(predicted[seed][i]), quiet_mean)
                    for seed in SEEDS},
                 "residual_ensemble_gate": entry_gate(row["candidate"], float(ensemble[i]), quiet_mean),
                 **{f"title_gate_seed_{seed}": entry_gate(row["candidate"], float(title_predicted[seed][i]), quiet_mean)
                    if title_predicted else False for seed in SEEDS},
                 "title_ensemble_gate": entry_gate(row["candidate"], float(title_ensemble[i]), quiet_mean)
                    if title_ensemble is not None else False}
        records.append({**{key: value for key, value in row.items() if key != "title_vector"},
                        "fold": fold["fold"], "fit_cutoff_exclusive": fold["first_entry_date"],
                        "fit_last_outcome_date": max(member["outcome_date"] for member in fit),
                        "fit_rows": len(fit), "fit_distinct_dates": len({member["entry_date"] for member in fit}),
                        "fit_flat_reference": quiet_mean,
                        "fit_same_stock_flat_count": reference["stocks"].get(row["stock_code"], {}).get("count", 0),
                        "predicted_market_return": float(market_test[i]),
                        "predicted_market_count_returns": {str(seed): float(count_predicted[seed][i]) for seed in SEEDS},
                        "predicted_market_count_ensemble_return": float(count_ensemble[i]),
                        "predicted_title_returns": {str(seed): float(title_predicted[seed][i]) for seed in SEEDS}
                            if title_predicted else {},
                        "predicted_title_ensemble_return": float(title_ensemble[i]) if title_ensemble is not None else None,
                        "predicted_residual_returns": seed_predictions,
                        "predicted_ensemble_return": float(ensemble[i]), "selections": gates})
    return records, {"fold": fold["fold"], "first_entry_date": fold["first_entry_date"],
                     "test_dates": fold["test_dates"], "fit_rows": len(fit), "test_rows": len(test),
                     "fit_distinct_dates": len({row["entry_date"] for row in fit}),
                     "fit_last_outcome_date": max(row["outcome_date"] for row in fit),
                     "fit_flat_reference": reference, "seeds": logs}


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def calibrate_out_of_sample(records):
    """Learn transfer/scale only from earlier forecasts whose outcomes matured.

    Never use in-sample scores. All test dates in a block share the calibration
    known before the first open. Warm-up has no calibrated selection.
    """
    logs = []
    keys = (("market_context", None),
            ("market_count", "predicted_market_count_ensemble_return"),
            ("residual", "predicted_ensemble_return"),
            ("title", "predicted_title_ensemble_return"))
    for fold_id in sorted({row["fold"] for row in records}):
        test = [row for row in records if row["fold"] == fold_id]
        first = min(row["entry_date"] for row in test)
        fit = [row for row in records if row["outcome_date"] < first]
        fit_dates = len({row["entry_date"] for row in fit})
        for kind, score in keys:
            eligible = [row for row in fit if score is None or row.get(score) is not None]
            enabled = fit_dates >= CALIBRATION_MIN_DATES and bool(eligible)
            model = None
            if enabled:
                x_fit = np.array([([row[score]] if score else []) + row["market_features"][:2]
                                  for row in eligible])
                y_fit = np.array([row["return"] for row in eligible])
                model = fit_market_ridge(x_fit, y_fit, CALIBRATION_ALPHA)
            for row in test:
                prediction = None
                if model is not None and (score is None or row.get(score) is not None):
                    inputs = np.array([([row[score]] if score else []) + row["market_features"][:2]])
                    prediction = float(market_predict(model, inputs)[0])
                row[f"predicted_{kind}_calibrated_return"] = prediction
                row["selections"][f"{kind}_calibrated_gate"] = entry_gate(
                    row["candidate"], prediction, row["fit_flat_reference"]) if prediction is not None else False
                row["calibration_fit_dates"] = fit_dates
            logs.append({"fold": fold_id, "kind": kind, "first_entry_date": first,
                         "fit_rows": len(eligible), "fit_distinct_dates": fit_dates,
                         "fit_last_outcome_date": max((row["outcome_date"] for row in eligible), default=None),
                         "status": "FITTED" if model is not None else "WARMUP_NO_SELECTION",
                         "model": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                   for key, value in model.items()} if model is not None else None})
    return logs


def nonoverlap_positions(rows, position, horizon):
    """First trigger opens a position; the next may open after its exit session."""
    selected, last_exit = [], -1
    for row in sorted(rows, key=lambda item: item["entry_date"]):
        entry = position[row["entry_date"]]
        if entry > last_exit:
            selected.append(row)
            last_exit = entry + horizon - 1
    return selected


def summarize(records, calendar):
    position = {day: i for i, day in enumerate(calendar)}
    by_stock = defaultdict(list)
    for row in records:
        by_stock[row["stock_code"]].append(row)
    per_stock = []
    for code, rows in sorted(by_stock.items()):
        quiet = [row for row in rows if row["emotion_flat"]]
        quiet_mean = mean_or_none([row["return"] for row in quiet])
        horizon = rows[0]["horizon"]
        quiet_positions = nonoverlap_positions(quiet, position, horizon)
        quiet_position_mean = mean_or_none([row["return"] for row in quiet_positions])
        groups = {}
        for variant in VARIANTS:
            variant_quiet = quiet
            if "calibrated" in variant:
                prediction_key = f"predicted_{variant.removesuffix('_gate')}_return"
                variant_quiet = [row for row in quiet if row.get(prediction_key) is not None]
            variant_quiet_mean = mean_or_none([row["return"] for row in variant_quiet])
            variant_quiet_positions = nonoverlap_positions(variant_quiet, position, horizon)
            variant_quiet_position_mean = mean_or_none([row["return"] for row in variant_quiet_positions])
            selected = [row for row in rows if row["selections"].get(variant, False)]
            indices = sorted(position[row["entry_date"]] for row in selected)
            episodes = int(bool(indices)) + sum(b != a + 1 for a, b in zip(indices, indices[1:]))
            actual = mean_or_none([row["return"] for row in selected])
            delta = actual - variant_quiet_mean if actual is not None and variant_quiet_mean is not None else None
            passes_absolute = actual > MIN_RETURN if actual is not None else None
            passes_flat = delta > 0 if delta is not None else None
            passes = bool(passes_absolute and passes_flat) if actual is not None and delta is not None else None
            positions = nonoverlap_positions(selected, position, horizon)
            position_actual = mean_or_none([row["return"] for row in positions])
            position_delta = (position_actual - variant_quiet_position_mean
                              if position_actual is not None and variant_quiet_position_mean is not None else None)
            position_passes_absolute = position_actual > MIN_RETURN if position_actual is not None else None
            position_passes_flat = position_delta > 0 if position_delta is not None else None
            position_passes = (bool(position_passes_absolute and position_passes_flat)
                               if position_actual is not None and position_delta is not None else None)
            groups[variant] = {"selected_count": len(selected), "episodes": episodes,
                               "mean_actual_return": actual, "difference_vs_heldout_flat": delta,
                               "passes_absolute": passes_absolute, "passes_flat": passes_flat,
                               "passes_both": passes,
                               "heldout_flat_count": len(variant_quiet), "heldout_flat_mean": variant_quiet_mean,
                               "selected_flat_overlap": sum(row["emotion_flat"] for row in selected),
                               "selected_dates": [row["entry_date"] for row in selected],
                               "primary_nonoverlap": {"position_count": len(positions),
                                   "entry_dates": [row["entry_date"] for row in positions],
                                   "mean_actual_return": position_actual,
                                   "heldout_flat_position_count": len(variant_quiet_positions),
                                   "heldout_flat_mean": variant_quiet_position_mean,
                                   "difference_vs_heldout_flat": position_delta,
                                   "passes_absolute": position_passes_absolute, "passes_flat": position_passes_flat,
                                   "passes_both": position_passes}}
        per_stock.append({"stock_code": code, "stock_name": rows[0]["stock_name"],
                          "evaluated_count": len(rows), "candidate_count": sum(row["candidate"] for row in rows),
                          "heldout_flat_count": len(quiet), "heldout_flat_mean": quiet_mean,
                          "primary_nonoverlap_flat": {"position_count": len(quiet_positions),
                              "entry_dates": [row["entry_date"] for row in quiet_positions],
                              "mean_actual_return": quiet_position_mean}, "variants": groups})
    aggregate = {}
    primary_aggregate = {}
    for variant in VARIANTS:
        groups = [stock["variants"][variant] for stock in per_stock]
        paired = [group for group in groups if group["difference_vs_heldout_flat"] is not None]
        aggregate[variant] = {"selected_count": sum(group["selected_count"] for group in groups),
                              "stocks_with_selection": sum(group["selected_count"] > 0 for group in groups),
                              "stocks_without_selection": sum(group["selected_count"] == 0 for group in groups),
                              "paired_stocks": len(paired), "passing_paired_stocks": sum(group["passes_both"] for group in paired),
                              "equal_stock_mean_return": mean_or_none([group["mean_actual_return"] for group in groups
                                                                        if group["mean_actual_return"] is not None]),
                              "equal_stock_paired_difference": mean_or_none([group["difference_vs_heldout_flat"] for group in paired])}
        primary = [group["primary_nonoverlap"] for group in groups]
        primary_paired = [group for group in primary if group["difference_vs_heldout_flat"] is not None]
        primary_aggregate[variant] = {"position_count": sum(group["position_count"] for group in primary),
                                     "stocks_with_selection": sum(group["position_count"] > 0 for group in primary),
                                     "stocks_without_selection": sum(group["position_count"] == 0 for group in primary),
                                     "paired_stocks": len(primary_paired),
                                     "passing_paired_stocks": sum(group["passes_both"] for group in primary_paired),
                                     "equal_stock_mean_return": mean_or_none([group["mean_actual_return"] for group in primary
                                                                             if group["mean_actual_return"] is not None]),
                                     "equal_stock_paired_difference": mean_or_none([group["difference_vs_heldout_flat"] for group in primary_paired])}
    return {"title_model_enabled": bool(records and records[0].get("predicted_title_ensemble_return") is not None),
            "evaluated_rows": len(records), "evaluated_dates": len({row["entry_date"] for row in records}),
            "candidate_rows": sum(row["candidate"] for row in records), "per_stock": per_stock,
            "secondary_equal_stock_aggregate": aggregate, "primary_nonoverlap_equal_stock_aggregate": primary_aggregate}


def pct(value):
    return "—" if value is None else f"{value * 100:+.3f}%"


def render_report(summary):
    lines = ["# 条件机会：历史时间顺序检验", "",
             "目标：所选入场日的平均实际收益大于0.1%，并高于同股、同一批评测日期中情绪平阶段的平均收益。10/20日为主，30日为辅助。", "",
             "入场t的已存状态来自t之前帖子；行情特征只到t前一交易日。极端为当前多空≤前20日Q10；冷却为前5日多空≤前20日Q25、近5日多空改善、完整观测帖量与入场前成交量均下降。候选是极端或冷却。情绪平代理为当前情绪平衡落入前20日Q25～Q75，表示相对常态，不等同于明确CALM。", "",
             "各折只拟合结果严格早于该折首个入场日的历史；参照收益仅用拟合期情绪平样本，至少2个同股已成熟情绪平样本才允许进入，同股均值以10个样本强度向全体情绪平均值收缩。进入条件同时要求预测收益>0.1%且>拟合期参照，可跳过整段。常量历史状态不算极端或情绪平机会。", "",
             "主结果逐股按最早触发入场，持有H交易日至收盘，在该仓位退出后才接受下一触发；情绪平对照独立采用同一无重叠规则。每日信号均值、相邻信号连续段为辅助统计；不同股票仍受共同市场影响。", "",
             "市场＋帖量对照采用完全相同容量、epoch和seed的小网络，仅输入已知行情及完整观测帖量，不输入语义概率；用它检验改善是否来自非线性量价，而非语义增量。", "",
             "本数据的语义概率在行情后生成，日期与结果此前已经研究过；本次属于回顾性条件假设检验。报告为开盘到第H交易日收盘的毛收益，未扣费用；没有入选的股票显示缺结果，不能算作0%成功。", ""]
    if summary["configuration"].get("title_vectors"):
        lines += ["另用同备份每天全部去重标题的官方冻结LERT原始日均向量，先取N日均值，再仅在每折拟合数据中做8维PCA；相同的市场＋帖量残差网络、种子和门槛。没有复用旧5%目标的PCA或价格权重。极端/冷却及情绪平定义仍固定来自七头状态，以保持机会定义不变；原标题网络不用七头概率作预测特征。", ""]
    if summary["configuration"].get("adaptive_calibration"):
        lines += ["延续实验：预测刻度校准只用此前已经到期的时间外预测，输入为原预测、已知个股N日收益及大盘N日收益，固定Ridge强度10。至少5个成熟预测日期才启用，之前不交易；每五日期块开始时拟合且块内不更新。另保留只看股价/大盘两输入的校准对照。该步骤学习预测可迁移的条件，不按当前或未来盈亏剔除坏案例。校准版的情绪平参照只取校准已经可用的同一批日期，原始版本与校准版本全部保留。这是看到首轮问题后的延续诊断，不是新盲测。", ""]
    for item in summary["experiments"]:
        lines += [f"## N={item['n']}，H={item['horizon']}（{'辅助' if item['horizon'] == 30 else '主观察'}）", "",
                  f"评测{item['evaluated_dates']}日期/{item['evaluated_rows']}股票日；候选{item['candidate_rows']}。模型为固定价格Ridge加市场/群体状态残差小网络，三个seed35/71/107全部展示；逐股表为固定三seed预测均值门槛。", "",
                  "| 股票 | 候选/评测 | 入选日/连续段/实际仓位 | 仓位实际均值 | 情绪平仓位数/均值 | 相对情绪平 | 两条件达到 |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for stock in item["per_stock"]:
            group = stock["variants"]["residual_ensemble_gate"]
            primary = group["primary_nonoverlap"]
            quiet = stock["primary_nonoverlap_flat"]
            result = "缺可比结果" if primary["passes_both"] is None else "是" if primary["passes_both"] else "否"
            lines.append(f"| {stock['stock_name']} {stock['stock_code']} | {stock['candidate_count']}/{stock['evaluated_count']} | {group['selected_count']}/{group['episodes']}/{primary['position_count']} | {pct(primary['mean_actual_return'])} | {quiet['position_count']}/{pct(quiet['mean_actual_return'])} | {pct(primary['difference_vs_heldout_flat'])} | {result} |")
        if item.get("title_model_enabled"):
            lines += ["", "原标题路径逐股结果（同一批机会与情绪平对照）：", "",
                      "| 股票 | 实际仓位 | 仓位实际均值 | 情绪平均值 | 相对情绪平 | 两条件达到 |",
                      "|---|---:|---:|---:|---:|---|"]
            for stock in item["per_stock"]:
                group = stock["variants"]["title_ensemble_gate"]["primary_nonoverlap"]
                quiet = stock["primary_nonoverlap_flat"]
                result = "缺可比结果" if group["passes_both"] is None else "是" if group["passes_both"] else "否"
                lines.append(f"| {stock['stock_name']} {stock['stock_code']} | {group['position_count']} | {pct(group['mean_actual_return'])} | {pct(quiet['mean_actual_return'])} | {pct(group['difference_vs_heldout_flat'])} | {result} |")
        if summary["configuration"].get("adaptive_calibration"):
            lines += ["", f"原标题校准版：{item['calibrated_evaluation_dates']}个有足够已成熟反馈的评测日期；不足时不入场。", "",
                      "| 股票 | 实际仓位 | 仓位实际均值 | 可用期情绪平仓位/均值 | 相对情绪平 | 两条件达到 |",
                      "|---|---:|---:|---:|---:|---|"]
            for stock in item["per_stock"]:
                group = stock["variants"]["title_calibrated_gate"]["primary_nonoverlap"]
                result = "缺可比结果" if group["passes_both"] is None else "是" if group["passes_both"] else "否"
                lines.append(f"| {stock['stock_name']} {stock['stock_code']} | {group['position_count']} | {pct(group['mean_actual_return'])} | {group['heldout_flat_position_count']}/{pct(group['heldout_flat_mean'])} | {pct(group['difference_vs_heldout_flat'])} | {result} |")
        lines += ["", "下表为无重叠仓位的辅助跨股汇总，均值只平均有入选的股票；无入选股票另列。各组不具有相同覆盖率。每日信号均值另存summary.json。", "",
                  "| 方法 | 实际仓位 | 有入选/无入选股 | 两条件达到/可比股 | 股票等权均值 | 等权相对情绪平 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant, data in item["primary_nonoverlap_equal_stock_aggregate"].items():
            if "calibrated" in variant and not summary["configuration"].get("adaptive_calibration"):
                continue
            if variant.startswith("title_") and not item.get("title_model_enabled"):
                continue
            lines.append(f"| {variant} | {data['position_count']} | {data['stocks_with_selection']}/{data['stocks_without_selection']} | {data['passing_paired_stocks']}/{data['paired_stocks']} | {pct(data['equal_stock_mean_return'])} | {pct(data['equal_stock_paired_difference'])} |")
        lines += ["", "口径对照：若把每个信号日都当成一次独立买入，会有重叠持仓；不能与上面的单股一仓位规则混称同一个收益。以下同时保留，防止仅展示较好的一种口径。", "",
                  "| 方法 | 每信号日假设：达标股/可比股 | 每信号日：股票等权均值 | 一仓位规则：股票等权均值 |",
                  "|---|---:|---:|---:|"]
        shown = ["market_count_ensemble_gate", "residual_ensemble_gate", "title_ensemble_gate"]
        if summary["configuration"].get("adaptive_calibration"):
            shown += ["market_count_calibrated_gate", "residual_calibrated_gate", "title_calibrated_gate"]
        for variant in shown:
            if variant.startswith("title_") and not item.get("title_model_enabled"):
                continue
            daily = item["secondary_equal_stock_aggregate"][variant]
            single = item["primary_nonoverlap_equal_stock_aggregate"][variant]
            lines.append(f"| {variant} | {daily['passing_paired_stocks']}/{daily['paired_stocks']} | {pct(daily['equal_stock_mean_return'])} | {pct(single['equal_stock_mean_return'])} |")
        lines += ["", "逐日预测、拟合截止、候选类别、预测参照与所有门槛在predictions.jsonl；逐股入选日期和各seed结果在summary.json。", ""]
    return "\n".join(lines)


def load_inputs(args):
    states = read_states(args.states)
    qfq, raw, benchmark = read_bars(args.qfq), read_bars(args.raw), read_bars(args.benchmark, benchmark=True)
    calendar = sorted(benchmark)
    if not calendar:
        raise ValueError("empty benchmark calendar")
    return states, calendar, qfq, raw, benchmark


def run(args):
    started = time.monotonic()
    if not 0 < args.max_seconds <= MAX_SECONDS:
        raise ValueError("max-seconds must be within the fixed two-hour limit")
    inputs = load_inputs(args)
    title_vectors = read_title_vectors(args.title_vectors) if args.title_vectors else None
    configuration = {"lookbacks": args.lookbacks, "horizons": args.horizons,
                     "min_fit_dates": args.min_fit_dates, "test_block": args.test_block,
                     "seeds": list(SEEDS), "reference_days": REFERENCE, "flat_shrink_count": SHRINK,
                     "minimum_return": MIN_RETURN, "ridge_alpha": 1.0, "residual_epochs": EPOCHS,
                     "hidden_units": HIDDEN, "learning_rate": LR, "weight_decay": L2,
                     "title_vectors": str(args.title_vectors.resolve()) if args.title_vectors else None,
                     "title_components_fit_only": TITLE_COMPONENTS if title_vectors is not None else None,
                     "adaptive_calibration": args.adaptive_calibration,
                     "calibration_alpha": CALIBRATION_ALPHA if args.adaptive_calibration else None,
                     "calibration_min_dates": CALIBRATION_MIN_DATES if args.adaptive_calibration else None,
                     "max_seconds": args.max_seconds, "max_bytes": MAX_BYTES}
    prepared, preflight = [], []
    for n in args.lookbacks:
        for horizon in args.horizons:
            check_budget(started, args.max_seconds)
            panel, skipped = build_panel(*inputs, n, horizon)
            if title_vectors is not None:
                attach_title_windows(panel, title_vectors, inputs[1], n)
            folds, skipped_dates = make_folds(panel, args.min_fit_dates, args.test_block)
            inventory = {"n": n, "horizon": horizon, "panel_rows": len(panel),
                         "status_counts": dict(Counter(row["status"] for row in panel)),
                         "candidate_rows": sum(row["candidate"] for row in panel),
                         "flat_rows": sum(row["emotion_flat"] for row in panel),
                         "folds": len(folds), "evaluation_rows": sum(len(fold["test"]) for fold in folds),
                         "evaluation_dates": sum(len(fold["test_dates"]) for fold in folds),
                         "skipped_history": skipped, "insufficient_fit_dates": skipped_dates}
            preflight.append(inventory)
            prepared.append((n, horizon, panel, folds, inventory))
    if args.preflight_only:
        result = {"configuration": configuration, "experiments": preflight,
                  "training_invoked": False, "calendar_last": inputs[1][-1]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --preflight-only")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    experiments, predictions = [], []
    with (output / "predictions.jsonl").open("w") as prediction_file:
        for n, horizon, panel, folds, inventory in prepared:
            records, fold_logs = [], []
            for fold in folds:
                check_budget(started, args.max_seconds)
                fold_records, log = evaluate_fold(fold, torch=torch, output=output, n=n, horizon=horizon,
                                                  started=started, max_seconds=args.max_seconds)
                records.extend(fold_records)
                fold_logs.append(log)
            calibration_logs = calibrate_out_of_sample(records) if args.adaptive_calibration else []
            for row in records:
                prediction_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            prediction_file.flush()
            experiments.append({**inventory, **summarize(records, inputs[1]), "fold_details": fold_logs,
                                "calibration_details": calibration_logs,
                                "calibrated_evaluation_dates": len({row["entry_date"] for row in records
                                    if row.get("predicted_market_context_calibrated_return") is not None}),
                                "pending_rows": sum(row["status"] == "PENDING_HORIZON" for row in panel)})
            predictions.extend(records)
            print(json.dumps({"n": n, "horizon": horizon, "folds": len(folds),
                              "evaluated_rows": len(records), "elapsed_seconds": time.monotonic() - started}), flush=True)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
               "sources": {key: str(Path(getattr(args, key)).resolve()) for key in ("states", "qfq", "raw", "benchmark")},
               "evidence_level": "RETROSPECTIVE_CONDITIONAL_HYPOTHESIS_ONLY", "device": "cpu",
               "configuration_selected_from_evaluation": False,
               "calendar_last": inputs[1][-1], "experiments": experiments,
               "elapsed_seconds": time.monotonic() - started}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(render_report(summary))
    print(json.dumps({"output_dir": str(output), "predictions": len(predictions),
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--states", type=Path, default=SOURCE / "daily-states.jsonl")
    result.add_argument("--qfq", type=Path, default=MARKET / "daily-bars-qfq.csv")
    result.add_argument("--raw", type=Path, default=MARKET / "daily-bars-raw.csv")
    result.add_argument("--benchmark", type=Path, default=MARKET / "benchmark-000300.csv")
    result.add_argument("--title-vectors", type=Path, help="Optional original frozen all-title daily vectors; PCA refit in each fold")
    result.add_argument("--adaptive-calibration", action="store_true", help="Fit score scale from already-matured previous out-of-sample forecasts")
    result.add_argument("--lookbacks", nargs="+", type=int, choices=(5, 20), default=[5, 20])
    result.add_argument("--horizons", nargs="+", type=int, choices=(10, 20, 30), default=[10, 20, 30])
    result.add_argument("--min-fit-dates", type=int, default=15)
    result.add_argument("--test-block", type=int, default=5)
    result.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.min_fit_dates < 1 or arguments.test_block < 1:
        raise ValueError("fit dates and test block must be positive")
    run(arguments)
