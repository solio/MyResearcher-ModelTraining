"""One fixed, retrospective market-supervised post-weight experiment.

This is NOT a point-in-time backtest: the May--September post snapshot and its
A0 semantic probabilities were assembled after many of the entry dates.  The
script never promotes its retrospective diagnostic to a trading signal.

Run with .encoder-venv/bin/python tools/market_text_gradient.py --output-dir ...
The only learned text parameters are post-attention weights and the mapping
from an anchored BEAR-minus-BULL score to a fixed-horizon return.  The seven-
head semantic model is frozen; this is not end-to-end language fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import market_signal_multiday as multiday
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as validation


PROJECT = Path(__file__).resolve().parents[2]
HISTORY = PROJECT / "pipeline-runs/market-extreme-unenriched-20260928/source"
MARKET = PROJECT / "pipeline-runs/market-data-backfill-20260928"
EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"
MARKET_KEYS = (
    "lag1_stock_return", "trailing5_stock_return", "lag1_benchmark_return",
    "trailing5_benchmark_return", "previous_turnover_rate",
    "previous_volume_ratio5", "previous_amount_ratio5",
)
POST_KEYS = (
    "bear", "bull", "mixed", "neutral", "unknown", "negative_emotion",
    "positive_emotion", "buy_add", "sell_reduce", "on_target",
    "log_text_length", "relative_age",
)
NEGATIVE = ("ANGER", "ANXIETY", "FEAR", "FRUSTRATION", "REGRET")
POSITIVE = ("EXCITEMENT", "HOPE", "FOMO")
MODES = ("market_count", "equal_semantic", "learned_post_weight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prob(pred: Mapping[str, Any], head: str, label: str) -> float:
    value = float(pred["predictions"][head]["probabilities"][label])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid probability {head}:{label}")
    return value


def post_representation(meta: Mapping[str, Any], pred: Mapping[str, Any], age: int, window: int) -> tuple[list[float], float]:
    if meta["sample_id"] != pred["sample_id"]:
        raise ValueError("sample_id mismatch")
    values = [
        _prob(pred, "stance", label)
        for label in ("BEAR", "BULL", "MIXED", "NEUTRAL", "UNKNOWN")
    ]
    values += [
        sum(_prob(pred, "emotion_primary", label) for label in NEGATIVE),
        sum(_prob(pred, "emotion_primary", label) for label in POSITIVE),
        sum(_prob(pred, "action_tendency", label) for label in ("BUY", "ADD")),
        sum(_prob(pred, "action_tendency", label) for label in ("SELL", "REDUCE")),
        _prob(pred, "target_mode", "ON_TARGET"),
        math.log1p(len(str(meta["model_text"]))) / 6.0,
        age / max(window - 1, 1),
    ]
    return values, values[0] - values[1]


@dataclass
class InputRows:
    rows: list[dict[str, Any]]
    dates: list[str]
    calendar: list[str]
    qfq: dict[tuple[str, str], dict[str, Any]]
    meta_count: int


def build_rows(metadata_path: Path, predictions_path: Path, daily_path: Path,
               qfq_path: Path, raw_path: Path, benchmark_path: Path,
               *, window: int, horizon: int) -> InputRows:
    metadata = validation.read_jsonl(metadata_path)
    predictions = validation.read_jsonl(predictions_path)
    if not metadata or len({row["sample_id"] for row in metadata}) != len(metadata):
        raise ValueError("metadata is empty or sample_id is duplicated")
    pred_by_id = {row["sample_id"]: row for row in predictions}
    if len(pred_by_id) != len(predictions) or set(pred_by_id) != {row["sample_id"] for row in metadata}:
        raise ValueError("predictions do not match metadata one-to-one")
    by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for meta in metadata:
        entry = str(meta["entry_date"])
        publication = datetime.fromisoformat(str(meta["published_at"]))
        if publication.date().isoformat() >= entry:
            raise ValueError(f"post is not strictly before entry: {meta['sample_id']}")
        by_unit[(str(meta["stock_code"]), entry)].append(meta)
    if max(len(items) for items in by_unit.values()) > 16:
        raise ValueError("historical frozen post cap exceeds 16")
    daily = validation.read_jsonl(daily_path)
    qfq = snapshot._read_stock_bars(qfq_path, adjustment="qfq")
    raw = snapshot._read_stock_bars(raw_path, adjustment="raw")
    benchmark = validation.read_csv(benchmark_path)
    calendar = sorted({str(row["trade_date"]) for row in benchmark})
    if not calendar:
        raise ValueError("no benchmark trading calendar")
    as_of = calendar[-1]
    panel = multiday.build_panel(daily, calendar, qfq, raw, benchmark, window=window,
                                 horizon=horizon, as_of=as_of, min_coverage=1.0,
                                 return_threshold=0.0)
    result: list[dict[str, Any]] = []
    for panel_row in panel:
        if panel_row["status"] != "COMPLETE":
            continue
        code, entry = str(panel_row["stock_code"]), str(panel_row["entry_date"])
        index = int(panel_row["entry_index"])
        interval = calendar[index - window + 1:index + 1]
        if not all((code, day) in by_unit for day in interval):
            continue
        market = panel_row["market_features"]
        if any(market.get(key) is None or not math.isfinite(float(market[key])) for key in MARKET_KEYS):
            continue
        posts = []
        for day_pos, day in enumerate(interval):
            for meta in sorted(by_unit[(code, day)], key=lambda row: (row["published_at"], row["sample_id"])):
                rep, polarity = post_representation(meta, pred_by_id[meta["sample_id"]],
                                                    window - 1 - day_pos, window)
                posts.append((meta["sample_id"], rep, polarity))
        if not posts:
            continue
        observed_count = sum(int(by_unit[(code, day)][0]["unit_total_post_count"]) for day in interval)
        market_vec = [float(market[key]) for key in MARKET_KEYS] + [math.log1p(observed_count)]
        result.append({
            "stock_code": code, "entry_date": entry, "outcome_date": str(panel_row["outcome_date"]),
            "return": float(panel_row["stock_return"]), "market": market_vec,
            "post_ids": [item[0] for item in posts], "post_x": [item[1] for item in posts],
            "polarity": [item[2] for item in posts], "sampled_posts": len(posts),
        })
    result.sort(key=lambda row: (row["entry_date"], row["stock_code"]))
    return InputRows(result, sorted({row["entry_date"] for row in result}), calendar, qfq, len(metadata))


def split_rows(rows: Sequence[Mapping[str, Any]], fit_end: str, diagnostic_start: str) -> tuple[list[dict], list[dict]]:
    if fit_end >= diagnostic_start:
        raise ValueError("fit_end must precede diagnostic_start")
    fit = [dict(row) for row in rows if row["outcome_date"] <= fit_end]
    diagnostic = [dict(row) for row in rows if row["entry_date"] >= diagnostic_start]
    if not fit or not diagnostic:
        raise ValueError("empty fit or diagnostic split")
    if max(row["outcome_date"] for row in fit) >= min(row["entry_date"] for row in diagnostic):
        raise ValueError("matured outcome/entry overlap between splits")
    return fit, diagnostic


def _arrays(rows: Sequence[Mapping[str, Any]], max_posts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    market = np.asarray([row["market"] for row in rows], dtype=np.float32)
    y = np.asarray([row["return"] for row in rows], dtype=np.float32)
    post = np.zeros((len(rows), max_posts, len(POST_KEYS)), dtype=np.float32)
    polarity = np.zeros((len(rows), max_posts), dtype=np.float32)
    mask = np.zeros((len(rows), max_posts), dtype=bool)
    for i, row in enumerate(rows):
        count = len(row["post_x"])
        if not 0 < count <= max_posts:
            raise ValueError("post bag exceeds fixed 5 x 16 capacity")
        post[i, :count] = row["post_x"]
        polarity[i, :count] = row["polarity"]
        mask[i, :count] = True
    return market, post, polarity, mask, y


def _scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def fit_models(fit: Sequence[Mapping[str, Any]], diagnostic: Sequence[Mapping[str, Any]],
               *, window: int, epochs: int, seed: int) -> tuple[dict, list[dict]]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    max_posts = window * 16
    m_fit, x_fit, p_fit, a_fit, y_fit = _arrays(fit, max_posts)
    m_diag, x_diag, p_diag, a_diag, y_diag = _arrays(diagnostic, max_posts)
    m_mean, m_std = _scaler(m_fit)
    y_mean, y_std = _scaler(y_fit)
    if float(y_std) < 1e-6:
        raise ValueError("no variation in fit returns")
    equal_fit = (p_fit * a_fit).sum(axis=1) / a_fit.sum(axis=1)
    s_mean, s_std = _scaler(equal_fit)
    if float(s_std) < 1e-6:
        raise ValueError("no variation in semantic scores")

    def tensors(m, x, p, a, y):
        return (torch.from_numpy((m - m_mean) / m_std), torch.from_numpy(x),
                torch.from_numpy(p), torch.from_numpy(a),
                torch.from_numpy((y - y_mean) / y_std))

    training = tensors(m_fit, x_fit, p_fit, a_fit, y_fit)
    evaluation = tensors(m_diag, x_diag, p_diag, a_diag, y_diag)
    checkpoint = {
        "evidence_level": EVIDENCE, "seed": seed, "epochs": epochs, "optimizer": "AdamW",
        "learning_rate": 0.03, "weight_decay": 0.05, "loss": "SmoothL1(beta=1) on train-standardized H10 return",
        "market_keys": [*MARKET_KEYS, "log1p_observed_posts_in_window"], "post_keys": list(POST_KEYS),
        "market_mean": m_mean.tolist(), "market_std": m_std.tolist(),
        "return_mean": float(y_mean), "return_std": float(y_std),
        "equal_semantic_mean": float(s_mean), "equal_semantic_std": float(s_std),
        "models": {},
    }
    all_predictions: dict[str, np.ndarray] = {}
    for mode in MODES:
        # A deliberately tiny nested model: market+count; add an equally
        # weighted anchored semantic score; then learn post attention.
        beta = torch.nn.Parameter(torch.zeros(training[0].shape[1], dtype=torch.float32))
        intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        gamma = torch.nn.Parameter(torch.zeros((), dtype=torch.float32)) if mode != "market_count" else None
        attention = torch.nn.Parameter(torch.zeros(len(POST_KEYS), dtype=torch.float32)) if mode == "learned_post_weight" else None
        parameters = [beta, intercept] + ([gamma] if gamma is not None else []) + ([attention] if attention is not None else [])
        optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0.05)

        def predict(batch):
            market, post, polarity, mask, _ = batch
            score = market @ beta + intercept
            if gamma is not None:
                if attention is None:
                    semantic = (polarity * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    logits = (post @ attention).masked_fill(~mask, -1e9)
                    weights = torch.softmax(logits, dim=1)
                    semantic = (polarity * weights).sum(dim=1)
                score = score + gamma * (semantic - float(s_mean)) / float(s_std)
            return score

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(predict(training), training[-1], beta=1.0)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            raw = predict(evaluation).numpy() * float(y_std) + float(y_mean)
            train_loss = float(torch.nn.functional.smooth_l1_loss(predict(training), training[-1], beta=1.0))
        if not np.isfinite(raw).all():
            raise ValueError(f"non-finite predictions in {mode}")
        all_predictions[mode] = raw
        checkpoint["models"][mode] = {
            "beta": beta.detach().numpy().tolist(), "intercept": float(intercept.detach()),
            "gamma": float(gamma.detach()) if gamma is not None else None,
            "attention": attention.detach().numpy().tolist() if attention is not None else None,
            "fit_loss": train_loss,
        }
    predictions = []
    for i, row in enumerate(diagnostic):
        predictions.append({
            "stock_code": row["stock_code"], "entry_date": row["entry_date"],
            "outcome_date": row["outcome_date"], "actual_horizon_return": row["return"],
            "sampled_posts": row["sampled_posts"], "evidence_level": EVIDENCE,
            **{f"predicted_return__{mode}": float(all_predictions[mode][i]) for mode in MODES},
        })
    return checkpoint, predictions


def predict_from_checkpoint(rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Framework-free replay of the saved trained weights on supplied bags."""
    if not rows:
        return {mode: np.asarray([], dtype=float) for mode in MODES}
    max_posts = max(len(row["post_x"]) for row in rows)
    market, post, polarity, mask, _ = _arrays(rows, max_posts)
    market = (market - np.asarray(checkpoint["market_mean"], dtype=np.float32)) / np.asarray(checkpoint["market_std"], dtype=np.float32)
    output = {}
    for mode in MODES:
        state = checkpoint["models"][mode]
        score = market @ np.asarray(state["beta"], dtype=np.float32) + np.float32(state["intercept"])
        if state["gamma"] is not None:
            if state["attention"] is None:
                semantic = (polarity * mask).sum(axis=1) / mask.sum(axis=1)
            else:
                logits = post @ np.asarray(state["attention"], dtype=np.float32)
                logits = np.where(mask, logits, -1e9)
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits) * mask
                weights /= weights.sum(axis=1, keepdims=True)
                semantic = (polarity * weights).sum(axis=1)
            score += np.float32(state["gamma"]) * (
                semantic - np.float32(checkpoint["equal_semantic_mean"])
            ) / np.float32(checkpoint["equal_semantic_std"])
        output[mode] = score * np.float32(checkpoint["return_std"]) + np.float32(checkpoint["return_mean"])
    return output


def regression_metrics(predictions: Sequence[Mapping[str, Any]], mode: str) -> dict[str, float]:
    actual = np.asarray([row["actual_horizon_return"] for row in predictions], dtype=float)
    predicted = np.asarray([row[f"predicted_return__{mode}"] for row in predictions], dtype=float)
    return {"mae": float(np.mean(np.abs(actual - predicted))),
            "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "pearson": float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 and np.std(predicted) > 0 else 0.0}


def per_stock_selection(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["stock_code"])].append(row)
    output = []
    for code, rows in sorted(grouped.items()):
        chosen = max(1, math.ceil(len(rows) / 4))
        result: dict[str, Any] = {"stock_code": code, "diagnostic_days": len(rows), "selected_days": chosen,
                                  "all_day_mean_return": statistics.fmean(float(r["actual_horizon_return"]) for r in rows)}
        for mode in MODES:
            top = sorted(rows, key=lambda row: (-float(row[f"predicted_return__{mode}"]), row["entry_date"]))[:chosen]
            result[f"top_quartile_mean_return__{mode}"] = statistics.fmean(float(r["actual_horizon_return"]) for r in top)
            result[f"top_quartile_positive_rate__{mode}"] = statistics.fmean(float(r["actual_horizon_return"]) > 0 for r in top)
        output.append(result)
    return output


def event_diagnostic(predictions: Sequence[Mapping[str, Any]], qfq: Mapping[tuple[str, str], Mapping[str, Any]],
                     calendar: Sequence[str], fit_end: str, *, prior_days: int = 5, top_k: int = 3) -> dict[str, Any]:
    by_key = {(str(row["stock_code"]), str(row["entry_date"])): row for row in predictions}
    codes = sorted({code for code, _ in qfq})
    fit_moves = []
    for code in codes:
        for i in range(1, len(calendar)):
            if calendar[i] > fit_end:
                break
            previous, current = qfq.get((code, calendar[i - 1])), qfq.get((code, calendar[i]))
            if previous and current and float(previous["close"]) > 0:
                fit_moves.append(abs(float(current["close"]) / float(previous["close"]) - 1.0))
    if not fit_moves:
        return {"events": [], "threshold_abs_daily_close_return": None}
    threshold = float(np.quantile(fit_moves, 0.90))
    events = []
    for code in codes:
        for i in range(prior_days, len(calendar)):
            day = calendar[i]
            if day <= fit_end:
                continue
            previous, current = qfq.get((code, calendar[i - 1])), qfq.get((code, day))
            if not previous or not current or float(previous["close"]) <= 0:
                continue
            move = float(current["close"]) / float(previous["close"]) - 1.0
            if abs(move) < threshold:
                continue
            candidates = [by_key.get((code, calendar[j])) for j in range(i - prior_days, i)]
            if any(row is None for row in candidates):
                continue
            oracle = sorted(candidates, key=lambda row: -float(row["actual_horizon_return"]))[:top_k]
            oracle_mean = statistics.fmean(float(row["actual_horizon_return"]) for row in oracle)
            event = {"stock_code": code, "shock_date": day, "shock_close_return": move,
                     "direction": "UP" if move > 0 else "DOWN", "oracle_top3_mean_return": oracle_mean}
            for mode in MODES:
                selected = sorted(candidates, key=lambda row: -float(row[f"predicted_return__{mode}"]))[:top_k]
                selected_mean = statistics.fmean(float(row["actual_horizon_return"]) for row in selected)
                event[f"selected_top3_mean_return__{mode}"] = selected_mean
                event[f"oracle_regret__{mode}"] = oracle_mean - selected_mean
            recent = candidates[-top_k:]
            event["recent3_mean_return"] = statistics.fmean(float(row["actual_horizon_return"]) for row in recent)
            events.append(event)
    return {"threshold_abs_daily_close_return": threshold, "threshold_source": "fit-period 90th percentile absolute close-to-close move",
            "event_conditioning": "RETROSPECTIVE_ONLY_N1_NOT_KNOWN_AT_CANDIDATE_ENTRY",
            "events": events}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    if args.window < 1 or args.horizon < 1 or args.epochs < 1 or args.event_top_k > args.window:
        raise ValueError("invalid window/horizon/epochs/event_top_k")
    paths = {"metadata": Path(args.metadata), "predictions": Path(args.predictions),
             "daily_states": Path(args.daily_states), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark)}
    inputs = build_rows(paths["metadata"], paths["predictions"], paths["daily_states"],
                        paths["qfq"], paths["raw"], paths["benchmark"],
                        window=args.window, horizon=args.horizon)
    fit, diagnostic = split_rows(inputs.rows, args.fit_end, args.diagnostic_start)
    checkpoint, predictions = fit_models(fit, diagnostic, window=args.window, epochs=args.epochs, seed=args.seed)
    replay = predict_from_checkpoint(diagnostic, checkpoint)
    for i, row in enumerate(predictions):
        for mode in MODES:
            if abs(row[f"predicted_return__{mode}"] - float(replay[mode][i])) > 1e-6:
                raise ValueError(f"saved checkpoint failed independent numpy replay: {mode} at {i}")
    by_stock = per_stock_selection(predictions)
    events = event_diagnostic(predictions, inputs.qfq, inputs.calendar, args.fit_end,
                              prior_days=args.event_prior_days, top_k=args.event_top_k)
    summary = {
        "evidence_level": EVIDENCE,
        "not_point_in_time_reasons": ["historical forum posts backfilled after their publication dates",
                                       "A0 seven-head probabilities inferred after historical entry dates",
                                       "diagnostic market period was previously inspected in this project"],
        "semantic_model": "SEM7-20260915-A0 frozen probabilities; no semantic encoder update",
        "target": f"qfq close of trading session H={args.horizon} divided by entry-session qfq open minus 1",
        "input_cutoff": "posts mapped to next actual entry session; market covariates end at previous session close",
        "configuration": {"window": args.window, "horizon": args.horizon, "fit_end": args.fit_end,
                          "diagnostic_start": args.diagnostic_start, "epochs": args.epochs, "seed": args.seed,
                          "event_prior_days": args.event_prior_days, "event_top_k": args.event_top_k},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
        "rows": {"metadata_posts": inputs.meta_count, "matured_panel_rows": len(inputs.rows),
                 "fit_rows": len(fit), "fit_entry_dates": len({r["entry_date"] for r in fit}),
                 "diagnostic_rows": len(diagnostic), "diagnostic_entry_dates": len({r["entry_date"] for r in diagnostic}),
                 "fit_last_outcome": max(r["outcome_date"] for r in fit),
                 "diagnostic_first_entry": min(r["entry_date"] for r in diagnostic)},
        "metrics": {mode: regression_metrics(predictions, mode) for mode in MODES},
        "per_stock": by_stock,
        "event_diagnostic": {key: value for key, value in events.items() if key != "events"},
        "event_count": len(events["events"]),
    }
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "checkpoint.json", checkpoint)
    _json(output / "summary.json", summary)
    _jsonl(output / "diagnostic-predictions.jsonl", predictions)
    _jsonl(output / "event-cases.jsonl", events["events"])
    _jsonl(output / "per-stock.jsonl", by_stock)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--predictions", type=Path, default=HISTORY / "m12-predictions.jsonl")
    parser.add_argument("--daily-states", type=Path, default=HISTORY / "daily-states.jsonl")
    parser.add_argument("--qfq", type=Path, default=MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=MARKET / "benchmark-000300.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fit-end", default="2026-07-31")
    parser.add_argument("--diagnostic-start", default="2026-08-03")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--event-prior-days", type=int, default=5)
    parser.add_argument("--event-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": summary["rows"],
                      "metrics": summary["metrics"], "event_count": summary["event_count"],
                      "evidence_level": EVIDENCE}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
