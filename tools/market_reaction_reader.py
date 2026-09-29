"""Two-stage, retrospective test of a price-shock-supervised title reader.

Stage 1 learns a *contemporaneous reaction* score from titles published after a
session's 15:00 close and before the next session's 09:30 open. Its weak target
is the just-finished session's adjusted close-to-close sign, weighted by move
magnitude. No post-close title can be used to forecast that already-known move.

Stage 2 freezes that reader and asks whether its preceding five reaction-day
scores add to a price/volume/same-backup-count baseline for future H10 outcomes.
All historical posts here were collected retrospectively; a chronological
split is a development diagnostic, never live or fully forward evidence.

The ``preflight`` command reads source metadata only. ``run`` is deliberately
separate and can be expensive; it never downloads a model or edits reports.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, time as wall_time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import market_signal_v1 as historical
from tools import market_text_encoder as base

PROJECT = REPO.parent
MARKET = PROJECT / "pipeline-runs/market-data-backfill-20260928"
COLLECTOR = PROJECT / "MyResearcher-DataCollector/runtime/backups/collector-20260928T023304Z.db"
STAGE1_FIT = ("2026-05-07", "2026-06-23")
STAGE1_GATE = ("2026-06-24", "2026-07-20")
STAGE1_LATE = ("2026-08-03", "2026-09-11")
STAGE2_FIT = ("2026-07-01", "2026-07-20")
STAGE2_DIAG = ("2026-08-03", "2026-09-11")
HORIZON = 10
BARRIER = 0.05
MAGNITUDE_CAP = 0.07
SEEDS = (20260929, 20260930, 20261001)
EPOCHS = 25
READER_LR = 0.003
READER_L2 = 0.01
BASELINE_C = 0.1
RESIDUAL_L2 = 0.5
GATE_MIN_BRIER_GAIN = 0.005
EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"


def read_bars(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    bars: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["instrument_code"]), str(row["trade_date"]))
            if key in bars:
                raise ValueError(f"duplicate market bar: {key}")
            bars[key] = {name: float(row[name]) for name in
                         ("open", "high", "low", "close", "volume", "amount",
                          "pct_change", "turnover_rate")}
    if not bars:
        raise ValueError(f"empty market bars: {path}")
    return bars


def load_market(market_root: Path) -> tuple[list[str], list[str], dict, dict, dict]:
    qfq = read_bars(market_root / "daily-bars-qfq.csv")
    raw = read_bars(market_root / "daily-bars-raw.csv")
    benchmark = read_bars(market_root / "benchmark-000300.csv")
    calendar = sorted({day for code, day in benchmark if code == "000300"})
    codes = sorted({code for code, _ in qfq})
    if len(codes) != 16 or not calendar:
        raise ValueError(f"expected 16 stocks and a benchmark calendar, got {len(codes)}")
    for code in codes:
        for day in calendar:
            if (code, day) not in qfq or (code, day) not in raw:
                raise ValueError(f"incomplete stock calendar: {code} {day}")
    return calendar, codes, qfq, raw, benchmark


def session_range(calendar: Sequence[str], bounds: tuple[str, str]) -> list[str]:
    dates = [day for day in calendar if bounds[0] <= day <= bounds[1]]
    if not dates or dates[0] != bounds[0] or dates[-1] != bounds[1]:
        raise ValueError(f"fixed date range missing from calendar: {bounds}")
    return dates


def preceding_units(code: str, entry: str, calendar: Sequence[str], n: int = 5) -> list[tuple[str, str]]:
    position = {day: index for index, day in enumerate(calendar)}
    index = position[entry]
    if index < n:
        raise ValueError("insufficient pre-entry sessions")
    return [(code, day) for day in calendar[index - n:index]]


def needed_units(calendar: Sequence[str], codes: Sequence[str]) -> set[tuple[str, str]]:
    needed = {(code, day) for code in codes for bounds in
              (STAGE1_FIT, STAGE1_GATE, STAGE1_LATE) for day in session_range(calendar, bounds)}
    for bounds in (STAGE2_FIT, STAGE2_DIAG):
        for day in session_range(calendar, bounds):
            for code in codes:
                # The reader uses five days, while the count baseline also
                # needs the preceding five days for a change control.
                needed.update(preceding_units(code, day, calendar, n=10))
    return needed


def reaction_day(published: datetime, calendar: Sequence[str]) -> str | None:
    """Map only [trade-day 15:00, next-session 09:30) into that trade day."""
    local = published.astimezone(historical.MARKET_TZ)
    day = local.date().isoformat()
    index = bisect.bisect_left(calendar, day)
    if index < len(calendar) and calendar[index] == day:
        if local.time() >= wall_time(15, 0):
            return day if index + 1 < len(calendar) else None
        if local.time() < wall_time(9, 30) and index > 0:
            return calendar[index - 1]
        return None
    if index == 0 or index >= len(calendar):
        return None
    # A weekend/holiday belongs to the preceding session's overnight window.
    return calendar[index - 1]


def load_title_bags(db_path: Path, calendar: Sequence[str],
                    units: set[tuple[str, str]]) -> tuple[dict[tuple[str, str], list[str]], dict]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if list(calendar) != sorted(set(calendar)):
        raise ValueError("invalid benchmark calendar")
    codes = {code for code, _ in units}
    bags: dict[tuple[str, str], set[str]] = {unit: set() for unit in units}
    raw_counts: dict[tuple[str, str], int] = {unit: 0 for unit in units}
    bad_time = 0
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for code, title, published_at in connection.execute(
                "SELECT stock_code,title,published_at FROM posts"):
            code = str(code or "")
            if code not in codes:
                continue
            title = historical._normalize_text(title)
            if not title:
                continue
            try:
                published = historical._parse_datetime(published_at)
            except (TypeError, ValueError):
                bad_time += 1
                continue
            day = reaction_day(published, calendar)
            key = (code, day) if day is not None else None
            if key in bags:
                raw_counts[key] += 1
                bags[key].add(title)
    ordered = {key: sorted(value) for key, value in sorted(bags.items())}
    summary = {
        "raw_posts": sum(raw_counts.values()),
        "unique_title_occurrences": sum(map(len, ordered.values())),
        "unique_title_texts": len({title for bag in ordered.values() for title in bag}),
        "empty_units": sum(not bag for bag in ordered.values()),
        "bad_publication_times": bad_time,
        "raw_counts": raw_counts,
        "unique_counts": {key: len(value) for key, value in ordered.items()},
    }
    return ordered, summary


def reaction_label(code: str, day: str, calendar: Sequence[str], qfq: Mapping) -> tuple[int | None, float]:
    index = calendar.index(day)
    if index == 0:
        raise ValueError("a previous close is needed for reaction supervision")
    before = qfq[(code, calendar[index - 1])]["close"]
    after = qfq[(code, day)]["close"]
    move = after / before - 1.0
    if not math.isfinite(move):
        raise ValueError("non-finite same-day market move")
    label = 1 if move > 0 else 0 if move < 0 else None
    return label, min(abs(move) / MAGNITUDE_CAP, 1.0)


def future_labels(code: str, entry: str, calendar: Sequence[str], qfq: Mapping) -> dict:
    index = calendar.index(entry)
    if index + HORIZON > len(calendar):
        raise ValueError("H10 outcome not mature")
    opened = qfq[(code, entry)]["open"]
    if opened <= 0:
        raise ValueError("invalid entry open")
    up = down = None
    horizon = calendar[index:index + HORIZON]
    for day in horizon:
        change = qfq[(code, day)]["close"] / opened - 1.0
        if up is None and change >= BARRIER:
            up = day
        if down is None and change <= -BARRIER:
            down = day
    return {
        "barrier_up_first": int(up is not None and (down is None or up < down)),
        "h10_positive": int(qfq[(code, horizon[-1])]["close"] > opened),
        "outcome_date": horizon[-1],
    }


def build_stage2_rows(calendar: Sequence[str], codes: Sequence[str], qfq: Mapping,
                      bounds: tuple[str, str]) -> list[dict]:
    rows = []
    for day in session_range(calendar, bounds):
        for code in codes:
            rows.append({"code": code, "entry": day, "prior_units": preceding_units(code, day, calendar),
                         **future_labels(code, day, calendar, qfq)})
    return rows


def partition_summary(calendar: Sequence[str], codes: Sequence[str], qfq: Mapping,
                      bags: Mapping, counts: Mapping) -> dict:
    def stage1(bounds: tuple[str, str]) -> dict:
        days = session_range(calendar, bounds)
        units = [(code, day) for day in days for code in codes]
        nonempty = [key for key in units if bags[key]]
        effective = [key for key in nonempty if reaction_label(*key, calendar, qfq)[0] is not None]
        return {"dates": len(days), "units": len(units), "nonempty_units": len(nonempty),
                "effective_signed_units": len(effective),
                "unique_title_occurrences": sum(len(bags[key]) for key in units),
                "raw_posts": sum(counts["raw_counts"][key] for key in units)}
    fit = build_stage2_rows(calendar, codes, qfq, STAGE2_FIT)
    diag = build_stage2_rows(calendar, codes, qfq, STAGE2_DIAG)
    def stage2(rows: Sequence[dict]) -> dict:
        return {"rows": len(rows), "complete_5_bag_rows": sum(all(bags[key] for key in row["prior_units"])
                                                      for row in rows),
                "barrier_positive": sum(row["barrier_up_first"] for row in rows),
                "h10_positive": sum(row["h10_positive"] for row in rows),
                "outcome_last": max(row["outcome_date"] for row in rows)}
    if max(row["outcome_date"] for row in fit) >= min(row["entry"] for row in diag):
        raise ValueError("stage-2 training outcomes overlap diagnostic entries")
    if max(session_range(calendar, STAGE1_FIT)) >= min(session_range(calendar, STAGE1_GATE)):
        raise ValueError("stage-1 fit and gate dates overlap")
    return {"evidence": EVIDENCE, "stage1_fit": stage1(STAGE1_FIT),
            "stage1_gate": stage1(STAGE1_GATE), "stage1_late": stage1(STAGE1_LATE),
            "stage2_fit": stage2(fit), "stage2_diagnostic": stage2(diag),
            "selected_bags": len(bags), "selected_unique_title_texts": counts["unique_title_texts"],
            "selected_raw_posts": counts["raw_posts"],
            "selected_unique_title_occurrences": counts["unique_title_occurrences"],
            "empty_selected_bags": counts["empty_units"],
            "bad_publication_times": counts["bad_publication_times"]}


def corpus_digest(bags: Mapping[tuple[str, str], Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for key, titles in sorted(bags.items()):
        digest.update(json.dumps([*key, list(titles)], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def encode_or_load(bags: Mapping[tuple[str, str], Sequence[str]], cache: Path,
                   *, batch_size: int = 64) -> tuple[dict[str, int], np.ndarray]:
    """One frozen per-title cache shared by every seed and both stages.

    Source-bag and model identity checks prevent accidental stale-cache reuse.
    Float16 storage is converted to float32 for pooling/training.
    """
    titles = sorted({title for bag in bags.values() for title in bag})
    digest = corpus_digest(bags)
    model = str(base.BASE_MODEL.resolve())
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["corpus_digest"].item()) != digest or str(data["model"].item()) != model:
                raise ValueError("title cache source/model identity mismatch")
            vectors = data["vectors"].astype(np.float32)
            stored_hashes = data["title_hashes"].tolist()
        hashes = [hashlib.sha256(title.encode("utf-8")).hexdigest() for title in titles]
        if stored_hashes != hashes or len(vectors) != len(titles):
            raise ValueError("title cache title identity mismatch")
        return {title: index for index, title in enumerate(titles)}, vectors
    if not base.BASE_MODEL.is_dir():
        raise FileNotFoundError(f"local LERT snapshot is unavailable: {base.BASE_MODEL}")
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                                               use_fast=True)
    encoder = AutoModel.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(titles), batch_size):
            tokens = tokenizer(titles[start:start + batch_size], padding=True, truncation=True,
                               max_length=64, return_tensors="pt")
            parts.append(encoder(**tokens).last_hidden_state[:, 0, :].cpu().numpy())
    vectors = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("non-finite frozen LERT vectors")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vectors=vectors.astype(np.float16), corpus_digest=digest, model=model,
                        title_hashes=np.asarray([hashlib.sha256(t.encode("utf-8")).hexdigest()
                                                 for t in titles]))
    # First run and cache reuse must train on bit-identical input precision.
    return {title: index for index, title in enumerate(titles)}, vectors.astype(np.float16).astype(np.float32)


def stage1_examples(bounds: tuple[str, str], calendar: Sequence[str], codes: Sequence[str],
                    qfq: Mapping, bags: Mapping, title_ids: Mapping[str, int]) -> list[dict]:
    rows = []
    for day in session_range(calendar, bounds):
        for code in codes:
            label, weight = reaction_label(code, day, calendar, qfq)
            titles = bags[(code, day)]
            if not titles or label is None:
                continue
            rows.append({"code": code, "day": day, "label": label, "weight": weight,
                         "ids": [title_ids[title] for title in titles]})
    return rows


def permute_fit_labels(rows: Sequence[dict]) -> list[dict]:
    """Within-stock cyclic-shift placebo uses only Stage-1 fit labels."""
    by_code: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_code[row["code"]].append(index)
    shuffled = [dict(row) for row in rows]
    for indices in by_code.values():
        # A fixed seven-session shift preserves stock-specific label frequency.
        for j, index in enumerate(indices):
            source = rows[indices[(j + 7) % len(indices)]]
            shuffled[index]["label"] = source["label"]
            shuffled[index]["weight"] = source["weight"]
    return shuffled


def train_reader(rows: Sequence[dict], vectors: np.ndarray, seed: int) -> tuple[Any, dict]:
    """Frozen title vectors; gradients reach only a ~2d-parameter attention/head."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    x = torch.from_numpy(vectors)
    dimension = x.shape[1]
    class Reader(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Parameter(torch.zeros(dimension))
            self.head = torch.nn.Linear(dimension, 1)

        def forward(self, ids: Sequence[int]) -> Any:
            posts = x[ids]
            weights = torch.softmax(posts @ self.query / math.sqrt(dimension), dim=0)
            return self.head(weights @ posts).squeeze()

    reader = Reader()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=READER_LR, weight_decay=READER_L2)
    ordered = sorted(rows, key=lambda row: (row["day"], row["code"]))
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = torch.stack([reader(row["ids"]) for row in ordered])
        labels = torch.tensor([row["label"] for row in ordered], dtype=torch.float32)
        weights = torch.tensor([row["weight"] for row in ordered], dtype=torch.float32)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (weights * losses).sum() / weights.sum()
        loss.backward()
        optimizer.step()
    return reader, {"last_train_loss": float(loss.detach()), "parameters": sum(p.numel() for p in reader.parameters())}


def reader_probabilities(reader: Any, rows: Sequence[dict]) -> np.ndarray:
    import torch
    with torch.inference_mode():
        return np.asarray([float(torch.sigmoid(reader(row["ids"]))) for row in rows], dtype=np.float64)


def weighted_brier(rows: Sequence[dict], probs: Sequence[float]) -> float:
    labels = np.asarray([row["label"] for row in rows], dtype=float)
    weights = np.asarray([row["weight"] for row in rows], dtype=float)
    values = np.asarray(probs, dtype=float)
    if len(rows) != len(values) or not len(rows) or weights.sum() <= 0:
        raise ValueError("invalid Brier inputs")
    return float(np.average((labels - values) ** 2, weights=weights))


def stage1_gate(fit: Sequence[dict], gate: Sequence[dict], vectors: np.ndarray) -> tuple[bool, list, dict]:
    prior = float(np.average([row["label"] for row in fit], weights=[row["weight"] for row in fit]))
    prior_brier = weighted_brier(gate, [prior] * len(gate))
    results = []
    readers = []
    placebo_rows = permute_fit_labels(fit)
    for seed in SEEDS:
        reader, train_info = train_reader(fit, vectors, seed)
        placebo, _ = train_reader(placebo_rows, vectors, seed)
        true_brier = weighted_brier(gate, reader_probabilities(reader, gate))
        placebo_brier = weighted_brier(gate, reader_probabilities(placebo, gate))
        results.append({"seed": seed, "gate_weighted_brier": true_brier,
                        "placebo_gate_weighted_brier": placebo_brier, **train_info})
        readers.append(reader)
    margins = [min(prior_brier, item["placebo_gate_weighted_brier"]) - item["gate_weighted_brier"]
               for item in results]
    passed = float(np.median(margins)) >= GATE_MIN_BRIER_GAIN and sum(
        margin >= GATE_MIN_BRIER_GAIN for margin in margins) >= 2
    return passed, readers, {"passed": passed, "minimum_brier_gain": GATE_MIN_BRIER_GAIN,
                             "fit_weighted_up_prior": prior, "gate_prior_weighted_brier": prior_brier,
                             "seeds": results, "seed_margins": margins}


def price_count_features(row: Mapping, calendar: Sequence[str], qfq: Mapping, raw: Mapping,
                         benchmark: Mapping, counts: Mapping) -> list[float]:
    """All controls end before entry 09:30; title counts use the same backup."""
    code, entry = row["code"], row["entry"]
    index = calendar.index(entry)
    if index < 20:
        raise ValueError("20 known-price sessions required")
    prior = calendar[index - 20:index]
    recent = prior[-5:]
    previous = calendar[index - 10:index - 5]
    def returns(instrument: str, source: Mapping) -> list[float]:
        return [source[(instrument, day)]["pct_change"] / 100.0 for day in prior]
    stock_returns = returns(code, qfq)
    bench_returns = returns("000300", benchmark)
    previous_raw = raw[(code, prior[-1])]
    before_raw = [raw[(code, day)] for day in prior[-6:-1]]
    if any(item["volume"] <= 0 or item["amount"] <= 0 for item in before_raw):
        raise ValueError("invalid prior trading activity")
    def count_level(key: str, dates: Sequence[str]) -> float:
        return float(np.mean([math.log1p(counts[key][(code, day)]) for day in dates]))
    raw_recent = count_level("raw_counts", recent)
    raw_previous = count_level("raw_counts", previous)
    unique_recent = count_level("unique_counts", recent)
    unique_previous = count_level("unique_counts", previous)
    ranges = [(qfq[(code, day)]["high"] - qfq[(code, day)]["low"]) /
              qfq[(code, day)]["open"] for day in recent]
    result = [stock_returns[-1], float(np.prod(1 + np.asarray(stock_returns[-5:])) - 1),
              float(np.prod(1 + np.asarray(stock_returns)) - 1),
              bench_returns[-1], float(np.prod(1 + np.asarray(bench_returns[-5:])) - 1),
              float(np.prod(1 + np.asarray(bench_returns)) - 1),
              float(np.std(stock_returns)), float(np.mean(ranges)),
              previous_raw["turnover_rate"] / 100.0,
              math.log(previous_raw["volume"] / float(np.mean([x["volume"] for x in before_raw]))),
              math.log(previous_raw["amount"] / float(np.mean([x["amount"] for x in before_raw]))),
              raw_recent, raw_recent - raw_previous,
              unique_recent, unique_recent - unique_previous,
              sum(not counts["unique_counts"][key] for key in row["prior_units"]) / 5.0]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"non-finite known control: {code} {entry}")
    return result


def reader_trajectory(row: Mapping, scores: Mapping[tuple[str, str], float], counts: Mapping) -> list[float]:
    """Predeclared low-capacity extreme/recovery/cooling summary, no future bar."""
    units = row["prior_units"]
    observed = [(index, scores[key]) for index, key in enumerate(units) if key in scores]
    if not observed:
        return [0.0, 0.0, 0.0]
    min_index, minimum = min(observed, key=lambda item: item[1])
    last_index, last = observed[-1]
    rebound = last - minimum
    extreme_count = counts["unique_counts"][units[min_index]]
    latest_count = counts["unique_counts"][units[last_index]]
    count_cooling = max(0.0, math.log1p(extreme_count) - math.log1p(latest_count))
    return [minimum, rebound, max(0.0, -minimum) * max(0.0, rebound) * count_cooling]


def fit_stage2(fit: Sequence[dict], diag: Sequence[dict], market: Mapping,
               scores: Mapping, counts: Mapping) -> dict:
    """Two separate train-only baseline + frozen-reader residual comparisons."""
    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    calendar, codes, qfq, raw, benchmark = (market[key] for key in
                                             ("calendar", "codes", "qfq", "raw", "benchmark"))
    def matrix(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([price_count_features(row, calendar, qfq, raw, benchmark, counts)
                        for row in rows], dtype=float)
        onehot = np.asarray([[float(row["code"] == code) for code in codes] for row in rows])
        t = np.asarray([reader_trajectory(row, scores, counts) for row in rows], dtype=float)
        return np.column_stack((x, onehot)), t
    x_fit, t_fit = matrix(fit)
    x_diag, t_diag = matrix(diag)
    n_numeric = x_fit.shape[1] - len(codes)
    mean = x_fit[:, :n_numeric].mean(axis=0)
    scale = x_fit[:, :n_numeric].std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_fit[:, :n_numeric] = (x_fit[:, :n_numeric] - mean) / scale
    x_diag[:, :n_numeric] = (x_diag[:, :n_numeric] - mean) / scale
    t_mean, t_scale = t_fit.mean(axis=0), t_fit.std(axis=0)
    t_scale[t_scale < 1e-8] = 1.0
    t_fit, t_diag = (t_fit - t_mean) / t_scale, (t_diag - t_mean) / t_scale
    result = {}
    for outcome in ("barrier_up_first", "h10_positive"):
        y_fit = np.asarray([row[outcome] for row in fit])
        y_diag = np.asarray([row[outcome] for row in diag])
        baseline = LogisticRegression(C=BASELINE_C, max_iter=1000)
        baseline.fit(x_fit, y_fit)
        base_fit = baseline.decision_function(x_fit)
        base_diag = baseline.decision_function(x_diag)
        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            logit = base_fit + t_fit @ beta
            p = expit(logit)
            loss = float(np.mean(np.logaddexp(0.0, logit) - y_fit * logit) +
                         RESIDUAL_L2 * (beta @ beta) / 2)
            gradient = t_fit.T @ (p - y_fit) / len(y_fit) + RESIDUAL_L2 * beta
            return loss, gradient
        optimum = minimize(objective, np.zeros(t_fit.shape[1]), jac=True, method="L-BFGS-B")
        if not optimum.success:
            raise RuntimeError(f"residual optimization failed: {optimum.message}")
        base_prob, text_prob = expit(base_diag), expit(base_diag + t_diag @ optimum.x)
        complete = np.asarray([all(counts["unique_counts"][key] > 0 for key in row["prior_units"])
                               for row in diag])
        def metrics(mask: np.ndarray) -> dict:
            return {"n": int(mask.sum()), "baseline_brier": float(np.mean((base_prob[mask] - y_diag[mask]) ** 2)),
                    "text_brier": float(np.mean((text_prob[mask] - y_diag[mask]) ** 2)),
                    "baseline_auc": float(roc_auc_score(y_diag[mask], base_prob[mask])),
                    "text_auc": float(roc_auc_score(y_diag[mask], text_prob[mask]))}
        result[outcome] = {"all_rows": metrics(np.ones(len(diag), dtype=bool)),
                           "complete_5_bag_rows": metrics(complete),
                           "residual_coefficients": optimum.x.tolist(),
                           "diagnostic_predictions": [
                               {"code": row["code"], "entry": row["entry"], "outcome": int(label),
                                "complete_5_bags": bool(valid), "baseline": float(p0), "reader": float(p1)}
                               for row, label, valid, p0, p1 in
                               zip(diag, y_diag, complete, base_prob, text_prob)]}
    return result


def run(market_root: Path, db_path: Path, output: Path,
        title_cache: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("run output directory must be new or empty")
    calendar, codes, qfq, raw, benchmark = load_market(market_root)
    bags, counts = load_title_bags(db_path, calendar, needed_units(calendar, codes))
    summary = partition_summary(calendar, codes, qfq, bags, counts)
    cache_path = title_cache or output / "frozen-title-cache.npz"
    title_ids, vectors = encode_or_load(bags, cache_path)
    summary["frozen_title_cache"] = str(cache_path.resolve())
    fit = stage1_examples(STAGE1_FIT, calendar, codes, qfq, bags, title_ids)
    gate = stage1_examples(STAGE1_GATE, calendar, codes, qfq, bags, title_ids)
    passed, readers, gate_result = stage1_gate(fit, gate, vectors)
    summary["stage1_gate_result"] = gate_result
    late = stage1_examples(STAGE1_LATE, calendar, codes, qfq, bags, title_ids)
    score_rows = [{"unit": key, "ids": [title_ids[t] for t in bags[key]]}
                  for key in sorted(bags) if bags[key]]
    daily_scores = []
    scores_by_seed = {}
    summary["stage1_late_weighted_brier"] = []
    for seed, reader in zip(SEEDS, readers):
        values = reader_probabilities(reader, score_rows)
        scores = {row["unit"]: float(2 * probability - 1)
                  for row, probability in zip(score_rows, values)}
        scores_by_seed[seed] = scores
        daily_scores.extend({"seed": seed, "code": key[0], "reaction_day": key[1],
                             "reaction_score": value}
                            for key, value in sorted(scores.items()))
        late_probabilities = [(scores[(row["code"], row["day"])] + 1) / 2 for row in late]
        summary["stage1_late_weighted_brier"].append(
            {"seed": seed, "weighted_brier": weighted_brier(late, late_probabilities)})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "daily-reader-scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "code", "reaction_day", "reaction_score"))
        writer.writeheader()
        writer.writerows(daily_scores)
    if passed:
        stage2_fit = build_stage2_rows(calendar, codes, qfq, STAGE2_FIT)
        stage2_diag = build_stage2_rows(calendar, codes, qfq, STAGE2_DIAG)
        summary["stage2"] = []
        for seed in SEEDS:
            summary["stage2"].append({"seed": seed,
                "outcomes": fit_stage2(stage2_fit, stage2_diag,
                                       {"calendar": calendar, "codes": codes, "qfq": qfq,
                                        "raw": raw, "benchmark": benchmark}, scores_by_seed[seed], counts)})
        # Predeclared stop criterion: both outcomes must improve paired Brier
        # on all 480 dates and the 468 complete-bag dates in every seed.
        robust = all(
            metrics[partition]["text_brier"] < metrics[partition]["baseline_brier"]
            for seed_result in summary["stage2"]
            for metrics in seed_result["outcomes"].values()
            for partition in ("all_rows", "complete_5_bag_rows")
        )
        summary["stop_decision"] = ("RETROSPECTIVE_INCREMENT_PASS_NOT_FORWARD_EVIDENCE"
                                    if robust else "STOP_NO_ROBUST_INCREMENTAL_SIGNAL")
    else:
        summary["stage2"] = "NOT_RUN_STAGE1_GATE_FAILED"
        summary["stop_decision"] = "STOP_STAGE1_REACTION_GATE_FAILED"
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--market-root", type=Path, default=MARKET)
    parser.add_argument("--collector-db", type=Path, default=COLLECTOR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--title-cache", type=Path,
                        help="reuse a matching frozen per-title embedding cache; run output stays new")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.command == "preflight":
        calendar, codes, qfq, _, _ = load_market(args.market_root)
        bags, counts = load_title_bags(args.collector_db, calendar, needed_units(calendar, codes))
        result = partition_summary(calendar, codes, qfq, bags, counts)
    else:
        if args.output_dir is None:
            parser.error("run requires --output-dir")
        result = run(args.market_root, args.collector_db, args.output_dir, args.title_cache)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result if args.command == "preflight" else
                     {key: value for key, value in result.items() if key != "stage2"},
                     ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
