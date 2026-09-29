"""Bounded retrospective test of a raw-title residual for a two-sided market barrier.

The label is observed after buying at the entry session's adjusted open: a
close at least 5% above that open must occur before a close at least 5% below
it within ten actual sessions. No upper crossing means label zero. The frozen
LERT encoder supplies title embeddings; only a small residual head receives
market-label gradients. Historical Collector posts were backfilled after many
outcomes, so even the chronological diagnostic is not a forward result.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import resource
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_event_title_coverage as coverage
from tools import market_signal_v1 as historical
from tools import market_signal_v2_snapshot as snapshot
from tools import market_text_encoder as base

WINDOW = 20
HORIZON = 10
BARRIER = 0.05
FIT_OUTCOME_END = "2026-07-31"
DIAGNOSTIC_ENTRY_START = "2026-08-03"
SEEDS = (20260929, 20260930, 20261001)
MARKET_C = 0.1
PCA_COMPONENTS = 32
TEXT_L2 = 0.05
TEXT_LR = 0.01
TEXT_EPOCHS = 80
SHIFT_SESSIONS = 10
MAX_SECONDS = 7200.0
MAX_BYTES = 10 * 1024**3
EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"


def check_budget(started: float, max_seconds: float) -> None:
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("barrier-title run exceeded its fixed time budget")
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    used_bytes = usage if sys.platform == "darwin" else usage * 1024
    if used_bytes > MAX_BYTES:
        raise MemoryError("barrier-title run exceeded 10 GiB resident memory")


def first_barrier_label(code: str, entry_index: int, calendar: Sequence[str],
                        qfq: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Use only entry open and the ten entry-to-exit *closes*, in order."""
    if entry_index < 0 or entry_index + HORIZON > len(calendar):
        raise ValueError("barrier horizon is not mature")
    entry_date = calendar[entry_index]
    entry = qfq.get((code, entry_date))
    if entry is None:
        raise ValueError(f"missing entry bar: {code} {entry_date}")
    entry_open = float(entry["open"])
    if not math.isfinite(entry_open) or entry_open <= 0:
        raise ValueError("invalid entry open")
    first_up = first_down = None
    for day in calendar[entry_index:entry_index + HORIZON]:
        bar = qfq.get((code, day))
        if bar is None:
            raise ValueError(f"missing horizon bar: {code} {day}")
        close = float(bar["close"])
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"invalid horizon close: {code} {day}")
        change = close / entry_open - 1.0
        if first_up is None and change >= BARRIER:
            first_up = day
        if first_down is None and change <= -BARRIER:
            first_down = day
    if first_up is not None and (first_down is None or first_up < first_down):
        label, path = 1, "UP_FIRST"
    elif first_down is not None:
        label, path = 0, "DOWN_FIRST"
    else:
        label, path = 0, "NEITHER"
    return {"label": label, "barrier_path": path, "first_up_date": first_up,
            "first_down_date": first_down,
            "outcome_date": calendar[entry_index + HORIZON - 1]}


def labeled_partitions(metadata: Path, qfq_path: Path, raw: Path,
                       benchmark: Path) -> tuple[list[dict], list[dict], list[str]]:
    rows, _, calendar, qfq = base.build_text_rows(
        metadata, qfq_path, raw, benchmark, window=WINDOW, horizon=HORIZON)
    for row in rows:
        index = calendar.index(row["entry_date"])
        outcome = first_barrier_label(row["stock_code"], index, calendar, qfq)
        if outcome["outcome_date"] != row["outcome_date"]:
            raise ValueError("barrier and market panel have different outcome dates")
        row.update(outcome)
    fit = [row for row in rows if row["outcome_date"] <= FIT_OUTCOME_END]
    diagnostic = [row for row in rows if row["entry_date"] >= DIAGNOSTIC_ENTRY_START]
    if len(fit) != 528 or len(diagnostic) != 480:
        raise ValueError(f"fixed N20/H10 partition changed: {len(fit)}/{len(diagnostic)}")
    if max(row["outcome_date"] for row in fit) >= min(row["entry_date"] for row in diagnostic):
        raise ValueError("fit label crosses diagnostic entry")
    if {(row["stock_code"], row["entry_date"]) for row in fit} & {
            (row["stock_code"], row["entry_date"]) for row in diagnostic}:
        raise ValueError("fit and diagnostic stock-days overlap")
    return fit, diagnostic, calendar


def required_units(rows: Sequence[Mapping[str, Any]], calendar: Sequence[str]) -> set[tuple[str, str]]:
    position = {day: index for index, day in enumerate(calendar)}
    needed: set[tuple[str, str]] = set()
    for row in rows:
        index = position[str(row["entry_date"])]
        if index < WINDOW - 1:
            raise ValueError("missing 20-session title history")
        needed.update((str(row["stock_code"]), day)
                      for day in calendar[index - WINDOW + 1:index + 1])
    return needed


def load_all_title_bags(db_path: Path, calendar: Sequence[str],
                        needed: set[tuple[str, str]]) -> tuple[dict, dict]:
    """Read one Collector backup; map publication to the next *later* session.

    Deduplication is by normalized title text within each stock-day. No old
    sampled-ID subset assertion is made: a duplicated title may have a
    different first ID in the later backup.
    """
    if not db_path.is_file() or not needed:
        raise ValueError("missing Collector backup or required units")
    if len(set(calendar)) != len(calendar) or list(calendar) != sorted(calendar):
        raise ValueError("invalid trading calendar")
    relevant = {code for code, _ in needed}
    buckets: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    raw_counts: dict[tuple[str, str], int] = defaultdict(int)
    skipped_bad_time = 0
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for code, title, published_at in connection.execute(
                "SELECT stock_code,title,published_at FROM posts"):
            code = str(code or "")
            if code not in relevant:
                continue
            title = historical._normalize_text(title)
            if not title:
                continue
            try:
                publication = historical._parse_datetime(published_at)
            except (TypeError, ValueError):
                skipped_bad_time += 1
                continue
            day = publication.date().isoformat()
            index = bisect.bisect_right(calendar, day)
            if index >= len(calendar):
                continue
            entry = calendar[index]
            if day >= entry:
                raise ValueError("publication is not before mapped entry")
            unit = (code, entry)
            if unit not in needed:
                continue
            raw_counts[unit] += 1
            buckets[unit].setdefault(title, title)
    missing = sorted(needed - buckets.keys())
    if missing:
        raise ValueError(f"Collector backup lacks {len(missing)} required title units: {missing[:3]}")
    bags = {unit: sorted(buckets[unit]) for unit in sorted(needed)}
    counts = {"required_units": len(needed), "raw_posts": sum(raw_counts.values()),
              "unique_titles_in_units": sum(map(len, bags.values())),
              "unique_title_texts": len({title for bag in bags.values() for title in bag}),
              "max_titles_per_unit": max(map(len, bags.values())),
              "skipped_bad_publication_time": skipped_bad_time,
              "raw_counts": dict(raw_counts),
              "unique_counts": {unit: len(bag) for unit, bag in bags.items()}}
    return bags, counts


def add_backup_count_features(rows: Sequence[dict], calendar: Sequence[str],
                              counts: Mapping[str, Any]) -> None:
    position = {day: index for index, day in enumerate(calendar)}
    for row in rows:
        code, entry = row["stock_code"], row["entry_date"]
        index = position[entry]
        days = calendar[index - WINDOW + 1:index + 1]
        raw_count = sum(counts["raw_counts"][(code, day)] for day in days)
        unique_count = sum(counts["unique_counts"][(code, day)] for day in days)
        if min(raw_count, unique_count) <= 0:
            raise ValueError("empty backup post-count window")
        previous = days[-10:-5]
        recent = days[-5:]
        def timing(count_key: str) -> tuple[float, float]:
            source = counts[count_key]
            prior_level = float(np.mean([math.log1p(source[(code, day)]) for day in previous]))
            recent_level = float(np.mean([math.log1p(source[(code, day)]) for day in recent]))
            return recent_level, recent_level - prior_level
        raw_level, raw_change = timing("raw_counts")
        unique_level, unique_change = timing("unique_counts")
        # First seven fields are unchanged price/volume features. The old
        # metadata count in field eight is replaced by same-backup counts.
        # Both sides see the same recent-five level/change timing as text.
        row["market_same_backup"] = row["market"][:7] + [math.log1p(raw_count),
                                                           math.log1p(unique_count),
                                                           raw_level, raw_change,
                                                           unique_level, unique_change]
        row["backup_raw_posts_n20"] = raw_count
        row["backup_unique_titles_n20"] = unique_count


def add_known_price_path_features(rows: Sequence[dict], calendar: Sequence[str],
                                  qfq: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
    """Three fixed controls from sessions strictly before the entry open.

    BaoStock's preceding-session pct_change makes all 20 daily close returns
    available even for the earliest N20 row, where the bar before the local
    calendar begins. The last five high-low ranges are scaled by same-day
    adjusted open; no entry-day or later bar is read.
    """
    position = {day: index for index, day in enumerate(calendar)}
    for row in rows:
        code, index = str(row["stock_code"]), position[str(row["entry_date"])]
        if index < WINDOW:
            raise ValueError("missing 20-session known price path")
        prior = calendar[index - WINDOW:index]
        daily_returns = []
        for day in prior:
            bar = qfq.get((code, day))
            if bar is None:
                raise ValueError(f"missing prior price bar: {code} {day}")
            change = float(bar["pct_change"]) / 100.0
            if not math.isfinite(change) or change <= -1:
                raise ValueError(f"invalid prior daily return: {code} {day}")
            daily_returns.append(change)
        ranges = []
        for day in prior[-5:]:
            bar = qfq[(code, day)]
            high, low, opened = (float(bar[field]) for field in ("high", "low", "open"))
            if not all(math.isfinite(x) for x in (high, low, opened)) or opened <= 0 or high < low:
                raise ValueError(f"invalid prior high-low bar: {code} {day}")
            ranges.append((high - low) / opened)
        features = [float(np.prod(1.0 + np.asarray(daily_returns)) - 1.0),
                    float(np.std(daily_returns, ddof=0)), float(np.mean(ranges))]
        if not all(math.isfinite(x) for x in features):
            raise ValueError("non-finite known price-path feature")
        row["market_same_backup"].extend(features)


def embed_titles(titles: Sequence[str], *, batch_size: int, started: float,
                 max_seconds: float) -> tuple[dict[str, int], np.ndarray, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not titles or batch_size < 1:
        raise ValueError("empty title corpus or invalid batch size")
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    encoder = AutoModel.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                       trust_remote_code=False).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    mapping = {title: index for index, title in enumerate(titles)}
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(titles), batch_size):
            check_budget(started, max_seconds)
            batch = titles[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=64,
                               return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            encoded = encoder(**tokens).last_hidden_state[:, 0, :]
            pieces.append(encoded.cpu().numpy().copy())
            if (start + len(batch)) % 4096 < batch_size:
                print(f"frozen_titles={min(start + batch_size, len(titles))}/{len(titles)}", flush=True)
    matrix = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite frozen title embedding")
    return mapping, matrix, str(device)


def daily_mean_vectors(bags: Mapping[tuple[str, str], Sequence[str]],
                       title_index: Mapping[str, int], embeddings: np.ndarray) -> dict:
    means = {}
    for unit, titles in bags.items():
        means[unit] = embeddings[[title_index[title] for title in titles]].mean(axis=0)
    return means


def save_daily_mean_vectors(path: Path, daily: Mapping[tuple[str, str], np.ndarray]) -> None:
    units = sorted(daily)
    vectors = np.asarray([daily[unit] for unit in units], dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("cannot save non-finite daily title vectors")
    np.savez_compressed(path, units=np.asarray([f"{code}|{day}" for code, day in units]),
                        vectors=vectors)


def load_daily_mean_vectors(path: Path, needed: set[tuple[str, str]]) -> dict:
    with np.load(path, allow_pickle=False) as cache:
        keys = cache["units"].tolist()
        vectors = np.asarray(cache["vectors"], dtype=np.float32)
    units = [tuple(key.split("|", 1)) for key in keys]
    if (len(units) != len(set(units)) or set(units) != needed or
            vectors.ndim != 2 or vectors.shape[0] != len(units) or
            not np.isfinite(vectors).all()):
        raise ValueError("daily title-vector cache does not match required stock-days")
    return {unit: vector for unit, vector in zip(units, vectors)}


def trajectory_vectors(rows: Sequence[Mapping[str, Any]], calendar: Sequence[str],
                       daily: Mapping[tuple[str, str], np.ndarray]) -> np.ndarray:
    """Equal-weight mean across days; last-five level and recent-minus-prior."""
    position = {day: index for index, day in enumerate(calendar)}
    vectors = []
    for row in rows:
        code = str(row["stock_code"])
        index = position[str(row["entry_date"])]
        previous = np.stack([daily[(code, day)] for day in calendar[index - 9:index - 4]])
        recent = np.stack([daily[(code, day)] for day in calendar[index - 4:index + 1]])
        prev_mean, recent_mean = previous.mean(axis=0), recent.mean(axis=0)
        vectors.append(np.concatenate((recent_mean, recent_mean - prev_mean)))
    result = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite title trajectory")
    return result


def stock_shifted_vectors(rows: Sequence[Mapping[str, Any]], vectors: np.ndarray,
                          shift: int = SHIFT_SESSIONS) -> np.ndarray:
    """Fixed placebo: circularly offset diagnostic text within each stock."""
    shifted = np.empty_like(vectors)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["stock_code"])].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: str(rows[i]["entry_date"]))
        if len(indices) <= shift:
            raise ValueError("not enough diagnostic dates for fixed text shift")
        shifted[indices] = np.roll(vectors[indices], shift=shift, axis=0)
    return shifted


def market_matrices(fit: Sequence[Mapping[str, Any]],
                    diagnostic: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.preprocessing import StandardScaler

    stock_codes = sorted({str(row["stock_code"]) for row in fit})
    if stock_codes != sorted({str(row["stock_code"]) for row in diagnostic}):
        raise ValueError("stock universe changed between partitions")
    scaler = StandardScaler().fit([row["market_same_backup"] for row in fit])
    def make(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        numeric = scaler.transform([row["market_same_backup"] for row in rows])
        onehot = np.asarray([[float(row["stock_code"] == code) for code in stock_codes]
                             for row in rows], dtype=float)
        return np.concatenate((numeric, onehot), axis=1)
    return make(fit), make(diagnostic), {"stock_codes": stock_codes,
                                        "numeric_mean": scaler.mean_.tolist(),
                                        "numeric_scale": scaler.scale_.tolist()}


def text_matrices(fit_vectors: np.ndarray,
                  diagnostic_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(fit_vectors)
    train_scaled = scaler.transform(fit_vectors)
    diagnostic_scaled = scaler.transform(diagnostic_vectors)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=35)
    fit_pca = pca.fit_transform(train_scaled)
    diagnostic_pca = pca.transform(diagnostic_scaled)
    component_scale = np.maximum(fit_pca.std(axis=0), 1e-6)
    return (fit_pca / component_scale).astype(np.float32), (diagnostic_pca / component_scale).astype(np.float32), {
        "components": PCA_COMPONENTS,
        "fit_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "fitted_on_fit_only": True,
    }


def fit_text_residual(fit_text: np.ndarray, fit_y: np.ndarray,
                      fit_market_logit: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[dict]]:
    import torch

    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    x = torch.tensor(fit_text, dtype=torch.float32)
    y = torch.tensor(fit_y, dtype=torch.float32)
    offset = torch.tensor(fit_market_logit, dtype=torch.float32)
    beta = torch.nn.Parameter(torch.randn(x.shape[1], dtype=torch.float32) * 1e-4)
    optimizer = torch.optim.Adam([beta], lr=TEXT_LR)
    log = []
    for epoch in range(TEXT_EPOCHS):
        indices = generator.permutation(len(fit_y))
        for start in range(0, len(indices), 64):
            batch = indices[start:start + 64]
            optimizer.zero_grad()
            logits = offset[batch] + x[batch] @ beta
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
            loss = loss + TEXT_L2 * beta.square().sum()
            loss.backward()
            optimizer.step()
        if epoch in (0, TEXT_EPOCHS - 1):
            with torch.no_grad():
                full = torch.nn.functional.binary_cross_entropy_with_logits(offset + x @ beta, y)
                log.append({"epoch": epoch + 1, "fit_bce": float(full),
                            "coefficient_l2": float(torch.linalg.vector_norm(beta))})
    return beta.detach().numpy().copy(), log


def probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60, 60)))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("invalid probability metric inputs")
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    bins = {}
    for name, mask in (("predicted_nonpositive", p < 0.5),
                       ("predicted_positive", p >= 0.5)):
        bins[name] = {"count": int(mask.sum()),
                      "mean_probability": float(p[mask].mean()) if mask.any() else None,
                      "observed_positive_rate": float(y[mask].mean()) if mask.any() else None}
    return {"rows": len(y), "positive": int(y.sum()),
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "sign_calibration": bins}


def diagnostic_report(rows: Sequence[Mapping[str, Any]], baseline: np.ndarray,
                      constant: np.ndarray, text_predictions: Mapping[int, np.ndarray],
                      shifted_predictions: Mapping[int, np.ndarray]) -> tuple[dict, list[dict]]:
    y = np.asarray([row["label"] for row in rows], dtype=int)
    controls = {"fit_prior": probability_metrics(y, constant),
                "market_count": probability_metrics(y, baseline)}
    records = []
    by_seed = {}
    for seed in SEEDS:
        text_p, shift_p = text_predictions[seed], shifted_predictions[seed]
        text_metrics = probability_metrics(y, text_p)
        shift_metrics = probability_metrics(y, shift_p)
        by_seed[str(seed)] = {
            "text": text_metrics,
            "same_stock_shifted_text": shift_metrics,
            "text_minus_market_brier_gain": controls["market_count"]["brier"] - text_metrics["brier"],
            "text_minus_prior_brier_gain": controls["fit_prior"]["brier"] - text_metrics["brier"],
            "text_minus_shifted_brier_gain": shift_metrics["brier"] - text_metrics["brier"],
        }
    per_stock = []
    for code in sorted({row["stock_code"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["stock_code"] == code]
        actual = y[indices]
        market_brier = probability_metrics(actual, baseline[indices])["brier"]
        prior_brier = probability_metrics(actual, constant[indices])["brier"]
        seed_gain = {str(seed): float(market_brier - probability_metrics(
            actual, text_predictions[seed][indices])["brier"]) for seed in SEEDS}
        per_stock.append({"stock_code": code, "rows": len(indices),
                          "positive": int(actual.sum()), "market_count_brier": market_brier,
                          "fit_prior_brier": prior_brier,
                          "text_minus_market_brier_gain_by_seed": seed_gain,
                          "mean_text_minus_market_brier_gain": float(np.mean(list(seed_gain.values())))})
    for index, row in enumerate(rows):
        records.append({"stock_code": row["stock_code"], "entry_date": row["entry_date"],
                        "outcome_date": row["outcome_date"], "label": row["label"],
                        "barrier_path": row["barrier_path"],
                        "first_up_date": row["first_up_date"],
                        "first_down_date": row["first_down_date"],
                        "probability_fit_prior": float(constant[index]),
                        "probability_market_count": float(baseline[index]),
                        "probability_text_by_seed": {str(seed): float(text_predictions[seed][index]) for seed in SEEDS},
                        "probability_shifted_text_by_seed": {
                            str(seed): float(shifted_predictions[seed][index]) for seed in SEEDS}})
    summary = {"controls": controls, "by_seed": by_seed, "per_stock": per_stock,
               "mean_text_minus_market_brier_gain": float(np.mean([
                   by_seed[str(seed)]["text_minus_market_brier_gain"] for seed in SEEDS])),
               "mean_text_minus_prior_brier_gain": float(np.mean([
                   by_seed[str(seed)]["text_minus_prior_brier_gain"] for seed in SEEDS])),
               "mean_text_minus_shifted_brier_gain": float(np.mean([
                   by_seed[str(seed)]["text_minus_shifted_brier_gain"] for seed in SEEDS]))}
    return summary, records


def run(args: argparse.Namespace) -> dict:
    from sklearn.linear_model import LogisticRegression

    if not 0 < args.max_seconds <= MAX_SECONDS:
        raise ValueError("max-seconds must be within the fixed two-hour cap")
    if args.encoder_batch_size < 1 or args.encoder_batch_size > 128:
        raise ValueError("encoder batch size must be in [1, 128]")
    output = Path(args.output_dir).resolve() if args.output_dir else None
    if not args.preflight_only and output is None:
        raise ValueError("--output-dir is required for a full run")
    if output and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new empty output directory: {output}")
    started = time.monotonic()
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark),
             "collector_backup": Path(args.collector_backup)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for filename in ("pytorch_model.bin", "tokenizer.json"):
        if not (base.BASE_MODEL / filename).is_file():
            raise FileNotFoundError(base.BASE_MODEL / filename)
    fit, diagnostic, calendar = labeled_partitions(paths["metadata"], paths["qfq"],
                                                    paths["raw"], paths["benchmark"])
    needed = required_units([*fit, *diagnostic], calendar)
    bags, counts = load_all_title_bags(paths["collector_backup"], calendar, needed)
    add_backup_count_features([*fit, *diagnostic], calendar, counts)
    qfq = snapshot._read_stock_bars(paths["qfq"], adjustment="qfq")
    add_known_price_path_features([*fit, *diagnostic], calendar, qfq)
    outcome_counts = lambda rows: {name: sum(row["barrier_path"] == name for row in rows)
                                   for name in ("UP_FIRST", "DOWN_FIRST", "NEITHER")}
    preflight = {"evidence_level": EVIDENCE, "fit_rows": len(fit),
                 "diagnostic_rows": len(diagnostic),
                 "fit_barrier_paths": outcome_counts(fit),
                 "diagnostic_barrier_paths": outcome_counts(diagnostic),
                 "fit_latest_outcome": max(row["outcome_date"] for row in fit),
                 "diagnostic_earliest_entry": min(row["entry_date"] for row in diagnostic),
                 "title_coverage": {key: value for key, value in counts.items()
                                    if key not in ("raw_counts", "unique_counts")},
                 "local_encoder": str(base.BASE_MODEL),
                 "retrospective_warning": "Collector backup and frozen Encoder are later than historical outcomes."}
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
        return preflight
    assert output is not None
    check_budget(started, args.max_seconds)
    market_fit, market_diag, market_config = market_matrices(fit, diagnostic)
    fit_y = np.asarray([row["label"] for row in fit], dtype=int)
    model = LogisticRegression(C=MARKET_C, solver="lbfgs", max_iter=1000, random_state=35)
    model.fit(market_fit, fit_y)
    if model.n_iter_[0] >= 1000:
        raise ValueError("market baseline did not converge")
    fit_offset = model.decision_function(market_fit)
    diag_offset = model.decision_function(market_diag)
    market_p = probability(diag_offset)
    prior_p = np.full(len(diagnostic), float(fit_y.mean()))
    if args.daily_vector_cache:
        daily = load_daily_mean_vectors(Path(args.daily_vector_cache), needed)
        device = "saved_frozen_daily_vectors"
    else:
        titles = sorted({title for bag in bags.values() for title in bag})
        title_index, embeddings, device = embed_titles(
            titles, batch_size=args.encoder_batch_size, started=started, max_seconds=args.max_seconds)
        check_budget(started, args.max_seconds)
        daily = daily_mean_vectors(bags, title_index, embeddings)
        del embeddings
    output.mkdir(parents=True, exist_ok=True)
    save_daily_mean_vectors(output / "daily-mean-title-vectors.npz", daily)
    fit_raw_text = trajectory_vectors(fit, calendar, daily)
    diag_raw_text = trajectory_vectors(diagnostic, calendar, daily)
    fit_text, diag_text, text_config = text_matrices(fit_raw_text, diag_raw_text)
    shifted_text = stock_shifted_vectors(diagnostic, diag_text)
    text_predictions = {}
    shifted_predictions = {}
    training = {}
    coefficients = {}
    for seed in SEEDS:
        check_budget(started, args.max_seconds)
        beta, train_log = fit_text_residual(fit_text, fit_y, fit_offset, seed=seed)
        text_predictions[seed] = probability(diag_offset + diag_text @ beta)
        shifted_predictions[seed] = probability(diag_offset + shifted_text @ beta)
        training[str(seed)] = train_log
        coefficients[str(seed)] = beta.tolist()
    diagnostic_summary, predictions = diagnostic_report(
        diagnostic, market_p, prior_p, text_predictions, shifted_predictions)
    summary = {"status": "RETROSPECTIVE_BARRIER_TITLE_RESIDUAL_DIAGNOSTIC",
               "preflight": preflight,
               "configuration": {"N": WINDOW, "H": HORIZON, "upper_barrier": BARRIER,
                                 "lower_barrier": -BARRIER, "fit_outcome_end": FIT_OUTCOME_END,
                                 "diagnostic_entry_start": DIAGNOSTIC_ENTRY_START,
                                 "seeds": list(SEEDS), "market_C": MARKET_C,
                                 "text_pca_components": PCA_COMPONENTS,
                                 "text_l2": TEXT_L2, "text_lr": TEXT_LR,
                                 "text_epochs": TEXT_EPOCHS,
                                 "same_stock_placebo_shift_sessions": SHIFT_SESSIONS,
                                 "encoder_frozen": True, "title_tokens_max": 64,
                                 "title_aggregation": "unique titles mean per stock-day; last5 mean and last5 minus previous5",
                                 "market_numeric": [*base.MARKET_KEYS[:7],
                                                    "backup_log1p_raw_post_count_n20",
                                                    "backup_log1p_unique_title_count_n20",
                                                    "backup_raw_daily_log1p_last5_level",
                                                    "backup_raw_daily_log1p_last5_minus_previous5",
                                                    "backup_unique_daily_log1p_last5_level",
                                                    "backup_unique_daily_log1p_last5_minus_previous5",
                                                    "known_stock_close_return_previous20_sessions",
                                                    "known_stock_daily_close_return_std_previous20_sessions",
                                                    "known_stock_high_low_over_open_mean_previous5_sessions"],
                                 "market_stock_intercepts": True},
               "market_fit": {"iterations": int(model.n_iter_[0]),
                              "fit_prior_positive": float(fit_y.mean()),
                              "feature_config": market_config},
               "text_fit": {"feature_config": text_config, "training_by_seed": training,
                            "coefficients_by_seed": coefficients},
               "diagnostic": diagnostic_summary,
               "encoder_device": device, "elapsed_seconds": time.monotonic() - started,
               "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "daily_vector_cache": str(output / "daily-mean-title-vectors.npz"),
               "input_paths": {key: str(path.resolve()) for key, path in paths.items()}}
    base._write_json(output / "summary.json", summary)
    base._write_jsonl(output / "diagnostic-predictions.jsonl", predictions)
    print(json.dumps({"output": str(output), "diagnostic": diagnostic_summary},
                     ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--collector-backup", type=Path, default=coverage.BACKUP_DB)
    parser.add_argument("--daily-vector-cache", type=Path)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
