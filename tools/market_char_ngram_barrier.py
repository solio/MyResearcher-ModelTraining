"""One fixed sparse-title alternative for retrospective H10 market outcomes.

The raw titles in each pre-entry N20 stock-day history become character
2--4-gram TF-IDF daily documents. IDF/vocabulary fit on the chronological
training histories only. A regularized *linear* sparse residual is added to
the exact price plus same-backup count/timing baseline. No LERT or other
encoder is run. August--September data are read only for final diagnosis.

This is a historical backfill diagnostic, not live or trading evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import market_barrier_title_residual as barrier
from tools import market_text_encoder as base
from tools import market_signal_v2_snapshot as snapshot
from tools import market_event_title_coverage as coverage

PROJECT = REPO.parent
DEFAULT_OUTPUT = PROJECT / "pipeline-runs/market-char-ngram-barrier-20260930"
MAX_FEATURES = 8192
MIN_DF = 5
NGRAM_RANGE = (2, 4)
MARKET_C = 0.1
RESIDUAL_C = 0.1
SHIFT_SESSIONS = 10
MAX_SECONDS = 7200.0
MAX_BYTES = 10 * 1024 ** 3
EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"


def resource_status(started: float) -> dict:
    peak_native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_native if sys.platform == "darwin" else peak_native * 1024
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        raise TimeoutError("sparse title experiment exceeded two hours")
    if peak_bytes > MAX_BYTES:
        raise MemoryError("sparse title experiment exceeded 10 GiB RSS")
    return {"elapsed_seconds": float(elapsed), "peak_rss_bytes": int(peak_bytes)}


def h10_positive_label(row: Mapping, calendar: Sequence[str], qfq: Mapping) -> int:
    """Entry adjusted open to close on the tenth chronological trading day."""
    index = calendar.index(str(row["entry_date"]))
    horizon = calendar[index:index + barrier.HORIZON]
    if len(horizon) != barrier.HORIZON or horizon[-1] != row["outcome_date"]:
        raise ValueError("H10 market label has not matured or horizon changed")
    entry_open = float(qfq[(row["stock_code"], row["entry_date"])]["open"])
    ending_close = float(qfq[(row["stock_code"], horizon[-1])]["close"])
    if not math.isfinite(entry_open) or not math.isfinite(ending_close) or entry_open <= 0:
        raise ValueError("invalid H10 entry/exit price")
    return int(ending_close > entry_open)


def daily_documents(bags: Mapping[tuple[str, str], Sequence[str]]) -> dict[tuple[str, str], str]:
    """Include every unique normalized title in each required day, no cap."""
    documents = {unit: " \n ".join(titles) for unit, titles in bags.items()}
    if not documents or any(not doc for doc in documents.values()):
        raise ValueError("all required stock-day title bags must be nonempty")
    return documents


def fit_daily_tfidf(fit_units: set[tuple[str, str]], documents: Mapping):
    """Train-only vocabulary and IDF, fixed capacity and one configuration."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not fit_units or not fit_units.issubset(documents):
        raise ValueError("missing fit-only daily documents")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=NGRAM_RANGE,
                                 min_df=MIN_DF, max_features=MAX_FEATURES,
                                 sublinear_tf=True, norm="l2", dtype=np.float32)
    vectorizer.fit([documents[unit] for unit in sorted(fit_units)])
    if not vectorizer.vocabulary_:
        raise ValueError("empty train-only character vocabulary")
    return vectorizer


def history_matrix(rows: Sequence[Mapping], calendar: Sequence[str],
                   daily: Mapping[tuple[str, str], Any], vocabulary_size: int):
    """Sparse N20 level plus last5-minus-previous5 trajectory.

    Each day's document is transformed just once. The pair of normalized
    blocks is scaled by sqrt(2), so row norm remains comparable to one and
    count cannot simply enter through document length.
    """
    from scipy import sparse
    from sklearn.preprocessing import normalize

    position = {day: index for index, day in enumerate(calendar)}
    levels = []
    trends = []
    for row in rows:
        code = str(row["stock_code"])
        index = position[str(row["entry_date"])]
        if index < barrier.WINDOW - 1:
            raise ValueError("N20 history is not available")
        days = calendar[index - barrier.WINDOW + 1:index + 1]
        matrices = [daily[(code, day)] for day in days]
        level = sparse.vstack(matrices).mean(axis=0)
        previous = sparse.vstack(matrices[-10:-5]).mean(axis=0)
        recent = sparse.vstack(matrices[-5:]).mean(axis=0)
        levels.append(sparse.csr_matrix(level, dtype=np.float32))
        trends.append(sparse.csr_matrix(recent - previous, dtype=np.float32))
    level_matrix = normalize(sparse.vstack(levels, format="csr"), norm="l2")
    trend_matrix = normalize(sparse.vstack(trends, format="csr"), norm="l2")
    result = sparse.hstack((level_matrix, trend_matrix), format="csr", dtype=np.float32) / math.sqrt(2)
    if result.shape != (len(rows), 2 * vocabulary_size) or not np.isfinite(result.data).all():
        raise ValueError("invalid sparse N20 history matrix")
    return result


def shifted_indices(rows: Sequence[Mapping], shift: int = SHIFT_SESSIONS) -> np.ndarray:
    """Barrier pilot's fixed same-stock circular diagnostic text shift."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["stock_code"])].append(index)
    indices = np.arange(len(rows))
    for group in groups.values():
        group.sort(key=lambda index: str(rows[index]["entry_date"]))
        if len(group) <= shift:
            raise ValueError("not enough dates for fixed same-stock shift")
        indices[group] = np.roll(group, shift)
    return indices


def sparse_offset_fit(x: Any, y: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, dict]:
    """One deterministic L2-regularized logistic residual, offset frozen."""
    from scipy.optimize import minimize
    from scipy.special import expit

    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if x.shape[0] != len(y) or len(y) != len(offset):
        raise ValueError("offset fit dimensions disagree")
    penalty = 1.0 / (RESIDUAL_C * len(y))
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = offset + x @ beta
        probabilities = expit(logits)
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits) +
                     penalty * np.dot(beta, beta) / 2)
        gradient = np.asarray(x.T @ (probabilities - y)).ravel() / len(y) + penalty * beta
        return loss, gradient
    optimized = minimize(objective, np.zeros(x.shape[1], dtype=float), jac=True,
                         method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-11})
    if not optimized.success:
        raise RuntimeError(f"sparse offset optimization did not converge: {optimized.message}")
    return optimized.x, {"iterations": int(optimized.nit), "coefficient_l2": float(np.linalg.norm(optimized.x)),
                         "penalty_per_parameter": penalty, "fit_objective": float(optimized.fun)}


def brier_by_stock(rows: Sequence[Mapping], y: np.ndarray, baseline: np.ndarray,
                   text: np.ndarray, shifted: np.ndarray) -> list[dict]:
    result = []
    for code in sorted({str(row["stock_code"]) for row in rows}):
        mask = np.asarray([row["stock_code"] == code for row in rows], dtype=bool)
        def brier(p: np.ndarray) -> float:
            return float(np.mean((p[mask] - y[mask]) ** 2))
        base, actual, placebo = brier(baseline), brier(text), brier(shifted)
        result.append({"stock_code": code, "rows": int(mask.sum()), "positive": int(y[mask].sum()),
                       "baseline_brier": base, "title_brier": actual,
                       "shifted_title_brier": placebo,
                       "title_minus_baseline_gain": base - actual,
                       "title_minus_shifted_gain": placebo - actual})
    return result


def diagnostic_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("invalid diagnostic probabilities")
    return {"n": len(y), "positive": int(y.sum()),
            "brier": float(np.mean((p - y) ** 2)),
            "auc": float(roc_auc_score(y, p))}


def run(metadata: Path, qfq_path: Path, raw_path: Path, benchmark_path: Path,
        collector_backup: Path, output: Path) -> dict:
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression

    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    started = time.monotonic()
    fit, diagnostic, calendar = barrier.labeled_partitions(metadata, qfq_path, raw_path, benchmark_path)
    needed = barrier.required_units([*fit, *diagnostic], calendar)
    bags, counts = barrier.load_all_title_bags(collector_backup, calendar, needed)
    barrier.add_backup_count_features([*fit, *diagnostic], calendar, counts)
    # This is the published count-timing baseline: 7 price/volume controls,
    # same-backup N20 counts and recent-five level/change, plus stock one-hot.
    # We do not add the later exploratory price-path fields, whose Aug/Sep
    # baseline was materially worse and would make a weaker comparison.
    market_fit, market_diag, market_config = barrier.market_matrices(fit, diagnostic)
    qfq = snapshot._read_stock_bars(qfq_path, adjustment="qfq")
    for row in [*fit, *diagnostic]:
        row["h10_positive"] = h10_positive_label(row, calendar, qfq)
    if max(row["outcome_date"] for row in fit) > "2026-07-31" or min(
            row["entry_date"] for row in diagnostic) < "2026-08-03":
        raise ValueError("fixed chronological outcome/diagnostic split changed")
    resource_status(started)
    documents = daily_documents(bags)
    fit_units = barrier.required_units(fit, calendar)
    vectorizer = fit_daily_tfidf(fit_units, documents)
    units = sorted(documents)
    daily_sparse = vectorizer.transform([documents[unit] for unit in units])
    daily = {unit: daily_sparse[index] for index, unit in enumerate(units)}
    x_fit = history_matrix(fit, calendar, daily, len(vectorizer.vocabulary_))
    x_diag = history_matrix(diagnostic, calendar, daily, len(vectorizer.vocabulary_))
    x_shifted = x_diag[shifted_indices(diagnostic)]
    resource_status(started)
    outcomes = {}
    predictions = []
    for name, field in (("up_first_barrier", "label"), ("h10_positive", "h10_positive")):
        fit_y = np.asarray([row[field] for row in fit], dtype=int)
        diag_y = np.asarray([row[field] for row in diagnostic], dtype=int)
        model = LogisticRegression(C=MARKET_C, solver="lbfgs", max_iter=1000, random_state=35)
        model.fit(market_fit, fit_y)
        if int(model.n_iter_[0]) >= 1000:
            raise RuntimeError("market/count baseline failed to converge")
        fit_logit = model.decision_function(market_fit)
        diag_logit = model.decision_function(market_diag)
        beta, optimization = sparse_offset_fit(x_fit, fit_y, fit_logit)
        base_p = expit(diag_logit)
        text_p = expit(diag_logit + x_diag @ beta)
        shifted_p = expit(diag_logit + x_shifted @ beta)
        outcomes[name] = {
            "fit_positive": int(fit_y.sum()), "diagnostic_positive": int(diag_y.sum()),
            "baseline": diagnostic_metrics(diag_y, base_p),
            "title": diagnostic_metrics(diag_y, text_p),
            "same_stock_shifted_title": diagnostic_metrics(diag_y, shifted_p),
            "title_minus_baseline_brier_gain": float(np.mean((base_p - diag_y) ** 2) -
                                                     np.mean((text_p - diag_y) ** 2)),
            "title_minus_shifted_brier_gain": float(np.mean((shifted_p - diag_y) ** 2) -
                                                    np.mean((text_p - diag_y) ** 2)),
            "per_stock": brier_by_stock(diagnostic, diag_y, base_p, text_p, shifted_p),
            "residual_fit": optimization,
        }
        predictions.extend({"target": name, "stock_code": row["stock_code"],
                            "entry_date": row["entry_date"], "outcome_date": row["outcome_date"],
                            "label": int(label), "baseline": float(p0),
                            "title": float(p1), "shifted_title": float(p2)}
                           for row, label, p0, p1, p2 in
                           zip(diagnostic, diag_y, base_p, text_p, shifted_p))
        resource_status(started)
    summary = {
        "evidence": EVIDENCE, "method": "fixed_character_tfidf_sparse_logistic_residual",
        "no_lert_run": True,
        "fit_rows": len(fit), "diagnostic_rows": len(diagnostic),
        "fit_latest_outcome": max(row["outcome_date"] for row in fit),
        "diagnostic_first_entry": min(row["entry_date"] for row in diagnostic),
        "title_coverage": {key: value for key, value in counts.items()
                           if key not in ("raw_counts", "unique_counts")},
        "configuration": {"n20": barrier.WINDOW, "h10": barrier.HORIZON,
                          "vectorizer": {"analyzer": "char", "ngram_range": NGRAM_RANGE,
                                         "min_df": MIN_DF, "max_features": MAX_FEATURES,
                                         "sublinear_tf": True, "fit_on_training_daily_documents_only": True,
                                         "learned_vocabulary_size": len(vectorizer.vocabulary_)},
                          "text_blocks": "L2-normalized N20 mean plus L2-normalized recent5-minus-previous5",
                          "market_C": MARKET_C, "text_residual_C": RESIDUAL_C,
                          "same_stock_diagnostic_shift_sessions": SHIFT_SESSIONS,
                          "market_feature_config": market_config},
        "outcomes": outcomes, "resource": resource_status(started),
        "input_paths": {"metadata": str(metadata.resolve()), "qfq": str(qfq_path.resolve()),
                        "raw": str(raw_path.resolve()), "benchmark": str(benchmark_path.resolve()),
                        "collector_backup": str(collector_backup.resolve())},
    }
    output.mkdir(parents=True, exist_ok=True)
    base._write_json(output / "summary.json", summary)
    base._write_jsonl(output / "diagnostic-predictions.jsonl", predictions)
    print(json.dumps({"output": str(output), "resource": summary["resource"],
                      "outcomes": {name: {key: value for key, value in item.items()
                                          if key not in ("per_stock", "residual_fit")}
                                   for name, item in outcomes.items()}},
                     ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--collector-backup", type=Path, default=coverage.BACKUP_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(args.metadata, args.qfq, args.raw, args.benchmark, args.collector_backup, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
