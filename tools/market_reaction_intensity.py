"""One bounded retrospective test of title-read *market-shock intensity*.

This deliberately does not retry the failed directional reader's Stage 2.
Frozen local LERT title vectors feed a low-capacity, gradient-trained
post-attention residual for the absolute just-finished close-to-close move.
The target is contemporaneous with, and known before, the after-close titles:
this is not a future-return predictor or evidence of a trading strategy.

The count/stock baseline and same-stock seven-session shifted-title placebo
are fixed on May 7--June 23. June 24--July 20 and August 3--September 11
are read exactly once as chronological diagnostics. The late period is never
used to select hyperparameters. All historical text was backfilled later;
even the chronological checks are retrospective, not live evidence.
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
from tools import market_reaction_reader as source

PROJECT = REPO.parent
DEFAULT_CACHE = PROJECT / "pipeline-runs/market-reaction-reader-20260930/frozen-title-cache.npz"
DEFAULT_OUTPUT = PROJECT / "pipeline-runs/market-reaction-intensity-20260930-seedfix"
TARGET_CAP = 0.15  # A 15% absolute close-to-close move maps to one; fixed before evaluation.
EXTREME_THRESHOLD = 0.07
SEEDS = source.SEEDS
SHIFT_SESSIONS = 7
EPOCHS = 25
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.01
COUNT_RIDGE_ALPHA = 20.0
MIN_INCREMENTAL_MSE = 0.001  # On [0,1] absolute-return target, fixed before evaluation.
MAX_SECONDS = 7200.0
MAX_BYTES = 10 * 1024 ** 3
EVIDENCE = source.EVIDENCE


def check_budget(started: float) -> None:
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("intensity experiment exceeded two-hour budget")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    used = peak if sys.platform == "darwin" else peak * 1024
    if used > MAX_BYTES:
        raise MemoryError("intensity experiment exceeded 10 GiB RSS budget")


def make_examples(bounds: tuple[str, str], calendar: Sequence[str], codes: Sequence[str],
                  qfq: Mapping, bags: Mapping, title_ids: Mapping[str, int] | None,
                  counts: Mapping) -> list[dict]:
    """One full, deduplicated post-close title bag per nonempty stock-day."""
    rows = []
    position = {day: index for index, day in enumerate(calendar)}
    for day in source.session_range(calendar, bounds):
        index = position[day]
        if index == 0:
            raise ValueError("the absolute move needs the prior close")
        for code in codes:
            key = (code, day)
            titles = bags[key]
            if not titles:
                continue
            previous = qfq[(code, calendar[index - 1])]["close"]
            current = qfq[(code, day)]["close"]
            move = current / previous - 1.0
            if not math.isfinite(move):
                raise ValueError(f"non-finite close-to-close move: {key}")
            rows.append({"code": code, "day": day, "absolute_move": abs(move),
                         "target": min(abs(move) / TARGET_CAP, 1.0),
                         "extreme": int(abs(move) >= EXTREME_THRESHOLD),
                         "raw_count": counts["raw_counts"][key],
                         "unique_count": counts["unique_counts"][key],
                         "ids": None if title_ids is None else [title_ids[title] for title in titles]})
    return rows


def shift_fit_titles(rows: Sequence[dict]) -> list[dict]:
    """Break text/target alignment *within stock* without moving any labels.

    Only May--June fit bags are shifted. Placebo models are evaluated on the
    same unshifted gate/late bags as the true reader; future diagnostic titles
    and labels never enter training.
    """
    by_code: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_code[row["code"]].append(index)
    shifted = [dict(row) for row in rows]
    for indices in by_code.values():
        for j, index in enumerate(indices):
            source_index = indices[(j + SHIFT_SESSIONS) % len(indices)]
            shifted[index]["ids"] = rows[source_index]["ids"]
    return shifted


def fit_count_baseline(fit: Sequence[dict], gate: Sequence[dict], late: Sequence[dict],
                       codes: Sequence[str]) -> tuple[dict[str, np.ndarray], dict]:
    """Train-only count+stock ridge; no held-out target enters preprocessing."""
    from sklearn.linear_model import Ridge

    def matrix(rows: Sequence[dict]) -> np.ndarray:
        numeric = np.asarray([[math.log1p(row["raw_count"]), math.log1p(row["unique_count"])]
                              for row in rows], dtype=float)
        stock = np.asarray([[float(row["code"] == code) for code in codes] for row in rows], dtype=float)
        return np.column_stack((numeric, stock))
    matrices = {name: matrix(rows) for name, rows in
                (("fit", fit), ("gate", gate), ("late", late))}
    mean = matrices["fit"][:, :2].mean(axis=0)
    scale = matrices["fit"][:, :2].std(axis=0)
    scale[scale < 1e-8] = 1.0
    for matrix_value in matrices.values():
        matrix_value[:, :2] = (matrix_value[:, :2] - mean) / scale
    model = Ridge(alpha=COUNT_RIDGE_ALPHA)
    model.fit(matrices["fit"], [row["target"] for row in fit])
    predictions = {name: model.predict(matrix_value) for name, matrix_value in matrices.items()}
    return predictions, {"ridge_alpha": COUNT_RIDGE_ALPHA, "numeric_fit_mean": mean.tolist(),
                         "numeric_fit_std": scale.tolist(), "stock_intercepts": len(codes)}


def train_residual(rows: Sequence[dict], baseline_fit: np.ndarray,
                   vectors: np.ndarray, seed: int, started: float) -> tuple[Any, dict]:
    """Attention pooling/head get gradient; LERT and the count ridge stay frozen."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    embeddings = torch.from_numpy(vectors)
    dimension = int(embeddings.shape[1])

    class IntensityReader(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Parameter(torch.empty(dimension))
            self.head = torch.nn.Linear(dimension, 1)
            # Small seeded starts keep the residual near zero while making
            # independent seeds genuinely independent (unlike a zero start).
            torch.nn.init.normal_(self.query, std=0.01)
            torch.nn.init.normal_(self.head.weight, std=0.01)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, ids: Sequence[int]) -> Any:
            posts = embeddings[ids]
            weights = torch.softmax(posts @ self.query / math.sqrt(dimension), dim=0)
            return self.head(weights @ posts).squeeze()

    if len(rows) != len(baseline_fit) or any(not row["ids"] for row in rows):
        raise ValueError("invalid train reader rows")
    reader = IntensityReader()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    y = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    baseline = torch.tensor(baseline_fit, dtype=torch.float32)
    for _ in range(EPOCHS):
        check_budget(started)
        optimizer.zero_grad()
        residual = torch.stack([reader(row["ids"]) for row in rows])
        loss = torch.mean((baseline + residual - y) ** 2)
        loss.backward()
        optimizer.step()
    return reader, {"train_mse": float(loss.detach()),
                    "trainable_parameters": sum(parameter.numel() for parameter in reader.parameters())}


def residual_predictions(model: Any, rows: Sequence[dict], baseline: np.ndarray) -> np.ndarray:
    import torch
    if len(rows) != len(baseline):
        raise ValueError("prediction rows/baseline length mismatch")
    with torch.inference_mode():
        residual = np.asarray([float(model(row["ids"])) for row in rows], dtype=float)
    return baseline + residual


def diagnostic_metrics(rows: Sequence[dict], predicted: Sequence[float]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray([row["target"] for row in rows], dtype=float)
    extreme = np.asarray([row["extreme"] for row in rows], dtype=int)
    # Clipping is fixed and identical for count-only, true and placebo heads.
    score = np.clip(np.asarray(predicted, dtype=float), 0.0, 1.0)
    if not len(rows) or len(score) != len(rows):
        raise ValueError("invalid diagnostic rows")
    return {"n": len(rows), "extreme_n": int(extreme.sum()),
            "scaled_target_mse": float(np.mean((score - y) ** 2)),
            "absolute_move_mae": float(TARGET_CAP * np.mean(abs(score - y))),
            "extreme_auc": float(roc_auc_score(extreme, score)) if len(set(extreme)) == 2 else None,
            "extreme_average_precision": float(average_precision_score(extreme, score))
            if len(set(extreme)) == 2 else None}


def evaluate(market_root: Path, collector_db: Path, cache: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    started = time.monotonic()
    calendar, codes, qfq, _, _ = source.load_market(market_root)
    bags, counts = source.load_title_bags(collector_db, calendar,
                                          source.needed_units(calendar, codes))
    title_ids, vectors = source.encode_or_load(bags, cache)
    partitions = {
        "fit": make_examples(source.STAGE1_FIT, calendar, codes, qfq, bags, title_ids, counts),
        "gate": make_examples(source.STAGE1_GATE, calendar, codes, qfq, bags, title_ids, counts),
        "late": make_examples(source.STAGE1_LATE, calendar, codes, qfq, bags, title_ids, counts),
    }
    if len(partitions["fit"]) != 527 or len(partitions["gate"]) != 304 or len(partitions["late"]) != 475:
        raise ValueError("fixed nonempty Stage-1 partitions changed")
    if max(row["day"] for row in partitions["fit"]) >= min(row["day"] for row in partitions["gate"]):
        raise ValueError("fit/gate chronology violated")
    if max(row["day"] for row in partitions["gate"]) >= min(row["day"] for row in partitions["late"]):
        raise ValueError("gate/late chronology violated")
    predictions, baseline_info = fit_count_baseline(partitions["fit"], partitions["gate"],
                                                    partitions["late"], codes)
    summary = {
        "evidence": EVIDENCE, "question": "contemporaneous_absolute_price_shock_readout_only",
        "sign_reader_gate": "FAILED_AND_NOT_BYPASSED",
        "stage2_future_market_test": "NOT_RUN",
        "initialization": "seeded_normal_std_0.01; supersedes zero-initialized debugging run",
        "frozen_title_cache": str(cache.resolve()),
        "target": {"scale": TARGET_CAP, "extreme_threshold": EXTREME_THRESHOLD},
        "partitions": {name: {"first_day": min(row["day"] for row in rows),
                              "last_day": max(row["day"] for row in rows),
                              "nonempty_stock_days": len(rows),
                              "extreme_stock_days": sum(row["extreme"] for row in rows)}
                       for name, rows in partitions.items()},
        "baseline": baseline_info,
        "count_only": {name: diagnostic_metrics(rows, predictions[name])
                       for name, rows in partitions.items()},
        "seeds": [],
        "decision_rule": "True-title scaled MSE must beat count-only and same-stock shifted-title by at least "
                         f"{MIN_INCREMENTAL_MSE} on both gate and late for every seed; otherwise no "
                         "incremental intensity-readout evidence. Extreme AUC/AP are secondary.",
    }
    shifted = shift_fit_titles(partitions["fit"])
    for seed in SEEDS:
        check_budget(started)
        true_model, true_info = train_residual(partitions["fit"], predictions["fit"], vectors,
                                                seed, started)
        placebo_model, placebo_info = train_residual(shifted, predictions["fit"], vectors,
                                                      seed, started)
        seed_result: dict[str, Any] = {"seed": seed, "true_train": true_info,
                                       "shifted_title_train": placebo_info}
        for name in ("gate", "late"):
            rows = partitions[name]
            seed_result[name] = {
                "true_title": diagnostic_metrics(
                    rows, residual_predictions(true_model, rows, predictions[name])),
                "shifted_title_placebo": diagnostic_metrics(
                    rows, residual_predictions(placebo_model, rows, predictions[name])),
            }
        summary["seeds"].append(seed_result)
    passed = all(
        seed_result[name]["true_title"]["scaled_target_mse"] + MIN_INCREMENTAL_MSE <=
        min(summary["count_only"][name]["scaled_target_mse"],
            seed_result[name]["shifted_title_placebo"]["scaled_target_mse"])
        for seed_result in summary["seeds"] for name in ("gate", "late")
    )
    summary["decision"] = ("RETROSPECTIVE_INTENSITY_READOUT_ONLY_NOT_FORECAST"
                           if passed else "NO_ROBUST_INCREMENTAL_INTENSITY_READOUT")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    check_budget(started)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", type=Path, default=source.MARKET)
    parser.add_argument("--collector-db", type=Path, default=source.COLLECTOR)
    parser.add_argument("--title-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = evaluate(args.market_root, args.collector_db, args.title_cache, args.output_dir)
    print(json.dumps({key: result[key] for key in
                      ("partitions", "count_only", "seeds", "decision", "elapsed_seconds")},
                     ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
