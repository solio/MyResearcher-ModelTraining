"""Retrospective raw-title market supervision with a genuinely trainable Encoder.

The loss flows through the final Transformer block of a locally installed
Chinese LERT-small model.  Historical inputs were backfilled after their
putative entry dates; results are method diagnostics, NEVER forward evidence.
No network, LLM, seven-head probability cache, or new human labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import market_signal_multiday as multiday
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as validation

PROJECT = REPO.parent
HISTORY = PROJECT / "pipeline-runs/market-extreme-unenriched-20260928/source"
MARKET = PROJECT / "pipeline-runs/market-data-backfill-20260928"
BASE_MODEL = REPO / ".encoder-artifacts/hf-cache-lert-small/official-snapshot/69e3e69ba258be5b301b26937e5b55a076c90460"
EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"
MARKET_KEYS = (
    "lag1_stock_return", "trailing5_stock_return", "lag1_benchmark_return",
    "trailing5_benchmark_return", "previous_turnover_rate",
    "previous_volume_ratio5", "previous_amount_ratio5",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_text_rows(metadata_path: Path, qfq_path: Path, raw_path: Path,
                    benchmark_path: Path, *, window: int, horizon: int) -> tuple[list[dict], dict[str, str], list[str], dict]:
    """Build one row per stock-entry day, with title-only past-session bags."""
    metadata = validation.read_jsonl(metadata_path)
    if not metadata or len({row["sample_id"] for row in metadata}) != len(metadata):
        raise ValueError("empty or duplicate post snapshot")
    posts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    title_by_id: dict[str, str] = {}
    daily: dict[tuple[str, str], dict] = {}
    for row in metadata:
        sid, code, entry = str(row["sample_id"]), str(row["stock_code"]), str(row["entry_date"])
        title = str(row.get("title") or "").strip()
        if not title:
            raise ValueError(f"empty frozen title: {sid}")
        if str(row["published_at"])[:10] >= entry:
            raise ValueError(f"post not prior to mapped entry: {sid}")
        title_by_id[sid] = title
        posts[(code, entry)].append(row)
    for (code, entry), items in posts.items():
        if len(items) > 16:
            raise ValueError("frozen unit exceeds 16-post sampling cap")
        total_counts = {int(item["unit_total_post_count"]) for item in items}
        if len(total_counts) != 1:
            raise ValueError("inconsistent daily source count")
        daily[(code, entry)] = {
            "stock_code": code, "entry_date": entry,
            "features": {name: 0.0 for name in multiday.CORE_STATE},
            "observed_posts": total_counts.pop(), "sampled_posts": len(items),
        }
    qfq = snapshot._read_stock_bars(qfq_path, adjustment="qfq")
    raw = snapshot._read_stock_bars(raw_path, adjustment="raw")
    benchmark = validation.read_csv(benchmark_path)
    calendar = sorted({str(item["trade_date"]) for item in benchmark})
    panel = multiday.build_panel(list(daily.values()), calendar, qfq, raw, benchmark,
                                 window=window, horizon=horizon, as_of=calendar[-1],
                                 min_coverage=1.0, return_threshold=0.0)
    rows = []
    for item in panel:
        if item["status"] != "COMPLETE":
            continue
        market = item["market_features"]
        if any(market.get(key) is None or not math.isfinite(float(market[key])) for key in MARKET_KEYS):
            continue
        code, index = str(item["stock_code"]), int(item["entry_index"])
        bag: list[str] = []
        ages: list[float] = []
        observed = 0
        for day_pos, day in enumerate(calendar[index - window + 1:index + 1]):
            unit = (code, day)
            if unit not in posts:
                raise ValueError("complete panel lacks a frozen post unit")
            selected = sorted(posts[unit], key=lambda row: (row["published_at"], row["sample_id"]))
            observed += daily[unit]["observed_posts"]
            for post in selected:
                bag.append(str(post["sample_id"]))
                ages.append((window - 1 - day_pos) / max(window - 1, 1))
        rows.append({
            "stock_code": code, "entry_date": str(item["entry_date"]),
            "outcome_date": str(item["outcome_date"]), "return": float(item["stock_return"]),
            "market": [float(market[key]) for key in MARKET_KEYS] + [math.log1p(observed)],
            "post_ids": bag, "ages": ages,
        })
    rows.sort(key=lambda row: (row["entry_date"], row["stock_code"]))
    return rows, title_by_id, calendar, qfq


def split(rows: Sequence[Mapping[str, Any]], fit_end: str, diagnostic_start: str) -> tuple[list[dict], list[dict]]:
    if fit_end >= diagnostic_start:
        raise ValueError("fit_end must precede diagnostic_start")
    fit = [dict(row) for row in rows if row["outcome_date"] <= fit_end]
    diagnostic = [dict(row) for row in rows if row["entry_date"] >= diagnostic_start]
    if not fit or not diagnostic:
        raise ValueError("empty fit or diagnostic split")
    if max(row["outcome_date"] for row in fit) >= min(row["entry_date"] for row in diagnostic):
        raise ValueError("matured label crosses diagnostic entry")
    return fit, diagnostic


def training_up_events(fit: Sequence[Mapping[str, Any]], calendar: Sequence[str], qfq: Mapping,
                       *, fit_end: str, prior_days: int = 5, top_gap_sessions: int = 10) -> tuple[float, list[list[int]]]:
    """Training-only UP shocks, greedily non-overlapping within stock."""
    positions = {day: i for i, day in enumerate(calendar)}
    by_key = {(str(row["stock_code"]), str(row["entry_date"])): i for i, row in enumerate(fit)}
    codes = sorted({code for code, _ in qfq})
    moves = []
    for code in codes:
        for i in range(1, len(calendar)):
            if calendar[i] > fit_end:
                break
            before, after = qfq.get((code, calendar[i - 1])), qfq.get((code, calendar[i]))
            if before and after and float(before["close"]) > 0:
                moves.append(abs(float(after["close"]) / float(before["close"]) - 1))
    if not moves:
        raise ValueError("no fit-period daily market moves")
    threshold = float(np.quantile(moves, 0.90))
    groups = []
    for code in codes:
        last_accepted = -10**9
        for i in range(prior_days, len(calendar)):
            day = calendar[i]
            if day > fit_end:
                break
            before, after = qfq.get((code, calendar[i - 1])), qfq.get((code, day))
            if not before or not after or float(before["close"]) <= 0:
                continue
            move = float(after["close"]) / float(before["close"]) - 1
            if move < threshold or i - last_accepted < top_gap_sessions:
                continue
            indices = [by_key.get((code, calendar[j])) for j in range(i - prior_days, i)]
            if any(value is None for value in indices):
                continue
            groups.append([int(value) for value in indices])
            last_accepted = positions[day]
    return threshold, groups


def pairwise_rank_loss(torch: Any, scores: Any, targets: Any, *, min_difference: float = 0.005) -> Any:
    """One event contributes one mean loss regardless of its ten ordered pairs."""
    terms = []
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            difference = float(targets[i] - targets[j])
            if abs(difference) < min_difference:
                continue
            sign = 1.0 if difference > 0 else -1.0
            terms.append(torch.nn.functional.softplus(-sign * (scores[i] - scores[j])))
    if not terms:
        return scores.sum() * 0
    return torch.stack(terms).mean()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


class TitleTokenCache:
    def __init__(self, tokenizer: Any, title_by_id: Mapping[str, str], *, max_length: int) -> None:
        if max_length < 8:
            raise ValueError("max_length is too small")
        self.pad_id = int(tokenizer.pad_token_id)
        self.ids = {
            sid: tokenizer(title, add_special_tokens=True, truncation=True,
                           max_length=max_length)["input_ids"]
            for sid, title in title_by_id.items()
        }

    def batch(self, sample_ids: Sequence[str], device: Any) -> dict[str, Any]:
        import torch

        sequences = [self.ids[sid] for sid in sample_ids]
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
        for row, ids in enumerate(sequences):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention[row, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


def make_model(snapshot_path: Path, market_width: int, *, baseline_beta: Sequence[float],
               baseline_intercept: float, device: Any) -> Any:
    import torch
    from transformers import AutoModel

    class MarketTitleEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True,
                                                       trust_remote_code=False)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            if not hasattr(self.backbone, "encoder") or not hasattr(self.backbone.encoder, "layer"):
                raise ValueError("expected BERT-style local Encoder with explicit layers")
            for parameter in self.backbone.encoder.layer[-1].parameters():
                parameter.requires_grad_(True)
            hidden = int(self.backbone.config.hidden_size)
            self.attention = torch.nn.Linear(hidden + 1, 1, bias=False)
            self.text_head = torch.nn.Linear(hidden, 1, bias=False)
            self.market_beta = torch.nn.Parameter(torch.tensor(baseline_beta, dtype=torch.float32))
            self.market_intercept = torch.nn.Parameter(torch.tensor(float(baseline_intercept), dtype=torch.float32))
            torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.005)
            torch.nn.init.normal_(self.text_head.weight, mean=0.0, std=0.001)

        def predict_rows(self, rows: Sequence[Mapping[str, Any]], tokens: TitleTokenCache,
                         *, chunk_size: int = 96, equal_weight: bool = False) -> Any:
            # Deduplicate repeated posts within an event's overlapping five-day windows.
            unique = list(dict.fromkeys(sid for row in rows for sid in row["post_ids"]))
            lookup = {sid: i for i, sid in enumerate(unique)}
            encoded = []
            for start in range(0, len(unique), chunk_size):
                batch = tokens.batch(unique[start:start + chunk_size], self.market_beta.device)
                encoded.append(self.backbone(**batch).last_hidden_state[:, 0, :])
            representations = torch.cat(encoded, dim=0)
            predictions = []
            for row in rows:
                selected = representations[[lookup[sid] for sid in row["post_ids"]]]
                ages = torch.tensor(row["ages"], dtype=selected.dtype, device=selected.device).unsqueeze(1)
                if equal_weight:
                    weights = torch.full((len(selected),), 1.0 / len(selected),
                                         dtype=selected.dtype, device=selected.device)
                else:
                    weights = torch.softmax(self.attention(torch.cat([selected, ages], dim=1)).squeeze(1), dim=0)
                pooled = (selected * weights[:, None]).sum(dim=0)
                market = torch.tensor(row["market_scaled"], dtype=selected.dtype, device=selected.device)
                predictions.append(market @ self.market_beta + self.market_intercept + self.text_head(pooled).squeeze())
            return torch.stack(predictions)

    return MarketTitleEncoder().to(device)


def train_market_baseline(fit: Sequence[Mapping[str, Any]], diagnostic: Sequence[Mapping[str, Any]],
                          *, epochs: int, seed: int) -> tuple[dict, np.ndarray]:
    import torch

    torch.manual_seed(seed)
    train_x = torch.tensor([row["market_scaled"] for row in fit], dtype=torch.float32)
    train_y = torch.tensor([row["target_scaled"] for row in fit], dtype=torch.float32)
    test_x = torch.tensor([row["market_scaled"] for row in diagnostic], dtype=torch.float32)
    beta = torch.nn.Parameter(torch.zeros(train_x.shape[1]))
    intercept = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([beta, intercept], lr=0.03, weight_decay=0.05)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(train_x @ beta + intercept, train_y, beta=1.0)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = (test_x @ beta + intercept).numpy()
        fit_loss = float(torch.nn.functional.smooth_l1_loss(train_x @ beta + intercept, train_y, beta=1.0))
    return {"beta": beta.detach().numpy().tolist(), "intercept": float(intercept.detach()),
            "fit_loss": fit_loss}, prediction


def _metrics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    actual = np.asarray([row["actual_horizon_return"] for row in rows], dtype=float)
    predicted = np.asarray([row[field] for row in rows], dtype=float)
    return {"mae": float(np.mean(np.abs(actual - predicted))),
            "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "pearson": float(np.corrcoef(actual, predicted)[0, 1]) if np.std(predicted) > 0 else 0.0}


def _per_stock(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stock_code"])].append(row)
    output = []
    for code, items in sorted(grouped.items()):
        k = max(1, math.ceil(len(items) / 4))
        record = {"stock_code": code, "dates": len(items), "selected_dates": k,
                  "all_date_mean_return": statistics.fmean(float(row["actual_horizon_return"]) for row in items)}
        for model in ("market_count", "raw_title_encoder"):
            selected = sorted(items, key=lambda row: -float(row[f"predicted_return__{model}"]))[:k]
            record[f"top_quartile_mean_return__{model}"] = statistics.fmean(float(row["actual_horizon_return"]) for row in selected)
        output.append(record)
    return output


def _event_diagnostic(rows: Sequence[Mapping[str, Any]], calendar: Sequence[str], qfq: Mapping,
                      threshold: float, *, fit_end: str, prior_days: int = 5, top_k: int = 3) -> list[dict]:
    by_key = {(str(row["stock_code"]), str(row["entry_date"])): row for row in rows}
    events = []
    for code in sorted({code for code, _ in qfq}):
        for i in range(prior_days, len(calendar)):
            day = calendar[i]
            if day <= fit_end:
                continue
            before, after = qfq.get((code, calendar[i - 1])), qfq.get((code, day))
            if not before or not after or float(before["close"]) <= 0:
                continue
            move = float(after["close"]) / float(before["close"]) - 1
            if abs(move) < threshold:
                continue
            candidates = [by_key.get((code, calendar[j])) for j in range(i - prior_days, i)]
            if any(item is None for item in candidates):
                continue
            best = sorted(candidates, key=lambda item: -float(item["actual_horizon_return"]))[:top_k]
            oracle = statistics.fmean(float(item["actual_horizon_return"]) for item in best)
            record = {"stock_code": code, "shock_date": day, "direction": "UP" if move > 0 else "DOWN",
                      "shock_close_return": move, "oracle_top3_mean_return": oracle,
                      "recent3_mean_return": statistics.fmean(float(item["actual_horizon_return"]) for item in candidates[-top_k:])}
            for model in ("market_count", "raw_title_encoder"):
                selected = sorted(candidates, key=lambda item: -float(item[f"predicted_return__{model}"]))[:top_k]
                mean_return = statistics.fmean(float(item["actual_horizon_return"]) for item in selected)
                record[f"selected_top3_mean_return__{model}"] = mean_return
                record[f"oracle_regret__{model}"] = oracle - mean_return
            events.append(record)
    return events


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new empty output directory: {output}")
    if args.window != 5 or args.horizon != 10 or args.epochs < 1 or args.max_length < 8:
        raise ValueError("this fixed research candidate requires N=5, H=10, positive epochs and max_length>=8")
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark)}
    for path in (*paths.values(), BASE_MODEL / "pytorch_model.bin", BASE_MODEL / "tokenizer.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows, title_by_id, calendar, qfq = build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"],
        window=args.window, horizon=args.horizon)
    fit, diagnostic = split(rows, args.fit_end, args.diagnostic_start)
    market_fit = np.asarray([row["market"] for row in fit], dtype=np.float32)
    return_fit = np.asarray([row["return"] for row in fit], dtype=np.float32)
    market_mean, market_std = market_fit.mean(axis=0), market_fit.std(axis=0)
    market_std = np.where(market_std < 1e-6, 1.0, market_std)
    return_mean, return_std = float(return_fit.mean()), float(return_fit.std())
    if return_std < 1e-6:
        raise ValueError("fit returns have no variation")
    for row in (*fit, *diagnostic):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - market_mean) / market_std).tolist()
        row["target_scaled"] = (row["return"] - return_mean) / return_std

    threshold, event_groups = training_up_events(fit, calendar, qfq, fit_end=args.fit_end,
                                                  prior_days=5, top_gap_sessions=10)
    baseline, baseline_scaled = train_market_baseline(fit, diagnostic, epochs=200, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    tokens = TitleTokenCache(tokenizer, title_by_id, max_length=args.max_length)
    model = make_model(BASE_MODEL, len(fit[0]["market_scaled"]), baseline_beta=baseline["beta"],
                       baseline_intercept=baseline["intercept"], device=device)
    model.eval()  # Disable dropout; this does not disable autograd.
    original_last_layer = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
    head_parameters = [*model.attention.parameters(), *model.text_head.parameters(),
                       model.market_beta, model.market_intercept]
    encoder_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 2e-5, "weight_decay": 0.01},
        {"params": head_parameters, "lr": 1e-4, "weight_decay": 0.01},
    ])
    rng = random.Random(args.seed)
    training_log = []
    for epoch in range(args.epochs):
        order = list(range(len(fit)))
        rng.shuffle(order)
        all_day_losses = []
        for position, index in enumerate(order, start=1):
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("research run exceeded the fixed wall-clock budget")
            optimizer.zero_grad()
            predicted = model.predict_rows([fit[index]], tokens)
            target = torch.tensor([fit[index]["target_scaled"]], dtype=predicted.dtype, device=device)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
            optimizer.step()
            all_day_losses.append(float(loss.detach()))
            if position % 128 == 0:
                print(f"epoch={epoch + 1}/{args.epochs} all_dates={position}/{len(fit)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        selected_events = list(event_groups)
        rng.shuffle(selected_events)
        event_losses = []
        for group in selected_events:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("research run exceeded the fixed wall-clock budget")
            candidates = [fit[index] for index in group]
            optimizer.zero_grad()
            predicted = model.predict_rows(candidates, tokens)
            targets = torch.tensor([row["target_scaled"] for row in candidates], dtype=predicted.dtype, device=device)
            raw_returns = [float(row["return"]) for row in candidates]
            regression = torch.nn.functional.smooth_l1_loss(predicted, targets, beta=1.0)
            ranking = pairwise_rank_loss(torch, predicted, raw_returns)
            loss = regression + 0.1 * ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
            optimizer.step()
            event_losses.append(float(loss.detach()))
        training_log.append({"epoch": epoch + 1, "all_date_mean_loss": statistics.fmean(all_day_losses),
                             "up_event_mean_combined_loss": statistics.fmean(event_losses) if event_losses else None,
                             "all_date_steps": len(all_day_losses), "up_event_steps": len(event_losses)})
        print(f"epoch={epoch + 1} complete all_date_loss={training_log[-1]['all_date_mean_loss']:.6f} "
              f"up_events={len(event_losses)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    last_layer_delta = float(torch.linalg.vector_norm(
        model.backbone.encoder.layer[-1].output.dense.weight.detach() - original_last_layer))
    if not math.isfinite(last_layer_delta) or last_layer_delta <= 0:
        raise ValueError("the raw-title Encoder's final layer did not update")

    predictions = []
    with torch.no_grad():
        for i, row in enumerate(diagnostic):
            score = float(model.predict_rows([row], tokens)[0].cpu())
            predictions.append({"stock_code": row["stock_code"], "entry_date": row["entry_date"],
                                "outcome_date": row["outcome_date"], "actual_horizon_return": row["return"],
                                "sampled_posts": len(row["post_ids"]), "evidence_level": EVIDENCE,
                                "predicted_return__market_count": float(baseline_scaled[i] * return_std + return_mean),
                                "predicted_return__raw_title_encoder": float(score * return_std + return_mean)})
            if (i + 1) % 128 == 0:
                print(f"diagnostic={i + 1}/{len(diagnostic)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    events = _event_diagnostic(predictions, calendar, qfq, threshold, fit_end=args.fit_end)
    per_stock = _per_stock(predictions)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "raw-title-encoder-state.pt"
    torch.save(model.state_dict(), checkpoint_path)
    # Load the saved weights into a fresh model and run one held-out row.  This
    # verifies that the inference path does not depend on the fit-time object.
    restored = make_model(BASE_MODEL, len(fit[0]["market_scaled"]), baseline_beta=baseline["beta"],
                          baseline_intercept=baseline["intercept"], device=device)
    restored.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        replay_score = float(restored.predict_rows([diagnostic[0]], tokens)[0].cpu()) * return_std + return_mean
    if abs(replay_score - predictions[0]["predicted_return__raw_title_encoder"]) > 1e-6:
        raise ValueError("saved Encoder checkpoint failed first-row replay")
    input_identity = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in paths.items()}
    summary = {
        "evidence_level": EVIDENCE,
        "not_forward_reasons": ["all fit texts were first collected after their historical entry dates",
                                "diagnostic market dates were inspected in previous project analyses"],
        "model": "hfl/chinese-lert-small, local official fixed revision; final Transformer block trainable",
        "model_revision": BASE_MODEL.name,
        "model_snapshot_weight_sha256": file_hash(BASE_MODEL / "pytorch_model.bin"),
        "model_snapshot_tokenizer_sha256": file_hash(BASE_MODEL / "tokenizer.json"),
        "script_sha256": file_hash(Path(__file__)), "checkpoint_sha256": file_hash(checkpoint_path),
        "text_field": "frozen title only; no model_text or later enriched body",
        "trainable_last_layer_weight_delta_l2": last_layer_delta,
        "device": str(device), "wall_seconds": time.monotonic() - started,
        "data": {"frozen_posts": len(title_by_id), "matured_panel_rows": len(rows),
                 "fit_rows": len(fit), "fit_dates": len({row["entry_date"] for row in fit}),
                 "diagnostic_rows": len(diagnostic), "diagnostic_dates": len({row["entry_date"] for row in diagnostic}),
                 "fit_last_outcome": max(row["outcome_date"] for row in fit),
                 "diagnostic_first_entry": min(row["entry_date"] for row in diagnostic)},
        "configuration": {"window": args.window, "horizon": args.horizon, "fit_end": args.fit_end,
                          "diagnostic_start": args.diagnostic_start, "epochs": args.epochs,
                          "max_length": args.max_length, "seed": args.seed,
                          "max_seconds": args.max_seconds, "up_event_threshold": threshold,
                          "up_event_groups": len(event_groups), "ranking_loss_weight": 0.1,
                          "pairwise_min_return_difference": 0.005,
                          "same_stock_event_gap_sessions": 10},
        "target": "H10 qfq close / entry qfq open - 1; all-day SmoothL1 plus train-only UP-event pairwise rank",
        "input_identity": input_identity,
        "scaler": {"market_keys": [*MARKET_KEYS, "log1p_observed_posts_in_window"],
                   "market_mean": market_mean.tolist(), "market_std": market_std.tolist(),
                   "return_mean": return_mean, "return_std": return_std},
        "market_baseline": baseline, "training_log": training_log,
        "metrics": {name: _metrics(predictions, f"predicted_return__{name}")
                    for name in ("market_count", "raw_title_encoder")},
        "event_count": len(events), "event_dates": len({event["shock_date"] for event in events}),
        "per_stock": per_stock,
    }
    _write_json(output / "summary.json", summary)
    _write_jsonl(output / "diagnostic-predictions.jsonl", predictions)
    _write_jsonl(output / "event-cases.jsonl", events)
    _write_jsonl(output / "per-stock.jsonl", per_stock)
    print(json.dumps({"output_dir": str(output), "metrics": summary["metrics"],
                      "last_layer_delta_l2": last_layer_delta, "elapsed_s": summary["wall_seconds"],
                      "evidence_level": EVIDENCE}, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=MARKET / "benchmark-000300.csv")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fit-end", default="2026-07-31")
    parser.add_argument("--diagnostic-start", default="2026-08-03")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
