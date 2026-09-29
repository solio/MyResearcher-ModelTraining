"""Event-centred raw-title gradient experiment: rank the best 3 of 5 buy days.

An ex-post price shock defines a *research event*, never a feature at inference.
For each already-matured training event, only the five pre-shock candidate days
and their H10 realized returns enter the loss. No all-day return regression,
seven-head labels, LLM, or later-enriched bodies are used. Historical posts
were backfilled, and the diagnostic dates have already been inspected: this is
a retrospective mechanism test, not a tradable or independent validation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_text_encoder as base

EVIDENCE = "RETROSPECTIVE_DIAGNOSTIC_ONLY"


def event_cases(rows: Sequence[Mapping[str, Any]], calendar: Sequence[str], qfq: Mapping,
                *, threshold: float, first_shock: str, last_shock: str,
                prior_days: int = 5, minimum_gap: int = 10) -> list[dict]:
    """Find a fixed UP/DOWN shock catalog with complete, mature pre-event buys."""
    if not (0 < threshold < 1) or prior_days < 2 or minimum_gap < 0:
        raise ValueError("invalid event definition")
    keyed = {(str(row["stock_code"]), str(row["entry_date"])): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("duplicate stock-entry row")
    cases: list[dict] = []
    for code in sorted({code for code, _ in keyed}):
        last_accepted = -10**9
        for position in range(prior_days, len(calendar)):
            shock_day = calendar[position]
            if shock_day < first_shock or shock_day > last_shock:
                continue
            before = qfq.get((code, calendar[position - 1]))
            after = qfq.get((code, shock_day))
            if before is None or after is None or float(before["close"]) <= 0:
                continue
            move = float(after["close"]) / float(before["close"]) - 1.0
            if abs(move) < threshold or position - last_accepted < minimum_gap:
                continue
            candidates = [keyed.get((code, calendar[index]))
                          for index in range(position - prior_days, position)]
            if any(row is None for row in candidates):
                continue
            if any(str(row["outcome_date"]) < str(row["entry_date"]) for row in candidates):
                raise ValueError("candidate has invalid outcome chronology")
            cases.append({"event_id": f"{code}:{shock_day}", "stock_code": code,
                          "shock_date": shock_day, "direction": "UP" if move > 0 else "DOWN",
                          "shock_close_return": move, "candidates": candidates})
            last_accepted = position
    return sorted(cases, key=lambda case: (case["shock_date"], case["stock_code"]))


def oracle_top_indices(returns: Sequence[float], top_k: int = 3) -> list[int]:
    if not 0 < top_k < len(returns):
        raise ValueError("top_k must be inside the candidate window")
    return sorted(range(len(returns)), key=lambda index: (-float(returns[index]), index))[:top_k]


def boundary_rank_loss(torch: Any, scores: Any, returns: Sequence[float],
                       *, top_k: int = 3, minimum_gap: float = 0.005,
                       temperature: float = 0.1) -> Any:
    """Only penalize pairs crossing the oracle top-k selection boundary."""
    if temperature <= 0 or minimum_gap < 0 or len(scores) != len(returns):
        raise ValueError("invalid rank-loss shape or scale")
    order = sorted(range(len(returns)), key=lambda index: (-float(returns[index]), index))
    terms = [torch.nn.functional.softplus((scores[other] - scores[best]) / temperature)
             for best in order[:top_k] for other in order[top_k:]
             if float(returns[best]) - float(returns[other]) >= minimum_gap]
    if not terms:
        return scores.sum() * 0.0
    return torch.stack(terms).mean()


def select_indices(scores: Sequence[float], top_k: int = 3) -> list[int]:
    if not 0 < top_k <= len(scores):
        raise ValueError("invalid selected count")
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:top_k]


def case_record(case: Mapping[str, Any], text_scores: Sequence[float],
                market_scores: Sequence[float], *, top_k: int = 3) -> dict:
    candidates = case["candidates"]
    if len(text_scores) != len(candidates) or len(market_scores) != len(candidates):
        raise ValueError("candidate score count changed")
    returns = [float(row["return"]) for row in candidates]
    # Rotating title scores within the same event deliberately breaks alignment
    # without changing the five text inputs, returns, prices or model weights.
    shifted = [float(text_scores[(index + 2) % len(candidates)])
               for index in range(len(candidates))]
    selections = {
        "oracle": oracle_top_indices(returns, top_k),
        "text": select_indices(text_scores, top_k),
        "market_count": select_indices(market_scores, top_k),
        "shifted_text": select_indices(shifted, top_k),
        "recent": list(range(len(candidates) - top_k, len(candidates))),
    }
    mean = {name: float(np.mean([returns[index] for index in chosen]))
            for name, chosen in selections.items()}
    mean["all_candidates"] = float(np.mean(returns))
    return {"event_id": case["event_id"], "stock_code": case["stock_code"],
            "shock_date": case["shock_date"], "direction": case["direction"],
            "shock_close_return": case["shock_close_return"],
            "candidate_entry_dates": [row["entry_date"] for row in candidates],
            "candidate_outcome_dates": [row["outcome_date"] for row in candidates],
            "candidate_horizon_returns": returns,
            "scores": {"text": [float(value) for value in text_scores],
                       "market_count": [float(value) for value in market_scores],
                       "shifted_text": shifted},
            "selected_indices": selections, "selected_mean_return": mean,
            "oracle_regret": {name: mean["oracle"] - value for name, value in mean.items()
                              if name != "oracle"}, "evidence_level": EVIDENCE}


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict:
    if not records:
        return {"events": 0}
    metrics = ["text", "market_count", "shifted_text", "recent", "all_candidates", "oracle"]
    return {"events": len(records),
            "shock_dates": len({row["shock_date"] for row in records}),
            "stocks": len({row["stock_code"] for row in records}),
            "mean_selected_h10_return": {name: float(np.mean([
                row["selected_mean_return"][name] for row in records])) for name in metrics},
            "mean_oracle_regret": {name: float(np.mean([
                row["oracle_regret"][name] for row in records]))
                for name in metrics if name != "oracle"},
            "text_vs_market_events": {
                "better": sum(row["selected_mean_return"]["text"] >
                              row["selected_mean_return"]["market_count"] + 1e-12 for row in records),
                "equal": sum(abs(row["selected_mean_return"]["text"] -
                                 row["selected_mean_return"]["market_count"]) <= 1e-12 for row in records),
                "worse": sum(row["selected_mean_return"]["text"] <
                             row["selected_mean_return"]["market_count"] - 1e-12 for row in records),
            },
            "positive_selected_return_events": {name: sum(
                row["selected_mean_return"][name] > 0 for row in records)
                for name in metrics}}


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.text_window != 1 or args.prior_days != 5 or args.top_k != 3 or args.horizon != 10:
        raise ValueError("fixed first event-centred candidate requires text=1, prior=5, top=3, H10")
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark)}
    rows, titles, calendar, qfq = base.build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"],
        window=args.text_window, horizon=args.horizon)
    fit, diagnostic = base.split(rows, args.fit_end, args.diagnostic_start)
    threshold, _ = base.training_up_events(fit, calendar, qfq, fit_end=args.fit_end,
                                           prior_days=args.prior_days)
    fit_cases = event_cases(fit, calendar, qfq, threshold=threshold,
                            first_shock=calendar[0], last_shock=args.fit_end,
                            prior_days=args.prior_days, minimum_gap=args.minimum_gap)
    all_diagnostic_cases = event_cases(diagnostic, calendar, qfq, threshold=threshold,
                                       first_shock=args.diagnostic_start, last_shock=calendar[-1],
                                       prior_days=args.prior_days, minimum_gap=0)
    diagnostic_cases = event_cases(diagnostic, calendar, qfq, threshold=threshold,
                                   first_shock=args.diagnostic_start, last_shock=calendar[-1],
                                   prior_days=args.prior_days, minimum_gap=args.minimum_gap)
    if not fit_cases or not diagnostic_cases:
        raise ValueError("no complete training or diagnostic shock events")
    market_fit = np.asarray([row["market"] for row in fit], dtype=np.float32)
    mean = market_fit.mean(axis=0)
    std = market_fit.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    for row in (*fit, *diagnostic):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()

    # Both models see exactly the same training events and oracle top-3 sets.
    beta = torch.nn.Parameter(torch.zeros(len(mean), dtype=torch.float32))
    control_optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.01)
    market_log = []
    for epoch in range(args.market_epochs):
        control_optimizer.zero_grad()
        losses = []
        for case in fit_cases:
            features = torch.tensor([row["market_scaled"] for row in case["candidates"]], dtype=torch.float32)
            scores = features @ beta
            returns = [row["return"] for row in case["candidates"]]
            losses.append(boundary_rank_loss(torch, scores, returns, top_k=args.top_k,
                                             temperature=args.temperature))
        loss = torch.stack(losses).mean()
        loss.backward()
        control_optimizer.step()
        if epoch in (0, args.market_epochs - 1):
            market_log.append({"epoch": epoch + 1, "mean_event_boundary_loss": float(loss.detach())})

    training_ids = {sid for case in fit_cases
                    for row in case["candidates"] for sid in row["post_ids"]}
    diagnostic_ids = {sid for case in all_diagnostic_cases
                      for row in case["candidates"] for sid in row["post_ids"]}
    selected_ids = training_ids | diagnostic_ids
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    token_cache = base.TitleTokenCache(tokenizer, {sid: titles[sid] for sid in selected_ids}, max_length=64)
    model = base.make_model(base.BASE_MODEL, len(mean), baseline_beta=[0.0] * len(mean),
                            baseline_intercept=0.0, device=device)
    model.market_beta.requires_grad_(False)
    model.market_intercept.requires_grad_(False)
    model.eval()  # Fixed dropout, but gradients remain enabled.
    original_layer = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
    encoder_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [*model.attention.parameters(), *model.text_head.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": 2e-4, "weight_decay": 0.01},
        {"params": head_parameters, "lr": 1e-3, "weight_decay": 0.01},
    ])
    rng = random.Random(args.seed)
    training_log = []
    for epoch in range(args.text_epochs):
        cases = list(fit_cases)
        rng.shuffle(cases)
        losses = []
        for case in cases:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("fixed event training budget exceeded")
            optimizer.zero_grad()
            scores = model.predict_rows(case["candidates"], token_cache)
            returns = [row["return"] for row in case["candidates"]]
            loss = boundary_rank_loss(torch, scores, returns, top_k=args.top_k,
                                      temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        training_log.append({"epoch": epoch + 1, "events": len(cases),
                             "mean_event_boundary_loss": float(np.mean(losses))})
        print(f"event_top3 epoch={epoch + 1}/{args.text_epochs} events={len(cases)} "
              f"loss={training_log[-1]['mean_event_boundary_loss']:.6f} "
              f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    weight_delta = float(torch.linalg.vector_norm(
        model.backbone.encoder.layer[-1].output.dense.weight.detach() - original_layer))
    if not math.isfinite(weight_delta) or weight_delta <= 0:
        raise ValueError("last Encoder block did not update")

    def score_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict]:
        records = []
        with torch.no_grad():
            for case in cases:
                text = model.predict_rows(case["candidates"], token_cache).detach().cpu().numpy().tolist()
                market = [float(np.dot(np.asarray(row["market_scaled"], dtype=np.float32),
                                               beta.detach().numpy())) for row in case["candidates"]]
                records.append(case_record(case, text, market, top_k=args.top_k))
        return records

    train_records = score_cases(fit_cases)
    diagnostic_records = score_cases(diagnostic_cases)
    all_diagnostic_records = score_cases(all_diagnostic_cases)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "event-top3-raw-title-state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = base.make_model(base.BASE_MODEL, len(mean), baseline_beta=[0.0] * len(mean),
                               baseline_intercept=0.0, device=device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        replay = restored.predict_rows(diagnostic_cases[0]["candidates"], token_cache).cpu().numpy()
    if not np.allclose(replay, diagnostic_records[0]["scores"]["text"], atol=1e-6, rtol=0):
        raise ValueError("checkpoint replay failed")
    summary = {
        "evidence_level": EVIDENCE,
        "research_question": "Can a raw-title score rank the best 3 of 5 pre-shock buy days across other shock events?",
        "warning": "Shock identity and oracle returns are ex-post labels, never available features; historical text is backfilled and diagnostic dates were already inspected.",
        "model": "hfl/chinese-lert-small, final Transformer block plus post attention/text head trainable; market branch frozen zero",
        "model_revision": base.BASE_MODEL.name,
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "last_layer_delta_l2": weight_delta,
        "configuration": {"text_window": args.text_window, "event_prior_days": args.prior_days,
                          "top_k": args.top_k, "horizon": args.horizon, "fit_end": args.fit_end,
                          "diagnostic_start": args.diagnostic_start, "absolute_shock_threshold_fit_p90": threshold,
                          "minimum_same_stock_shock_gap_sessions": args.minimum_gap,
                          "text_epochs": args.text_epochs, "market_epochs": args.market_epochs,
                          "rank_temperature": args.temperature, "seed": args.seed},
        "data": {"frozen_posts": len(titles), "mature_stock_days": len(rows),
                 "fit_stock_days": len(fit), "diagnostic_stock_days": len(diagnostic),
                 "fit_events": len(fit_cases), "diagnostic_events_primary": len(diagnostic_cases),
                 "diagnostic_events_all_overlap_sensitivity": len(all_diagnostic_cases),
                 "training_event_title_count": len(training_ids),
                 "diagnostic_event_title_count": len(diagnostic_ids)},
        "training_log": training_log, "market_control_log": market_log,
        "train": {"all": aggregate(train_records),
                  "UP": aggregate([row for row in train_records if row["direction"] == "UP"]),
                  "DOWN": aggregate([row for row in train_records if row["direction"] == "DOWN"])},
        "diagnostic_primary": {"all": aggregate(diagnostic_records),
                               "UP": aggregate([row for row in diagnostic_records if row["direction"] == "UP"]),
                               "DOWN": aggregate([row for row in diagnostic_records if row["direction"] == "DOWN"])},
        "diagnostic_all_overlap_sensitivity": aggregate(all_diagnostic_records),
        "market_scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "market_control_beta": beta.detach().numpy().tolist(),
        "input_identity": {name: {"path": str(path.resolve()), "sha256": base.file_hash(path)}
                           for name, path in paths.items()},
        "model_snapshot_weight_sha256": base.file_hash(base.BASE_MODEL / "pytorch_model.bin"),
        "script_sha256": base.file_hash(Path(__file__)), "checkpoint_sha256": base.file_hash(checkpoint),
    }
    base._write_json(output / "summary.json", summary)
    base._write_jsonl(output / "train-events.jsonl", train_records)
    base._write_jsonl(output / "diagnostic-events-primary.jsonl", diagnostic_records)
    base._write_jsonl(output / "diagnostic-events-all.jsonl", all_diagnostic_records)
    print(json.dumps({"output_dir": str(output), "fit_events": len(fit_cases),
                      "diagnostic_events": len(diagnostic_cases),
                      "train": summary["train"]["all"],
                      "diagnostic": summary["diagnostic_primary"]["all"]},
                     ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--text-window", type=int, default=1)
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fit-end", default="2026-07-31")
    parser.add_argument("--diagnostic-start", default="2026-08-03")
    parser.add_argument("--minimum-gap", type=int, default=10)
    parser.add_argument("--text-epochs", type=int, default=3)
    parser.add_argument("--market-epochs", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
