"""Existing-data learning curve for the event-top3 raw-title Encoder.

Uses an embargoed May/June fit and July internal validation, with fixed
snapshots and three seeds. The source files include later rows, but those
August/September rows never enter a fit, snapshot metric or epoch selection.
This distinguishes undertraining from sample memorization;
it is not a new independent forward test or a hyperparameter search.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_event_top3_encoder as event
from tools import market_text_encoder as base

TRAIN_OUTCOME_END = "2026-06-23"
VALID_ENTRY_START = "2026-06-26"
VALID_OUTCOME_END = "2026-07-31"
SEEDS = (20260929, 20260930, 20261001)
SNAPSHOTS = (0, 1, 2, 3, 5, 8, 12, 20)


def internal_partition(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [row for row in rows if row["outcome_date"] <= TRAIN_OUTCOME_END]
    validation = [row for row in rows if VALID_ENTRY_START <= row["entry_date"]
                  and row["outcome_date"] <= VALID_OUTCOME_END]
    if not train or not validation:
        raise ValueError("empty embargoed internal partition")
    if max(row["outcome_date"] for row in train) >= min(row["entry_date"] for row in validation):
        raise ValueError("fit label overlaps validation candidate entry")
    if any(row["entry_date"] >= "2026-08-01" for row in (*train, *validation)):
        raise ValueError("later diagnostic entry crossed the internal partition")
    return train, validation


def select_best_epoch(by_epoch: list[dict]) -> int:
    if not by_epoch:
        raise ValueError("empty epoch curve")
    return int(min(by_epoch, key=lambda row: (row["mean_validation_regret"], row["epoch"]))["epoch"])


def rank_snapshot(torch, model, token_cache, cases, market_beta, *, seed: int,
                  epoch: int, partition: str, initial_layer) -> tuple[dict, list[dict]]:
    records = []
    losses, spans, margins = [], [], []
    with torch.no_grad():
        for case in cases:
            raw_scores = model.predict_rows(case["candidates"], token_cache)
            scores = raw_scores.cpu().numpy().tolist()
            market = [float(np.dot(np.asarray(row["market_scaled"], dtype=np.float32), market_beta))
                      for row in case["candidates"]]
            record = event.case_record(case, scores, market)
            records.append({"seed": seed, "epoch": epoch, "partition": partition, **record})
            returns = [row["return"] for row in case["candidates"]]
            losses.append(float(event.boundary_rank_loss(torch, raw_scores, returns).cpu()))
            spans.append(max(scores) - min(scores))
            top = set(event.oracle_top_indices(returns))
            margins.append(min(scores[index] for index in top) -
                           max(scores[index] for index in range(len(scores)) if index not in top))
    aggregate = event.aggregate(records)
    aggregate.update({"seed": seed, "epoch": epoch, "partition": partition,
                      "boundary_loss": float(np.mean(losses)),
                      "exact_top3_events": sum(set(row["selected_indices"]["text"]) ==
                                               set(row["selected_indices"]["oracle"]) for row in records),
                      "mean_text_score_span": float(np.mean(spans)),
                      "mean_oracle_boundary_margin": float(np.mean(margins)),
                      "text_head_weight_l2": float(torch.linalg.vector_norm(model.text_head.weight.detach())),
                      "last_layer_delta_l2": float(torch.linalg.vector_norm(
                          model.backbone.encoder.layer[-1].output.dense.weight.detach() - initial_layer))})
    return aggregate, records


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.maximum_epochs != SNAPSHOTS[-1]:
        raise ValueError("this fixed learning-curve contract has exactly 20 epochs")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark)}
    rows, titles, calendar, qfq = base.build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"], window=1, horizon=10)
    train, validation = internal_partition(rows)
    threshold, _ = base.training_up_events(train, calendar, qfq,
                                           fit_end=TRAIN_OUTCOME_END, prior_days=5)
    train_cases = event.event_cases(train, calendar, qfq, threshold=threshold,
                                    first_shock=calendar[0], last_shock=TRAIN_OUTCOME_END,
                                    minimum_gap=10)
    validation_cases = event.event_cases(validation, calendar, qfq, threshold=threshold,
                                         first_shock=VALID_ENTRY_START, last_shock=VALID_OUTCOME_END,
                                         minimum_gap=10)
    if len(train_cases) < 8 or len(validation_cases) < 8:
        raise ValueError("too few internal events for a learning curve")
    market_train = np.asarray([row["market"] for row in train], dtype=np.float32)
    mean = market_train.mean(axis=0)
    std = market_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    for row in (*train, *validation):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
    # Same event-only top-3 target, refit without July labels.
    beta = torch.nn.Parameter(torch.zeros(len(mean), dtype=torch.float32))
    control_optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.01)
    for _ in range(300):
        control_optimizer.zero_grad()
        losses = []
        for case in train_cases:
            features = torch.tensor([row["market_scaled"] for row in case["candidates"]], dtype=torch.float32)
            returns = [row["return"] for row in case["candidates"]]
            losses.append(event.boundary_rank_loss(torch, features @ beta, returns))
        loss = torch.stack(losses).mean()
        loss.backward()
        control_optimizer.step()
    market_beta = beta.detach().numpy()

    selected_ids = {sid for case in (*train_cases, *validation_cases)
                    for row in case["candidates"] for sid in row["post_ids"]}
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    token_cache = base.TitleTokenCache(tokenizer, {sid: titles[sid] for sid in selected_ids}, max_length=64)
    snapshots, case_records, checkpoint_paths = [], [], []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = base.make_model(base.BASE_MODEL, len(mean), baseline_beta=[0.0] * len(mean),
                                baseline_intercept=0.0, device=device)
        model.market_beta.requires_grad_(False)
        model.market_intercept.requires_grad_(False)
        model.eval()
        initial_layer = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
        encoder_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
        head_parameters = [*model.attention.parameters(), *model.text_head.parameters()]
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": 2e-4, "weight_decay": 0.01},
            {"params": head_parameters, "lr": 1e-3, "weight_decay": 0.01},
        ])
        rng = random.Random(seed)
        for epoch in range(args.maximum_epochs + 1):
            if epoch in SNAPSHOTS:
                for partition, cases in (("train", train_cases), ("validation", validation_cases)):
                    stats, records = rank_snapshot(torch, model, token_cache, cases, market_beta,
                                                   seed=seed, epoch=epoch, partition=partition,
                                                   initial_layer=initial_layer)
                    snapshots.append(stats)
                    case_records.extend(records)
                output.mkdir(parents=True, exist_ok=True)
                checkpoint = output / f"trainable-state-seed-{seed}-epoch-{epoch}.pt"
                torch.save({
                    "last_block": {key: value.detach().cpu().clone() for key, value in
                                   model.backbone.encoder.layer[-1].state_dict().items()},
                    "attention": {key: value.detach().cpu().clone() for key, value in
                                  model.attention.state_dict().items()},
                    "text_head": {key: value.detach().cpu().clone() for key, value in
                                  model.text_head.state_dict().items()},
                }, checkpoint)
                checkpoint_paths.append({"seed": seed, "epoch": epoch,
                                         "path": str(checkpoint), "sha256": base.file_hash(checkpoint)})
                recent = snapshots[-1]
                print(f"seed={seed} epoch={epoch} validation_loss={recent['boundary_loss']:.6f} "
                      f"validation_regret={recent['mean_oracle_regret']['text']:.6f} "
                      f"validation_exact={recent['exact_top3_events']}/{len(validation_cases)} "
                      f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
            if epoch == args.maximum_epochs:
                break
            cases = list(train_cases)
            rng.shuffle(cases)
            for case in cases:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("fixed learning-curve time budget exceeded")
                optimizer.zero_grad()
                scores = model.predict_rows(case["candidates"], token_cache)
                returns = [row["return"] for row in case["candidates"]]
                loss = event.boundary_rank_loss(torch, scores, returns)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
                optimizer.step()

    by_epoch = []
    for epoch in SNAPSHOTS:
        subset = [row for row in snapshots if row["partition"] == "validation" and row["epoch"] == epoch]
        by_epoch.append({"epoch": epoch,
                         "mean_validation_regret": float(np.mean([row["mean_oracle_regret"]["text"] for row in subset])),
                         "mean_validation_selected_h10_return": float(np.mean([
                             row["mean_selected_h10_return"]["text"] for row in subset])),
                         "mean_validation_boundary_loss": float(np.mean([row["boundary_loss"] for row in subset])),
                         "mean_validation_exact_top3_events": float(np.mean([
                             row["exact_top3_events"] for row in subset]))})
    selected_epoch = select_best_epoch(by_epoch)
    market_validation = next(row for row in snapshots if row["partition"] == "validation")
    result = {"evidence_level": event.EVIDENCE,
              "purpose": "existing-data optimization/generalization diagnostic, never August/September model selection",
              "configuration": {"train_outcome_end": TRAIN_OUTCOME_END,
                                "validation_entry_start": VALID_ENTRY_START,
                                "validation_outcome_end": VALID_OUTCOME_END,
                                "embargo_sessions": ["2026-06-24", "2026-06-25"],
                                "text_window": 1, "event_prior_days": 5, "top_k": 3,
                                "horizon": 10, "minimum_same_stock_event_gap": 10,
                                "snapshot_epochs": list(SNAPSHOTS), "seeds": list(SEEDS),
                                "last_block_lr": 2e-4, "text_head_lr": 1e-3,
                                "rank_temperature": 0.1,
                                "internal_validation_selection": "mean oracle regret across all 3 seeds; earliest epoch on tie"},
              "data": {"train_stock_days": len(train), "validation_stock_days": len(validation),
                       "train_events": len(train_cases), "validation_events": len(validation_cases),
                       "train_label_latest": max(row["outcome_date"] for row in train),
                       "validation_entry_earliest": min(row["entry_date"] for row in validation),
                       "fit_only_abs_shock_threshold_p90": threshold},
              "market_control_validation": {"selected_h10_return": market_validation["mean_selected_h10_return"]["market_count"],
                                            "oracle_regret": market_validation["mean_oracle_regret"]["market_count"],
                                            "positive_events": market_validation["positive_selected_return_events"]["market_count"]},
              "by_epoch_three_seed_means": by_epoch,
              "internal_selected_epoch_diagnostic_only": selected_epoch,
              "snapshots": snapshots, "checkpoint_paths": checkpoint_paths,
              "device": str(device), "elapsed_seconds": time.monotonic() - started,
              "model_snapshot_weight_sha256": base.file_hash(base.BASE_MODEL / "pytorch_model.bin"),
              "script_sha256": base.file_hash(Path(__file__)),
              "input_identity": {name: {"path": str(path.resolve()), "sha256": base.file_hash(path)}
                                 for name, path in paths.items()}}
    base._write_json(output / "summary.json", result)
    base._write_jsonl(output / "snapshot-event-cases.jsonl", case_records)
    print(json.dumps({"output_dir": str(output), "train_events": len(train_cases),
                      "validation_events": len(validation_cases),
                      "selected_epoch": selected_epoch,
                      "validation_by_epoch": by_epoch}, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--maximum-epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
