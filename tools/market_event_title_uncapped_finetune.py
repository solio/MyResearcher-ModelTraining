"""Matched last-block LERT experiment: 16 sampled versus all available titles.

The May/June fit and embargoed July diagnostic are identical in both arms;
only the titles in each pre-open stock/day bag change. This is retrospective
evidence, not an independent forward result or a deployable trading model.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_event_title_coverage as coverage
from tools import market_event_top3_encoder as event
from tools import market_event_top3_learning_curve as curve
from tools import market_text_encoder as base

SNAPSHOTS = (0, 1, 2, 3, 5, 8)


def rebag_cases(cases, bags, arm):
    """Keep event outcomes/features fixed; replace only the pre-open titles."""
    changed = []
    for case in cases:
        rewritten = dict(case)
        rewritten["candidates"] = []
        for row in case["candidates"]:
            item = dict(row)
            ids = list(bags[(item["stock_code"], item["entry_date"])][arm])
            if not ids:
                raise ValueError("empty title bag")
            item["post_ids"] = ids
            item["ages"] = [0.0] * len(ids)
            rewritten["candidates"].append(item)
        changed.append(rewritten)
    return changed


def prepare(args, torch):
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark),
             "collector_backup": Path(args.collector_backup)}
    rows, frozen_titles, calendar, qfq = base.build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"],
        window=1, horizon=10)
    train, validation = curve.internal_partition(rows)
    threshold, _ = base.training_up_events(train, calendar, qfq,
                                           fit_end=curve.TRAIN_OUTCOME_END, prior_days=5)
    train_cases = event.event_cases(train, calendar, qfq, threshold=threshold,
                                    first_shock=calendar[0], last_shock=curve.TRAIN_OUTCOME_END,
                                    minimum_gap=10)
    validation_cases = event.event_cases(validation, calendar, qfq, threshold=threshold,
                                         first_shock=curve.VALID_ENTRY_START,
                                         last_shock=curve.VALID_OUTCOME_END, minimum_gap=10)
    if (len(train_cases), len(validation_cases)) != (16, 12):
        raise ValueError("internal event identity unexpectedly changed")
    units = {(row["stock_code"], row["entry_date"])
             for case in (*train_cases, *validation_cases) for row in case["candidates"]}
    sampled = {(row["stock_code"], row["entry_date"]): row["post_ids"]
               for case in (*train_cases, *validation_cases) for row in case["candidates"]}
    bags, titles, counts = coverage.title_universe(paths["collector_backup"], calendar,
                                                   units, frozen_titles, sampled)
    market_train = np.asarray([row["market"] for row in train], dtype=np.float32)
    mean = market_train.mean(axis=0)
    std = np.where(market_train.std(axis=0) < 1e-6, 1.0, market_train.std(axis=0))
    for row in (*train, *validation):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
    beta = torch.nn.Parameter(torch.zeros(len(mean), dtype=torch.float32))
    optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.01)
    for _ in range(300):
        optimizer.zero_grad()
        losses = []
        for case in train_cases:
            features = torch.tensor([row["market_scaled"] for row in case["candidates"]],
                                    dtype=torch.float32)
            losses.append(event.boundary_rank_loss(
                torch, features @ beta, [row["return"] for row in case["candidates"]]))
        torch.stack(losses).mean().backward()
        optimizer.step()
    return paths, train_cases, validation_cases, bags, titles, counts, threshold, len(mean), beta.detach().numpy()


def run(args):
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if args.maximum_epochs != SNAPSHOTS[-1]:
        raise ValueError("fixed experiment requires exactly eight epochs")
    if not args.seeds or not args.arms or not set(args.arms).issubset(coverage.ARMS):
        raise ValueError("invalid arms or seeds")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths, train_cases, validation_cases, bags, titles, counts, threshold, market_width, market_beta = prepare(args, torch)
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    token_cache = base.TitleTokenCache(tokenizer, titles, max_length=64)
    output.mkdir(parents=True, exist_ok=True)
    snapshots, records, checkpoint_paths = [], [], []
    for arm in args.arms:
        fit = rebag_cases(train_cases, bags, arm)
        validation = rebag_cases(validation_cases, bags, arm)
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = base.make_model(base.BASE_MODEL, market_width,
                                    baseline_beta=[0.0] * market_width,
                                    baseline_intercept=0.0, device=device)
            model.market_beta.requires_grad_(False)
            model.market_intercept.requires_grad_(False)
            model.eval()  # Keep the old deterministic, dropout-free fit.
            initial_layer = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
            encoder_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
            head_parameters = [*model.attention.parameters(), *model.text_head.parameters()]
            optimizer = torch.optim.AdamW([
                {"params": encoder_parameters, "lr": 2e-4, "weight_decay": 0.01},
                {"params": head_parameters, "lr": 1e-3, "weight_decay": 0.01},
            ])
            rng = random.Random(seed)
            for epoch in range(args.maximum_epochs + 1):
                if epoch in SNAPSHOTS:
                    for partition, cases in (("train", fit), ("validation", validation)):
                        stats, samples = curve.rank_snapshot(
                            torch, model, token_cache, cases, market_beta,
                            seed=seed, epoch=epoch, partition=partition,
                            initial_layer=initial_layer)
                        stats["arm"] = arm
                        snapshots.append(stats)
                        records.extend({"arm": arm, **sample} for sample in samples)
                    if epoch in (5, 8):
                        checkpoint = output / f"model-{arm}-seed-{seed}-epoch-{epoch}.pt"
                        torch.save({
                            "last_block": {k: v.detach().cpu() for k, v in
                                           model.backbone.encoder.layer[-1].state_dict().items()},
                            "attention": {k: v.detach().cpu() for k, v in model.attention.state_dict().items()},
                            "text_head": {k: v.detach().cpu() for k, v in model.text_head.state_dict().items()},
                        }, checkpoint)
                        checkpoint_paths.append({"arm": arm, "seed": seed, "epoch": epoch,
                                                 "path": str(checkpoint), "sha256": base.file_hash(checkpoint)})
                    recent = snapshots[-1]
                    print(f"arm={arm} seed={seed} epoch={epoch} "
                          f"validation_h10={recent['mean_selected_h10_return']['text']:.5f} "
                          f"train_loss={snapshots[-2]['boundary_loss']:.5f} "
                          f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
                if epoch == args.maximum_epochs:
                    break
                ordered = list(fit)
                rng.shuffle(ordered)
                for case in ordered:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("full-title fine-tune exceeded fixed time budget")
                    optimizer.zero_grad()
                    scores = model.predict_rows(case["candidates"], token_cache)
                    loss = event.boundary_rank_loss(torch, scores,
                                                    [row["return"] for row in case["candidates"]])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
                    optimizer.step()
            del model, optimizer
            if device.type == "mps":
                torch.mps.empty_cache()
    by_epoch = []
    for epoch in SNAPSHOTS:
        entry = {"epoch": epoch}
        for arm in args.arms:
            subset = [row for row in snapshots if row["arm"] == arm and
                      row["partition"] == "validation" and row["epoch"] == epoch]
            entry[arm] = {
                "mean_selected_h10_return": float(np.mean([
                    row["mean_selected_h10_return"]["text"] for row in subset])),
                "mean_oracle_regret": float(np.mean([
                    row["mean_oracle_regret"]["text"] for row in subset])),
                "mean_exact_top3_events": float(np.mean([
                    row["exact_top3_events"] for row in subset])),
            }
        if set(coverage.ARMS).issubset(args.arms):
            entry["all_minus_16_h10_return"] = (entry["all_titles"]["mean_selected_h10_return"] -
                                               entry["sample16"]["mean_selected_h10_return"])
        by_epoch.append(entry)
    control = next(row for row in snapshots if row["partition"] == "validation")
    summary = {
        "evidence_level": event.EVIDENCE,
        "purpose": "same last-block fine-tune comparison of 16 sampled vs all available titles",
        "warning": "Retrospective later backup; validation dates were previously inspected.",
        "configuration": {"arms": list(args.arms), "seeds": list(args.seeds),
                          "snapshot_epochs": list(SNAPSHOTS), "train_outcome_end": curve.TRAIN_OUTCOME_END,
                          "validation_entry_start": curve.VALID_ENTRY_START,
                          "validation_outcome_end": curve.VALID_OUTCOME_END,
                          "last_block_lr": 2e-4, "head_lr": 1e-3, "rank_temperature": 0.1,
                          "encoder_gradient_checkpointing": False, "dropout_disabled": True},
        "data": {"train_events": len(train_cases), "validation_events": len(validation_cases),
                 "fit_only_abs_shock_threshold_p90": threshold,
                 "collector_backup_sha256": base.file_hash(paths["collector_backup"]), **counts},
        "market_control_validation": {"selected_h10_return": control["mean_selected_h10_return"]["market_count"],
                                      "oracle_regret": control["mean_oracle_regret"]["market_count"]},
        "by_epoch_three_seed_means": by_epoch,
        "snapshots": snapshots, "checkpoint_paths": checkpoint_paths,
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "script_sha256": base.file_hash(Path(__file__)),
        "model_snapshot_weight_sha256": base.file_hash(base.BASE_MODEL / "pytorch_model.bin"),
        "input_identity": {name: {"path": str(path.resolve()), "sha256": base.file_hash(path)}
                           for name, path in paths.items()},
    }
    base._write_json(output / "summary.json", summary)
    base._write_jsonl(output / "snapshot-event-cases.jsonl", records)
    print(json.dumps({"output_dir": str(output), "coverage": counts,
                      "market_control": summary["market_control_validation"],
                      "by_epoch": by_epoch}, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--collector-backup", type=Path, default=coverage.BACKUP_DB)
    parser.add_argument("--arms", nargs="+", default=list(coverage.ARMS), choices=coverage.ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(curve.SEEDS))
    parser.add_argument("--maximum-epochs", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
