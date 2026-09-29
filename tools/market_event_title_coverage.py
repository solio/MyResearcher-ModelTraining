"""Existing-data ablation: 16 sampled titles/day versus every unique title/day.

Both arms use the SAME frozen official LERT-small CLS representations, the
SAME trainable attention/text head, events, seeds and top-3 ranking loss.
This isolates the old post-cap within this fixed-representation screen. It
does not reproduce last-block fine-tuning or establish forward evidence.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_event_top3_encoder as event
from tools import market_event_top3_learning_curve as curve
from tools import market_signal_v1 as historical
from tools import market_text_encoder as base

BACKUP_DB = (base.PROJECT / "MyResearcher-DataCollector/runtime/backups/"
             "collector-20260928T023304Z.db")
ARMS = ("sample16", "all_titles")


def title_universe(db_path: Path, calendar: Sequence[str], required_units: set[tuple[str, str]],
                   frozen_titles: Mapping[str, str], sampled_bags: Mapping[tuple[str, str], Sequence[str]]) -> tuple[dict, dict, dict]:
    """Map one read-only collector vintage to pre-open units, dedup by title."""
    uri = db_path.resolve().as_uri() + "?mode=ro"
    grouped: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    relevant_codes = {unit[0] for unit in required_units}
    source_count = 0
    with sqlite3.connect(uri, uri=True) as connection:
        for source, source_item_id, code, title, published_at in connection.execute(
                "SELECT source,source_item_id,stock_code,title,published_at FROM posts"):
            code = str(code or "")
            if code not in relevant_codes:
                continue
            normalized = historical._normalize_text(title)
            if not normalized:
                continue
            try:
                day = historical._parse_datetime(published_at).date().isoformat()
            except (TypeError, ValueError):
                continue
            position = bisect.bisect_right(calendar, day)
            if position >= len(calendar):
                continue
            unit = (code, calendar[position])
            if unit not in required_units:
                continue
            sid = historical._stable_sample_id(str(source or ""), str(source_item_id or ""),
                                               str(published_at))
            grouped[unit].append((str(published_at), sid, str(title).strip()))
            source_count += 1
    bags: dict[tuple[str, str], dict[str, list[str]]] = {}
    titles_by_id: dict[str, str] = {}
    for unit in sorted(required_units):
        ordered = sorted(grouped.get(unit, []), key=lambda row: (row[0], row[1]))
        first_by_title: dict[str, tuple[str, str, str]] = {}
        for row in ordered:
            first_by_title.setdefault(historical._normalize_text(row[2]), row)
        all_ids = [row[1] for row in first_by_title.values()]
        sampled_ids = list(sampled_bags[unit])
        if not all_ids or not sampled_ids or len(sampled_ids) > 16:
            raise ValueError(f"missing title unit or invalid old sample cap: {unit}")
        if not set(sampled_ids).issubset(all_ids):
            raise ValueError(f"old sampled ID not found in same-vintage unique titles: {unit}")
        for row in first_by_title.values():
            titles_by_id[row[1]] = row[2]
        for sid in sampled_ids:
            if titles_by_id[sid] != frozen_titles[sid]:
                raise ValueError(f"frozen sampled title changed: {sid}")
        bags[unit] = {"sample16": sampled_ids, "all_titles": all_ids}
    counts = {"raw_posts_in_required_units": source_count,
              "unique_titles_in_required_units": sum(len(value["all_titles"]) for value in bags.values()),
              "sample16_titles_in_required_units": sum(len(value["sample16"]) for value in bags.values()),
              "units": len(bags),
              "max_all_titles_per_unit": max(len(value["all_titles"]) for value in bags.values())}
    return bags, titles_by_id, counts


def frozen_embeddings(torch, titles_by_id: Mapping[str, str], *, batch_size: int,
                      output: Path) -> tuple[dict[str, int], Any, dict]:
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    backbone = AutoModel.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                         trust_remote_code=False).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    unique_texts = sorted(set(titles_by_id.values()))
    text_to_index = {value: index for index, value in enumerate(unique_texts)}
    sid_to_index = {sid: text_to_index[value] for sid, value in titles_by_id.items()}
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(unique_texts), batch_size):
            batch = unique_texts[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=64,
                               return_tensors="pt")
            tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
            encoded = backbone(**tokens).last_hidden_state[:, 0, :]
            blocks.append(encoded.cpu().clone())
            if (start + len(batch)) % 2048 < batch_size:
                print(f"frozen LERT titles={start+len(batch)}/{len(unique_texts)}", flush=True)
    matrix = torch.cat(blocks, dim=0).contiguous()
    if not torch.isfinite(matrix).all():
        raise ValueError("non-finite frozen title embedding")
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "frozen-title-cls.pt"
    torch.save(matrix, matrix_path)
    identities = [{"sample_id": sid, "embedding_index": index,
                   "title_sha256": hashlib.sha256(titles_by_id[sid].encode("utf-8")).hexdigest()}
                  for sid, index in sorted(sid_to_index.items())]
    base._write_jsonl(output / "title-embedding-index.jsonl", identities)
    return sid_to_index, matrix, {"unique_title_texts": len(unique_texts),
                                  "embedding_width": int(matrix.shape[1]),
                                  "encoder_device": str(device),
                                  "embedding_cache_path": str(matrix_path),
                                  "embedding_cache_sha256": base.file_hash(matrix_path)}


def make_head(torch, width: int):
    class BagHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Linear(width, 1, bias=False)
            self.text_head = torch.nn.Linear(width, 1, bias=False)
            torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.005)
            torch.nn.init.normal_(self.text_head.weight, mean=0.0, std=0.001)

        def score(self, case, *, arm: str, bags, indices, matrix):
            scores = []
            for row in case["candidates"]:
                unit = (row["stock_code"], row["entry_date"])
                ids = bags[unit][arm]
                vectors = matrix[[indices[sid] for sid in ids]]
                weights = torch.softmax(self.attention(vectors).squeeze(1), dim=0)
                pooled = (weights[:, None] * vectors).sum(dim=0)
                scores.append(self.text_head(pooled).squeeze())
            return torch.stack(scores)

    return BagHead()


def score_partition(torch, head, cases, *, arm: str, bags, indices, matrix,
                    market_beta, seed: int, epoch: int, partition: str) -> tuple[dict, list[dict]]:
    records, losses = [], []
    with torch.no_grad():
        for case in cases:
            text_scores = head.score(case, arm=arm, bags=bags, indices=indices, matrix=matrix)
            returns = [row["return"] for row in case["candidates"]]
            market_scores = [float(np.dot(np.asarray(row["market_scaled"], dtype=np.float32), market_beta))
                             for row in case["candidates"]]
            record = event.case_record(case, text_scores.numpy().tolist(), market_scores)
            records.append({"arm": arm, "seed": seed, "epoch": epoch, "partition": partition, **record})
            losses.append(float(event.boundary_rank_loss(torch, text_scores, returns)))
    result = event.aggregate(records)
    result.update({"arm": arm, "seed": seed, "epoch": epoch, "partition": partition,
                   "boundary_loss": float(np.mean(losses)),
                   "exact_top3_events": sum(set(row["selected_indices"]["text"]) ==
                                            set(row["selected_indices"]["oracle"]) for row in records),
                   "text_head_weight_l2": float(torch.linalg.vector_norm(head.text_head.weight.detach()))})
    return result, records


def run(args: argparse.Namespace) -> dict:
    import torch

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark), "collector_backup": Path(args.collector_backup)}
    rows, sampled_titles, calendar, qfq = base.build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"], window=1, horizon=10)
    train, validation = curve.internal_partition(rows)
    threshold, _ = base.training_up_events(train, calendar, qfq, fit_end=curve.TRAIN_OUTCOME_END)
    train_cases = event.event_cases(train, calendar, qfq, threshold=threshold,
                                    first_shock=calendar[0], last_shock=curve.TRAIN_OUTCOME_END,
                                    minimum_gap=10)
    validation_cases = event.event_cases(validation, calendar, qfq, threshold=threshold,
                                         first_shock=curve.VALID_ENTRY_START,
                                         last_shock=curve.VALID_OUTCOME_END, minimum_gap=10)
    if (len(train_cases), len(validation_cases)) != (16, 12):
        raise ValueError("internal event identity unexpectedly changed")
    required_units = {(row["stock_code"], row["entry_date"])
                      for case in (*train_cases, *validation_cases) for row in case["candidates"]}
    sampled_bags = {(row["stock_code"], row["entry_date"]): row["post_ids"]
                    for case in (*train_cases, *validation_cases) for row in case["candidates"]}
    bags, titles_by_id, coverage = title_universe(paths["collector_backup"], calendar,
                                                    required_units, sampled_titles, sampled_bags)
    indices, matrix, embedding_info = frozen_embeddings(
        torch, titles_by_id, batch_size=args.encoder_batch_size, output=output)
    market_train = np.asarray([row["market"] for row in train], dtype=np.float32)
    mean = market_train.mean(axis=0)
    std = np.where(market_train.std(axis=0) < 1e-6, 1.0, market_train.std(axis=0))
    for row in (*train, *validation):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
    beta = torch.nn.Parameter(torch.zeros(len(mean), dtype=torch.float32))
    control_optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.01)
    for _ in range(300):
        control_optimizer.zero_grad()
        losses = []
        for case in train_cases:
            features = torch.tensor([row["market_scaled"] for row in case["candidates"]], dtype=torch.float32)
            losses.append(event.boundary_rank_loss(torch, features @ beta,
                                                   [row["return"] for row in case["candidates"]]))
        loss = torch.stack(losses).mean()
        loss.backward()
        control_optimizer.step()
    market_beta = beta.detach().numpy()
    snapshots, records, head_paths = [], [], []
    for arm in ARMS:
        for seed in curve.SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            head = make_head(torch, matrix.shape[1])
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
            rng = random.Random(seed)
            for epoch in range(curve.SNAPSHOTS[-1] + 1):
                if epoch in curve.SNAPSHOTS:
                    for partition, cases in (("train", train_cases), ("validation", validation_cases)):
                        summary, samples = score_partition(
                            torch, head, cases, arm=arm, bags=bags, indices=indices,
                            matrix=matrix, market_beta=market_beta, seed=seed,
                            epoch=epoch, partition=partition)
                        snapshots.append(summary)
                        records.extend(samples)
                    head_path = output / f"head-{arm}-{seed}-epoch-{epoch}.pt"
                    torch.save(head.state_dict(), head_path)
                    head_paths.append({"arm": arm, "seed": seed, "epoch": epoch,
                                       "path": str(head_path), "sha256": base.file_hash(head_path)})
                    recent = snapshots[-1]
                    print(f"arm={arm} seed={seed} epoch={epoch} "
                          f"validation_return={recent['mean_selected_h10_return']['text']:.5f} "
                          f"validation_exact={recent['exact_top3_events']}/12 "
                          f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
                if epoch == curve.SNAPSHOTS[-1]:
                    break
                cases = list(train_cases)
                rng.shuffle(cases)
                for case in cases:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("coverage screen exceeded fixed time budget")
                    optimizer.zero_grad()
                    scores = head.score(case, arm=arm, bags=bags, indices=indices, matrix=matrix)
                    loss = event.boundary_rank_loss(torch, scores,
                                                    [row["return"] for row in case["candidates"]])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
    by_epoch = []
    for epoch in curve.SNAPSHOTS:
        entry = {"epoch": epoch}
        for arm in ARMS:
            subset = [row for row in snapshots if row["arm"] == arm and
                      row["partition"] == "validation" and row["epoch"] == epoch]
            entry[arm] = {"mean_selected_h10_return": float(np.mean([
                row["mean_selected_h10_return"]["text"] for row in subset])),
                "mean_oracle_regret": float(np.mean([row["mean_oracle_regret"]["text"] for row in subset])),
                "mean_exact_top3_events": float(np.mean([row["exact_top3_events"] for row in subset]))}
        entry["all_minus_16_h10_return"] = (entry["all_titles"]["mean_selected_h10_return"] -
                                           entry["sample16"]["mean_selected_h10_return"])
        by_epoch.append(entry)
    market_validation = next(row for row in snapshots if row["partition"] == "validation")
    result = {"evidence_level": event.EVIDENCE,
              "purpose": "same-frozen-encoder architecture comparison of 16 sampled vs all existing title coverage",
              "warning": "Retrospective later backup, originally exposed dates; not a final model or independent validation.",
              "configuration": {"seeds": list(curve.SEEDS), "snapshot_epochs": list(curve.SNAPSHOTS),
                                "train_outcome_end": curve.TRAIN_OUTCOME_END,
                                "validation_entry_start": curve.VALID_ENTRY_START,
                                "validation_outcome_end": curve.VALID_OUTCOME_END,
                                "title_encoder": "full hfl/chinese-lert-small frozen CLS, 64 tokens",
                                "trainable": "same attention plus text head in both arms",
                                "rank_temperature": 0.1, "head_lr": 1e-3,
                                "comparison_primary_epoch": 5},
              "data": {"train_events": len(train_cases), "validation_events": len(validation_cases),
                       "train_label_latest": max(row["outcome_date"] for row in train),
                       "validation_entry_earliest": min(row["entry_date"] for row in validation),
                       "fit_only_abs_shock_threshold_p90": threshold,
                       "collector_backup_sha256": base.file_hash(paths["collector_backup"]),
                       **coverage, **embedding_info},
              "market_control_validation": {"selected_h10_return": market_validation["mean_selected_h10_return"]["market_count"],
                                            "oracle_regret": market_validation["mean_oracle_regret"]["market_count"]},
              "by_epoch_three_seed_means": by_epoch,
              "snapshots": snapshots, "head_checkpoints": head_paths,
              "elapsed_seconds": time.monotonic() - started,
              "script_sha256": base.file_hash(Path(__file__)),
              "model_snapshot_weight_sha256": base.file_hash(base.BASE_MODEL / "pytorch_model.bin"),
              "input_identity": {name: {"path": str(path.resolve()), "sha256": base.file_hash(path)}
                                 for name, path in paths.items()}}
    base._write_json(output / "summary.json", result)
    base._write_jsonl(output / "snapshot-event-cases.jsonl", records)
    print(json.dumps({"output_dir": str(output), "coverage": coverage,
                      "market_control": result["market_control_validation"],
                      "by_epoch": by_epoch}, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--collector-backup", type=Path, default=BACKUP_DB)
    parser.add_argument("--encoder-batch-size", type=int, default=96)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
