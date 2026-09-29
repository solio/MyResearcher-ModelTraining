"""Retrospective all-date top-3 ranking check with uncapped raw titles.

The fixed LERT-small last-block model and boundary loss match the earlier
shock-window experiment. This run fits complete, disjoint five-session windows
from every mature May/June stock-day. July windows and the old shock catalog
are diagnostics only. Historical titles were backfilled and these dates have
already been inspected; neither diagnostic view is independent validation.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import market_event_title_coverage as coverage
from tools import market_event_title_uncapped_finetune as uncapped
from tools import market_event_top3_encoder as event
from tools import market_event_top3_learning_curve as curve
from tools import market_text_encoder as base

EPOCHS = 5
WINDOW = 5
EXPECTED_FIT_DAYS = 320
EXPECTED_DIAGNOSTIC_DAYS = 272
EXPECTED_FIT_WINDOWS = 64
EXPECTED_DIAGNOSTIC_WINDOWS = 48
EXPECTED_SHOCK_EVENTS = 12
MAX_SECONDS = 7200
MAX_BYTES = 10 * 1024**3


def complete_windows(rows: Sequence[Mapping[str, Any]], calendar: Sequence[str],
                     *, role: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """Take earliest complete chronological blocks; record the unused tail."""
    if role not in ("fit", "july_diagnostic"):
        raise ValueError("invalid window role")
    position = {day: index for index, day in enumerate(calendar)}
    if len(position) != len(calendar):
        raise ValueError("duplicate trading session")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for row in rows:
        unit = (str(row["stock_code"]), str(row["entry_date"]))
        if unit in identities or unit[1] not in position:
            raise ValueError("duplicate or unknown stock-session unit")
        identities.add(unit)
        grouped[unit[0]].append(row)
    cases: list[dict] = []
    unused: list[tuple[str, str]] = []
    for code, stock_rows in sorted(grouped.items()):
        ordered = sorted(stock_rows, key=lambda row: position[str(row["entry_date"])])
        indices = [position[str(row["entry_date"])] for row in ordered]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"nonconsecutive mature sessions for {code}")
        complete = len(ordered) // WINDOW * WINDOW
        for start in range(0, complete, WINDOW):
            candidates = ordered[start:start + WINDOW]
            first = str(candidates[0]["entry_date"])
            last = str(candidates[-1]["entry_date"])
            cases.append({"event_id": f"{role}:{code}:{first}:{last}",
                          "stock_code": code, "case_kind": "all_dates_window",
                          "window_start": first, "window_end": last,
                          "candidates": candidates})
        unused.extend((code, str(row["entry_date"])) for row in ordered[complete:])
    cases.sort(key=lambda case: (case["window_start"], case["stock_code"]))
    return cases, sorted(unused)


def validate_contract(fit: Sequence[Mapping[str, Any]], diagnostic: Sequence[Mapping[str, Any]],
                      fit_cases: Sequence[Mapping[str, Any]],
                      diagnostic_cases: Sequence[Mapping[str, Any]],
                      diagnostic_unused: Sequence[tuple[str, str]],
                      shock_cases: Sequence[Mapping[str, Any]]) -> None:
    if (len(fit), len(diagnostic), len(fit_cases), len(diagnostic_cases),
            len(diagnostic_unused), len(shock_cases)) != (
            EXPECTED_FIT_DAYS, EXPECTED_DIAGNOSTIC_DAYS,
            EXPECTED_FIT_WINDOWS, EXPECTED_DIAGNOSTIC_WINDOWS, 32,
            EXPECTED_SHOCK_EVENTS):
        raise ValueError("fixed stock-day, window, or shock-event count changed")
    fit_stocks = {str(row["stock_code"]) for row in fit}
    diagnostic_stocks = {str(row["stock_code"]) for row in diagnostic}
    if len(fit_stocks) != 16 or fit_stocks != diagnostic_stocks:
        raise ValueError("fixed 16-stock universe changed")
    if max(str(row["outcome_date"]) for row in fit) != curve.TRAIN_OUTCOME_END:
        raise ValueError("fit labels no longer end June 23")
    if min(str(row["entry_date"]) for row in diagnostic) != curve.VALID_ENTRY_START:
        raise ValueError("diagnostic entry no longer starts June 26")
    if max(str(row["outcome_date"]) for row in diagnostic) != curve.VALID_OUTCOME_END:
        raise ValueError("diagnostic labels no longer end July 31")
    if any(row["entry_date"] in {"2026-06-24", "2026-06-25"} for row in (*fit, *diagnostic)):
        raise ValueError("embargo entry included")
    fit_units = [(row["stock_code"], row["entry_date"])
                 for case in fit_cases for row in case["candidates"]]
    diagnostic_units = [(row["stock_code"], row["entry_date"])
                        for case in diagnostic_cases for row in case["candidates"]]
    if (len(fit_units) != len(set(fit_units)) or len(diagnostic_units) != len(set(diagnostic_units))
            or set(fit_units) != {(row["stock_code"], row["entry_date"]) for row in fit}
            or set(fit_units) & set(diagnostic_units)
            or set(diagnostic_units) & set(diagnostic_unused)
            or set(diagnostic_units) | set(diagnostic_unused) != {
                (row["stock_code"], row["entry_date"]) for row in diagnostic}):
        raise ValueError("window coverage or fit/diagnostic isolation changed")
    if max(row["outcome_date"] for row in fit) >= min(row["entry_date"] for row in diagnostic):
        raise ValueError("fit label crosses embargo")


def scale_market(fit: Sequence[dict], diagnostic: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fit the market-plus-count scaler on May/June rows only."""
    matrix = np.asarray([row["market"] for row in fit], dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = np.where(matrix.std(axis=0) < 1e-6, 1.0, matrix.std(axis=0))
    for row in (*fit, *diagnostic):
        row["market_scaled"] = ((np.asarray(row["market"], dtype=np.float32) - mean) / std).tolist()
    return mean, std


def fit_market_control(torch: Any, fit_cases: Sequence[Mapping[str, Any]], width: int,
                       started: float, max_seconds: float) -> tuple[np.ndarray, list[dict]]:
    """Match the existing market/count rank baseline on the same 64 windows."""
    beta = torch.nn.Parameter(torch.zeros(width, dtype=torch.float32))
    optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.01)
    batches = [(torch.tensor([row["market_scaled"] for row in case["candidates"]],
                             dtype=torch.float32),
                [row["return"] for row in case["candidates"]]) for case in fit_cases]
    log = []
    for epoch in range(300):
        check_budget(started, max_seconds)
        optimizer.zero_grad()
        loss = torch.stack([event.boundary_rank_loss(torch, features @ beta, returns)
                            for features, returns in batches]).mean()
        loss.backward()
        optimizer.step()
        if epoch in (0, 299):
            log.append({"epoch": epoch + 1, "mean_window_boundary_loss": float(loss.detach())})
    return beta.detach().numpy(), log


def check_budget(started: float, max_seconds: float) -> None:
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("all-date rank run exceeded the fixed two-hour budget")
    # macOS ru_maxrss is in bytes. MPS allocations are checked separately by
    # the caller while a model is resident.
    if sys.platform == "darwin" and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_BYTES:
        raise MemoryError("all-date rank run exceeded 10 GiB resident memory")


def score_view(torch: Any, model: Any, token_cache: Any,
               cases: Sequence[Mapping[str, Any]], market_beta: np.ndarray,
               *, view: str, seed: int, started: float,
               max_seconds: float) -> tuple[dict, list[dict]]:
    records = []
    losses = []
    with torch.no_grad():
        for case in cases:
            check_budget(started, max_seconds)
            raw = model.predict_rows(case["candidates"], token_cache)
            scores = raw.detach().cpu().numpy().tolist()
            market = [float(np.dot(np.asarray(row["market_scaled"], dtype=np.float32), market_beta))
                      for row in case["candidates"]]
            if view == "july_windows":
                source = {**case, "shock_date": case["window_end"],
                          "direction": "ALL_DATES", "shock_close_return": None}
            else:
                source = case
            record = event.case_record(source, scores, market)
            if view == "july_windows":
                for key in ("shock_date", "direction", "shock_close_return"):
                    record.pop(key)
                record.update({"window_start": case["window_start"],
                               "window_end": case["window_end"]})
            record.update({"case_kind": "all_dates_window" if view == "july_windows" else "shock_event",
                           "view": view, "seed": seed, "epoch": EPOCHS})
            records.append(record)
            losses.append(float(event.boundary_rank_loss(
                torch, raw, [row["return"] for row in case["candidates"]]).detach().cpu()))
    # event.aggregate counts shock dates, so supply the window end under that
    # internal key only while aggregating the all-date diagnostic.
    aggregate_rows = [{**row, "shock_date": row.get("shock_date", row.get("window_end"))}
                      for row in records]
    stats = event.aggregate(aggregate_rows)
    if view == "july_windows":
        stats.pop("shock_dates")
    stats.update({"view": view, "seed": seed, "epoch": EPOCHS,
                  "boundary_loss": float(np.mean(losses)),
                  "exact_top3_cases": sum(set(row["selected_indices"]["text"]) ==
                                          set(row["selected_indices"]["oracle"]) for row in records),
                  "text_minus_market_h10": (stats["mean_selected_h10_return"]["text"] -
                                            stats["mean_selected_h10_return"]["market_count"]),
                  "text_minus_shifted_h10": (stats["mean_selected_h10_return"]["text"] -
                                             stats["mean_selected_h10_return"]["shifted_text"])})
    return stats, records


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    seeds = tuple(args.seeds)
    if not seeds or seeds != tuple(seed for seed in curve.SEEDS if seed in seeds):
        raise ValueError("seeds must be an ordered subset of the fixed seed trio")
    if not 0 < args.max_seconds <= MAX_SECONDS:
        raise ValueError("max-seconds must be within the two-hour limit")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths = {"metadata": Path(args.metadata), "qfq": Path(args.qfq),
             "raw": Path(args.raw), "benchmark": Path(args.benchmark),
             "collector_backup": Path(args.collector_backup)}
    rows, frozen_titles, calendar, qfq = base.build_text_rows(
        paths["metadata"], paths["qfq"], paths["raw"], paths["benchmark"],
        window=1, horizon=10)
    fit, diagnostic = curve.internal_partition(rows)
    fit_cases, fit_unused = complete_windows(fit, calendar, role="fit")
    diagnostic_cases, diagnostic_unused = complete_windows(diagnostic, calendar,
                                                            role="july_diagnostic")
    threshold, _ = base.training_up_events(fit, calendar, qfq,
                                           fit_end=curve.TRAIN_OUTCOME_END, prior_days=5)
    shock_cases = event.event_cases(diagnostic, calendar, qfq, threshold=threshold,
                                    first_shock=curve.VALID_ENTRY_START,
                                    last_shock=curve.VALID_OUTCOME_END, minimum_gap=10)
    validate_contract(fit, diagnostic, fit_cases, diagnostic_cases,
                      diagnostic_unused, shock_cases)
    if fit_unused:
        raise ValueError("May/June fit did not consume every mature stock-day")
    selected = [row for case in (*fit_cases, *diagnostic_cases, *shock_cases)
                for row in case["candidates"]]
    sampled = {(row["stock_code"], row["entry_date"]): row["post_ids"] for row in selected}
    if any(sampled[(row["stock_code"], row["entry_date"])] != row["post_ids"] for row in selected):
        raise ValueError("same stock-day has conflicting sampled title IDs")
    bags, titles, counts = coverage.title_universe(
        paths["collector_backup"], calendar, set(sampled), frozen_titles, sampled)
    mean, std = scale_market(fit, diagnostic)
    fit_cases = uncapped.rebag_cases(fit_cases, bags, "all_titles")
    diagnostic_cases = uncapped.rebag_cases(diagnostic_cases, bags, "all_titles")
    shock_cases = uncapped.rebag_cases(shock_cases, bags, "all_titles")
    market_beta, market_log = fit_market_control(torch, fit_cases, len(mean), started,
                                                  args.max_seconds)
    check_budget(started, args.max_seconds)
    tokenizer = AutoTokenizer.from_pretrained(str(base.BASE_MODEL), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    token_cache = base.TitleTokenCache(tokenizer, titles, max_length=64)
    output.mkdir(parents=True, exist_ok=True)
    snapshots, checkpoint_paths, training_logs = [], [], []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = base.make_model(base.BASE_MODEL, len(mean), baseline_beta=[0.0] * len(mean),
                                baseline_intercept=0.0, device=device)
        model.market_beta.requires_grad_(False)
        model.market_intercept.requires_grad_(False)
        model.eval()  # Identical deterministic dropout-free last-block fit.
        original = model.backbone.encoder.layer[-1].output.dense.weight.detach().clone()
        encoder_parameters = [parameter for parameter in model.backbone.parameters()
                              if parameter.requires_grad]
        head_parameters = [*model.attention.parameters(), *model.text_head.parameters()]
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": 2e-4, "weight_decay": 0.01},
            {"params": head_parameters, "lr": 1e-3, "weight_decay": 0.01},
        ])
        rng = random.Random(seed)
        for epoch in range(1, EPOCHS + 1):
            ordered = list(fit_cases)
            rng.shuffle(ordered)
            losses = []
            for case in ordered:
                check_budget(started, args.max_seconds)
                optimizer.zero_grad()
                scores = model.predict_rows(case["candidates"], token_cache)
                loss = event.boundary_rank_loss(
                    torch, scores, [row["return"] for row in case["candidates"]])
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            training_logs.append({"seed": seed, "epoch": epoch,
                                  "windows": len(ordered),
                                  "mean_boundary_loss": float(np.mean(losses))})
            print(f"seed={seed} epoch={epoch}/{EPOCHS} windows={len(ordered)} "
                  f"fit_loss={training_logs[-1]['mean_boundary_loss']:.6f} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
        delta = float(torch.linalg.vector_norm(
            model.backbone.encoder.layer[-1].output.dense.weight.detach() - original))
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("last Encoder block did not update")
        seed_records = []
        for view, cases in (("july_windows", diagnostic_cases), ("july_shock_events", shock_cases)):
            check_budget(started, args.max_seconds)
            stats, records = score_view(torch, model, token_cache, cases, market_beta,
                                        view=view, seed=seed, started=started,
                                        max_seconds=args.max_seconds)
            stats["last_layer_delta_l2"] = delta
            snapshots.append(stats)
            seed_records.extend(records)
            print(f"seed={seed} view={view} text_h10={stats['mean_selected_h10_return']['text']:.6f} "
                  f"market_h10={stats['mean_selected_h10_return']['market_count']:.6f} "
                  f"shifted_h10={stats['mean_selected_h10_return']['shifted_text']:.6f} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
        checkpoint = output / f"all-dates-seed-{seed}-epoch-5.pt"
        torch.save({
            "last_block": {key: value.detach().cpu().clone() for key, value in
                           model.backbone.encoder.layer[-1].state_dict().items()},
            "attention": {key: value.detach().cpu().clone() for key, value in
                          model.attention.state_dict().items()},
            "text_head": {key: value.detach().cpu().clone() for key, value in
                          model.text_head.state_dict().items()},
        }, checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = base.make_model(base.BASE_MODEL, len(mean), baseline_beta=[0.0] * len(mean),
                                   baseline_intercept=0.0, device=torch.device("cpu"))
        restored.backbone.encoder.layer[-1].load_state_dict(saved["last_block"])
        restored.attention.load_state_dict(saved["attention"])
        restored.text_head.load_state_dict(saved["text_head"])
        restored.eval()
        smoke_row = diagnostic_cases[0]["candidates"][0]
        with torch.no_grad():
            expected = model.predict_rows([smoke_row], token_cache).detach().cpu().numpy()
            actual = restored.predict_rows([smoke_row], token_cache).detach().cpu().numpy()
        if not np.allclose(expected, actual, rtol=0, atol=1e-4):
            raise ValueError("offline CPU checkpoint replay changed score")
        checkpoint_paths.append({"seed": seed, "epoch": EPOCHS, "path": str(checkpoint),
                                 "sha256": base.file_hash(checkpoint),
                                 "cpu_reload_score_absolute_difference": float(abs(expected[0] - actual[0]))})
        base._write_jsonl(output / f"diagnostic-cases-seed-{seed}.jsonl", seed_records)
        base._write_json(output / f"seed-{seed}-snapshot.json",
                         {"seed": seed, "snapshots": snapshots[-2:],
                          "checkpoint": checkpoint_paths[-1]})
        del restored, saved, model, optimizer
        if device.type == "mps":
            torch.mps.empty_cache()
    by_view = {}
    for view in ("july_windows", "july_shock_events"):
        subset = [row for row in snapshots if row["view"] == view]
        by_view[view] = {
            "seeds": len(subset),
            "text_h10_mean": float(np.mean([row["mean_selected_h10_return"]["text"] for row in subset])),
            "text_h10_min": float(min(row["mean_selected_h10_return"]["text"] for row in subset)),
            "text_h10_max": float(max(row["mean_selected_h10_return"]["text"] for row in subset)),
            "market_count_h10": subset[0]["mean_selected_h10_return"]["market_count"],
            "shifted_text_h10_mean": float(np.mean([
                row["mean_selected_h10_return"]["shifted_text"] for row in subset])),
            "text_minus_market_h10_mean": float(np.mean([row["text_minus_market_h10"] for row in subset])),
            "text_minus_shifted_h10_mean": float(np.mean([row["text_minus_shifted_h10"] for row in subset])),
            "text_oracle_regret_mean": float(np.mean([row["mean_oracle_regret"]["text"] for row in subset])),
            "exact_top3_mean": float(np.mean([row["exact_top3_cases"] for row in subset])),
        }
    summary = {
        "evidence_level": event.EVIDENCE,
        "purpose": "test whether 16 ex-post shock training windows were insufficient",
        "warning": "Retrospective, previously inspected historical dates and later collector backup; diagnostic views are not independent validation or a trading result.",
        "complete_fixed_seed_trio": seeds == curve.SEEDS,
        "configuration": {"seeds_requested": list(seeds), "fixed_seed_trio": list(curve.SEEDS),
                          "primary_epoch": EPOCHS, "window_sessions": WINDOW,
                          "top_k": 3, "horizon_sessions": 10,
                          "fit_label_end": curve.TRAIN_OUTCOME_END,
                          "diagnostic_entry_start": curve.VALID_ENTRY_START,
                          "diagnostic_outcome_end": curve.VALID_OUTCOME_END,
                          "embargo_sessions": ["2026-06-24", "2026-06-25"],
                          "incomplete_july_tail_policy": "discard final two mature entry sessions per stock",
                          "last_block_lr": 2e-4, "head_lr": 1e-3,
                          "market_control_lr": 0.03, "market_control_steps": 300,
                          "rank_temperature": 0.1, "dropout_disabled": True,
                          "title_max_tokens": 64,
                          "july_use": "diagnostic scoring only; no gradients, scaler fit or epoch selection"},
        "data": {"fit_stock_days": len(fit), "diagnostic_mature_stock_days": len(diagnostic),
                 "fit_windows": len(fit_cases), "diagnostic_disjoint_windows": len(diagnostic_cases),
                 "diagnostic_unused_stock_days": len(diagnostic_unused),
                 "diagnostic_unused_units": diagnostic_unused,
                 "diagnostic_shock_events": len(shock_cases),
                 "fit_only_abs_shock_threshold_p90": threshold,
                 "collector_backup_sha256": base.file_hash(paths["collector_backup"]), **counts},
        "market_scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "market_control_beta": market_beta.tolist(), "market_control_log": market_log,
        "training_log": training_logs, "snapshots": snapshots,
        "by_view": by_view, "checkpoint_paths": checkpoint_paths,
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "script_sha256": base.file_hash(Path(__file__)),
        "model_snapshot_weight_sha256": base.file_hash(base.BASE_MODEL / "pytorch_model.bin"),
        "model_snapshot_tokenizer_sha256": base.file_hash(base.BASE_MODEL / "tokenizer.json"),
        "input_identity": {name: {"path": str(path.resolve()), "sha256": base.file_hash(path)}
                           for name, path in paths.items()},
    }
    base._write_json(output / "summary.json", summary)
    print(json.dumps({"output_dir": str(output), "complete_fixed_seed_trio":
                      summary["complete_fixed_seed_trio"], "by_view": by_view},
                     ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=base.HISTORY / "sampled-post-metadata.jsonl")
    parser.add_argument("--qfq", type=Path, default=base.MARKET / "daily-bars-qfq.csv")
    parser.add_argument("--raw", type=Path, default=base.MARKET / "daily-bars-raw.csv")
    parser.add_argument("--benchmark", type=Path, default=base.MARKET / "benchmark-000300.csv")
    parser.add_argument("--collector-backup", type=Path, default=coverage.BACKUP_DB)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(curve.SEEDS))
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
