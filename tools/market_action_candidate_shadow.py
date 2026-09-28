"""Replay an isolated Action checkpoint against a frozen post/market snapshot.

This is a retrospective output-sensitivity study, not an as-of trading
backtest: the candidate may have trained on text from the historical period.
The accepted predictions, daily states, market files, and other six heads stay
untouched.  All generated files live in a new output directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import market_extreme_events as extremes
from tools import market_signal_multiday as multiday
from tools import market_signal_v1 as v1
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as io


def swap_action_prediction(original: dict, order: list[str], values: list[float]) -> dict:
    if len(order) != len(values):
        raise ValueError("Action class order and probability vector differ")
    probabilities = dict(zip(order, values))
    head = {"prediction": order[max(range(len(values)), key=values.__getitem__)],
            "probabilities": probabilities}
    return {**original, "predictions": {**original["predictions"], "action_tendency": head}}


def changed_extreme_dates(original: list[dict], candidate: list[dict], *, count: int) -> list[dict]:
    old_by_stock: dict[str, list[dict]] = defaultdict(list)
    new_by_stock: dict[str, list[dict]] = defaultdict(list)
    for row in original:
        old_by_stock[row["stock_code"]].append(row)
    for row in candidate:
        new_by_stock[row["stock_code"]].append(row)
    if set(old_by_stock) != set(new_by_stock):
        raise ValueError("stock universe changed")
    result = []
    for code in sorted(old_by_stock):
        old_dates = {row["entry_date"] for row in extremes.extreme_rows(old_by_stock[code], "action_index", count, "low")}
        new_dates = {row["entry_date"] for row in extremes.extreme_rows(new_by_stock[code], "action_index", count, "low")}
        result.append({"stock_code": code, "stock_name": old_by_stock[code][0]["stock_name"],
                       "old_low_dates": sorted(old_dates), "candidate_low_dates": sorted(new_dates),
                       "common_dates": sorted(old_dates & new_dates)})
    return result


def replay(source: Path, candidate_dir: Path, output: Path, *, device_name: str, batch_size: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new empty output directory: {output}")
    source_config = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    inputs = io.read_jsonl(source / "m12-inference-input.jsonl")
    accepted = io.read_jsonl(source / "m12-predictions.jsonl")
    original_daily = io.read_jsonl(source / "daily-states.jsonl")
    by_id = {str(row["sample_id"]): row for row in accepted}
    if len(by_id) != len(inputs) or {str(row["sample_id"]) for row in inputs} != set(by_id):
        raise ValueError("frozen input and accepted prediction IDs differ")

    from semantic_model import encoder_action_contrastive as action
    from semantic_model import encoder_m1 as m1
    from semantic_model import encoder_m12_human_first_cv as m12
    from semantic_model import encoder_m2_1_softlabel as m21
    from semantic_model import encoder_m2_lert_final as lert
    from semantic_model import encoder_m2_selected_runtime as selected

    _np, torch, AutoModel, AutoTokenizer = m21._load_runtime()
    torch.set_num_threads(4)
    schema, _, _ = m1.load_m1_partitions(m1.ProjectConfig.load(m12.DEFAULT_CONFIG))
    order = list(schema.class_order[action.HEAD])
    frozen = lert._contract_requirements(selected.CONTRACT_PATH)
    config = lert._config(frozen["contract"], schema.class_order)
    model_snapshot, _identity = m21._snapshot(m12.DEFAULT_CACHE)
    tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    parent = torch.load(m12.DEFAULT_SELECTED_ARTIFACT / "unified-runtime-bundle.pt",
                        map_location="cpu", weights_only=False)
    saved = torch.load(candidate_dir / "action-checkpoint.pt", map_location="cpu", weights_only=True)
    if saved.get("head") != action.HEAD:
        raise ValueError("checkpoint is not an Action specialist")
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model = action._make_model(torch, AutoModel, model_snapshot, schema, config, parent, device,
                               two_block=bool(saved.get("two_block", False)))
    model.specialist_blocks[action.HEAD].load_state_dict(saved["specialist_block_state_dict"])
    model.heads[action.HEAD].load_state_dict(saved["head_state_dict"])
    model.eval()

    generated = output / "source"
    generated.mkdir(parents=True)
    shadow_path = generated / "action-shadow-predictions.jsonl"
    old_labels: Counter[str] = Counter()
    candidate_labels: Counter[str] = Counter()
    changed_post_predictions = 0
    started = time.monotonic()
    with shadow_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for offset in range(0, len(inputs), batch_size):
            rows = inputs[offset:offset + batch_size]
            records = [m1.M1Record(str(row["sample_id"]), str(row["stock_code"]),
                                   str(row["stock_name"]), str(row["model_text"]),
                                   {action.HEAD: "NO_ACTION_SIGNAL"}, {action.HEAD: 1.0}) for row in rows]
            batch = action.prior._batch(torch, tokenizer, records, config, schema, device)
            probs = torch.softmax(model(batch["input_ids"], batch["attention_mask"])[action.HEAD], dim=-1)
            for row, vector in zip(rows, probs.detach().cpu().tolist()):
                original = by_id[str(row["sample_id"])]
                shadow = swap_action_prediction(original, order, vector)
                old_labels[str(original["predictions"][action.HEAD]["prediction"])] += 1
                candidate_labels[str(shadow["predictions"][action.HEAD]["prediction"])] += 1
                changed_post_predictions += original["predictions"][action.HEAD]["prediction"] != shadow["predictions"][action.HEAD]["prediction"]
                handle.write(json.dumps(shadow, ensure_ascii=False, allow_nan=False) + "\n")
            if (offset + len(rows)) % 2048 < batch_size or offset + len(rows) == len(inputs):
                print(f"Action-only shadow {offset + len(rows)}/{len(inputs)} {device_name}", flush=True)

    calendar = v1.benchmark_calendar(source / "entry-calendar-dates.csv")
    args = SimpleNamespace(history_metadata=source / "sampled-post-metadata.jsonl",
                           history_predictions=shadow_path, post_cap=int(source_config["post_cap"]),
                           early_forward="", forward_dir=[], forward_glob="")
    shadow_daily, _input_audit = multiday.prepare_daily_states(args, calendar)
    old_by_key = {(row["stock_code"], row["entry_date"]): row for row in original_daily}
    new_by_key = {(row["stock_code"], row["entry_date"]): row for row in shadow_daily}
    if set(old_by_key) != set(new_by_key):
        raise ValueError("stock/date universe changed")
    if any(old_by_key[key]["features"][head] != new_by_key[key]["features"][head]
           for key in old_by_key for head in ("stance_index", "emotion_balance")):
        raise ValueError("non-Action crowd direction changed")
    io.write_jsonl(generated / "daily-states.jsonl", shadow_daily)
    io.write_json(generated / "run-config.json", {**source_config,
                  "history_predictions": str(shadow_path),
                  "history_metadata": str(args.history_metadata)})

    report_dir = output / "action-extreme-descriptive"
    extremes.main(["--source-run", str(generated), "--metrics", "action_index", "--count", "3",
                   "--horizons", "10", "20", "--as-of", "latest", "--output-dir", str(report_dir)])
    candidate_report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    observed_calendar = sorted({row["trade_date"] for row in io.read_csv(source_config["benchmark_bars"])})
    qfq = snapshot._read_stock_bars(source_config["qfq_bars"], adjustment="qfq")
    raw = snapshot._read_stock_bars(source_config["raw_bars"], adjustment="raw")
    _, _, old_summary = extremes.build_study(original_daily, observed_calendar, qfq, raw,
                                              count=3, horizons=[10, 20], metrics=["action_index"])
    def selected_summary(rows: list[dict]) -> list[dict]:
        return [{"horizon": row["horizon"], "low_complete": row["low_complete"],
                 "paired_central_stocks": row["paired_central_stocks"],
                 "paired_low_mean_return": row["paired_central_low_mean_return"],
                 "paired_central_mean_return": row["paired_central_mean_return"],
                 "low_minus_central": row["low_minus_central"]}
                for row in rows if row["scope"] == "full_history"]
    date_changes = changed_extreme_dates(original_daily, shadow_daily, count=3)
    differences = [new_by_key[key]["features"]["action_index"] - old_by_key[key]["features"]["action_index"]
                   for key in sorted(old_by_key)]
    result = {"status": "RETROSPECTIVE_ACTION_ONLY_SHADOW_NOT_TRADING_BACKTEST",
              "source": str(source), "candidate_dir": str(candidate_dir), "device": device_name,
              "posts": len(inputs), "daily_stock_dates": len(original_daily),
              "changed_post_predictions": changed_post_predictions,
              "old_label_distribution": dict(old_labels), "candidate_label_distribution": dict(candidate_labels),
              "mean_absolute_action_index_change": statistics.fmean(abs(value) for value in differences),
              "sign_flipped_daily_states": sum(old_by_key[key]["features"]["action_index"] * new_by_key[key]["features"]["action_index"] < 0 for key in old_by_key),
              "extreme_date_comparison": date_changes,
              "old_descriptive_action_extremes": selected_summary(old_summary),
              "candidate_descriptive_action_extremes": selected_summary(candidate_report["summary"]),
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "limitations": ["candidate trained using text from this historical period",
                              "full-history lowest-three dates selected in hindsight",
                              "no new human semantic truth or forward market outcome in this study"]}
    io.write_json(output / "shadow-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    result = replay(args.source_run.resolve(), args.candidate_dir.resolve(), args.output_dir.resolve(),
                    device_name=args.device, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
