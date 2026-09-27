"""Refresh the existing extreme-state report from locally available post text.

Only inference and aggregation: no collection, enrich, training or parameter fit.
The selected input text and cohort are saved so an enriched edition can be
compared with this edition without erasing the original draft.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import market_extreme_events as extremes
from tools import market_signal_multiday as multiday
from tools import market_signal_v1 as v1
from tools import market_validation_v0 as io


def input_key(row):
    return (row["stock_code"], row.get("stock_name", ""), row["model_text"])


def existing_probabilities(additional_sources=()):
    """Exact input memoization of the accepted M12 run, never fuzzy matching."""
    cache = {}
    history = PROJECT / "pipeline-runs/market-signal-v1-20260917"
    meta = {row["sample_id"]: row for row in io.read_jsonl(history / "sampled-post-metadata.jsonl")}
    for row in io.read_jsonl(history / "m12-predictions.jsonl"):
        cache[input_key(meta[row["sample_id"]])] = row
    forward_paths = [PROJECT / "pipeline-runs/market-data-v0-20260916/forward-shadow-predictions.jsonl"]
    forward_paths.extend(sorted((PROJECT / "pipeline-runs").glob("market-signal-v2-forward-*/forward-shadow-predictions.jsonl")))
    for path in forward_paths:
        if path.exists():
            for row in io.read_jsonl(path):
                if row.get("model_text") and row.get("predictions"):
                    cache.setdefault(input_key(row), row)
    for source in additional_sources:
        metadata = {row["sample_id"]: row for row in io.read_jsonl(Path(source) / "sampled-post-metadata.jsonl")}
        for row in io.read_jsonl(Path(source) / "m12-predictions.jsonl"):
            cache[input_key(metadata[row["sample_id"]])] = row
    return cache


def refreshed_predictions(inputs, output, device, reuse_sources=()):
    cache = existing_probabilities(reuse_sources)
    reuse, pending = {}, []
    for row in inputs:
        donor = cache.get(input_key(row))
        if donor is None:
            pending.append(row)
        else:
            reuse[row["sample_id"]] = {**donor, "sample_id": row["sample_id"]}
    print(json.dumps({"selected_posts": len(inputs), "reused_exact_inputs": len(reuse),
                      "new_inference_posts": len(pending)}, ensure_ascii=False), flush=True)
    audit = {"reused_exact_inputs": len(reuse), "new_inference_posts": len(pending), "device": None}
    if pending:
        import torch
        from semantic_model import encoder_m12_hybrid_runtime as hybrid
        from semantic_model import encoder_m2_selected_runtime as selected

        torch.set_num_threads(4)
        chosen = "mps" if device == "auto" and torch.backends.mps.is_available() else ("cpu" if device == "auto" else device)
        runtime = hybrid.M12HybridInferenceRuntime.from_artifacts()
        runtime.selected_runtime.model.to(torch.device(chosen))
        runtime.m12_model.to(torch.device(chosen))
        audit["device"] = chosen
        started = time.monotonic()
        progress = output / "new-m12-predictions.jsonl"
        with progress.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(pending), 64):
                batch = pending[offset:offset + 64]
                records = [selected._validate_inference_record(row) for row in batch]
                result = runtime.predict_records(records)
                for row in result:
                    reuse[row["sample_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                done = min(offset + 64, len(pending))
                if done % 1024 == 0 or done == len(pending):
                    print(f"M12 inference {done}/{len(pending)} {chosen}, {time.monotonic() - started:.1f}s", flush=True)
        audit["inference_seconds"] = time.monotonic() - started
    rows = [reuse[row["sample_id"]] for row in inputs]
    io.write_jsonl(output / "m12-predictions.jsonl", rows)
    return audit


def annotate_report(report_dir, summary):
    stage = summary["text_stage"]
    counts = summary["sampled_text_sources"]
    total = summary["sampled_input_posts"]
    stage_title = "未 enrich 初稿" if stage == "pre-enrich-draft" else f"语料版本 {stage}"
    late = [f"{code} 从 {row['first_publication_date']} 开始" for code, row in summary["post_ranges_by_stock"].items()
            if row["first_publication_date"] > "2026-05-06"]
    note = (f"语料版本：{stage}；数据快照 {summary['snapshot_generated_at']}。本工具不等待或执行 enrich；"
            f"模型、抽样与极端分析口径不变。抽样{total:,}帖，其中已有正文{counts.get('content', 0):,}帖、"
            f"仅标题{counts.get('title', 0):,}帖（{counts.get('title', 0) / total:.1%}）。"
            + ("现有历史覆盖较晚的股票：" + "；".join(late) + "。" if late else "")
            + "使用已有正文，否则标题；没有用缺正文作为删帖条件。各语料版本独立保存，不覆盖前稿。")
    report_path = report_dir / "report.json"
    report = json.loads(report_path.read_text())
    report["text_snapshot"] = summary
    io.write_json(report_path, report)
    # Re-render from results only: annotation never reselects cases or reruns M12.
    cases = io.read_jsonl(report_dir / "event-cases.jsonl")
    html_text = extremes.render_html(report, cases).replace("<h1>最极端群体状态出现后，股票怎样了？</h1>",
                  "<h1>最极端群体状态与后续行情 · " + extremes.html.escape(stage_title)
                  + "</h1><p class=\"note\">" + extremes.html.escape(note) + "</p>")
    (report_dir / "REPORT.html").write_text(html_text, encoding="utf-8")
    (report_dir / "REPORT.md").write_text("# " + stage_title + "\n\n" + note + "\n\n"
                                          + extremes.render_markdown(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-dir", type=Path, default=PROJECT / "pipeline-runs/market-data-backfill-20260928")
    parser.add_argument("--collector-db", type=Path, default=v1.DEFAULT_COLLECTOR_DB)
    parser.add_argument("--targets", type=Path, default=v1.DEFAULT_TARGETS)
    parser.add_argument("--calendar-json", type=Path, default=PROJECT / "pipeline-runs/market-signal-v2-forward-20260924/trading-calendar-dates.json")
    parser.add_argument("--start-date", default="2026-05-04")
    parser.add_argument("--text-stage", default="pre-enrich-draft")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--reuse-source", type=Path, action="append", default=[], help="additional known M12 source directory; identical inputs only")
    parser.add_argument("--post-cap", type=int, default=16)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30, 60])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new output directory: {output}")
    source = output / "source"
    source.mkdir(parents=True)
    observed = v1.benchmark_calendar(args.market_dir / "benchmark-000300.csv")
    future = json.loads(args.calendar_json.read_text())["trade_dates"]
    calendar = sorted(set(observed + future))
    # Calendar-only file: upcoming session dates have no fabricated prices.
    calendar_path = source / "entry-calendar-dates.csv"
    with calendar_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trade_date"])
        writer.writeheader()
        writer.writerows({"trade_date": day} for day in calendar)
    inference, metadata, summary = v1.select_panel_posts(
        collector_db=args.collector_db, benchmark_path=calendar_path,
        targets_path=args.targets, post_cap=args.post_cap,
    )
    metadata = [row for row in metadata if row["signal_date"] >= args.start_date]
    selected_ids = {row["sample_id"] for row in metadata}
    inference = [row for row in inference if row["sample_id"] in selected_ids]
    source_counts = Counter(row["text_source"] for row in metadata)
    ranges = {}
    for code in sorted({row["stock_code"] for row in metadata}):
        rows = [row for row in metadata if row["stock_code"] == code]
        ranges[code] = {"first_publication_date": min(row["signal_date"] for row in rows),
                        "last_publication_date": max(row["signal_date"] for row in rows),
                        "sampled_posts": len(rows), "body_posts": sum(row["text_source"] == "content" for row in rows)}
    summary.update({"text_stage": args.text_stage, "snapshot_generated_at": datetime.now().astimezone().isoformat(),
                    "requested_post_start": args.start_date, "sampled_text_sources": dict(source_counts),
                    "sampled_input_posts": len(inference), "post_ranges_by_stock": ranges,
                    "text_policy": "existing normalized content, otherwise title; unchanged from prior report",
                    "enrich_invoked": False, "model_retrained": False})
    io.write_jsonl(source / "m12-inference-input.jsonl", inference)
    io.write_jsonl(source / "sampled-post-metadata.jsonl", metadata)
    io.write_json(source / "input-summary.json", summary)
    audit = refreshed_predictions(inference, source, args.device, args.reuse_source)
    daily_args = SimpleNamespace(history_metadata=source / "sampled-post-metadata.jsonl",
                                 history_predictions=source / "m12-predictions.jsonl", post_cap=args.post_cap,
                                 early_forward="", forward_dir=[], forward_glob="")
    daily, input_audit = multiday.prepare_daily_states(daily_args, calendar)
    io.write_jsonl(source / "daily-states.jsonl", daily)
    config = {"history_metadata": str(daily_args.history_metadata), "history_predictions": str(daily_args.history_predictions),
              "qfq_bars": str((args.market_dir / "daily-bars-qfq.csv").resolve()),
              "raw_bars": str((args.market_dir / "daily-bars-raw.csv").resolve()),
              "benchmark_bars": str((args.market_dir / "benchmark-000300.csv").resolve()),
              "post_cap": args.post_cap, "early_forward": "", "forward_dir": [], "forward_glob": "",
              "text_stage": args.text_stage, "input_summary": str(source / "input-summary.json")}
    io.write_json(source / "run-config.json", config)
    io.write_json(source / "inference-summary.json", {**audit, **input_audit})
    report_dir = output / "analysis"
    extremes.main(["--source-run", str(source), "--count", str(args.count), "--horizons",
                   *map(str, args.horizons), "--as-of", "latest", "--output-dir", str(report_dir)])
    annotate_report(report_dir, summary)
    io.write_json(output / "refresh-summary.json", {"input": summary, "inference": audit,
                  "daily_states": len(daily), "source_run": str(source), "report_dir": str(report_dir)})
    print(json.dumps({"report_dir": str(report_dir), "selected_posts": len(inference),
                      "daily_states": len(daily), "text_stage": args.text_stage}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
