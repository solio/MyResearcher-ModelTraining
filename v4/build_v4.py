#!/usr/bin/env python3
"""Build the v4.x local research report data.

Usage:
    python3 v4/build_v4.py
    python3 v4/build_v4.py --check
    python3 v4/build_v4.py --output static/v4-data.json

The build is read-only against the collector database and the frozen market
CSVs.  It writes aggregate data only; no post title, content, author id or URL
is included in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v4.lib.analysis import (  # noqa: E402
    build_observation,
    build_prediction,
    build_primary,
    build_stock_days,
    build_training,
)
from v4.lib.loaders import (  # noqa: E402
    aggregate_posts,
    author_share,
    compile_vocab,
    load_json,
    load_prices,
    sha256_file,
    stocks_config,
)
from v4.lib.timebase import index_of, next_trading_day, regime_series  # noqa: E402


def latest_complete_info_day(aggregate: dict, calendar: list[str], cutoff: str) -> str | None:
    """Last trading day whose 15:00 information set is fully collected."""
    watermark = aggregate.get("watermark") or ""
    if not watermark:
        return None
    candidates = []
    for i, day in enumerate(calendar):
        if i + 1 >= len(calendar):
            continue
        if watermark >= f"{day}T{cutoff}":
            candidates.append(day)
    return max(candidates) if candidates else None


def build(config_path: Path) -> dict:
    config = load_json(config_path)
    meta_stocks = stocks_config(ROOT / "v4" / "stocks.json")
    stocks = {s["code"] for s in meta_stocks}
    evidence = load_json(ROOT / "v4" / "external-evidence.json")
    lineage = load_json(ROOT / "v4" / "lineage.json")
    hypotheses = load_json(ROOT / "v4" / "hypotheses.json")
    reference = load_json(ROOT / "v4" / "reference" / "k3_h1_v34_events.json")
    protocol = config["protocol"]

    bars, calendar, benchmark = load_prices(config)
    vocab = compile_vocab(ROOT / "v4" / "vocab")
    aggregate = aggregate_posts(config, calendar, stocks, vocab)
    latest = latest_complete_info_day(aggregate, calendar, protocol["info_cutoff"])
    if latest is None:
        raise SystemExit("no complete information day; check collector watermark and trading calendar")
    market_latest = calendar[-1]
    next_day = next_trading_day(calendar, latest)
    if next_day is None:
        raise SystemExit("no next trading day for the latest complete information day")

    regimes = regime_series(benchmark.get("000300", {}), calendar, 200)
    series_map = {}
    for stock in sorted(stocks):
        if stock not in bars:
            continue
        series_map[stock] = build_stock_days(stock, aggregate["cells"], bars, calendar, latest, protocol)

    primary = build_primary(series_map, protocol)
    training = build_training(
        series_map, primary, regimes, protocol, reference, evidence
    )
    training["lineage"] = lineage
    training["hypotheses"] = hypotheses

    auth = author_share(aggregate["author_counter"], 0.01)
    observation = build_observation(
        series_map,
        aggregate["cells"],
        bars,
        calendar,
        latest,
        market_latest,
        regimes,
        benchmark,
        meta_stocks,
        aggregate,
        auth,
        protocol,
    )
    prediction = build_prediction(
        series_map,
        bars,
        calendar,
        latest,
        regimes,
        primary,
        training,
        observation,
        protocol,
        meta_stocks,
    )

    posts_after = sum(cell["n"] for (stock, day), cell in aggregate["cells"].items() if day > latest)
    latest_posts = sum(cell["n"] for (stock, day), cell in aggregate["cells"].items() if day == latest)
    first_dates = [v[0] for v in aggregate["first_last"].values() if v]
    last_dates = [v[1] for v in aggregate["first_last"].values() if v]
    warnings = []
    if market_latest != latest:
        warnings.append(
            f"帖子的完整信息日只到 {latest}；{market_latest} 仅有行情，不能作为信号日。"
        )
    if latest_posts == 0:
        warnings.append(f"{latest} 的信息日没有帖子，当前状态可能是采集缺口而不是真实安静。")
    if aggregate["content_nonempty"] < aggregate["total"] * 0.2:
        warnings.append("绝大多数记录只有标题；正文缺失，无法做正文语义复核。")

    collector_sha = sha256_file(config["paths"]["collector"])
    input_paths = {
        "collector": config["paths"]["collector"],
        "bars_main": config["paths"]["bars_main"],
        "bars_patch": config["paths"]["bars_patch"],
        "benchmark": config["paths"]["benchmark"],
        "k3_panel": config["paths"]["k3_panel"],
        "k3_forward": config["paths"]["k3_forward"],
        "config": str(config_path),
        "stocks": str(ROOT / "v4" / "stocks.json"),
        "external_evidence": str(ROOT / "v4" / "external-evidence.json"),
        "hypotheses": str(ROOT / "v4" / "hypotheses.json"),
        "lineage": str(ROOT / "v4" / "lineage.json"),
        "k3_reference_events": str(ROOT / "v4" / "reference" / "k3_h1_v34_events.json"),
    }
    inputs = []
    for key, path in input_paths.items():
        inputs.append(
            {
                "id": key,
                "path": path,
                "sha256": collector_sha if key == "collector" else sha256_file(path),
                "exists": Path(path).is_file(),
            }
        )
    fingerprint_basis = json.dumps(
        {
            "inputs": [(i["id"], i["sha256"]) for i in inputs],
            "posts_total": aggregate["total"],
            "watermark": aggregate["watermark"],
            "stocks": len(stocks),
            "vocab": [(v["file"], v["sha256"]) for v in vocab["files"]],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(fingerprint_basis.encode("utf-8")).hexdigest()

    data = {
        "schema_version": "v4-data/1.0.0",
        "report": config["report"],
        "as_of": {
            "info_day": latest,
            "info_cutoff": f"{latest}T{protocol['info_cutoff']}+08:00",
            "post_watermark": aggregate["watermark"],
            "bars_through": market_latest,
            "benchmark_through": max(benchmark.get("000300", {})) if benchmark.get("000300") else None,
            "next_trading_day": next_day,
            "market_as_of": market_latest,
            "signal_day_complete": True,
            "market_context_complete": market_latest in benchmark.get("000300", {}),
            "stale_info_days": (index_of(calendar, market_latest) or 0) - (index_of(calendar, latest) or 0),
            "warnings": warnings,
        },
        "meta": {
            "posts_total": aggregate["total"],
            "posts_used_for_asof": aggregate["total"] - posts_after,
            "posts_after_cutoff": posts_after,
            "latest_info_day_posts": latest_posts,
            "stocks": len(series_map),
            "stocks_with_posts": sum(1 for s in aggregate["first_last"]),
            "authors": len(aggregate["author_counter"]),
            "content_nonempty": aggregate["content_nonempty"],
            "content_nonempty_pct": round(aggregate["content_nonempty"] / aggregate["total"] * 100, 2) if aggregate["total"] else None,
            "published_min": min(aggregate["first_last"].values(), key=lambda v: v[0])[0] if aggregate["first_last"] else None,
            "published_max": max(aggregate["first_last"].values(), key=lambda v: v[1])[1] if aggregate["first_last"] else None,
            "data_fingerprint": fingerprint,
            "inputs": inputs,
            "vocab": vocab["files"],
            "reference": {
                "panel_sha256": sha256_file(config["paths"]["k3_panel"]),
                "panel_mtime": datetime.fromtimestamp(Path(config["paths"]["k3_panel"]).stat().st_mtime).isoformat() if Path(config["paths"]["k3_panel"]).is_file() else None,
                "forward_tracking_sha256": sha256_file(config["paths"]["k3_forward"]),
                "events": len(reference),
            },
        },
        "ui": {
            "default_tab": "observation",
            "tabs": [
                {"id": "observation", "label": "观察", "version": "4.0", "hash": "#observation"},
                {"id": "training", "label": "训练", "version": "4.1", "hash": "#training"},
                {"id": "prediction", "label": "预测", "version": "4.2", "hash": "#prediction"},
            ],
            "disclaimer": "本页是本地研究审计，不是投资建议；历史关系未经验证，不可部署。",
            "empty_states": {
                "no_candidate": "当前没有 S∩D 可执行候选；这不是买入建议。",
                "no_forward": "尚无前瞻事件；0 不等于失败。",
                "warmup": "expanding z 尚未达到 60 个有效观测。",
                "no_breadth": "近 250 个交易日没有足够的 S∩D 广度观测。",
                "missing_bars": "行情缺失，不插值。",
            },
        },
        "observation": observation,
        "training": training,
        "prediction": prediction,
    }

    # Data quality warnings are assembled here because they depend on the
    # observation health table.
    low_coverage = [s for s in observation["stocks"] if s["coverage_pct"] is not None and s["coverage_pct"] < 90]
    observation["data_quality"] = {
        "posts_total": aggregate["total"],
        "posts_used_for_asof": aggregate["total"] - posts_after,
        "posts_after_cutoff": posts_after,
        "content_nonempty": aggregate["content_nonempty"],
        "low_coverage_stocks": len(low_coverage),
        "polluted_days": sum(s["polluted_days"] for s in observation["stocks"]),
        "warnings": warnings
        + (
            [f"{len(low_coverage)} 只股票覆盖率 <90%，其个股统计只作描述。"]
            if low_coverage
            else []
        )
        + ["K3 健康表基于 377,163 帖快照；当前 613,985 帖，部分历史断档可能已被回填，不能逐项对比。"],
    }
    return data


def check_data(data: dict) -> list[str]:
    errors = []
    if data.get("schema_version") != "v4-data/1.0.0":
        errors.append("unexpected schema_version")
    if data["prediction"]["headline"]["candidate_count"] != len(data["prediction"]["candidates"]):
        errors.append("candidate_count does not match candidates")
    spec = data["training"]["specification_curve"]
    if len(spec["points"]) != spec["n_tests"]:
        errors.append("specification_curve length != n_tests")
    if spec["n_tests"] != 48:
        errors.append("n_tests should be 48")
    for point in spec["points"]:
        if point["n_events"] < 3 and point.get("t_cross") is not None:
            errors.append(f"{point['id']}: t shown with fewer than 3 events")
    for event in data["prediction"]["forward_ledger"]:
        if event["status"] in {"pending_entry", "pending_exit"} and event.get("gross_pp") is not None:
            errors.append(f"pending event has a return: {event['event_id']}")
    if data["meta"]["posts_total"] <= 0:
        errors.append("posts_total is empty")
    if data["as_of"]["info_day"] > data["as_of"]["market_as_of"]:
        errors.append("info day is after market day")
    forbidden = {"title", "content", "author_id", "author_name"}
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in forbidden:
                    errors.append(f"forbidden key at {path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
    walk(data)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the v4 observation/training/prediction report data")
    parser.add_argument("--config", type=Path, default=ROOT / "v4" / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "static" / "v4-data.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "v4" / "out" / "manifest.json")
    parser.add_argument("--check", action="store_true", help="validate the existing JSON instead of building")
    args = parser.parse_args(argv)

    if args.check:
        if not args.output.is_file():
            print(f"missing {args.output}; run without --check first", file=sys.stderr)
            return 2
        data = load_json(args.output)
        errors = check_data(data)
        if errors:
            print("v4 check failed:", file=sys.stderr)
            for error in errors:
                print(f" - {error}", file=sys.stderr)
            return 1
        print(f"v4 check passed: {args.output} ({args.output.stat().st_size} bytes)")
        return 0

    data = build(args.config)
    errors = check_data(data)
    if errors:
        print("v4 build produced invalid data:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.output.write_text(payload, encoding="utf-8")
    output_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "v4-manifest/1.0.0",
        "generated_at": data["report"]["generated_at"],
        "as_of": data["as_of"],
        "data_fingerprint": data["meta"]["data_fingerprint"],
        "output": str(args.output.relative_to(ROOT)) if str(args.output).startswith(str(ROOT)) else str(args.output),
        "output_sha256": output_sha,
        "output_bytes": len(payload.encode("utf-8")),
        "inputs": data["meta"]["inputs"],
        "vocab": data["meta"]["vocab"],
        "training": {
            "n_events": data["training"]["primary"]["n_events"],
            "n_clusters": data["training"]["primary"]["n_clusters"],
            "n_stocks": data["training"]["primary"]["n_stocks"],
            "mean_pp": data["training"]["primary"]["mean_pp"],
            "t_event": data["training"]["primary"]["t_event"],
            "matched_pairs": data["training"]["matched_controls"]["summary"]["n_pairs"],
            "matched_mean_pp": data["training"]["matched_controls"]["summary"]["mean_diff_pp"],
            "n_specs": data["training"]["specification_curve"]["n_tests"],
        },
        "prediction": {
            "candidates": data["prediction"]["headline"]["candidate_count"],
            "h4_gross_pp": (data["prediction"]["headline"]["h4"] or {}).get("gross_pp"),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} ({len(payload.encode('utf-8'))} bytes, sha {output_sha[:12]}); "
        f"events={data['training']['primary']['n_events']} clusters={data['training']['primary']['n_clusters']} "
        f"candidates={data['prediction']['headline']['candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
