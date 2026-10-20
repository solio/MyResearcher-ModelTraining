"""Causal signal audit and report sections for v4.

The module deliberately keeps two layers separate:

* ``build_stock_days`` produces a full daily panel with strict information
  timing and forward returns.
* ``events_for`` derives event sets for a pre-declared z/drawdown/horizon
  specification.  Every event is entered at the next open and exited at the
  close ``horizon`` sessions later.

No title, content or author identifier is written to the report JSON.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from .stats import (
    bootstrap_cluster_mean,
    bootstrap_mean_by_stock,
    cluster_by_gap,
    cross_stock_mean_and_t,
    mean,
    percentile,
    quantile,
    stdev,
    tstat,
)
from .timebase import (
    forward_return,
    index_of,
    next_trading_day,
    prior_drawdown,
    regime_series,
    same_close_return,
    trailing_return,
)


HORIZON_LABELS = {1: "1 个交易日", 3: "3 个交易日", 5: "5 个交易日"}


def _pp(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def _t(value: list[float] | None) -> float | None:
    return round(tstat(value), 2) if value else None


def build_stock_days(
    stock: str,
    cells: Mapping[tuple[str, str], dict],
    bars: Mapping[str, Mapping[str, dict]],
    calendar: Sequence[str],
    latest: str,
    protocol: dict,
) -> dict:
    """Build one stock's full causal daily panel."""
    stock_bars = bars.get(stock, {})
    info_days = sorted(d for (s, d) in cells if s == stock and d in stock_bars and d <= latest)
    if not info_days:
        return {"code": stock, "days": [], "events": [], "baselines": {}}
    start = info_days[0]
    start_pos = index_of(calendar, start)
    end_pos = index_of(calendar, latest)
    if start_pos is None or end_pos is None:
        return {"code": stock, "days": [], "events": [], "baselines": {}}

    closes = {d: stock_bars[d]["close"] for d in stock_bars}
    volumes = {d: stock_bars[d].get("volume") for d in stock_bars}
    days: list[dict] = []
    for pos in range(start_pos, end_pos + 1):
        day = calendar[pos]
        cell = cells.get((stock, day), {"n": 0, "authors": Counter(), "bear": 0, "excite": 0, "declare": 0})
        n = cell["n"]
        authors = cell["authors"]
        top1 = max(authors.values()) / n if n else None
        eligible = bool(n >= protocol["min_posts"] and top1 is not None and top1 <= protocol["pollution_top1_share"] and day in stock_bars)
        density = cell["bear"] / n if n else None
        dd = prior_drawdown(closes, calendar, day, protocol["dd_window"])
        p5 = trailing_return(closes, calendar, day, 5)
        p60 = trailing_return(closes, calendar, day, 60)
        vol = volumes.get(day)
        hist_vol = [volumes.get(calendar[i]) for i in range(max(0, pos - 59), pos + 1)]
        hist_vol = [v for v in hist_vol if v]
        vol_ratio = vol / statistics.median(hist_vol) if vol and hist_vol else None
        returns = {}
        statuses = {}
        entry_days = {}
        exit_days = {}
        for h in protocol["horizons"]:
            entry, exit_, ret, status = forward_return(stock_bars, calendar, day, h)
            returns[h] = ret
            statuses[h] = status
            entry_days[h] = entry
            exit_days[h] = exit_
        days.append(
            {
                "d": day,
                "pos": pos,
                "n": n,
                "authors": len(authors),
                "top1": top1,
                "eligible": eligible,
                "density": density,
                "excite": cell["excite"] / n if n else None,
                "declare": cell["declare"] / n if n else None,
                "close": stock_bars[day]["close"],
                "open": stock_bars[day]["open"],
                "dd": dd,
                "p5": p5,
                "p60": p60,
                "vol_ratio": vol_ratio,
                "returns": returns,
                "statuses": statuses,
                "entry": entry_days,
                "exit": exit_days,
                "z": None,
                "z_status": "warmup",
                "s": False,
                "deep": dd is not None and dd <= protocol["dd_deep"],
                "s_run": None,
            }
        )

    # Expanding z uses prior eligible days only.
    prior: list[float] = []
    for row in days:
        if row["eligible"] and row["density"] is not None:
            if len(prior) >= protocol["min_prior_obs"]:
                mu = sum(prior) / len(prior)
                sd = stdev(prior)
                row["z"] = (row["density"] - mu) / sd if sd else None
                row["z_status"] = "ok" if row["z"] is not None else "na"
            else:
                row["z_status"] = "warmup"
            prior.append(row["density"])
        elif row["n"]:
            row["z_status"] = "thin_or_polluted"

    # Primary S and F-run (z >= 2.0).  Run counts extreme days inside a
    # <= 5-session cluster; the third and later are not entered.
    prev = None
    run = 0
    for row in days:
        row["s"] = bool(row["eligible"] and row["z"] is not None and row["z"] >= protocol["z_min"])
        if row["s"]:
            run = 1 if prev is None or row["pos"] - prev > protocol["cluster_gap_td"] else run + 1
            prev = row["pos"]
            row["s_run"] = run

    baselines = {}
    for h in protocol["horizons"]:
        vals = [r["returns"][h] for r in days if r["eligible"] and r["returns"][h] is not None]
        baselines[h] = {"mean": mean(vals), "n": len(vals)}
    return {"code": stock, "days": days, "baselines": baselines, "first": start, "last": latest}


def events_for(
    series: dict,
    z_gate: float,
    dd_gate: float | None,
    horizon: int,
    gap: int,
    include_pending: bool = False,
    cutoff: str | None = None,
) -> list[dict]:
    """Derive an event set for one specification and assign clusters."""
    events: list[dict] = []
    prev = None
    run = 0
    for row in series["days"]:
        is_s = bool(row["eligible"] and row["z"] is not None and row["z"] >= z_gate)
        if is_s:
            run = 1 if prev is None or row["pos"] - prev > gap else run + 1
            prev = row["pos"]
            row["_spec_run"] = run
            if cutoff is not None and row["d"] >= cutoff:
                continue
            if run > 2:
                continue
            if dd_gate is not None and (row["dd"] is None or row["dd"] > dd_gate):
                continue
            ret = row["returns"].get(horizon)
            status = row["statuses"].get(horizon, "MISSING")
            if ret is None and not include_pending:
                continue
            events.append(
                {
                    "stock": series["code"],
                    "info_day": row["d"],
                    "entry": row["entry"].get(horizon),
                    "exit": row["exit"].get(horizon),
                    "ret": ret,
                    "status": status,
                    "z": row["z"],
                    "dd": row["dd"],
                    "p5": row["p5"],
                    "density": row["density"],
                    "n": row["n"],
                    "run": run,
                    "deep": bool(row["dd"] is not None and row["dd"] <= -0.20),
                    "_index": row["pos"],
                    "horizon": horizon,
                }
            )
        else:
            row["_spec_run"] = None
    # Cluster assignment within stock.
    clusters: list[list[dict]] = []
    for event in events:
        if clusters and event["_index"] - clusters[-1][-1]["_index"] <= gap:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    for cid, cluster in enumerate(clusters):
        for event in cluster:
            event["cluster"] = cid
    return events


def _baseline_for(series: dict, horizon: int) -> float | None:
    return series["baselines"].get(horizon, {}).get("mean")


def _event_excess(series: dict, events: list[dict], horizon: int) -> list[float]:
    base = _baseline_for(series, horizon)
    if base is None:
        return []
    return [e["ret"] - base for e in events if e["ret"] is not None]


def build_primary(series_map: dict[str, dict], protocol: dict) -> dict:
    primary: dict[str, list[dict]] = {}
    for stock, series in series_map.items():
        primary[stock] = events_for(
            series,
            protocol["z_min"],
            protocol["dd_deep"],
            protocol["primary_horizon"],
            protocol["cluster_gap_td"],
        )
    all_events = [e for events in primary.values() for e in events]
    all_events.sort(key=lambda e: (e["info_day"], e["stock"]))
    return {"by_stock": primary, "all": all_events}


def per_stock_stats(series_map: dict[str, dict], primary: dict, horizon: int) -> list[dict]:
    rows = []
    for stock, series in series_map.items():
        events = [e for e in primary["by_stock"].get(stock, []) if e["ret"] is not None]
        base = _baseline_for(series, horizon)
        vals = [e["ret"] - base for e in events] if base is not None else []
        clusters = defaultdict(list)
        for e in events:
            clusters[e["cluster"]].append(e["ret"] - base if base is not None else e["ret"])
        cluster_means = [mean(v) for v in clusters.values() if v]
        rows.append(
            {
                "code": stock,
                "n_events": len(events),
                "n_clusters": len(clusters),
                "baseline_pp": _pp(base),
                "mean_pp": _pp(mean([e["ret"] for e in events])) if events else None,
                "excess_pp": _pp(mean(vals)) if vals else None,
                "t_event": round(tstat(vals), 2) if tstat(vals) is not None else None,
                "cluster_excess_pp": _pp(mean(cluster_means)) if cluster_means else None,
            }
        )
    return rows


def specification_curve(series_map: dict[str, dict], protocol: dict, cutoff: str | None = None) -> dict:
    points = []
    for z in protocol["spec_z"]:
        for dd in protocol["spec_dd"]:
            for h in protocol["spec_h"]:
                values_by_stock = {}
                n_events = 0
                n_clusters = 0
                pooled = []
                cluster_vals = []
                for stock, series in series_map.items():
                    events = [
                        e
                        for e in events_for(series, z, dd, h, protocol["cluster_gap_td"], cutoff=cutoff)
                        if e["ret"] is not None
                    ]
                    if not events:
                        continue
                    base = _baseline_for(series, h)
                    if base is None:
                        continue
                    vals = [e["ret"] - base for e in events]
                    values_by_stock[stock] = mean(vals)
                    pooled.extend(vals)
                    n_events += len(events)
                    n_clusters += len({e["cluster"] for e in events})
                    clusters = defaultdict(list)
                    for e in events:
                        clusters[e["cluster"]].append(e["ret"] - base)
                    cluster_vals.extend(mean(v) for v in clusters.values() if v)
                cross_mean, cross_t, n_stocks = cross_stock_mean_and_t(values_by_stock)
                points.append(
                    {
                        "id": f"z{z:g}_dd{int(dd*100)}_h{h}",
                        "z": z,
                        "dd_pp": int(dd * 100),
                        "horizon": h,
                        "n_events": n_events,
                        "n_clusters": n_clusters,
                        "n_stocks": n_stocks,
                        "mean_pp": _pp(cross_mean),
                        "t_cross": round(cross_t, 2) if cross_t is not None else None,
                        "event_mean_pp": _pp(mean(pooled)) if pooled else None,
                        "event_t": round(tstat(pooled), 2) if tstat(pooled) is not None else None,
                        "cluster_t": round(tstat(cluster_vals), 2) if tstat(cluster_vals) is not None else None,
                        "is_primary": abs(z - protocol["z_min"]) < 1e-9
                        and abs(dd - protocol["dd_deep"]) < 1e-9
                        and h == protocol["primary_horizon"],
                    }
                )
    primary = next((p for p in points if p["is_primary"]), None)
    return {"n_tests": len(points), "primary": primary, "points": points}


def _matched_controls(series_map: dict[str, dict], primary_events: list[dict], protocol: dict) -> dict:
    pairs = []
    used = defaultdict(set)
    for event in primary_events:
        if event["ret"] is None:
            continue
        series = series_map[event["stock"]]
        if series["baselines"][protocol["primary_horizon"]]["mean"] is None:
            continue
        best = None
        for row in series["days"]:
            if row["d"] == event["info_day"] or row["s"]:
                continue
            if protocol.get("training_cutoff") and row["d"] >= protocol["training_cutoff"]:
                continue
            if row["returns"][protocol["primary_horizon"]] is None or row["dd"] is None:
                continue
            if row["dd"] > protocol["dd_deep"]:
                continue
            if abs(row["dd"] - event["dd"]) > 0.05:
                continue
            if event["p5"] is not None and row["p5"] is not None and abs(row["p5"] - event["p5"]) > 0.03:
                continue
            if abs(row["pos"] - event["_index"]) <= protocol["cluster_gap_td"]:
                continue
            if abs(row["pos"] - event["_index"]) > protocol.get("match_max_gap_td", 60):
                continue
            # Avoid controls within five sessions of any extreme day.
            if any(abs(row["pos"] - e["_index"]) <= protocol["cluster_gap_td"] for e in primary_events if e["stock"] == event["stock"]):
                continue
            if row["d"] in used[event["stock"]]:
                continue
            score = abs(row["dd"] - event["dd"]) + (abs(row["p5"] - event["p5"]) if event["p5"] is not None and row["p5"] is not None else 0)
            if best is None or score < best[0]:
                best = (score, row)
        if best is None:
            continue
        control = best[1]
        used[event["stock"]].add(control["d"])
        diff = event["ret"] - control["returns"][protocol["primary_horizon"]]
        pairs.append(
            {
                "stock": event["stock"],
                "signal_date": event["info_day"],
                "control_date": control["d"],
                "signal_return_pp": _pp(event["ret"]),
                "control_return_pp": _pp(control["returns"][protocol["primary_horizon"]]),
                "diff_pp": _pp(diff),
                "dd_signal_pp": _pp(event["dd"]),
                "dd_control_pp": _pp(control["dd"]),
                "p5_signal_pp": _pp(event["p5"]),
                "p5_control_pp": _pp(control["p5"]),
            }
        )
    diffs = [p["diff_pp"] / 100 for p in pairs if p["diff_pp"] is not None]
    by_stock = defaultdict(list)
    for p in pairs:
        if p["diff_pp"] is not None:
            by_stock[p["stock"]].append(p["diff_pp"] / 100)
    boot_mean, boot_lo, boot_hi = bootstrap_mean_by_stock(
        by_stock, protocol["bootstrap_reps"], protocol["bootstrap_seed"]
    )
    enough = len(pairs) >= 3
    summary = {
        "n_pairs": len(pairs),
        "n_stocks": len(by_stock),
        "mean_diff_pp": _pp(mean(diffs)) if diffs else None,
        "t": round(tstat(diffs), 2) if enough and tstat(diffs) is not None else None,
        "ci_low_pp": _pp(boot_lo) if enough else None,
        "ci_high_pp": _pp(boot_hi) if enough else None,
        "bootstrap_seed": protocol["bootstrap_seed"],
        "bootstrap_reps": protocol["bootstrap_reps"],
        "note": "同股匹配：|dd差|<=5pp、|过去5日差|<=3pp、非极端、间隔 5–60 个交易日、一对一；n<3 不报 t/CI。",
    }
    if not enough:
        summary["note"] += " 当前匹配对不足 3 对，不能据此判断情绪增量。"
    return {"pairs": pairs, "summary": summary}


def _quadrants(series_map: dict[str, dict], protocol: dict, cutoff: str | None = None) -> dict:
    h = protocol["primary_horizon"]
    cells: dict[str, list[float]] = defaultdict(list)
    per_stock: dict[str, dict[str, float]] = defaultdict(dict)
    for stock, series in series_map.items():
        for row in series["days"]:
            ret = row["returns"][h]
            if not row["eligible"] or ret is None:
                continue
            if cutoff is not None and row["d"] >= cutoff:
                continue
            s = bool(row["z"] is not None and row["z"] >= protocol["z_min"])
            d = bool(row["dd"] is not None and row["dd"] <= protocol["dd_deep"])
            key = ("S+" if s else "S-") + ("D+" if d else "D-")
            cells[key].append(ret)
            per_stock[stock].setdefault(key, []).append(ret)
    out = {}
    for key in ("S+D+", "S+D-", "S-D+", "S-D-"):
        vals = cells.get(key, [])
        out[key] = {
            "label": {"S+D+": "极端情绪 × 深回撤", "S+D-": "极端情绪 × 非深回撤", "S-D+": "非极端 × 深回撤", "S-D-": "非极端 × 非深回撤"}[key],
            "n_days": len(vals),
            "n_stocks": sum(1 for stock in per_stock if key in per_stock[stock]),
            "mean_pp": _pp(mean(vals)) if vals else None,
            "median_pp": _pp(quantile(vals, 0.5)) if vals else None,
        }
    contrasts = {}
    for name, a, b in (
        ("S+D+ vs S-D+", "S+D+", "S-D+"),
        ("S+D- vs S-D-", "S+D-", "S-D-"),
    ):
        vals = {}
        for stock, by_cell in per_stock.items():
            if a in by_cell and b in by_cell:
                vals[stock] = mean(by_cell[a]) - mean(by_cell[b])
        m, t, n = cross_stock_mean_and_t(vals)
        contrasts[name] = {"mean_pp": _pp(m), "t": round(t, 2) if t is not None else None, "n_stocks": n}
    return {"cells": out, "contrasts": contrasts, "note": "单元格是日度池化，股票构成不同；对比是逐股先平均再跨股，不能把两个单元格均值直接相减。日度观测有重叠，推断以匹配对照和簇统计为准。"}


def _regime_for_day(regimes: dict[str, dict], day: str) -> str:
    return regimes.get(day, {}).get("label", "unknown")


def _regime_condition(series_map: dict[str, dict], primary_events: list[dict], regimes: dict[str, dict], protocol: dict) -> list[dict]:
    h = protocol["primary_horizon"]
    groups = defaultdict(list)
    for event in primary_events:
        if event["ret"] is None:
            continue
        series = series_map[event["stock"]]
        base = _baseline_for(series, h)
        if base is None:
            continue
        groups[_regime_for_day(regimes, event["info_day"])].append(event["ret"] - base)
    out = []
    for label in ("risk_on", "risk_off", "mixed", "unknown"):
        vals = groups.get(label, [])
        out.append(
            {
                "regime": label,
                "n_events": len(vals),
                "n_stocks": len({e["stock"] for e in primary_events if e["ret"] is not None and _regime_for_day(regimes, e["info_day"]) == label}),
                "mean_pp": _pp(mean(vals)) if vals else None,
                "t": _t(vals),
            }
        )
    return out


def _walk_forward(series_map: dict[str, dict], primary_events: list[dict], protocol: dict) -> dict:
    h = protocol["primary_horizon"]
    cost = protocol["cost_one_way"] * 2
    rows = []
    cumulative_gross = 0.0
    cumulative_net = 0.0
    cluster_rows = []
    by_cluster = defaultdict(list)
    for event in sorted(primary_events, key=lambda e: (e["info_day"], e["stock"])):
        if event["ret"] is None:
            continue
        base = _baseline_for(series_map[event["stock"]], h)
        if base is None:
            continue
        excess = event["ret"] - base
        cumulative_gross += excess
        cumulative_net += excess - cost
        rows.append(
            {
                "d": event["info_day"],
                "stock": event["stock"],
                "gross_pp": _pp(excess),
                "net_pp": _pp(excess - cost),
                "cum_gross_pp": _pp(cumulative_gross),
                "cum_net_pp": _pp(cumulative_net),
                "n": len(rows) + 1,
            }
        )
        by_cluster[(event["stock"], event["cluster"])].append(excess)
    cum = 0.0
    for (stock, cid), vals in sorted(by_cluster.items(), key=lambda item: min(e["info_day"] for e in primary_events if e["stock"] == item[0][0] and e["cluster"] == item[0][1])):
        cm = mean(vals)
        cum += cm - cost
        cluster_rows.append({"stock": stock, "cluster": cid, "mean_pp": _pp(cm), "cum_net_pp": _pp(cum)})
    vals = [r["net_pp"] / 100 for r in rows]
    return {
        "series": rows,
        "clusters": cluster_rows,
        "summary": {
            "n_events": len(rows),
            "n_clusters": len(by_cluster),
            "mean_net_pp": _pp(mean(vals)) if vals else None,
            "t": round(tstat(vals), 2) if tstat(vals) is not None else None,
            "cost_bps_roundtrip": protocol["cost_one_way"] * 2 * 10000,
        },
        "note": "逐事件等权累计，不复合；扣除单边 10bp、双边 20bp 的费用假设，不含滑点/冲击成本。",
    }


def build_training(
    series_map: dict[str, dict],
    primary: dict,
    regimes: dict[str, dict],
    protocol: dict,
    reference_events: list[dict],
    external_evidence: list[dict],
) -> dict:
    h = protocol["primary_horizon"]
    cutoff = protocol.get("training_cutoff")
    all_events = [e for e in primary["all"] if cutoff is None or e["info_day"] < cutoff]
    complete = [e for e in all_events if e["ret"] is not None]
    training_primary = {
        "by_stock": {
            stock: [e for e in events if cutoff is None or e["info_day"] < cutoff]
            for stock, events in primary["by_stock"].items()
        },
        "all": all_events,
    }
    per_stock = per_stock_stats(series_map, training_primary, h)
    spec = specification_curve(series_map, protocol, cutoff)
    matched = _matched_controls(series_map, complete, protocol)
    quadrants = _quadrants(series_map, protocol, cutoff)
    regime_condition = _regime_condition(series_map, complete, regimes, protocol)
    walk = _walk_forward(series_map, complete, protocol)

    # Primary event-level summary.
    base_values = []
    for event in complete:
        base = _baseline_for(series_map[event["stock"]], h)
        if base is not None:
            base_values.append(event["ret"] - base)
    cluster_vals = defaultdict(list)
    for event in complete:
        base = _baseline_for(series_map[event["stock"]], h)
        if base is not None:
            cluster_vals[(event["stock"], event["cluster"])].append(event["ret"] - base)
    cluster_means = [mean(v) for v in cluster_vals.values() if v]
    boot_mean, boot_lo, boot_hi = bootstrap_cluster_mean(
        list(cluster_vals.values()), protocol["bootstrap_reps"], protocol["bootstrap_seed"]
    )
    stock_vals = [r["excess_pp"] / 100 for r in per_stock if r.get("excess_pp") is not None]
    primary_summary = {
        "n_events": len(complete),
        "n_clusters": len(cluster_means),
        "n_stocks": len({e["stock"] for e in complete}),
        "mean_pp": _pp(mean(base_values)) if base_values else None,
        "t_event": round(tstat(base_values), 2) if tstat(base_values) is not None else None,
        "cluster_mean_pp": _pp(mean(cluster_means)) if cluster_means else None,
        "cluster_t": round(tstat(cluster_means), 2) if tstat(cluster_means) is not None else None,
        "stock_equal_mean_pp": _pp(mean(stock_vals)) if stock_vals else None,
        "stock_equal_t": round(tstat(stock_vals), 2) if tstat(stock_vals) is not None else None,
        "n_stocks_equal": len(stock_vals),
        "ci_low_pp": _pp(boot_lo),
        "ci_high_pp": _pp(boot_hi),
        "boot_mean_pp": _pp(boot_mean),
        "training_cutoff": cutoff,
        "note": "H-4 已作为前瞻排除；mean_pp/t_event 为逐事件等权；stock_equal 为逐股先平均再跨股。",
    }

    waterfall = build_waterfall(
        series_map,
        reference_events,
        primary_summary,
        matched["summary"],
        protocol,
    )

    return {
        "protocol": {
            "id": "v4.1-protocol-20261020",
            "density": "bearw.v2 + 3 个个股黑称",
            "z": "expanding prior-valid, min 60",
            "z_min": protocol["z_min"],
            "dd_window": protocol["dd_window"],
            "dd_deep_pp": protocol["dd_deep"] * 100,
            "min_posts": protocol["min_posts"],
            "pollution_top1_share": protocol["pollution_top1_share"],
            "cluster_gap_td": protocol["cluster_gap_td"],
            "training_cutoff": cutoff,
            "match_max_gap_td": protocol.get("match_max_gap_td", 60),
            "entry": "T+1 open",
            "exit_h5": "T+6 close",
            "primary_horizon": h,
            "cost_bps_roundtrip": protocol["cost_one_way"] * 2 * 10000,
            "bootstrap": {"reps": protocol["bootstrap_reps"], "seed": protocol["bootstrap_seed"], "method": "按股票/事件簇重采样"},
            "n_specs": spec["n_tests"],
            "epistemic": "retrospective_contaminated",
            "note": "H-4（2026-10-09）是预声明前瞻，已从训练统计中排除，只出现在前瞻台账。研究者已看过 K3 结果；本页训练段不是样本外。",
        },
        "primary": primary_summary,
        "per_stock": per_stock,
        "specification_curve": spec,
        "matched_controls": matched,
        "quadrants": quadrants,
        "regime_condition": regime_condition,
        "walk_forward": walk,
        "waterfall": waterfall,
        "external_evidence": external_evidence,
    }


def build_waterfall(
    series_map: dict[str, dict],
    reference_events: list[dict],
    primary_summary: dict,
    matched_summary: dict,
    protocol: dict,
) -> list[dict]:
    """Same-event clock comparisons plus the causal event set.

    The first row is the published K3 number; the next two rows use the same
    K3 event dates but our bars and baselines.  The last rows are v4 causal
    estimates and must not be interpreted as a continuous performance series.
    """
    h = protocol["primary_horizon"]
    ref = [e for e in reference_events if e.get("xs") is not None]
    same_vals = []
    next_vals = []
    same_by_stock = defaultdict(list)
    next_by_stock = defaultdict(list)
    same_base_by_stock = {}
    for stock, series in series_map.items():
        rows = series["days"]
        closes = [r["close"] for r in rows]
        vals = [
            closes[i + h] / closes[i] - 1
            for i, row in enumerate(rows)
            if row["eligible"] and i + h < len(rows) and closes[i] and closes[i + h]
        ]
        same_base_by_stock[stock] = mean(vals)
    for event in ref:
        stock = event["stock"]
        series = series_map.get(stock)
        if not series:
            continue
        rows = series["days"]
        index = next((i for i, r in enumerate(rows) if r["d"] == event["d"]), None)
        if index is None:
            continue
        row = rows[index]
        same = rows[index + h]["close"] / row["close"] - 1 if index + h < len(rows) and row["close"] else None
        nxt = row["statuses"][h] == "COMPLETE" and row["returns"][h] is not None
        base_same = same_base_by_stock.get(stock)
        base_next = series["baselines"][h]["mean"]
        if same is not None and base_same is not None:
            same_vals.append(same - base_same)
            same_by_stock[stock].append(same - base_same)
        if nxt and base_next is not None:
            next_vals.append(row["returns"][h] - base_next)
            next_by_stock[stock].append(row["returns"][h] - base_next)
    surviving_ref = []
    surviving_vals = []
    for event in ref:
        stock = event["stock"]
        series = series_map.get(stock)
        if not series:
            continue
        row = next((r for r in series["days"] if r["d"] == event["d"]), None)
        if row is None or not row["eligible"] or row["z"] is None or row["z"] < protocol["z_min"]:
            continue
        if row["returns"][h] is None:
            continue
        base = series["baselines"][h]["mean"]
        if base is None:
            continue
        surviving_ref.append(event)
        surviving_vals.append(row["returns"][h] - base)
    def _cluster_count(events):
        count = 0
        last = None
        for e in sorted(events, key=lambda x: (x["stock"], x["d"])):
            pos = next((r["pos"] for r in series_map[e["stock"]]["days"] if r["d"] == e["d"]), None)
            if pos is None:
                continue
            if last is None or e["stock"] != last[0] or pos - last[1] > protocol["cluster_gap_td"]:
                count += 1
            last = (e["stock"], pos)
        return count
    rows = [
        {
            "order": 0,
            "id": "K3_H1_PUBLISHED",
            "label": "K3 v3.4 H-1 发布值",
            "change_axis": "参考",
            "n_events": 26,
            "n_clusters": _cluster_count(ref),
            "n_stocks": len({e["stock"] for e in ref}),
            "mean_pp": 3.19,
            "t": 2.06,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "epistemic": "reference",
            "note": "同日收盘、全样本 z、簇内重复计数；只作参照。",
        },
        {
            "order": 1,
            "id": "K3_SAME_EVENTS_SAME_CLOSE",
            "label": "同一批事件 · 同日收盘",
            "change_axis": "复算",
            "n_events": len(same_vals),
            "n_clusters": _cluster_count(ref),
            "n_stocks": len(same_by_stock),
            "mean_pp": _pp(mean(same_vals)) if same_vals else None,
            "t": _t(same_vals),
            "ci_low_pp": None,
            "ci_high_pp": None,
            "epistemic": "reference_recomputed",
            "note": "使用我们冻结的行情；事件仍是 K3 的 26 笔。",
        },
        {
            "order": 2,
            "id": "K3_SAME_EVENTS_NEXT_OPEN",
            "label": "同一批事件 · 次日开盘",
            "change_axis": "执行时点",
            "n_events": len(next_vals),
            "n_clusters": _cluster_count(ref),
            "n_stocks": len(next_by_stock),
            "mean_pp": _pp(mean(next_vals)) if next_vals else None,
            "t": _t(next_vals),
            "ci_low_pp": None,
            "ci_high_pp": None,
            "epistemic": "reference_timing_shift",
            "note": "只改执行时点：T+1 开盘入场，T+6 收盘出场。",
        },
        {
            "order": 3,
            "id": "K3_EVENTS_CAUSAL_FILTER",
            "label": "同一批事件 · 因果 z 过滤",
            "change_axis": "阈值改为 expanding z",
            "n_events": len(surviving_ref),
            "n_clusters": _cluster_count(surviving_ref),
            "n_stocks": len({e["stock"] for e in surviving_ref}),
            "mean_pp": _pp(mean(surviving_vals)) if surviving_vals else None,
            "t": _t(surviving_vals),
            "ci_low_pp": None,
            "ci_high_pp": None,
            "epistemic": "reference_threshold_filter",
            "note": "事件日期仍是 K3 的；只保留“在当天用过去 60+ 个有效观测算出的 z≥2”的事件。",
        },
        {
            "order": 4,
            "id": "V4_CAUSAL_DEEP",
            "label": "v4 因果事件集 · 深回撤",
            "change_axis": "事件集重算（因果 z + F）",
            "n_events": primary_summary["n_events"],
            "n_clusters": primary_summary["n_clusters"],
            "n_stocks": primary_summary["n_stocks"],
            "mean_pp": primary_summary["mean_pp"],
            "t": primary_summary["t_event"],
            "ci_low_pp": primary_summary["ci_low_pp"],
            "ci_high_pp": primary_summary["ci_high_pp"],
            "epistemic": "retrospective_v4",
            "note": "expanding z（只用过去 60+ 个有效观测），不再使用未来信息。",
        },
        {
            "order": 5,
            "id": "V4_MATCHED_INCREMENT",
            "label": "v4 同股匹配 · 情绪增量",
            "change_axis": "价格状态对照",
            "n_events": matched_summary["n_pairs"],
            "n_clusters": None,
            "n_stocks": matched_summary["n_stocks"],
            "mean_pp": matched_summary["mean_diff_pp"],
            "t": matched_summary["t"],
            "ci_low_pp": matched_summary["ci_low_pp"],
            "ci_high_pp": matched_summary["ci_high_pp"],
            "epistemic": "retrospective_control",
            "note": "与同股、相近回撤/过去收益、非极端日相减；这才是情绪的增量。",
        },
        {
            "order": 6,
            "id": "V4_FORWARD_H4",
            "label": "前瞻 H-4 · 江丰单例",
            "change_axis": "前向",
            "n_events": 1,
            "n_clusters": 1,
            "n_stocks": 1,
            "mean_pp": -15.58,
            "t": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "epistemic": "forward_single",
            "note": "单例不可与左侧均值并列比较；仅记录流程已跑通、结果失败。",
        },
    ]
    return rows


def build_observation(
    series_map: dict[str, dict],
    cells: Mapping[tuple[str, str], dict],
    bars: Mapping[str, Mapping[str, dict]],
    calendar: Sequence[str],
    latest: str,
    market_latest: str,
    regimes: dict[str, dict],
    benchmark: Mapping[str, Mapping[str, dict]],
    stocks_meta: list[dict],
    aggregate: dict,
    auth: dict,
    protocol: dict,
) -> dict:
    h = protocol["primary_horizon"]
    meta_by_code = {s["code"]: s for s in stocks_meta}
    health = []
    current = []
    for stock, series in series_map.items():
        info_days = [r["d"] for r in series["days"] if r["n"] > 0]
        market_days = len(series["days"])
        covered = len(info_days)
        coverage = covered / market_days if market_days else None
        gaps = []
        for a, b in zip(info_days, info_days[1:]):
            pa, pb = index_of(calendar, a), index_of(calendar, b)
            if pa is not None and pb is not None:
                gaps.append(max(0, pb - pa - 1))
        thin = sum(1 for r in series["days"] if 0 < r["n"] < protocol["min_posts"])
        polluted = sum(1 for r in series["days"] if r["n"] and r["top1"] is not None and r["top1"] > protocol["pollution_top1_share"])
        recent = series["days"][-20:]
        recent_avg = mean([r["n"] for r in recent]) if recent else None
        full_avg = sum(r["n"] for r in series["days"]) / market_days if market_days else None
        decay = recent_avg / full_avg - 1 if recent_avg is not None and full_avg else None
        concentrations = [r["top1"] for r in series["days"] if r["top1"] is not None]
        warnings = []
        if coverage is not None and coverage < 0.90:
            warnings.append("覆盖率<90%")
        if decay is not None and decay <= -0.40:
            warnings.append("近20日帖量骤降")
        if polluted:
            warnings.append(f"{polluted}个污染日已剔除")
        health.append(
            {
                "code": stock,
                "name": meta_by_code.get(stock, {}).get("name", stock),
                "sector": meta_by_code.get(stock, {}).get("sector", "未分类"),
                "total_posts": sum(r["n"] for r in series["days"]),
                "market_days": market_days,
                "covered_days": covered,
                "coverage_pct": _pp(coverage),
                "longest_gap": max(gaps) if gaps else 0,
                "thin_days": thin,
                "polluted_days": polluted,
                "full_avg": round(full_avg, 1) if full_avg is not None else None,
                "recent20_avg": round(recent_avg, 1) if recent_avg is not None else None,
                "decay_pct": _pp(decay),
                "top1_p90_pct": _pp(quantile(concentrations, 0.9)) if concentrations else None,
                "warnings": warnings,
            }
        )
        row = series["days"][-1] if series["days"] else None
        if row is None:
            continue
        eligible = row["eligible"]
        z = row["z"]
        s = bool(eligible and z is not None and z >= protocol["z_min"])
        d = bool(row["dd"] is not None and row["dd"] <= protocol["dd_deep"])
        run = row.get("s_run")
        if row["n"] == 0:
            data_status = "no_posts"
        elif row["n"] < protocol["min_posts"]:
            data_status = "thin"
        elif row["top1"] is not None and row["top1"] > protocol["pollution_top1_share"]:
            data_status = "polluted"
        elif z is None:
            data_status = "warmup"
        else:
            data_status = "ok"
        entry_day = next_trading_day(calendar, latest)
        entry_price = bars.get(stock, {}).get(entry_day, {}).get("open") if entry_day else None
        mark_price = bars.get(stock, {}).get(market_latest, {}).get("close")
        day_ret = mark_price / entry_price - 1 if entry_price and mark_price and entry_day else None
        reasons = []
        if s and d:
            reasons.append("S∩D：极端情绪且深回撤")
        elif s:
            reasons.append("仅情绪极端，未达深回撤")
        elif d:
            reasons.append("仅深回撤，未达情绪极端")
        else:
            reasons.append("未达任一极端条件")
        if run is not None and run > 2:
            reasons.append("簇内第3个极端日，规则停手")
        current.append(
            {
                "code": stock,
                "name": meta_by_code.get(stock, {}).get("name", stock),
                "sector": meta_by_code.get(stock, {}).get("sector", "未分类"),
                "as_of": latest,
                "market_as_of": market_latest,
                "n_posts": row["n"],
                "density_pct": _pp(row["density"]),
                "z": round(z, 2) if z is not None else None,
                "z_status": row["z_status"],
                "dd20_pct": _pp(row["dd"]),
                "p5_pct": _pp(row["p5"]),
                "p60_pct": _pp(row["p60"]),
                "vol_ratio": round(row["vol_ratio"], 2) if row["vol_ratio"] is not None else None,
                "s_plus": s,
                "d_plus": d,
                "quadrant": ("S+" if s else "S-") + ("D+" if d else "D-"),
                "data_status": data_status,
                "gates": {
                    "info_complete": True,
                    "posts_ok": row["n"] >= protocol["min_posts"],
                    "not_polluted": row["top1"] is not None and row["top1"] <= protocol["pollution_top1_share"],
                    "z_ok": bool(z is not None and z >= protocol["z_min"]),
                    "deep": d,
                    "cluster_ok": run is None or run <= 2,
                },
                "reason": "；".join(reasons),
                "entry_day": entry_day,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "mark_return_pct": _pp(day_ret),
                "outcome_status": "PENDING_EXIT" if entry_price and mark_price else "PENDING_ENTRY",
            }
        )

    chart = {}
    for stock, series in series_map.items():
        rows = series["days"][-250:]
        chart[stock] = [
            [
                r["d"],
                round(r["close"], 4) if r["close"] is not None else None,
                _pp(r["dd"]),
                r["n"],
                _pp(r["density"]),
                round(r["z"], 2) if r["z"] is not None else None,
                _pp(r["p5"]),
                1 if r["eligible"] else 0,
            ]
            for r in rows
        ]

    regime_rows = [regimes[d] for d in calendar if d in regimes][-250:]
    # Breadth: number of stocks in S / S∩D each day and CSI300 forward 5d.
    day_s = defaultdict(int)
    day_sd = defaultdict(int)
    for stock, series in series_map.items():
        for row in series["days"]:
            if row["s"]:
                day_s[row["d"]] += 1
                if row["deep"]:
                    day_sd[row["d"]] += 1
    breadth = []
    bench_days = sorted(benchmark.get("000300", {}))
    for i, day in enumerate(bench_days):
        if i + 5 >= len(bench_days):
            fwd = None
        else:
            start = benchmark["000300"][day]["close"]
            end = benchmark["000300"][bench_days[i + 5]]["close"]
            fwd = end / start - 1 if start else None
        breadth.append(
            {
                "d": day,
                "n_s": day_s.get(day, 0),
                "n_sd": day_sd.get(day, 0),
                "fwd5_pp": _pp(fwd),
            }
        )
    breadth = breadth[-250:]
    pairs = [(b["n_sd"], b["fwd5_pp"]) for b in breadth if b["fwd5_pp"] is not None]
    corr = None
    if len(pairs) > 3:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = mean(xs), mean(ys)
        sx, sy = stdev(xs), stdev(ys)
        if sx and sy:
            corr = sum((x - mx) * (y - my) for x, y in pairs) / ((len(pairs) - 1) * sx * sy)
    breadth_summary = {
        "n_days": len(pairs),
        "corr_sd_fwd5": round(corr, 3) if corr is not None else None,
        "note": "日度重叠，相关系数只作描述，不作显著性结论。",
    }

    concentration_bins = Counter()
    for (stock, day), cell in cells.items():
        if cell["n"] < protocol["min_posts"]:
            continue
        share = max(cell["authors"].values()) / cell["n"]
        if share < 0.2:
            concentration_bins["<20%"] += 1
        elif share < 0.4:
            concentration_bins["20–40%"] += 1
        elif share < 0.6:
            concentration_bins["40–60%"] += 1
        else:
            concentration_bins["≥60%"] += 1

    latest_regime = regimes.get(market_latest, {})
    return {
        "thresholds": {
            "min_posts": protocol["min_posts"],
            "pollution_top1_share": protocol["pollution_top1_share"],
            "min_prior_obs": protocol["min_prior_obs"],
            "z_min": protocol["z_min"],
            "dd_deep_pct": protocol["dd_deep"] * 100,
            "cluster_gap_td": protocol["cluster_gap_td"],
        },
        "stocks": health,
        "current": current,
        "series": {"window_trading_days": 250, "by_stock": chart},
        "regime": {
            "index": "000300",
            "as_of": market_latest,
            "signal_day": latest,
            "close": latest_regime.get("close"),
            "ma200": latest_regime.get("ma200"),
            "below_ma200": (latest_regime.get("close") is not None and latest_regime.get("ma200") is not None and latest_regime["close"] < latest_regime["ma200"]),
            "ret20_pp": _pp(latest_regime.get("ret20")),
            "dd60_pp": _pp(latest_regime.get("dd60")),
            "label": latest_regime.get("label", "unknown"),
            "series": [
                {"d": r["d"], "close": r["close"], "ma200": r["ma200"], "dd60_pp": _pp(r["dd60"]), "label": r["label"]}
                for r in regime_rows
            ],
            "definition": "close < MA200 且 20 日收益 < 0 → risk_off；否则 mixed / risk_on / unknown。",
        },
        "breadth": {"series": breadth, "summary": breadth_summary},
        "author": {
            "total_authors": auth.get("total_authors"),
            "singleton_authors": auth.get("n_authors"),
            "top1pct_post_share_pct": _pp(auth.get("top1pct_post_share")),
            "top_author_post_share_pct": _pp(auth.get("top_author_share")),
            "concentration_bins": [{"bin": k, "days": v} for k, v in concentration_bins.items()],
            "note": "当前 613,985 帖快照；与 K3 的 377,163 帖快照不能混用。",
        },
    }


def build_prediction(
    series_map: dict[str, dict],
    bars: Mapping[str, Mapping[str, dict]],
    calendar: Sequence[str],
    latest: str,
    regimes: dict[str, dict],
    primary: dict,
    training: dict,
    observation: dict,
    protocol: dict,
    stocks_meta: list[dict],
) -> dict:
    h = protocol["primary_horizon"]
    meta_by_code = {s["code"]: s for s in stocks_meta}
    current = observation["current"]
    candidates = []
    watchlist = []
    for row in current:
        gates = row["gates"]
        if row["s_plus"] and row["d_plus"] and all(gates.values()):
            candidates.append(
                {
                    **row,
                    "bucket": "candidate",
                    "missing_gate": None,
                    "next_check": row.get("entry_day"),
                }
            )
        elif row["s_plus"] or row["d_plus"] or (row["z"] is not None and row["z"] >= 1.5):
            bucket = "fear_not_deep" if row["s_plus"] and not row["d_plus"] else ("deep_not_fear" if row["d_plus"] and not row["s_plus"] else "near")
            if row["d_plus"] and row["z"] is not None and row["z"] >= 1.5:
                bucket = "deep_near_fear"
            watchlist.append(
                {
                    "code": row["code"],
                    "name": row["name"],
                    "sector": row["sector"],
                    "bucket": bucket,
                    "z": row["z"],
                    "dd20_pct": row["dd20_pct"],
                    "density_pct": row["density_pct"],
                    "n_posts": row["n_posts"],
                    "missing_gate": "dd>-20%" if row["s_plus"] and not row["d_plus"] else ("z<2" if row["d_plus"] else "距离门槛"),
                    "as_of": latest,
                }
            )

    # H-4: the first pre-registered forward case.
    h4 = None
    entry_day = next_trading_day(calendar, "2026-10-09")
    exit_day = calendar[index_of(calendar, entry_day) + h] if entry_day and index_of(calendar, entry_day) is not None and index_of(calendar, entry_day) + h < len(calendar) else None
    entry_price = bars.get("300666", {}).get(entry_day, {}).get("open") if entry_day else None
    exit_price = bars.get("300666", {}).get(exit_day, {}).get("close") if exit_day else None
    if entry_price and exit_price:
        gross = exit_price / entry_price - 1
        h4 = {
            "event_id": "K3-H4-300666-20261009",
            "hypothesis": "K3-H-1 / V4-H-1",
            "stock": "300666",
            "name": "江丰电子",
            "signal_date": "2026-10-09",
            "entry_date": entry_day,
            "exit_date": exit_day,
            "status": "matured",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pp": _pp(gross),
            "net_pp": _pp(gross - protocol["cost_one_way"] * 2),
            "cost_bps_roundtrip": protocol["cost_one_way"] * 2 * 10000,
            "included_in_stats": True,
            "note": "预声明后首个前瞻 S∩D 案例；流程跑通，单例失败，不据此否决假设。",
        }

    # Full primary event ledger, including pending.
    ledger = []
    for event in sorted(primary["all"], key=lambda e: (e["info_day"], e["stock"]), reverse=True):
        base = series_map[event["stock"]]["baselines"].get(h, {}).get("mean")
        ledger.append(
            {
                "event_id": f"v4-{event['stock']}-{event['info_day'].replace('-', '')}",
                "stock": event["stock"],
                "name": meta_by_code.get(event["stock"], {}).get("name", event["stock"]),
                "signal_date": event["info_day"],
                "entry_date": event["entry"],
                "exit_date": event["exit"],
                "status": event["status"].lower(),
                "gross_pp": _pp(event["ret"]),
                "baseline_pp": _pp(base),
                "excess_pp": _pp(event["ret"] - base) if event["ret"] is not None and base is not None else None,
                "net_pp": _pp(event["ret"] - base - protocol["cost_one_way"] * 2) if event["ret"] is not None and base is not None else None,
                "z": round(event["z"], 2),
                "dd20_pct": _pp(event["dd"]),
                "p5_pct": _pp(event["p5"]),
                "cluster": event["cluster"],
                "included_in_stats": event["ret"] is not None,
                "is_forward": event["info_day"] >= "2026-10-09",
            }
        )
    if h4:
        match = next((i for i, row in enumerate(ledger) if row["stock"] == h4["stock"] and row["signal_date"] == h4["signal_date"]), None)
        if match is not None:
            row = ledger.pop(match)
            row.update(
                {
                    "event_id": h4["event_id"],
                    "hypothesis": h4["hypothesis"],
                    "entry_price": h4["entry_price"],
                    "exit_price": h4["exit_price"],
                    "cost_bps_roundtrip": h4["cost_bps_roundtrip"],
                    "status": "matured",
                    "note": h4["note"],
                    "is_forward": True,
                }
            )
            ledger.insert(0, row)
        else:
            ledger.insert(0, {**h4, "baseline_pp": None, "excess_pp": None, "z": None, "dd20_pct": None, "p5_pct": None, "cluster": None, "is_forward": True})

    # Event distribution.
    cutoff = protocol.get("training_cutoff")
    complete = [e for e in primary["all"] if e["ret"] is not None and (cutoff is None or e["info_day"] < cutoff)]
    rets = sorted(e["ret"] for e in complete)
    excess_rets = []
    for event in complete:
        base = series_map[event["stock"]]["baselines"].get(h, {}).get("mean")
        if base is not None:
            excess_rets.append(event["ret"] - base)
    matched_pairs = training.get("matched_controls", {}).get("pairs", [])
    matched_rets = [p["diff_pp"] / 100 for p in matched_pairs if p.get("diff_pp") is not None]
    def dist(vals):
        if not vals:
            return {"n": 0, "status": "insufficient_sample"}
        return {
            "n": len(vals),
            "mean_pp": _pp(mean(vals)),
            "median_pp": _pp(quantile(vals, 0.5)),
            "win_rate_pct": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            "p10_pp": _pp(quantile(vals, 0.10)),
            "p25_pp": _pp(quantile(vals, 0.25)),
            "p75_pp": _pp(quantile(vals, 0.75)),
            "p90_pp": _pp(quantile(vals, 0.90)),
            "status": "ok" if len(vals) >= 3 else "insufficient_sample",
        }
    event_distribution = {
        "primary": dist(rets),
        "primary_excess": dist(excess_rets),
        "matched_increment": dist(matched_rets),
        "histogram": [{"bin_pp": b, "count": sum(1 for v in rets if b <= v * 100 < b + 2)} for b in range(-20, 22, 2)],
        "excess_histogram": [{"bin_pp": b, "count": sum(1 for v in excess_rets if b <= v * 100 < b + 2)} for b in range(-20, 22, 2)],
        "training_cutoff": cutoff,
        "note": "H-4 是预声明前瞻，单独列在台账，不进入历史分布。",
    }

    # Hypothesis progress.  K3-H-2/H-3 are carried from the frozen K3 register;
    # their forward counts are not recomputed here.
    forward_events = [h4] if h4 else []
    hypotheses = [
        {
            "id": "K3-H-1",
            "source": "K3",
            "label": "深回撤 × 极端骂声（H-1）",
            "status": "forward_pending",
            "in_sample": {"n_events": 26, "n_stocks": 8, "mean_pp": 3.19, "t": 2.06, "note": "同日收盘、全样本 z；仅参考"},
            "forward": {"n": 1 if h4 else 0, "required": 20, "last_pp": h4["gross_pp"] if h4 else None, "status": "1 例已结算，失败"},
            "release": "≥20 个新独立簇，均值 > +0.5pp 且 t ≥ 1.5",
            "kill": "n≥20 且均值 ≤ 0，或 CI 上界 ≤ +0.5pp",
            "last_checked": latest,
        },
        {
            "id": "K3-H-2",
            "source": "K3",
            "label": "极端放量高危（H-2）",
            "status": "forward_pending",
            "in_sample": {"n_events": 5, "mean_pp": -3.34, "t": None, "note": "K3 v3.9.1 快照"},
            "forward": {"n": 0, "required": 20, "last_pp": None, "status": "跟踪中"},
            "release": "新事件 ≥20 且均值 < 0",
            "kill": "满 6 个月无增量则归档",
            "last_checked": latest,
        },
        {
            "id": "K3-H-3",
            "source": "K3",
            "label": "涨后骂轻正（H-3）",
            "status": "forward_pending",
            "in_sample": {"n_events": 21, "mean_pp": 0.77, "t": 0.48, "note": "K3 快照，纯噪声态"},
            "forward": {"n": 0, "required": 25, "last_pp": None, "status": "低优先"},
            "release": "新事件 ≥25 且 t ≥ 1.5",
            "kill": "满 6 个月自动归档",
            "last_checked": latest,
        },
        {
            "id": "V4-H-1",
            "source": "v4",
            "label": "因果时点后，S∩D 相对同股深回撤仍有增量",
            "status": "analysis_only",
            "in_sample": training.get("primary", {}),
            "forward": {"n": 1 if h4 else 0, "required": 20, "last_pp": h4["gross_pp"] if h4 else None, "status": "未过前瞻"},
            "release": "≥20 个新独立簇，匹配增量 > +0.5pp 且 CI 下界 > 0",
            "kill": "n≥20 且匹配增量 ≤ 0",
            "last_checked": latest,
        },
        {
            "id": "V4-H-2",
            "source": "v4",
            "label": "市场 regime 会改变深回撤反弹的赔率",
            "status": "analysis_only",
            "in_sample": {"regime_condition": training.get("regime_condition")},
            "forward": {"n": 0, "required": 20, "status": "样本不足"},
            "release": "risk_on/risk_off 至少各 10 个独立簇且方向稳定",
            "kill": "方向相反或无法分层",
            "last_checked": latest,
        },
        {
            "id": "V4-H-3",
            "source": "v4",
            "label": "跨股恐慌广度对 CSI300 有前瞻信息",
            "status": "analysis_only",
            "in_sample": observation.get("breadth", {}).get("summary", {}),
            "forward": {"n": 0, "required": 20, "status": "描述性"},
            "release": "≥20 个不重叠周，相关性稳定且增量显著",
            "kill": "相关接近 0 或符号不稳定",
            "last_checked": latest,
        },
    ]

    decision_ladder = [
        {"level": "L0", "label": "观察级", "state": "open", "criteria": "数据/口径可复核", "current": "开放"},
        {"level": "L1", "label": "研究级", "state": "partial", "criteria": "事件研究 + 对照 + CI", "current": "仅回顾性"},
        {"level": "L2", "label": "前瞻级", "state": "locked", "criteria": "预声明后达到放行线", "current": "K3-H-1 1/20；V4-H-1 1/20"},
        {"level": "L3", "label": "可执行级", "state": "locked", "criteria": "t>3、成本/执行已评估、跨 regime 稳定", "current": "不满足"},
    ]

    headline = {
        "as_of": latest,
        "candidate_count": len(candidates),
        "deployable": False,
        "regime": observation["regime"].get("label"),
        "h4": h4,
        "statement": "当前没有可执行候选；不部署任何情绪信号。",
    }
    return {
        "headline": headline,
        "candidates": candidates,
        "watchlist": sorted(watchlist, key=lambda x: (x["dd20_pct"] is None, x["dd20_pct"] or 0)),
        "hypotheses": hypotheses,
        "event_distribution": event_distribution,
        "forward_ledger": ledger,
        "decision_ladder": decision_ladder,
        "execution": {
            "entry": "T+1 open",
            "exit_h5": "T+6 close",
            "t_plus_1": True,
            "price_limits_pct": {s["code"]: s["price_limit_pct"] for s in stocks_meta},
            "cost_bps_roundtrip": protocol["cost_one_way"] * 2 * 10000,
            "unmodeled": ["滑点", "涨跌停无法成交", "停牌", "冲击成本", "仓位/资金约束"],
            "note": "所有收益未经真实执行验证；只使用 open/close，不处理盘口。",
        },
        "kill_criteria": [
            "新前瞻簇 ≥20 且均值 ≤0，或 CI 上界 ≤ +0.5pp → 归档",
            "匹配增量在 n≥20 时仍 ≤0 → 停止情绪增量路线",
            "覆盖率 <90% 或信息日不可验证 → 暂停发布",
            "发现前视/时点错误 → 撤回页面数字并修复",
        ],
        "change_my_mind": [
            "≥20 个独立前瞻簇，扣 20bp 后均值 > +0.5pp，CI 下界 >0，≥5 股且单股 <40%",
            "深回撤 + 情绪相对深回撤对照的增量 > +0.5pp",
            "跨两个季度、跨 regime 稳定；否则不升级",
        ],
    }
