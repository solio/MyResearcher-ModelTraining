"""Inspect strongest existing crowd states; descriptive research, no fit/search.

Full-history extrema are the primary cases. Observable-history extrema are
reported separately so a recent, pending minimum never silently disappears.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import market_signal_v1 as v1
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as io

DEFAULT_SOURCE = ROOT.parent / "pipeline-runs/market-signal-state-contrast-20260927-final"
METRICS = {"stance_index": "多空方向", "action_index": "操作倾向", "emotion_balance": "情绪平衡"}
LABEL_NAMES = {"ANGER": "愤怒", "ANXIETY": "焦虑", "CALM": "平静", "EXCITEMENT": "兴奋",
               "FEAR": "恐惧", "FOMO": "怕踏空", "FRUSTRATION": "挫败", "HOPE": "希望",
               "NONE_EXPLICIT": "未明确表达情绪", "REGRET": "后悔", "UNKNOWN": "不明",
               "BEAR": "看空", "BULL": "看多", "MIXED": "多空混合", "NEUTRAL": "中性",
               "ADD": "加仓", "BUY": "买入", "DO_T": "做T", "HOLD": "持有",
               "NO_ACTION_SIGNAL": "无操作信号", "REDUCE": "减仓", "SELL": "卖出", "WATCH": "观望"}
STATUS_TEXT = {"COMPLETE": "已到期", "PENDING_ENTRY": "尚未入场", "PENDING_HORIZON": "未到期",
               "MISSING_PRICE": "缺行情", "ADJUSTMENT_VINTAGE_MISMATCH": "复权区间不一致"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    items = [x for x in values if x is not None]
    return statistics.fmean(items) if items else None


def pct(value):
    return "—" if value is None else f"{value:+.2%}"


def extreme_rows(rows, metric, count, tail):
    """Select solely by the named state, never a future return or maturity."""
    sign = 1 if tail == "low" else -1
    return sorted(rows, key=lambda r: (sign * float(r["features"][metric]), r["entry_date"]))[:count]


def central_reference(rows, metric, count):
    """Full-snapshot state quartiles, fixed before inspecting any outcomes.

    Inclusive linear percentiles keep tied states together. Original selected
    low/high dates are excluded; the resulting count need not be exactly 50%.
    A constant state has no identifiable extreme-versus-central comparison.
    """
    values = sorted(float(r["features"][metric]) for r in rows)

    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    low = extreme_rows(rows, metric, count, "low")
    high = extreme_rows(rows, metric, count, "high")
    excluded = {r["entry_date"] for r in low + high}
    q25, q75 = quantile(.25), quantile(.75)
    dates = {r["entry_date"] for r in rows
             if q25 <= float(r["features"][metric]) <= q75 and r["entry_date"] not in excluded}
    if values[0] == values[-1]:
        dates = set()
    return {"q25": q25, "q75": q75, "dates": dates,
            "low_dates": {r["entry_date"] for r in low}, "high_dates": {r["entry_date"] for r in high},
            "state_dates": len(rows), "constant_state": values[0] == values[-1]}


def reference_memberships(states, metrics, count):
    by_stock = defaultdict(list)
    for row in states:
        by_stock[row["stock_code"]].append(row)
    members = []
    for code, rows in sorted(by_stock.items()):
        for metric in metrics:
            reference = central_reference(rows, metric, count)
            for row in sorted(rows, key=lambda r: r["entry_date"]):
                date = row["entry_date"]
                members.append({"stock_code": code, "entry_date": date, "metric": metric,
                                "state_value": row["features"][metric],
                                "q25": reference["q25"], "q75": reference["q75"],
                                "selected_low": date in reference["low_dates"],
                                "selected_high": date in reference["high_dates"],
                                "central_reference": date in reference["dates"],
                                "remaining_entry_day": date not in (reference["low_dates"] | reference["high_dates"]),
                                "constant_state": reference["constant_state"]})
    return members


def compatible(qfq, raw, code, entry, end, entry_price, close):
    raw_open = number(raw.get((code, entry), {}).get("open"))
    raw_close = number(raw.get((code, end), {}).get("close"))
    if not raw_open or not raw_close:
        return False
    a, b = qfq[(code, entry)], qfq[(code, end)]
    return a.get("fetched_at") == b.get("fetched_at") or abs(close / entry_price - raw_close / raw_open) <= 0.001


def price_path(code, entry, calendar, qfq, raw, max_horizon):
    if entry not in calendar:
        return []
    start = calendar.index(entry)
    opening = number(qfq.get((code, entry), {}).get("open"))
    if not opening or opening <= 0:
        return []
    points = [{"session": 0, "trade_date": entry, "return": 0.0}]
    for offset, day in enumerate(calendar[start:start + max_horizon], 1):
        close = number(qfq.get((code, day), {}).get("close"))
        if not close or close <= 0 or not compatible(qfq, raw, code, entry, day, opening, close):
            # Do not connect a curve across unknown or inconsistent prices.
            break
        points.append({"session": offset, "trade_date": day, "return": close / opening - 1})
    return points


def outcome(code, entry, horizon, calendar, qfq, raw):
    result = {"horizon": horizon, "status": "PENDING_ENTRY", "return": None,
              "positive": None, "worst_from_entry": None, "best_from_entry": None, "exit_date": None}
    if entry not in calendar:
        return result
    start = calendar.index(entry)
    if start + horizon > len(calendar):
        return dict(result, status="PENDING_HORIZON", observed_sessions=len(calendar) - start)
    days = calendar[start:start + horizon]
    opening = number(qfq.get((code, entry), {}).get("open"))
    result["exit_date"] = days[-1]
    if not opening or opening <= 0:
        return dict(result, status="MISSING_PRICE")
    highs, lows = [], []
    for day in days:
        bar = qfq.get((code, day), {})
        close, high, low = (number(bar.get(key)) for key in ("close", "high", "low"))
        if not close or not high or not low or min(close, high, low) <= 0:
            return dict(result, status="MISSING_PRICE")
        if not raw.get((code, day)):
            return dict(result, status="MISSING_PRICE")
        if not compatible(qfq, raw, code, entry, day, opening, close):
            return dict(result, status="ADJUSTMENT_VINTAGE_MISMATCH")
        highs.append(high)
        lows.append(low)
    ret = number(qfq[(code, days[-1])]["close"]) / opening - 1
    return dict(result, status="COMPLETE", entry_open=opening,
                exit_close=number(qfq[(code, days[-1])]["close"]),
                **{"return": ret, "positive": ret > 0, "worst_from_entry": min(0.0, min(lows) / opening - 1),
                   "best_from_entry": max(0.0, max(highs) / opening - 1)})


def past_return(code, entry, lag, calendar, qfq):
    if entry not in calendar:
        return None
    start = calendar.index(entry)
    if start < lag + 1:
        return None
    newer = number(qfq.get((code, calendar[start - 1]), {}).get("close"))
    older = number(qfq.get((code, calendar[start - lag - 1]), {}).get("close"))
    return newer / older - 1 if newer and older else None


def sample_examples(case, config, cache, cap):
    """Replay the existing daily sampling, then show three strongest examples."""
    source = str(case["source_file"])
    if source not in cache:
        predictions = io.read_jsonl(source)
        if Path(source).resolve() == Path(config["history_predictions"]).resolve():
            pred = {r["sample_id"]: r for r in predictions}
            metadata = io.read_jsonl(config["history_metadata"])
            members = [dict(r, predictions=pred[r["sample_id"]]["predictions"]) for r in metadata]
            groups = defaultdict(list)
            for row in members:
                groups[(row["stock_code"], row["entry_date"])].append(row)
        else:
            groups = defaultdict(list)
            for row in predictions:
                # A forward file belongs to one frozen entry per stock.
                groups[(row["stock_code"], None)].append(row)
        cache[source] = groups
    groups = cache[source]
    members = groups.get((case["stock_code"], case["entry_date"]), groups.get((case["stock_code"], None), []))
    members = sorted(members, key=lambda r: (r["published_at"], r["sample_id"]))
    if Path(source).resolve() != Path(config["history_predictions"]).resolve():
        unique = {}
        for row in members:
            unique.setdefault(v1._normalize_text(row.get("model_text")), row)
        members = list(unique.values())
    members = v1.systematic_sample(members, cap)

    def state(row):
        p = row["predictions"]
        if case["metric"] == "stance_index":
            q = p["stance"]["probabilities"]
            return q.get("BULL", 0) - q.get("BEAR", 0)
        if case["metric"] == "action_index":
            q = p["action_tendency"]["probabilities"]
            return q.get("BUY", 0) + q.get("ADD", 0) - q.get("SELL", 0) - q.get("REDUCE", 0)
        q = p["emotion_primary"]["probabilities"]
        return sum(q.get(k, 0) for k in ("HOPE", "EXCITEMENT")) - sum(q.get(k, 0) for k in ("ANGER", "ANXIETY", "FEAR", "FRUSTRATION", "REGRET"))

    sign = 1 if case["tail"] == "low" else -1
    chosen = sorted(members, key=lambda r: (sign * state(r), r["sample_id"]))[:3]
    return [{"sample_id": r["sample_id"], "text": r.get("model_text", ""),
             "model_state": state(r), "model_labels": {k: r["predictions"][k]["prediction"] for k in ("stance", "action_tendency", "emotion_primary")},
             "url": f"https://guba.eastmoney.com/news,{case['stock_code']},{r['source_item_id']}.html" if r.get("source_item_id") else None}
            for r in chosen]


def adjacent_groups(rows, calendar):
    positions = {day: pos for pos, day in enumerate(calendar)}
    ordered = sorted(rows, key=lambda r: r["entry_date"])
    groups = 0
    previous = None
    for row in ordered:
        pos = positions.get(row["entry_date"])
        if previous is None or pos is None or pos != previous + 1:
            groups += 1
        previous = pos
    return groups


def build_study(states, calendar, qfq, raw, *, count, horizons, metrics):
    by_stock = defaultdict(list)
    for row in states:
        by_stock[row["stock_code"]].append(row)
    results = {(r["stock_code"], r["entry_date"], h): outcome(r["stock_code"], r["entry_date"], h, calendar, qfq, raw)
               for r in states for h in horizons}
    common_horizons = [h for h in horizons if any(value["status"] == "COMPLETE" for key, value in results.items() if key[2] == h)]
    cases, stock_stats = [], []
    for code, rows in sorted(by_stock.items()):
        for metric in metrics:
            central = central_reference(rows, metric, count)
            scopes = [("full_history", None, rows)]
            scopes += [("observable_history", h, [r for r in rows if results[(code, r["entry_date"], h)]["status"] == "COMPLETE"]) for h in horizons]
            if common_horizons:
                scopes.append(("common_matured", None, [r for r in rows if all(results[(code, r["entry_date"], h)]["status"] == "COMPLETE" for h in common_horizons)]))
            for scope, scope_horizon, eligible in scopes:
                selected = {tail: extreme_rows(eligible, metric, count, tail) for tail in ("low", "high")}
                if scope == "common_matured":
                    eligible_days = {r["entry_date"] for r in eligible}
                    selected = {tail: [r for r in extreme_rows(rows, metric, count, tail) if r["entry_date"] in eligible_days] for tail in ("low", "high")}
                excluded = {r["entry_date"] for group in selected.values() for r in group}
                for tail, chosen in selected.items():
                    for rank, row in enumerate(chosen, 1):
                        cases.append({"case_id": f"{scope}:{scope_horizon}:{code}:{metric}:{tail}:{rank}",
                                      "scope": scope, "scope_horizon": scope_horizon, "metric": metric, "tail": tail,
                                      "rank": rank, "state_value": row["features"][metric],
                                      "stock_code": code, "stock_name": row.get("stock_name", ""), "entry_date": row["entry_date"],
                                      "sampled_posts": row["sampled_posts"], "observed_posts": row["observed_posts"],
                                      "source_file": row["source_file"], "evidence_type": row["evidence_type"],
                                      "distribution": {k: v for k, v in row["features"].items() if k.startswith("mean::")},
                                      "group_disagreement": row["features"]["group_disagreement"],
                                      "past5_return": past_return(code, row["entry_date"], 5, calendar, qfq),
                                      "past20_return": past_return(code, row["entry_date"], 20, calendar, qfq),
                                      "outcomes": {str(h): results[(code, row["entry_date"], h)] for h in horizons},
                                      "path": price_path(code, row["entry_date"], calendar, qfq, raw, max(horizons))})
                scope_horizons = [scope_horizon] if scope_horizon else common_horizons if scope == "common_matured" else horizons
                for h in scope_horizons:
                    observed = [r for r in eligible if results[(code, r["entry_date"], h)]["status"] == "COMPLETE"]
                    ordinary_outcomes = [results[(code, r["entry_date"], h)] for r in observed if r["entry_date"] not in excluded]
                    central_outcomes = [results[(code, r["entry_date"], h)] for r in observed if r["entry_date"] in central["dates"]]
                    selected_outcomes = {tail: [results[(code, r["entry_date"], h)] for r in chosen if results[(code, r["entry_date"], h)]["status"] == "COMPLETE"] for tail, chosen in selected.items()}
                    ordinary = [r["return"] for r in ordinary_outcomes]
                    values = {tail: [r["return"] for r in outcomes] for tail, outcomes in selected_outcomes.items()}
                    correlation = None
                    if len(observed) >= 5 and len({r["features"][metric] for r in observed}) > 1:
                        from scipy.stats import spearmanr
                        y = [results[(code, r["entry_date"], h)]["return"] for r in observed]
                        if len(set(y)) > 1:
                            correlation = number(spearmanr([r["features"][metric] for r in observed], y).statistic)
                    stock_stats.append({"scope": scope, "horizon": h, "metric": metric, "stock_code": code,
                                        "stock_name": rows[0].get("stock_name", ""), "eligible_dates": len(eligible),
                                        "eligible_start": min((r["entry_date"] for r in eligible), default=None),
                                        "eligible_end": max((r["entry_date"] for r in eligible), default=None),
                                        "low_selected": len(selected["low"]), "high_selected": len(selected["high"]),
                                        "low_adjacent_groups": adjacent_groups(selected["low"], calendar),
                                        "low_complete": len(values["low"]), "high_complete": len(values["high"]),
                                        "low_mean_return": mean(values["low"]), "high_mean_return": mean(values["high"]),
                                        "ordinary_dates": len(ordinary), "ordinary_mean_return": mean(ordinary),
                                        "low_mean_best_from_entry": mean([r["best_from_entry"] for r in selected_outcomes["low"]]),
                                        "high_mean_best_from_entry": mean([r["best_from_entry"] for r in selected_outcomes["high"]]),
                                        "ordinary_mean_best_from_entry": mean([r["best_from_entry"] for r in ordinary_outcomes]),
                                        "central_q25": central["q25"], "central_q75": central["q75"],
                                        "central_state_dates": central["state_dates"],
                                        "central_selected": len(central["dates"]), "central_complete": len(central_outcomes),
                                        "central_mean_return": mean([r["return"] for r in central_outcomes]),
                                        "central_mean_best_from_entry": mean([r["best_from_entry"] for r in central_outcomes]),
                                        "low_win_rate": mean([v > 0 for v in values["low"]]),
                                        "high_win_rate": mean([v > 0 for v in values["high"]]),
                                        "all_dates_state_return_spearman": correlation})
    summary = []
    for scope in ("full_history", "observable_history", "common_matured"):
        for metric in metrics:
            for h in horizons:
                group = [r for r in stock_stats if r["scope"] == scope and r["metric"] == metric and r["horizon"] == h]
                if not group:
                    continue
                paired = [r for r in group if r["low_mean_return"] is not None and r["ordinary_mean_return"] is not None]
                paired_tails = [r for r in group if r["low_mean_return"] is not None and r["high_mean_return"] is not None]
                paired_central = [r for r in group if r["low_mean_return"] is not None and r["central_mean_return"] is not None]
                central_differences = [r["low_mean_return"] - r["central_mean_return"] for r in paired_central]
                central_peak_differences = [r["low_mean_best_from_entry"] - r["central_mean_best_from_entry"] for r in paired_central]
                differences = [r["low_mean_return"] - r["ordinary_mean_return"] for r in paired]
                summary.append({"scope": scope, "metric": metric, "horizon": h,
                                "low_selected": sum(r["low_selected"] for r in group),
                                "low_complete": sum(r["low_complete"] for r in group),
                                "low_adjacent_groups": sum(r["low_adjacent_groups"] for r in group),
                                "low_mean_return": mean([r["low_mean_return"] for r in group]),
                                "low_mean_best_from_entry": mean([r["low_mean_best_from_entry"] for r in group]),
                                "low_win_rate": mean([r["low_win_rate"] for r in group]),
                                "high_mean_return": mean([r["high_mean_return"] for r in group]),
                                "high_mean_best_from_entry": mean([r["high_mean_best_from_entry"] for r in group]),
                                "paired_ordinary_stocks": len(paired),
                                "paired_low_mean_return": mean([r["low_mean_return"] for r in paired]),
                                "paired_ordinary_mean_return": mean([r["ordinary_mean_return"] for r in paired]),
                                "paired_ordinary_mean_best_from_entry": mean([r["ordinary_mean_best_from_entry"] for r in paired]),
                                "low_minus_ordinary": mean(differences),
                                "paired_central_stocks": len(paired_central),
                                "paired_central_dates": sum(r["central_complete"] for r in paired_central),
                                "paired_central_low_mean_return": mean([r["low_mean_return"] for r in paired_central]),
                                "paired_central_mean_return": mean([r["central_mean_return"] for r in paired_central]),
                                "paired_central_low_mean_best_from_entry": mean([r["low_mean_best_from_entry"] for r in paired_central]),
                                "paired_central_mean_best_from_entry": mean([r["central_mean_best_from_entry"] for r in paired_central]),
                                "low_minus_central": mean(central_differences),
                                "low_peak_minus_central": mean(central_peak_differences),
                                "stocks_low_above_central": sum(v > 0 for v in central_differences),
                                "stocks_low_peak_above_central": sum(v > 0 for v in central_peak_differences),
                                "low_median_stock_mean_return": statistics.median([r["low_mean_return"] for r in group if r["low_mean_return"] is not None]) if any(r["low_mean_return"] is not None for r in group) else None,
                                "low_minus_ordinary_median": statistics.median(differences) if differences else None,
                                "leave_one_stock_out_gain_min": min((statistics.fmean(differences[:i] + differences[i + 1:]) for i in range(len(differences))), default=None) if len(differences) > 1 else None,
                                "leave_one_stock_out_gain_max": max((statistics.fmean(differences[:i] + differences[i + 1:]) for i in range(len(differences))), default=None) if len(differences) > 1 else None,
                                "stocks_low_above_ordinary": sum(d > 0 for d in differences),
                                "paired_tail_stocks": len(paired_tails),
                                "low_minus_high": mean([r["low_mean_return"] - r["high_mean_return"] for r in paired_tails]),
                                "all_dates_spearman_stock_median": statistics.median([r["all_dates_state_return_spearman"] for r in group if r["all_dates_state_return_spearman"] is not None]) if any(r["all_dates_state_return_spearman"] is not None for r in group) else None})
    return cases, stock_stats, summary


def path_svg(cases, max_horizon):
    colors = ("#2563eb", "#d97706", "#059669", "#8b5cf6", "#db2777")
    points = [p["return"] for case in cases for p in case["path"]]
    if not points:
        return "<p>尚无入场后价格路径。</p>"
    bottom, top = min(0.0, min(points)), max(0.0, max(points))
    pad = max((top - bottom) * .1, .01)
    bottom, top = bottom - pad, top + pad
    x = lambda value: 65 + 665 * value / max_horizon
    y = lambda value: 220 - 185 * (value - bottom) / (top - bottom)
    parts = ['<svg viewBox="0 0 780 275" role="img" aria-label="最低状态出现后，以入场价为零的实际价格路径">']
    for value in (bottom, 0, top):
        parts.append(f'<line x1="65" x2="730" y1="{y(value):.2f}" y2="{y(value):.2f}" stroke="#cbd5e1"/><text x="3" y="{y(value):.2f}" font-size="12">{value:+.1%}</text>')
    for h in (10, 20, 30, 60):
        if h <= max_horizon:
            parts.append(f'<line x1="{x(h):.2f}" x2="{x(h):.2f}" y1="35" y2="220" stroke="#e2e8f0"/><text x="{x(h):.2f}" y="240" font-size="12">{h}日</text>')
    for index, case in enumerate(cases):
        color = colors[index % len(colors)]
        coordinates = " ".join(f'{x(p["session"]):.2f},{y(p["return"]):.2f}' for p in case["path"])
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{65 + index * 190}" y="265" font-size="12" fill="{color}">{html.escape(case["entry_date"])} · 最低#{case["rank"]}</text>')
    return "".join(parts) + "</svg>"


def render_markdown(report):
    focus_metric = "stance_index" if "stance_index" in report["config"]["metrics"] else report["config"]["metrics"][0]
    focus_horizon = 20 if 20 in report["config"]["horizons"] else report["config"]["horizons"][0]
    stock_lines = [f"## 先看个股：{METRICS[focus_metric]} {focus_horizon}日", "",
                   "这张表不是选股排名。已到期次数不能省略，只有1次的均值就是那一次；其余期限和逐事件日期见HTML中的个股区块。", "",
                   "| 股票 | 极端已到期/选中 | 极端期末收益 | 同股常态期末收益 | 极端平均最大浮盈 | 同股常态平均最大浮盈 |",
                   "|---|---:|---:|---:|---:|---:|"]
    for row in sorted((r for r in report.get("per_stock", []) if r["scope"] == "full_history" and r["metric"] == focus_metric and r["horizon"] == focus_horizon), key=lambda r: r["stock_code"]):
        stock_lines.append(f"| {row['stock_name']} {row['stock_code']} | {row['low_complete']}/{row['low_selected']} | {pct(row['low_mean_return'])} | {pct(row['central_mean_return'])} | {pct(row['low_mean_best_from_entry'])} | {pct(row['central_mean_best_from_entry'])} |")
    stock_lines += ["", "只比较同一只股票的两种状态，不要求买入全部股票；历史赢家不等于现在可买。", ""]
    central_lines = ["## 主对照：同股相对常态", "",
                     "常态按每股、每个状态指标分别定义：完整状态快照的Q25至Q75（含边界），剔除原入选最低/最高日期。分位数使用线性插值，同值不按日期强行拆分，数量不保证恰好50%。",
                     "分组只读取状态，不读取未来收益；各观察期使用同一组成员，仅排除未到期结果。常态是该股状态的中间区间，不代表股价平稳或帖子语义全为中性。常态期末收益与最大浮盈，都与极端组在相同股票内配对、股票等权。", "",
                     "| 状态 | 观察期 | 极端已到期/选中 | 配对常态日数 | 极端期末收益 | 常态期末收益 | 极端平均最大浮盈 | 常态平均最大浮盈 | 期末差值 | 高于常态的股票 |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["scope"] == "full_history":
            central_lines.append(f"| {METRICS[row['metric']]} | {row['horizon']}日 | {row['low_complete']}/{row['low_selected']} | {row['paired_central_dates']} | {pct(row['paired_central_low_mean_return'])} | {pct(row['paired_central_mean_return'])} | {pct(row['paired_central_low_mean_best_from_entry'])} | {pct(row['paired_central_mean_best_from_entry'])} | {pct(row['low_minus_central'])} | {row['stocks_low_above_central']}/{row['paired_central_stocks']} |")
    central_lines += ["", "逐股边界与逐日成员见 reference-groups.jsonl；无可区分状态或无完整常态行情时显示为空，不回填其他日期。该定义用于当前回顾性研究，不是实时可交易阈值。", ""]
    lines = ["# 最极端群体状态出现后，股票怎样了？", "",
             f"行情截至 {report['data']['calendar_end']}，{report['data']['calendar_sessions']} 个交易日；{report['data']['stocks']} 只股票。",
             f"每股每个指标取最低 {report['config']['count']} 个时点；最高同样数量仅作参照。没有训练、阈值搜索或概率模型。", "",
             "选点只读取状态值，不读取未来收益；这是回顾性信号存在性检查，不是可执行策略业绩。3 是本次案例数量，不是长期规则。",
             "价格按状态对应入场日开盘买入、第 H 个实际交易日收盘卖出的统一观察口径；未扣成本。",
             "期间最高收益 = 观察期内日线最高价的最大值 ÷ 入场日开盘价 − 1，包含入场日，共 H 个交易日。先对每个事件计算，再在每股内平均、对股票等权；不是挑出所有案例中最赚钱的一次，也不代表实际能卖在最高价。未到期不计入该期限均值。", "",
             *stock_lines,
             *central_lines,
             "## 背景对照：其余入场日（原表，不是相对常态）", "",
             "先在全部状态日期选点，再回填结果。最新极端尚未到期时保留为空，不用较早案例替换。", "",
             "| 状态 | 观察期 | 已到期/选中低值 | 低值后平均收益 | 低值后期间最高收益（均值） | 低值后正收益比例 | 其余入场日期末收益 | 其余入场日期间最高收益（均值） | 低值相对其余入场日期末收益 | 高于其余入场日的股票 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["scope"] == "full_history":
            lines.append(f"| {METRICS[row['metric']]} | {row['horizon']}日 | {row['low_complete']}/{row['low_selected']} | {pct(row['low_mean_return'])} | {pct(row['low_mean_best_from_entry'])} | {pct(row['low_win_rate'])} | {pct(row['paired_ordinary_mean_return'])} | {pct(row['paired_ordinary_mean_best_from_entry'])} | {pct(row['low_minus_ordinary'])} | {row['stocks_low_above_ordinary']}/{row['paired_ordinary_stocks']} |")
    lines += ["", "均值先在每股内计算，再对有数据股票等权。其余入场日仅排除该股该指标选中的低/高值日；它可能仍包含接近极端的日期，不等于相对常态。差值在相同股票内配对。收益观察区间有重叠，比例不是独立试验胜率。", "",
              "## 背景对照的均值是否由一只股票拉高", "",
              "以下仍是主结果；股票中位数用于看典型幅度，每次剔除一只股票后的差值范围用于检查单股影响，不是置信区间。", "",
              "| 状态 | 观察期 | 各股低值后收益中位数 | 相对其余入场日差值中位数 | 每次剔除一股后的平均差值范围 |",
              "|---|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["scope"] == "full_history":
            lines.append(f"| {METRICS[row['metric']]} | {row['horizon']}日 | {pct(row['low_median_stock_mean_return'])} | {pct(row['low_minus_ordinary_median'])} | {pct(row['leave_one_stock_out_gain_min'])} ～ {pct(row['leave_one_stock_out_gain_max'])} |")
    lines += ["",
              "## 同一批极端时点的不同持有期", "",
              "只保留主结果中已具备所有可观察期限结果的原始时点，不重选极端；其余入场日也使用同样完整的历史区间。60日无结果时不加入此交集。避免用各期限不同样本的均值差异冒充滞后规律。此表仍是背景对照。", "",
              "| 状态 | 观察期 | 同批低值案例 | 平均收益 | 期间最高收益（均值） | 其余入场日平均收益 | 其余入场日期间最高收益（均值） | 相对其余入场日期末收益 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["scope"] == "common_matured":
            lines.append(f"| {METRICS[row['metric']]} | {row['horizon']}日 | {row['low_complete']} | {pct(row['low_mean_return'])} | {pct(row['low_mean_best_from_entry'])} | {pct(row['paired_ordinary_mean_return'])} | {pct(row['paired_ordinary_mean_best_from_entry'])} | {pct(row['low_minus_ordinary'])} |")
    lines += ["",
              "## 当前已能观察完整结果的历史区间：补充，不替代主结果", "",
              "每个 H 单独保留已有完整行情结果的日期，再按同样规则选最低/最高值。它回答较早可观察历史中的极端表现，不能冒充全历史最极端结果。", "",
              "| 状态 | 观察期 | 低值案例 | 低值后平均收益 | 低值后期间最高收益（均值） | 其余入场日平均收益 | 其余入场日期间最高收益（均值） | 低值相对其余入场日期末收益 | 低值相对高值期末收益 | 高于其余入场日的股票 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["scope"] == "observable_history":
            lines.append(f"| {METRICS[row['metric']]} | {row['horizon']}日 | {row['low_complete']} | {pct(row['low_mean_return'])} | {pct(row['low_mean_best_from_entry'])} | {pct(row['paired_ordinary_mean_return'])} | {pct(row['paired_ordinary_mean_best_from_entry'])} | {pct(row['low_minus_ordinary'])} | {pct(row['low_minus_high'])} | {row['stocks_low_above_ordinary']}/{row['paired_ordinary_stocks']} |")
    lines += ["", "## 怎样读", "",
              "- 负相关也可以是信息：本报告不预设低值之后必须上涨。",
              "- 每股完整事件日期、状态数值、情绪分布、典型帖子、期末收益、期间最高收益和相对入场价期间最大跌幅见 REPORT.html。",
              "- 价格曲线使用每日收盘价；期间最高收益使用日内最高价，两者不是同一种价格。最高收益是事后观察到的价格上界，不是可实现策略收益。",
              "- 60日没有完整结果时仅显示未到期；图中的曲线止于真实已知日期，不补齐。",
              "- 连续低值可能属于同一事件；选中的3日不保证3个独立机会。",
              "- 相对常态和其余入场日都没有匹配此前涨跌/成交量和市场阶段，收益差不能证明情绪提供独立增量。",
              "- 每股每天最多16条已有概率，代表模型对抽样帖子的状态，不保证全股吧总体；样本和类别概率全部保留，不新增语义假设。",
              "- 历史语义预测部分在行情之后生成；本轮不声称前瞻、因果、可交易或显著性结论。", ""]
    return "\n".join(lines)


def render_html(report, cases):
    esc = html.escape
    groups = defaultdict(list)
    for case in cases:
        if case["scope"] == "full_history":
            groups[(case["stock_code"], case["metric"])].append(case)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>极端群体状态与后续行情</title>',
             '<style>body{font:16px/1.6 system-ui,sans-serif;color:#172033;max-width:1100px;margin:24px auto;padding:0 18px}h1{font-size:27px}h2{font-size:22px}h3{font-size:18px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dbe2ec;padding:7px;text-align:left}th{background:#eef2f7}section{border-top:2px solid #dbe2ec;margin-top:26px;padding-top:8px}.note{color:#59667c}svg{width:100%;max-height:310px}select{padding:8px;font-size:16px;margin:8px}a{color:#245ccc}.tools{position:sticky;top:0;background:white;padding:8px 0;border-bottom:1px solid #cbd5e1;z-index:2}details{margin:12px 0}.pending{color:#7b8798}</style>',
             '<h1>最极端群体状态出现后，股票怎样了？</h1>',
             f'<p>全部已有状态；行情截至 {report["data"]["calendar_end"]}。每股每个指标最低{report["config"]["count"]}次；不训练模型，不挑最好看的股票。</p>',
             '<p class="note">低值不自动等于恐慌，低值之后不必上涨。结果是开盘入场到第H个交易日收盘的绝对收益；期间跌幅是相对入场价，不是高点回撤。曲线未到期部分不补齐。</p>',
             '<p class="note">期间最高收益＝期内日线最高价的最大值÷入场开盘价−1，包含入场日，共H个交易日。先逐事件计算，再按股票等权平均；未到期不计入。不是所有案例中最赚钱的一次，也不代表实际能卖在最高价。价格曲线仍为每日收盘价。</p>',
             '<h2>主对照：同股相对常态</h2><p>每股、每个状态指标分别取完整状态快照的Q25至Q75（含边界），剔除原入选最低/最高日期。分位数线性插值，同值不按日期拆分。分组不读取未来收益，各观察期成员不变，只排除未到期结果。这里的常态是状态中段，不代表股价平稳或所有帖子语义中性。</p>',
             '<table><tr><th>状态</th><th>期限</th><th>极端已到期/选中</th><th>配对常态日数</th><th>极端期末收益</th><th>常态期末收益</th><th>极端平均最大浮盈</th><th>常态平均最大浮盈</th><th>期末差值</th></tr>']
    summary_start = next(i for i, value in enumerate(parts) if value.startswith('<h2>主对照'))
    for row in report["summary"]:
        if row["scope"] == "full_history":
            parts.append(f'<tr><td>{METRICS[row["metric"]]}</td><td>{row["horizon"]}日</td><td>{row["low_complete"]}/{row["low_selected"]}</td><td>{row["paired_central_dates"]}</td><td>{pct(row["paired_central_low_mean_return"])}</td><td>{pct(row["paired_central_mean_return"])}</td><td>{pct(row["paired_central_low_mean_best_from_entry"])}</td><td>{pct(row["paired_central_mean_best_from_entry"])}</td><td>{pct(row["low_minus_central"])}</td></tr>')
    parts.append('</table><p class="note">极端与常态在相同股票内配对、股票等权。逐日成员见reference-groups.jsonl；没有完整常态结果时留空。当前为回顾性定义，不是实时交易阈值，仍未匹配此前行情。</p><h2>背景对照：其余入场日（原表，不是相对常态）</h2><table><tr><th>状态</th><th>期限</th><th>已到期/选中</th><th>低值后期末收益</th><th>低值后期间最高收益（均值）</th><th>其余入场日期末收益</th><th>其余入场日期间最高收益（均值）</th><th>期末收益差值</th><th>高于其余入场日的股票</th></tr>')
    for row in report["summary"]:
        if row["scope"] == "full_history":
            parts.append(f'<tr><td>{METRICS[row["metric"]]}</td><td>{row["horizon"]}日</td><td>{row["low_complete"]}/{row["low_selected"]}</td><td>{pct(row["low_mean_return"])}</td><td>{pct(row["low_mean_best_from_entry"])}</td><td>{pct(row["paired_ordinary_mean_return"])}</td><td>{pct(row["paired_ordinary_mean_best_from_entry"])}</td><td>{pct(row["low_minus_ordinary"])}</td><td>{row["stocks_low_above_ordinary"]}/{row["paired_ordinary_stocks"]}</td></tr>')
    parts.append('</table><p class="note">按股票等权。其余入场日只排除所选低/高值日，仍可能包含接近极端的日期，不再称为普通日。对照未匹配此前行情，因此不是独立情绪收益。补充的可观察历史结果见 REPORT.md，不替换这里的未到期案例。</p>')
    summary_parts = parts[summary_start:]
    parts = parts[:summary_start]
    parts.append('<h2>先看个股</h2><p>按股票筛选，先看本股极端与本股常态；下方再看逐次事件。全体汇总移至文末，只作背景，不是让你配置所有股票。</p><div class="tools">股票<select id="stock"><option value="all">全部股票</option>')
    names = {r["stock_code"]: r["stock_name"] for r in cases}
    for code, name in sorted(names.items()):
        parts.append(f'<option value="{esc(code)}">{esc(name)} {esc(code)}</option>')
    parts.append('</select>状态<select id="metric"><option value="all">全部状态</option>')
    for metric in report["config"]["metrics"]:
        parts.append(f'<option value="{metric}">{METRICS[metric]}</option>')
    parts.append('</select></div>')
    for (code, metric), items in sorted(groups.items()):
        low = [r for r in items if r["tail"] == "low"]
        parts.append(f'<section data-stock="{code}" data-metric="{metric}"><h2>{esc(names[code])} {code} · {METRICS[metric]}</h2>')
        reference = next((r for r in report.get("per_stock", []) if r["scope"] == "full_history" and r["stock_code"] == code and r["metric"] == metric), None)
        if reference is not None:
            parts.append(f'<p class="note">该股相对常态范围：{reference["central_q25"]:+.4f}至{reference["central_q75"]:+.4f}；完整状态快照{reference["central_state_dates"]}日，常态成员{reference["central_selected"]}日（各期限只计已到期）。</p>')
            parts.append('<table><tr><th>期限</th><th>极端已到期/选中</th><th>常态已到期日</th><th>极端期末收益</th><th>本股常态期末收益</th><th>极端平均最大浮盈</th><th>本股常态平均最大浮盈</th></tr>')
            for row in sorted((r for r in report["per_stock"] if r["scope"] == "full_history" and r["stock_code"] == code and r["metric"] == metric), key=lambda r: r["horizon"]):
                parts.append(f'<tr><td>{row["horizon"]}日</td><td>{row["low_complete"]}/{row["low_selected"]}</td><td>{row["central_complete"]}</td><td>{pct(row["low_mean_return"])}</td><td>{pct(row["central_mean_return"])}</td><td>{pct(row["low_mean_best_from_entry"])}</td><td>{pct(row["central_mean_best_from_entry"])}</td></tr>')
            parts.append('</table>')
        parts.append(path_svg(low, max(report["config"]["horizons"])))
        headers = ''.join(f'<th>{h}日期末／期间最高</th>' for h in report["config"]["horizons"])
        parts.append('<p class="note">每条线从该事件的入场开盘价开始。线止于已有行情；相邻日期可能是同一个事件。</p><table><tr><th>低/高值排名</th><th>入场日</th><th>状态值</th><th>此前5日涨跌</th>' + headers + '</tr>')
        for case in items:
            cells = ''.join(f'<td>{pct(case["outcomes"][str(h)]["return"])}／{pct(case["outcomes"][str(h)]["best_from_entry"])}</td>' if case["outcomes"][str(h)]["status"] == "COMPLETE" else f'<td>{STATUS_TEXT[case["outcomes"][str(h)]["status"]]}</td>' for h in report["config"]["horizons"])
            parts.append(f'<tr><td>{"最低" if case["tail"] == "low" else "最高"}#{case["rank"]}</td><td>{case["entry_date"]}</td><td>{case["state_value"]:+.3f}</td><td>{pct(case["past5_return"])}</td>{cells}</tr>')
        parts.append('</table>')
        for case in low:
            parts.append(f'<details><summary>{case["entry_date"]} 最低#{case["rank"]}：情绪分布、期间跌幅和帖子</summary><p>观察到{case["observed_posts"]}帖，参与状态聚合{case["sampled_posts"]}帖；群体分歧度{case["group_disagreement"]:.3f}。</p><p>')
            emotion = sorted(((k.split("::")[-1], v) for k, v in case["distribution"].items() if k.startswith("mean::emotion_primary::")), key=lambda p: -p[1])
            parts.append('；'.join(f'{esc(LABEL_NAMES.get(label, label))} {value:.1%}' for label, value in emotion) + '</p><p>')
            parts.append('；'.join(f'{h}日相对入场价最低跌幅 {pct(result["worst_from_entry"])}' for h, result in case["outcomes"].items()) + '</p><p class="note">以下是模型判为该方向最强的抽样帖子，不是人工确认的情绪真值。</p>')
            for example in case.get("examples", []):
                link = f'<a href="{esc(example["url"], quote=True)}" target="_blank" rel="noopener">原帖</a>' if example["url"] else ''
                labels = '／'.join(LABEL_NAMES.get(value, value) for value in example["model_labels"].values())
                parts.append(f'<blockquote>{esc(example["text"])}<br><small>模型判断：{esc(labels)} · {link}</small></blockquote>')
            parts.append('</details>')
        parts.append('</section>')
    parts.extend(summary_parts)
    parts.append('<script>function filter(){const s=document.querySelector("#stock").value,m=document.querySelector("#metric").value;document.querySelectorAll("section").forEach(e=>e.hidden=!((s==="all"||e.dataset.stock===s)&&(m==="all"||e.dataset.metric===m)));}document.querySelectorAll("select").forEach(e=>e.addEventListener("change",filter));</script></html>')
    return ''.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30, 60])
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--as-of", default="latest")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.count < 1 or min(args.horizons) < 1:
        parser.error("count and horizons must be positive")
    source = args.source_run.resolve()
    config = json.loads((source / "run-config.json").read_text())
    output = args.output_dir or ROOT.parent / "pipeline-runs" / f"market-extreme-events-{datetime.now():%Y%m%d-%H%M%S}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    states = io.read_jsonl(source / "daily-states.jsonl")
    if len({(r["stock_code"], r["entry_date"]) for r in states}) != len(states):
        raise ValueError("duplicate stock/date states")
    calendar = sorted({r["trade_date"] for r in io.read_csv(config["benchmark_bars"])})
    as_of = calendar[-1] if args.as_of == "latest" else args.as_of
    calendar = [day for day in calendar if day <= as_of]
    if not calendar:
        raise ValueError("empty observed market calendar")
    qfq = {k: v for k, v in snapshot._read_stock_bars(config["qfq_bars"], adjustment="qfq").items() if k[1] <= as_of}
    raw = {k: v for k, v in snapshot._read_stock_bars(config["raw_bars"], adjustment="raw").items() if k[1] <= as_of}
    cases, stocks, summary = build_study(states, calendar, qfq, raw, count=args.count, horizons=args.horizons, metrics=args.metrics)
    cache = {}
    for case in cases:
        if case["scope"] == "full_history":
            case["examples"] = sample_examples(case, config, cache, config["post_cap"])
    report = {"status": "DESCRIPTIVE_EXTREME_STATE_INSPECTION_COMPLETE", "config": {"source_run": str(source), "count": args.count,
              "horizons": args.horizons, "metrics": args.metrics, "as_of": as_of},
              "data": {"calendar_start": calendar[0], "calendar_end": calendar[-1], "calendar_sessions": len(calendar),
                       "stocks": len({r["stock_code"] for r in states}), "daily_states": len(states),
                       "state_start": min(r["entry_date"] for r in states), "state_end": max(r["entry_date"] for r in states)},
              "summary": summary, "per_stock": stocks,
              "reference_definitions": {
                  "remaining_entry_days": {"label": "其余入场日", "legacy_json_field_prefix": "ordinary",
                                           "rule": "all remaining dates after selected low/high exclusions; not a central-state group"},
                  "central_state_days": {"label": "同股相对常态", "quantiles": [.25, .75],
                                         "quantile_method": "inclusive_linear_interpolation", "include_boundaries": True,
                                         "reference_scope": "full_state_snapshot_per_stock_per_metric",
                                         "exclude_original_selected_low_high": True, "ties_split_by_date": False,
                                         "constant_state_has_no_comparison": True, "membership_fixed_across_horizons": True,
                                         "reads_future_returns": False, "retrospective_not_a_live_threshold": True}},
              "boundaries": {"model_retrained": False, "parameter_search": False, "trading_backtest": False,
                             "extrema_selection_reads_returns": False, "full_history_extrema_are_retrospective": True,
                             "causal_or_statistical_significance_claim": False, "count_is_configurable_not_a_trading_rule": True}}
    output.mkdir(parents=True, exist_ok=True)
    io.write_json(output / "report.json", report)
    io.write_json(output / "run-config.json", report["config"])
    io.write_jsonl(output / "event-cases.jsonl", cases)
    io.write_jsonl(output / "reference-groups.jsonl", reference_memberships(states, args.metrics, args.count))
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    (output / "REPORT.html").write_text(render_html(report, cases), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "data": report["data"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
