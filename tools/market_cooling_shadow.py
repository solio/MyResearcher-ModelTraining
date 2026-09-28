"""One frozen, retrospective check of cooling after pessimistic forum activity.

Uses saved stock-day semantic states and existing market bars only. This does
not retrain the semantic model, select thresholds from returns, or change the
accepted extreme-state report.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import market_extreme_events as events
from tools import market_signal_v2_snapshot as snapshot
from tools import market_validation_v0 as io

DEFAULT_SOURCE = ROOT.parent / "pipeline-runs/market-extreme-unenriched-20260928/source"
REFERENCE_DAYS = 20
WINDOW_DAYS = 5
HORIZONS = (10, 20)


def find_candidates(states, calendar, raw):
    """Classify at the signal day's close; entry is the *next* session's open."""
    by_stock = defaultdict(dict)
    for row in states:
        by_stock[row["stock_code"]][row["entry_date"]] = row
    candidates = []
    for code, series in sorted(by_stock.items()):
        for index in range(REFERENCE_DAYS + WINDOW_DAYS - 1, len(calendar) - 1):
            history = calendar[index - REFERENCE_DAYS - WINDOW_DAYS + 1:index + 1]
            if not all(day in series and (code, day) in raw for day in history):
                continue
            reference = history[:REFERENCE_DAYS]
            prior, recent = history[-2 * WINDOW_DAYS:-WINDOW_DAYS], history[-WINDOW_DAYS:]
            q25 = statistics.quantiles([series[day]["features"]["stance_index"] for day in reference],
                                       n=4, method="inclusive")[0]
            stance_prior = statistics.fmean(series[day]["features"]["stance_index"] for day in prior)
            if stance_prior > q25:
                continue
            stance_recent = statistics.fmean(series[day]["features"]["stance_index"] for day in recent)
            posts_prior = sum(series[day]["observed_posts"] for day in prior)
            posts_recent = sum(series[day]["observed_posts"] for day in recent)
            volume_prior = sum(float(raw[(code, day)]["volume"]) for day in prior)
            volume_recent = sum(float(raw[(code, day)]["volume"]) for day in recent)
            if min(posts_prior, volume_prior) <= 0:
                continue
            cooled = stance_recent > stance_prior and posts_recent < posts_prior and volume_recent < volume_prior
            candidates.append({"stock_code": code, "stock_name": series[calendar[index]]["stock_name"],
                               "signal_date": calendar[index], "entry_date": calendar[index + 1],
                               "cooled": cooled, "previous_pessimism_q25": q25,
                               "stance_previous_5": stance_prior, "stance_recent_5": stance_recent,
                               "posts_previous_5": posts_prior, "posts_recent_5": posts_recent,
                               "volume_previous_5": volume_prior, "volume_recent_5": volume_recent})
    return candidates


def summarize(candidates, calendar, qfq, raw):
    positions = {day: index for index, day in enumerate(calendar)}

    def episodes(rows):
        indexes = sorted(positions[row["signal_date"]] for row in rows)
        return int(bool(indexes)) + sum(right != left + 1 for left, right in zip(indexes, indexes[1:]))

    for row in candidates:
        row["outcomes"] = {str(h): events.outcome(row["stock_code"], row["entry_date"], h, calendar, qfq, raw)
                           for h in HORIZONS}
    by_stock = defaultdict(list)
    for row in candidates:
        by_stock[row["stock_code"]].append(row)
    stock_results = []
    for code, members in sorted(by_stock.items()):
        for horizon in HORIZONS:
            groups = {}
            for cooled in (True, False):
                selected = [row for row in members if row["cooled"] == cooled]
                complete_cases = [row for row in selected if row["outcomes"][str(horizon)]["status"] == "COMPLETE"]
                completed = [row["outcomes"][str(horizon)] for row in complete_cases]
                groups["cooled" if cooled else "other"] = {
                    "signal_dates": len(selected), "complete": len(completed), "complete_episodes": episodes(complete_cases),
                    "mean_return": events.mean([row["return"] for row in completed]),
                    "mean_best_from_entry": events.mean([row["best_from_entry"] for row in completed]),
                    "mean_worst_from_entry": events.mean([row["worst_from_entry"] for row in completed]),
                }
            stock_results.append({"stock_code": code, "stock_name": members[0]["stock_name"],
                                  "horizon": horizon, "cooled": groups["cooled"], "other": groups["other"],
                                  "paired_return_difference": (groups["cooled"]["mean_return"] - groups["other"]["mean_return"]
                                                               if groups["cooled"]["mean_return"] is not None and groups["other"]["mean_return"] is not None else None)})
    return stock_results


def render_markdown(report):
    lines = ["# 悲观舆论之后的冷却：独立影子观察", "",
             "先看个股，不把16股平均收益当买入结果。原模型、原预测、原极端状态报告均不改变。", "",
             "定义（当日收盘可确认，次一交易日开盘作统一入场假设）：先看此前20个有状态的交易日；若前一段5日平均多空状态低于该20日的Q25，记为‘此前较悲观’。再比较最近5日与紧前5日：平均多空状态上升、帖子总数减少、原始成交量总数减少，三者均成立才叫‘冷却’。对照是同样此前较悲观但未同时满足三条件的日期。", "",
             "这不是对已知收益的阈值搜索。情绪缓和不等于变成看多，成交量下降不等于确定见底。价格观察为次日开盘至第10/20个实际交易日收盘；最大浮盈是事后最高价，非实际可获收益。连续日期可能属于同一段事件，不是独立机会。", "",
             f"语义输入：{report['source_run']}；行情截至{report['calendar_end']}。", ""]
    for horizon in HORIZONS:
        lines += [f"## {horizon}日：逐股", "",
                  "| 股票 | 冷却已到期/候选（连续段） | 未完整冷却已到期/候选（连续段） | 冷却期末收益 | 对照期末收益 | 冷却平均最大浮盈 | 对照平均最大浮盈 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in report["per_stock"]:
            if row["horizon"] == horizon:
                c, o = row["cooled"], row["other"]
                lines.append(f"| {row['stock_name']} {row['stock_code']} | {c['complete']}/{c['signal_dates']}（{c['complete_episodes']}段） | {o['complete']}/{o['signal_dates']}（{o['complete_episodes']}段） | {events.pct(c['mean_return'])} | {events.pct(o['mean_return'])} | {events.pct(c['mean_best_from_entry'])} | {events.pct(o['mean_best_from_entry'])} |")
        lines += ["", "— 表示没有可比较的完整结果，不是0%。括号内连续段只合并相邻交易日；即便不相邻，10/20日观察期仍可能重叠。表中不能据已知历史收益挑现在买哪只。", ""]
    lines += ["## 这份影子观察能与不能回答什么", "",
              "它只检查‘冷却’这种现象在现有已存历史中是否存在、逐股之后的行情是否有可见差别。",
              "未匹配此前价格跌幅、量价环境和市场共同变化；帖量下降也可能受采集覆盖影响。语义状态在行情之后生成，属于回顾性观察。",
              "它不是已验证的交易信号，不能因平均最大浮盈为正宣称可赚钱；也不替代原最看空/常态对照。",
              "逐日条件、入场时间、到期/缺行情状态见 cases.jsonl。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new output directory: {output}")
    config = json.loads((source / "run-config.json").read_text())
    states = io.read_jsonl(source / "daily-states.jsonl")
    calendar = sorted({row["trade_date"] for row in io.read_csv(config["benchmark_bars"])})
    qfq = snapshot._read_stock_bars(config["qfq_bars"], adjustment="qfq")
    raw = snapshot._read_stock_bars(config["raw_bars"], adjustment="raw")
    candidates = find_candidates(states, calendar, raw)
    per_stock = summarize(candidates, calendar, qfq, raw)
    report = {"status": "RETROSPECTIVE_COOLING_SHADOW_ONLY", "source_run": str(source),
              "calendar_end": calendar[-1], "definition": {"reference_days": REFERENCE_DAYS,
              "window_days": WINDOW_DAYS, "horizons": HORIZONS,
              "group_rule": "prior-5 stance mean <= q25 of prior-20 stance; recent-5 stance up, posts down, volume down",
              "signal_available": "signal date close", "entry_assumption": "next observed trading date open",
              "profit_selection_uses_future_returns": False}, "per_stock": per_stock,
              "candidate_dates": len(candidates), "cooled_dates": sum(row["cooled"] for row in candidates),
              "complete_by_horizon": {str(h): {"cooled_dates": sum(row["cooled"]["complete"] for row in per_stock if row["horizon"] == h),
                                             "cooled_adjacent_episodes": sum(row["cooled"]["complete_episodes"] for row in per_stock if row["horizon"] == h),
                                             "paired_stocks": sum(row["paired_return_difference"] is not None for row in per_stock if row["horizon"] == h)}
                                      for h in HORIZONS}}
    output.mkdir(parents=True, exist_ok=True)
    io.write_json(output / "report.json", report)
    io.write_jsonl(output / "cases.jsonl", candidates)
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output / "REPORT.md"), "candidate_dates": report["candidate_dates"],
                      "cooled_dates": report["cooled_dates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
