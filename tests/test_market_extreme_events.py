from copy import deepcopy

import pytest

from tools import market_extreme_events as study


def fixture():
    calendar = [f"2026-01-{n:02d}" for n in range(1, 9)]
    qfq = {("000001", d): {"open": 10, "close": 10 + i / 10, "low": 9.5,
                           "high": 11 + i / 10, "fetched_at": "a"}
           for i, d in enumerate(calendar)}
    raw = deepcopy(qfq)
    states = [{"stock_code": "000001", "stock_name": "Example", "entry_date": d,
               "features": {"stance_index": val, "group_disagreement": .2},
               "sampled_posts": 16, "observed_posts": 100, "source_file": "unused",
               "evidence_type": "RETROSPECTIVE_SEMANTIC_FEATURE"}
              for d, val in [(calendar[0], .8), (calendar[1], .3), (calendar[6], -.7),
                             (calendar[7], -.8), ("2026-01-09", -.9)]]
    return states, calendar, qfq, raw


def test_extrema_do_not_read_future_returns_or_drop_pending_minimum():
    states, calendar, qfq, raw = fixture()
    selected = study.extreme_rows(states, "stance_index", 3, "low")
    assert [r["entry_date"] for r in selected] == ["2026-01-09", "2026-01-08", "2026-01-07"]
    cases, _, summary = study.build_study(states, calendar, qfq, raw, count=3, horizons=[3, 60], metrics=["stance_index"])
    primary = [r for r in cases if r["scope"] == "full_history" and r["tail"] == "low"]
    assert [r["entry_date"] for r in primary] == [r["entry_date"] for r in selected]
    assert all(r["outcomes"]["3"]["return"] is None for r in primary)
    assert next(r for r in summary if r["scope"] == "full_history" and r["horizon"] == 3)["low_complete"] == 0
    observed = [r for r in cases if r["scope"] == "observable_history" and r["scope_horizon"] == 3 and r["tail"] == "low"]
    assert {r["entry_date"] for r in observed} == {"2026-01-01", "2026-01-02"}
    assert not any(r for r in cases if r["scope"] == "observable_history" and r["scope_horizon"] == 60)
    qfq[("000001", "2026-01-08")]["close"] = 999
    after, _, _ = study.build_study(states, calendar, qfq, raw, count=3, horizons=[3], metrics=["stance_index"])
    assert [(r["entry_date"], r["rank"]) for r in after if r["scope"] == "full_history" and r["tail"] == "low"] == [(r["entry_date"], r["rank"]) for r in primary]


def test_horizon_counts_entry_day_and_worst_loss_is_relative_to_entry():
    _, calendar, qfq, raw = fixture()
    result = study.outcome("000001", calendar[0], 3, calendar, qfq, raw)
    assert result["exit_date"] == calendar[2]
    assert result["return"] == pytest.approx(.02)
    assert result["worst_from_entry"] == pytest.approx(-.05)
    assert result["best_from_entry"] == pytest.approx(.12)
    assert result["status"] == "COMPLETE"


def test_missing_or_inconsistent_path_does_not_become_zero_or_a_connected_curve():
    _, calendar, qfq, raw = fixture()
    del qfq[("000001", calendar[2])]
    assert study.outcome("000001", calendar[0], 3, calendar, qfq, raw)["status"] == "MISSING_PRICE"
    assert len(study.price_path("000001", calendar[0], calendar, qfq, raw, 60)) == 3
    _, calendar, qfq, raw = fixture()
    qfq[("000001", calendar[2])].update(fetched_at="b", close=20)
    assert study.outcome("000001", calendar[0], 3, calendar, qfq, raw)["status"] == "ADJUSTMENT_VINTAGE_MISMATCH"


def test_count_is_configurable_and_adjacent_dates_are_visible():
    states, calendar, _, _ = fixture()
    assert len(study.extreme_rows(states, "stance_index", 5, "low")) == 5
    assert study.adjacent_groups(states[2:4], calendar) == 1


def test_equal_stock_comparison_is_paired_not_post_count_weighted():
    states, calendar, qfq, raw = fixture()
    second = [dict(r, stock_code="000002", observed_posts=10000) for r in states]
    qfq.update({("000002", d): dict(qfq[("000001", d)]) for d in calendar})
    raw.update({("000002", d): dict(raw[("000001", d)]) for d in calendar})
    _, _, first = study.build_study(states, calendar, qfq, raw, count=1, horizons=[1], metrics=["stance_index"])
    _, _, both = study.build_study(states + second, calendar, qfq, raw, count=1, horizons=[1], metrics=["stance_index"])
    a = next(r for r in first if r["scope"] == "observable_history")
    b = next(r for r in both if r["scope"] == "observable_history")
    assert a["low_mean_return"] == b["low_mean_return"]
    assert a["low_minus_ordinary"] == b["low_minus_ordinary"]
    assert b["paired_ordinary_stocks"] == 2


def test_common_horizons_use_original_extrema_without_replacement():
    states, calendar, qfq, raw = fixture()
    cases, _, summary = study.build_study(states, calendar, qfq, raw, count=3, horizons=[1, 3, 60], metrics=["stance_index"])
    common = [r for r in cases if r["scope"] == "common_matured" and r["tail"] == "low"]
    # None of the actual three global minima have 3-session outcomes.
    assert common == []
    assert {r["horizon"] for r in summary if r["scope"] == "common_matured"} == {1, 3}
    assert all(r["low_complete"] == 0 for r in summary if r["scope"] == "common_matured")


def test_window_high_is_aggregated_separately_from_exit_return_and_rendered():
    states, calendar, qfq, raw = fixture()
    states = [dict(states[0], entry_date=day, features={"stance_index": i, "group_disagreement": .2})
              for i, day in enumerate(calendar)]
    cases, stocks, summary = study.build_study(states, calendar, qfq, raw, count=1, horizons=[3], metrics=["stance_index"])
    primary = next(r for r in summary if r["scope"] == "full_history")
    assert primary["low_complete"] == 1
    assert primary["low_mean_return"] == pytest.approx(.02)
    assert primary["low_mean_best_from_entry"] == pytest.approx(.12)
    assert primary["paired_ordinary_mean_return"] == pytest.approx(.05)
    assert primary["paired_ordinary_mean_best_from_entry"] == pytest.approx(.15)
    assert primary["high_mean_best_from_entry"] is None
    stock = next(r for r in stocks if r["scope"] == "full_history")
    assert stock["ordinary_mean_best_from_entry"] == pytest.approx(.15)
    # Highs outside the requested horizon must not affect its observed maximum.
    qfq[("000001", calendar[3])]["high"] = 999
    assert study.outcome("000001", calendar[0], 3, calendar, qfq, raw)["best_from_entry"] == pytest.approx(.12)
    report = {"config": {"count": 1, "horizons": [3], "metrics": ["stance_index"]},
              "data": {"calendar_end": calendar[-1], "calendar_sessions": len(calendar), "stocks": 1},
              "summary": summary}
    markdown = study.render_markdown(report)
    html = study.render_html(report, cases)
    assert "低值后期间最高收益（均值）" in markdown
    assert "普通日期间最高收益（均值）" in html
    assert "+2.00%／+12.00%" in html
    assert "未到期" in html


def test_window_high_means_average_events_within_stock_then_equal_weight_stocks():
    states, calendar, qfq, raw = fixture()
    first = dict(states[0], entry_date=calendar[0])
    second = [dict(first, stock_code="000002", entry_date=calendar[i],
                   features={"stance_index": i, "group_disagreement": .2}) for i in range(3)]
    for i, day in enumerate(calendar):
        qfq[("000002", day)] = dict(qfq[("000001", day)], high=12 + 2 * i)
        raw[("000002", day)] = dict(qfq[("000002", day)])
    _, _, summary = study.build_study([first] + second, calendar, qfq, raw,
                                      count=3, horizons=[1], metrics=["stance_index"])
    primary = next(r for r in summary if r["scope"] == "full_history")
    assert primary["low_complete"] == 4
    # Stock A: .10; stock B: mean(.20, .40, .60) = .40. Macro mean = .25,
    # not the event-weighted .325 and not the best single event .60.
    assert primary["low_mean_best_from_entry"] == pytest.approx(.25)
