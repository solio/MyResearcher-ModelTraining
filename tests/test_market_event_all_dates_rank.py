from __future__ import annotations

import copy

import numpy as np
import pytest

from tools import market_event_all_dates_rank as experiment


def _fixed_panel():
    fit_dates = [f"2026-05-{day:02}" for day in range(1, 21)]
    diagnostic_dates = [f"2026-06-{day:02}" for day in range(26, 31)] + [
        f"2026-07-{day:02}" for day in range(1, 13)]
    calendar = fit_dates + ["2026-06-24", "2026-06-25"] + diagnostic_dates
    fit = [{"stock_code": f"{stock:06}", "entry_date": day,
            "outcome_date": "2026-06-23", "market": [float(index), 1.0],
            "return": float(index) / 100, "post_ids": [f"fit-{stock}-{index}"],
            "ages": [0.0]}
           for stock in range(16) for index, day in enumerate(fit_dates)]
    diagnostic = [{"stock_code": f"{stock:06}", "entry_date": day,
                   "outcome_date": "2026-07-31", "market": [100.0 + index, 1.0],
                   "return": float(index) / 100, "post_ids": [f"july-{stock}-{index}"],
                   "ages": [0.0]}
                  for stock in range(16) for index, day in enumerate(diagnostic_dates)]
    return calendar, fit, diagnostic


def test_complete_windows_cover_all_fit_days_and_disjoint_july_blocks():
    calendar, fit, diagnostic = _fixed_panel()
    fit_cases, fit_unused = experiment.complete_windows(fit, calendar, role="fit")
    july_cases, july_unused = experiment.complete_windows(
        diagnostic, calendar, role="july_diagnostic")
    shock_cases = [{"event_id": str(index)} for index in range(12)]

    experiment.validate_contract(fit, diagnostic, fit_cases, july_cases,
                                 july_unused, shock_cases)
    assert (len(fit_cases), len(july_cases), len(fit_unused), len(july_unused)) == (64, 48, 0, 32)
    assert len({(row["stock_code"], row["entry_date"])
                for case in fit_cases for row in case["candidates"]}) == 320
    assert len({(row["stock_code"], row["entry_date"])
                for case in july_cases for row in case["candidates"]}) == 240
    assert all(len(case["candidates"]) == 5 for case in (*fit_cases, *july_cases))
    assert july_cases[0]["window_start"] == "2026-06-26"
    assert july_cases[-1]["window_end"] == "2026-07-10"
    assert {date for _, date in july_unused} == {"2026-07-11", "2026-07-12"}
    assert [(case["event_id"], case["window_start"], case["window_end"])
            for case in july_cases] == [
                (case["event_id"], case["window_start"], case["window_end"])
                for case in experiment.complete_windows(diagnostic, calendar,
                                                        role="july_diagnostic")[0]]


def test_complete_windows_reject_gaps_duplicates_and_unknown_dates():
    calendar, fit, _ = _fixed_panel()
    with pytest.raises(ValueError, match="nonconsecutive"):
        experiment.complete_windows([*fit[:1], *fit[2:]], calendar, role="fit")
    with pytest.raises(ValueError, match="duplicate"):
        experiment.complete_windows([*fit, fit[0]], calendar, role="fit")
    unknown = copy.deepcopy(fit)
    unknown[0]["entry_date"] = "2026-04-30"
    with pytest.raises(ValueError, match="unknown"):
        experiment.complete_windows(unknown, calendar, role="fit")


def test_contract_rejects_fit_label_after_fixed_june_23_end():
    calendar, fit, diagnostic = _fixed_panel()
    fit_cases, _ = experiment.complete_windows(fit, calendar, role="fit")
    july_cases, july_unused = experiment.complete_windows(
        diagnostic, calendar, role="july_diagnostic")
    fit[0]["outcome_date"] = "2026-06-27"
    with pytest.raises(ValueError, match="fit labels no longer end June 23"):
        experiment.validate_contract(fit, diagnostic, fit_cases, july_cases,
                                     july_unused, [{}] * 12)


def test_market_scaler_uses_only_fit_rows_and_keeps_count_feature():
    fit = [{"market": [1.0, 2.0]}, {"market": [3.0, 2.0]}]
    diagnostic = [{"market": [1000.0, 2000.0]}]
    mean, std = experiment.scale_market(fit, diagnostic)
    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(std, [1.0, 1.0])
    np.testing.assert_allclose(fit[0]["market_scaled"], [-1.0, 0.0])
    np.testing.assert_allclose(fit[1]["market_scaled"], [1.0, 0.0])
    np.testing.assert_allclose(diagnostic[0]["market_scaled"], [998.0, 1998.0])
