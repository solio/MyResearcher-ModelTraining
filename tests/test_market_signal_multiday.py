from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import market_signal_multiday as signal


def states_and_bars():
    # No 9/25 bar: the horizon must follow actual sessions across the holiday.
    calendar = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-28", "2026-09-29"]
    states, bars, benchmark = [], {}, []
    for i, day in enumerate(calendar):
        states.append({"stock_code": "600000", "entry_date": day,
                       "features": {key: float(i + 1) for key in signal.CORE_STATE},
                       "observed_posts": 20, "sampled_posts": 16})
        bars[("600000", day)] = {"instrument_code": "600000", "trade_date": day, "open": 10,
                                  "close": 10 + i, "volume": 100 + i, "amount": 1000, "fetched_at": "same"}
        benchmark.append({"instrument_code": "000300", "trade_date": day, "open": 10, "close": 10, "volume": 100})
    return calendar, states, bars, benchmark


def panel_for(states, calendar, bars, benchmark, *, coverage=1.0):
    return signal.build_panel(states, calendar, bars, bars, benchmark, window=2, horizon=3,
                              as_of="2026-09-29", min_coverage=coverage, return_threshold=0.0)


def test_rolling_state_uses_last_n_signal_sessions_and_holiday_horizon():
    calendar, states, bars, benchmark = states_and_bars()
    rows = panel_for(states, calendar, bars, benchmark)
    row = next(r for r in rows if r["entry_date"] == "2026-09-24")
    assert row["crowd_features"]["mean::stance_index"] == 3.5
    assert row["crowd_features"]["slope::stance_index"] == 1.0
    assert row["outcome_date"] == "2026-09-29"
    assert row["stock_return"] == pytest.approx(0.5)
    altered = copy.deepcopy(states)
    for state in altered:
        if state["entry_date"] > "2026-09-24":
            state["features"] = {key: 999999.0 for key in signal.CORE_STATE}
    other = next(r for r in panel_for(altered, calendar, bars, benchmark) if r["entry_date"] == "2026-09-24")
    assert row["crowd_features"] == other["crowd_features"]
    assert row["market_features"] == other["market_features"]


def test_missing_signal_day_is_not_compressed_or_imputed_as_zero():
    calendar, states, bars, benchmark = states_and_bars()
    states = [r for r in states if r["entry_date"] != "2026-09-23"]
    row = next(r for r in panel_for(states, calendar, bars, benchmark) if r["entry_date"] == "2026-09-24")
    assert row["status"] == "MISSING_SIGNAL_SESSIONS"
    assert row["missing_signal_sessions"] == ["2026-09-23"]
    assert row["signal_coverage"] == 0.5
    permissive = next(r for r in panel_for(states, calendar, bars, benchmark, coverage=0.5) if r["entry_date"] == "2026-09-24")
    assert permissive["crowd_features"]["mean::stance_index"] == 4.0
    assert permissive["signal_coverage"] == 0.5


def synthetic_panel(horizon=3):
    rows = []
    for i in range(16):
        for stock in range(2):
            rows.append({"stock_code": f"60000{stock}", "entry_date": f"2026-08-{i + 1:02d}",
                         "entry_index": i, "outcome_date": f"2026-08-{i + horizon:02d}",
                         "outcome_index": i + horizon - 1, "status": "COMPLETE",
                         "window_trading_days": 5, "horizon_trading_days": horizon,
                         "stock_positive": bool((i + stock) % 2), "stock_return": 0.02 if (i + stock) % 2 else -0.01,
                         "crowd_features": {"state": float(i % 3)}, "market_features": {"price": float(stock)}})
    return rows


def run_cv(panel, *, block=1, embargo=0):
    return signal.time_cross_validation(panel, min_train_dates=3, test_block_dates=block,
               train_window_dates=0, embargo_sessions=embargo, c=0.1, seed=123)


def test_time_cv_purges_unmatured_labels_and_keeps_test_blocks_disjoint():
    predictions, folds = run_cv(synthetic_panel(), block=2, embargo=1)
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    assert predictions and evaluated
    assert len({(r["stock_code"], r["entry_date"]) for r in predictions}) == len(predictions)
    assert all(fold["train_outcome_max"] < fold["test_start"] for fold in evaluated)
    assert all(set(r["probabilities"]) == set(signal.MODELS) for r in predictions)
    assert all(0 <= p <= 1 for r in predictions for p in r["probabilities"].values())
    # The first test date in each block is index i; embargo=1 requires outcome < i-1.
    for fold in evaluated:
        first = int(fold["test_start"][-2:]) - 1
        latest_outcome = int(fold["train_outcome_max"][-2:]) - 1
        assert latest_outcome < first - 1


def test_future_feature_and_label_changes_do_not_change_earlier_predictions():
    panel = synthetic_panel()
    before, _ = run_cv(panel)
    first_date = min(r["entry_date"] for r in before)
    changed = copy.deepcopy(panel)
    for row in changed:
        if row["entry_date"] > first_date:
            row["stock_positive"] = not row["stock_positive"]
            row["crowd_features"]["state"] = 100000.0
    after, _ = run_cv(changed)
    assert [r for r in before if r["entry_date"] == first_date] == [r for r in after if r["entry_date"] == first_date]


def test_long_horizon_with_no_eligible_training_returns_explicit_empty_cv():
    panel = synthetic_panel(horizon=30)
    predictions, folds = run_cv(panel)
    assert not predictions
    assert {r["status"] for r in folds} == {"INSUFFICIENT_MATURED_TRAIN_DATES"}


def test_no_bootstrap_interval_when_there_are_not_two_time_blocks():
    rows = [{"entry_date": f"2026-08-{i + 1:02d}", "stock_code": "600000", "stock_positive": True,
             "probabilities": {"candidate": 0.8, "baseline": 0.5}} for i in range(8)]
    result = signal.paired_gain(rows, "candidate", "baseline", block_dates=20, repeats=100, seed=1)
    assert result["brier_gain"] == pytest.approx(0.21)
    assert result["ci95"] is None
    assert result["bootstrap_status"] == "TOO_FEW_DATES_FOR_TWO_BLOCKS"


def test_equal_probability_ties_do_not_manufacture_top_quartile_performance():
    rows = [{"entry_date": "2026-08-01", "stock_code": f"60000{i}", "stock_positive": bool(i % 2),
             "stock_return": float(i), "probabilities": {"model": 0.5}} for i in range(4)]
    result = signal.rank_metrics(rows, "model")
    assert result["top_quartile_positive_rate"] is None
    assert result["mean_daily_spearman"] is None


def test_partial_quartile_boundary_ties_share_weight_instead_of_stock_code_selection():
    rows = [{"entry_date": "2026-08-01", "stock_code": f"60000{i}", "stock_positive": i == 1,
             "stock_return": float(i), "probabilities": {"model": 0.1 if i == 0 else 0.7}} for i in range(4)]
    result = signal.rank_metrics(rows, "model")
    assert result["top_quartile_positive_rate"] == pytest.approx(1 / 3)
    assert result["top_quartile_mean_return"] == pytest.approx(2.0)


def test_config_parameters_can_be_overridden_on_command_line(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"windows": [5, 20], "horizons": [10], "min_train_dates": 5}), encoding="utf-8")
    args = signal.parse_args(["--config", str(config), "--horizons", "10", "20", "--min-train-dates", "8"])
    assert args.windows == [5, 20]
    assert args.horizons == [10, 20]
    assert args.min_train_dates == 8


def test_common_window_control_has_identical_training_and_market_baselines():
    panels = {}
    for window in (5, 20):
        rows = synthetic_panel()
        for r in rows:
            r["window_trading_days"] = window
            r["market_features"]["window_volume_mean"] = float(window)
            for feature in ("stance_index", "action_index", "emotion_balance"):
                r["crowd_features"][f"mean::{feature}"] = r["crowd_features"]["state"]
        if window == 20:
            rows = [r for r in rows if r["entry_index"] >= 2]
        panels[(window, 3)] = rows
    args = signal.parse_args(["--windows", "5", "20", "--horizons", "3", "--min-train-dates", "3", "--bootstrap-repeats", "0"])
    reports, predictions, folds = signal.common_sample_control(panels, args, 16)
    assert reports[0]["training_and_test_identities_identical_across_N"]
    one = {(r["stock_code"], r["entry_date"]): r for r in predictions if r["window_trading_days"] == 5}
    two = {(r["stock_code"], r["entry_date"]): r for r in predictions if r["window_trading_days"] == 20}
    assert set(one) == set(two) and one
    for key in one:
        assert one[key]["training_rows"] == two[key]["training_rows"]
        assert one[key]["probabilities"]["market_only"] == two[key]["probabilities"]["market_only"]
        assert one[key]["probabilities"]["stock_prior"] == two[key]["probabilities"]["stock_prior"]


def test_standalone_help_from_another_directory_needs_no_pythonpath(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([os.sys.executable, str(Path(signal.__file__).resolve()), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--windows" in result.stdout and "--config" in result.stdout
