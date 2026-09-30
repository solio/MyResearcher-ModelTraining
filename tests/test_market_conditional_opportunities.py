"""Behavioral checks of timing, skipped entries, and nonoverlapping evaluation."""
import math
import time
from datetime import date, timedelta

import numpy as np
import pytest

from tools import market_conditional_opportunities as opportunities


def fixture_panel(days=90):
    calendar = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(days)]
    states, qfq, raw, benchmark = [], {}, {}, {}
    for i, day in enumerate(calendar):
        features = {key: math.sin(i / (j + 3)) / 2 for j, key in enumerate(opportunities.CROWD_KEYS)}
        states.append({"stock_code": "000001", "stock_name": "Example", "entry_date": day,
                       "observed_posts": 15 + i % 5, "features": features})
        bar = {"open": 10 + i / 10, "close": 10.05 + i / 10, "high": 10.2 + i / 10,
               "low": 9.9 + i / 10, "volume": 100 + i % 7,
               "turnover_rate": 1.0, "fetched_at": "one_snapshot"}
        qfq[("000001", day)] = dict(bar)
        raw[("000001", day)] = dict(bar)
        benchmark[day] = dict(bar)
    return states, calendar, qfq, raw, benchmark


def row_at(panel, day):
    return next(row for row in panel if row["entry_date"] == day)


def test_features_and_conditions_do_not_read_entry_or_future_prices_and_states():
    inputs = fixture_panel()
    states, calendar, qfq, raw, benchmark = inputs
    first = row_at(opportunities.build_panel(*inputs, 20, 10)[0], calendar[40])
    for i in range(40, len(calendar)):
        qfq[("000001", calendar[i])]["close"] *= 30
        raw[("000001", calendar[i])]["volume"] = 999999
        benchmark[calendar[i]]["close"] *= 30
        if i > 40:
            states[i]["features"]["stance_index"] = -99
            states[i]["features"]["emotion_balance"] = 99
            states[i]["observed_posts"] = 999999
    changed = row_at(opportunities.build_panel(*inputs, 20, 10)[0], calendar[40])
    for key in ("market_features", "crowd_features", "candidate", "extreme", "cooling",
                "emotion_flat", "stance_prior20_q10", "emotion_prior20_q25", "volume_recent5"):
        assert changed[key] == first[key]
    assert first["market_feature_last_date"] == calendar[39]
    assert changed["return"] != first["return"]


def test_quantile_references_are_prior_single_day_values_and_constant_guard():
    inputs = fixture_panel()
    states, calendar, *_ = inputs
    current = 40
    expected_q10 = np.quantile([row["features"]["stance_index"] for row in states[20:40]], 0.1)
    states[current]["features"]["stance_index"] = -99
    first = row_at(opportunities.build_panel(*inputs, 5, 10)[0], calendar[current])
    assert first["extreme"] is True
    assert first["stance_prior20_q10"] == pytest.approx(expected_q10)
    for row in states:
        row["features"]["stance_index"] = 0
        row["features"]["emotion_balance"] = 0
    constant = row_at(opportunities.build_panel(*inputs, 5, 10)[0], calendar[current])
    assert constant["extreme"] is False
    assert constant["cooling"] is False
    assert constant["emotion_flat"] is False


def test_cooling_volume_matches_each_state_to_preceding_session_and_full_posts():
    inputs = fixture_panel()
    states, calendar, _, raw, _ = inputs
    for row in states:
        row["features"]["stance_index"] = 0.3
        row["observed_posts"] = 16
    t = 40
    for i in range(t - 9, t - 4):
        states[i]["features"]["stance_index"] = -0.8
        states[i]["observed_posts"] = 160
    for i in range(t - 4, t + 1):
        states[i]["features"]["stance_index"] = -0.2
        states[i]["observed_posts"] = 40
    for i in range(t - 10, t - 5):
        raw[("000001", calendar[i])]["volume"] = 100
    for i in range(t - 5, t):
        raw[("000001", calendar[i])]["volume"] = 50
    candidate = row_at(opportunities.build_panel(*inputs, 5, 10)[0], calendar[t])
    assert candidate["cooling"] is True
    assert candidate["posts_previous5"] == 800
    assert candidate["posts_recent5"] == 200
    assert candidate["volume_previous5"] == 500
    assert candidate["volume_recent5"] == 250
    raw[("000001", calendar[t])]["volume"] = 999999
    assert row_at(opportunities.build_panel(*inputs, 5, 10)[0], calendar[t])["cooling"] is True
    for i in range(t - 5, t):
        raw[("000001", calendar[i])]["volume"] = 150
    assert row_at(opportunities.build_panel(*inputs, 5, 10)[0], calendar[t])["cooling"] is False


def test_outcome_entry_open_and_hth_close_pending_is_not_zero():
    _, calendar, qfq, raw, _ = fixture_panel(30)
    result = opportunities.outcome("000001", 5, 10, calendar, qfq, raw)
    assert result["outcome_date"] == calendar[14]
    assert result["return"] == pytest.approx(qfq[("000001", calendar[14])]["close"] /
                                               qfq[("000001", calendar[5])]["open"] - 1)
    assert opportunities.outcome("000001", 25, 10, calendar, qfq, raw)["return"] is None


def test_walk_forward_fits_only_labels_strictly_mature_before_first_test_day():
    calendar = fixture_panel(40)[1]
    panel = [{"stock_code": "000001", "entry_date": calendar[i],
              "outcome_date": calendar[i + 9], "status": "COMPLETE"} for i in range(25)]
    folds, skipped = opportunities.make_folds(panel, min_fit_dates=3, test_block=5)
    first = folds[0]
    assert first["first_entry_date"] == calendar[12]
    assert len(first["fit"]) == 3
    assert first["test_dates"] == calendar[12:17]
    assert all(row["outcome_date"] < first["first_entry_date"] for row in first["fit"])
    assert panel[3] not in first["fit"]  # matures exactly at test entry, therefore excluded
    assert not set(skipped) & {day for fold in folds for day in fold["test_dates"]}
    assert len({day for fold in folds for day in fold["test_dates"]}) == sum(len(fold["test_dates"]) for fold in folds)


def test_flat_reference_requires_two_same_stock_mature_samples_and_gate_uses_only_fit():
    fit = [{"stock_code": "A", "emotion_flat": True, "return": 0.03},
           {"stock_code": "A", "emotion_flat": True, "return": 0.05},
           {"stock_code": "B", "emotion_flat": True, "return": -0.02},
           {"stock_code": "A", "emotion_flat": False, "return": 0.99}]
    reference = opportunities.fit_flat_reference(fit)
    expected = (0.08 + 10 * (0.06 / 3)) / 12
    assert opportunities.flat_reference_for(reference, "A") == pytest.approx(expected)
    assert opportunities.flat_reference_for(reference, "B") is None
    assert opportunities.flat_reference_for(reference, "C") is None
    assert not opportunities.entry_gate(True, 0.8, opportunities.flat_reference_for(reference, "B"))
    assert not opportunities.entry_gate(True, 0.001, expected)
    assert not opportunities.entry_gate(True, expected, expected)
    assert opportunities.entry_gate(True, expected + 0.001, expected)
    assert not opportunities.entry_gate(False, 0.8, expected)


def test_ridge_scaling_is_fit_only_and_unseen_extreme_test_does_not_change_fit():
    x = np.array([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
    y = np.array([0.01, 0.02, 0.03, 0.04])
    model = opportunities.fit_market_ridge(x, y)
    assert model["mean"].tolist() == [1.5, 1.0]
    assert model["std"][1] == 1
    before = opportunities.market_predict(model, x)
    opportunities.market_predict(model, np.array([[99999., -99999.]]))
    assert np.array_equal(before, opportunities.market_predict(model, x))


def test_nonoverlap_main_result_skips_hold_period_and_none_is_not_success():
    calendar = fixture_panel(40)[1]
    rows = []
    for i, ret in ((5, .02), (6, .50), (7, -.40), (15, .03), (16, .01)):
        selections = {variant: False for variant in opportunities.VARIANTS}
        selections["all_rows"] = True
        selections["residual_ensemble_gate"] = i in (5, 6, 7, 15)
        rows.append({"stock_code": "A", "stock_name": "A", "entry_date": calendar[i],
                     "horizon": 10, "emotion_flat": i in (6, 7, 16), "candidate": True,
                     "return": ret, "selections": selections})
    summary = opportunities.summarize(rows, calendar)
    stock = summary["per_stock"][0]
    chosen = stock["variants"]["residual_ensemble_gate"]
    assert chosen["selected_count"] == 4
    assert chosen["episodes"] == 2
    assert chosen["primary_nonoverlap"]["entry_dates"] == [calendar[5], calendar[15]]
    assert chosen["primary_nonoverlap"]["mean_actual_return"] == pytest.approx(.025)
    assert stock["primary_nonoverlap_flat"]["entry_dates"] == [calendar[6], calendar[16]]
    assert chosen["primary_nonoverlap"]["passes_both"] is False
    assert stock["variants"]["market_gate"]["mean_actual_return"] is None
    assert stock["variants"]["market_gate"]["primary_nonoverlap"]["passes_both"] is None
    aggregate = summary["primary_nonoverlap_equal_stock_aggregate"]["market_gate"]
    assert aggregate["stocks_without_selection"] == 1
    assert aggregate["passing_paired_stocks"] == 0


def test_fold_residual_seed_mean_precedes_gate_checkpoint_cpu_reload(tmp_path):
    torch = pytest.importorskip("torch")
    inputs = fixture_panel(70)
    panel, _ = opportunities.build_panel(*inputs, 5, 10)
    folds, _ = opportunities.make_folds(panel, 3, 2)
    records, log = opportunities.evaluate_fold(folds[0], torch=torch, output=tmp_path,
                                              n=5, horizon=10, started=time.monotonic(), max_seconds=60)
    assert len(log["seeds"]) == 6
    assert len(list(tmp_path.glob("*.pt"))) == 6
    assert all(item["cpu_reload_max_abs_delta"] <= 1e-7 for item in log["seeds"])
    for row in records:
        expected = np.mean(list(row["predicted_residual_returns"].values()))
        assert row["predicted_ensemble_return"] == pytest.approx(expected)
        assert row["selections"]["residual_ensemble_gate"] == opportunities.entry_gate(
            row["candidate"], expected, row["fit_flat_reference"])
        count_expected = np.mean(list(row["predicted_market_count_returns"].values()))
        assert row["predicted_market_count_ensemble_return"] == pytest.approx(count_expected)
        assert row["selections"]["market_count_ensemble_gate"] == opportunities.entry_gate(
            row["candidate"], count_expected, row["fit_flat_reference"])
        assert row["fit_last_outcome_date"] < row["fit_cutoff_exclusive"]


def test_title_window_uses_only_available_dates_and_projection_fits_training_only():
    inputs = fixture_panel(70)
    panel, _ = opportunities.build_panel(*inputs, 5, 10)
    calendar = inputs[1]
    vectors = {("000001", day): np.array([i, i**2, 1.0]) for i, day in enumerate(calendar)}
    code = panel[0]["stock_code"]
    vectors = {(code, day): vector for (_, day), vector in vectors.items()}
    opportunities.attach_title_windows(panel, vectors, calendar, 5)
    target = panel[0]
    t = target["entry_index"]
    expected = np.mean([vectors[(code, day)] for day in calendar[t-4:t+1]], axis=0)
    assert target["title_vector"] == pytest.approx(expected)
    frozen = list(target["title_vector"])
    for day in calendar[t+1:]:
        vectors[(code, day)] = np.array([1e9, -1e9, 1e9])
    opportunities.attach_title_windows(panel, vectors, calendar, 5)
    assert panel[0]["title_vector"] == pytest.approx(frozen)
    fit = np.array([[1., 2., 3.], [2., 1., 3.], [3., 0., 4.]])
    projection = opportunities.title_projection_fit(fit, 2)
    assert projection["mean"] == pytest.approx(np.mean(fit, axis=0))
    before = opportunities.title_project(projection, fit).copy()
    opportunities.title_project(projection, np.full((2, 3), 1e9))
    assert opportunities.title_project(projection, fit) == pytest.approx(before)


def test_title_branch_ensemble_and_fold_specific_projection_are_saved(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(opportunities, "EPOCHS", 2)
    inputs = fixture_panel(70)
    panel, _ = opportunities.build_panel(*inputs, 5, 10)
    for row in panel:
        row["title_vector"] = [float(row["entry_index"]), float(row["entry_index"] % 3), 1.0]
    folds, _ = opportunities.make_folds(panel, 3, 2)
    records, log = opportunities.evaluate_fold(folds[0], torch=torch, output=tmp_path,
                                              n=5, horizon=10, started=time.monotonic(), max_seconds=60)
    assert len(log["seeds"]) == 9
    for row in records:
        assert "title_vector" not in row
        expected = np.mean(list(row["predicted_title_returns"].values()))
        assert row["predicted_title_ensemble_return"] == pytest.approx(expected)
        assert row["selections"]["title_ensemble_gate"] == opportunities.entry_gate(
            row["candidate"], expected, row["fit_flat_reference"])
    checkpoint = next(tmp_path.glob("*-all-title-seed*.pt"))
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert stored["title_projection"]["mean"] == pytest.approx(np.mean(
        [row["title_vector"] for row in folds[0]["fit"]], axis=0))


def test_adaptive_calibration_uses_only_matured_prior_oos_and_warmup_is_no_selection():
    import copy
    calendar = [f"2026-07-{i+1:02d}" for i in range(20)]
    records = []
    for i in range(12):
        records.append({"fold": i, "entry_date": calendar[i], "outcome_date": calendar[i+2],
                        "market_features": [i / 100, -i / 200], "return": i / 1000 - .003,
                        "candidate": True, "fit_flat_reference": -.01, "selections": {},
                        "predicted_market_count_ensemble_return": i / 10,
                        "predicted_ensemble_return": i / 8,
                        "predicted_title_ensemble_return": i / 6})
    changed = copy.deepcopy(records)
    logs = opportunities.calibrate_out_of_sample(records)
    for row in records[:7]:
        assert row["predicted_title_calibrated_return"] is None
        assert row["selections"]["title_calibrated_gate"] is False
    target = records[8]
    assert target["calibration_fit_dates"] == 6
    for row in changed:
        if row["outcome_date"] >= target["entry_date"]:
            row["return"] = 1000.
    opportunities.calibrate_out_of_sample(changed)
    assert changed[8]["predicted_title_calibrated_return"] == pytest.approx(
        target["predicted_title_calibrated_return"])
    assert all(item["fit_last_outcome_date"] < item["first_entry_date"]
               for item in logs if item["status"] == "FITTED")
