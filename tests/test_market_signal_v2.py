from __future__ import annotations

import pytest

from tools import market_signal_v2 as signal


def test_disagreement_separates_identical_uncertainty_from_group_conflict():
    identical_ambiguous = signal.distribution_state_features(
        [{"BULL": 0.5, "BEAR": 0.5}, {"BULL": 0.5, "BEAR": 0.5}],
        head="stance",
    )
    opposing_confident = signal.distribution_state_features(
        [{"BULL": 1.0, "BEAR": 0.0}, {"BULL": 0.0, "BEAR": 1.0}],
        head="stance",
    )
    assert identical_ambiguous["model_uncertainty::stance"] == pytest.approx(1.0)
    assert identical_ambiguous["group_disagreement::stance"] == pytest.approx(0.0)
    assert identical_ambiguous["cross_post_agreement::stance"] == pytest.approx(1.0)
    assert identical_ambiguous["group_consensus::stance"] == pytest.approx(0.0)
    assert opposing_confident["model_uncertainty::stance"] == pytest.approx(0.0)
    assert opposing_confident["group_disagreement::stance"] == pytest.approx(1.0)
    assert opposing_confident["cross_post_agreement::stance"] == pytest.approx(0.0)
    assert opposing_confident["group_consensus::stance"] == pytest.approx(0.0)


def test_distribution_state_rejects_inconsistent_or_invalid_probabilities():
    with pytest.raises(ValueError, match="inconsistent labels"):
        signal.distribution_state_features(
            [{"BULL": 0.5, "BEAR": 0.5}, {"BULL": 1.0}],
            head="stance",
        )
    with pytest.raises(ValueError, match="invalid probability value"):
        signal.distribution_state_features(
            [{"BULL": 1.1, "BEAR": -0.1}],
            head="stance",
        )


def test_crowd_aggregation_rejects_duplicate_metadata_ids():
    metadata = [
        {"sample_id": "duplicate", "stock_code": "A", "entry_date": "2026-09-01"},
        {"sample_id": "duplicate", "stock_code": "A", "entry_date": "2026-09-01"},
    ]
    predictions = [{"sample_id": "duplicate"}]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        signal.aggregate_crowd_state(metadata, predictions)


def test_smoothed_prior_is_stock_history_safe():
    assert signal.smoothed_prior([]) == pytest.approx(0.5)
    assert signal.smoothed_prior([1, 1, 0]) == pytest.approx(3 / 5)


def test_walk_forward_targets_stock_positive_not_excess_positive():
    rows = []
    for date_index in range(1, 7):
        entry_date = f"2026-09-{date_index:02d}"
        for stock_index, stock_code in enumerate(("000001", "000002")):
            stock_positive = (date_index + stock_index) % 2 == 0
            rows.append(
                {
                    "market_unit_id": f"{stock_code}:{entry_date}",
                    "stock_code": stock_code,
                    "entry_date": entry_date,
                    "outcome_date": entry_date,
                    "horizon_trading_days": 1,
                    "outcome_status": "COMPLETE",
                    "stock_positive": stock_positive,
                    "excess_positive": not stock_positive,
                    "realized_stock_return": 0.01 if stock_positive else -0.01,
                    "crowd_features": {"state": 1.0 if stock_positive else -1.0},
                    "market_features": {"lag": float(stock_index)},
                }
            )
    predictions = signal.walk_forward_absolute_predictions(
        rows,
        feature_sets={"crowd": [("crowd_features", "state")]},
        min_train_dates=2,
    )
    assert predictions
    assert all(set(row["probabilities"]) == {"global_prior", "stock_prior", "crowd"} for row in predictions)
    expected = {
        (row["market_unit_id"], row["entry_date"]): bool(row["stock_positive"])
        for row in rows
    }
    assert all(
        row["stock_positive"] == expected[(row["market_unit_id"], row["entry_date"])]
        for row in predictions
    )
    assert all(
        (row["probabilities"]["crowd"] >= 0.5) is row["stock_positive"]
        for row in predictions
    )


def test_calibration_bins_preserve_absolute_return_mean():
    rows = [
        {
            "stock_positive": False,
            "stock_return": -0.02,
            "probabilities": {"model": 0.1},
        },
        {
            "stock_positive": True,
            "stock_return": 0.03,
            "probabilities": {"model": 0.9},
        },
    ]
    bins = signal.calibration_bins(rows, "model")
    assert len(bins) == 2
    assert bins[0]["positive_rate"] == 0.0
    assert bins[0]["mean_stock_return"] == pytest.approx(-0.02)
    assert bins[1]["positive_rate"] == 1.0
    assert bins[1]["mean_stock_return"] == pytest.approx(0.03)


def test_macro_stock_brier_weights_stocks_equally_when_row_counts_differ():
    rows = [
        {"stock_code": "A", "stock_positive": True, "probabilities": {"model": 1.0}},
        {"stock_code": "A", "stock_positive": True, "probabilities": {"model": 1.0}},
        {"stock_code": "A", "stock_positive": True, "probabilities": {"model": 1.0}},
        {"stock_code": "B", "stock_positive": True, "probabilities": {"model": 0.0}},
    ]
    assert signal.macro_stock_brier(rows, "model") == pytest.approx(0.5)


def test_date_block_bootstrap_keeps_five_date_blocks():
    rows = []
    for day in range(1, 11):
        for stock_code in ("A", "B"):
            rows.append(
                {
                    "entry_date": f"2026-09-{day:02d}",
                    "stock_code": stock_code,
                    "stock_positive": day % 2 == 0,
                    "probabilities": {"baseline": 0.5, "candidate": 0.6 if day % 2 == 0 else 0.4},
                }
            )
    interval = signal._date_block_gain(rows, "baseline", "candidate")
    assert interval["draws"] == 2000
    assert interval["block_entry_dates"] == 5
    assert interval["lower_95"] > 0.0
