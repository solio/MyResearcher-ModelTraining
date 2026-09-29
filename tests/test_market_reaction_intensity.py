"""Focused checks for the retrospective absolute-shock readout."""

from __future__ import annotations

import numpy as np
import pytest

from tools import market_reaction_intensity as intensity


def test_absolute_target_ignores_future_prices_and_signed_direction():
    calendar = ["d0", "d1", "d2"]
    bags = {("000001", "d1"): ["shock"], ("000001", "d2"): ["quiet"]}
    counts = {"raw_counts": {key: 1 for key in bags},
              "unique_counts": {key: 1 for key in bags}}
    qfq = {("000001", "d0"): {"close": 100.0},
           ("000001", "d1"): {"close": 108.0},
           ("000001", "d2"): {"close": 108.0}}
    rows = intensity.make_examples(("d1", "d1"), calendar, ["000001"], qfq,
                                   bags, {"shock": 0}, counts)
    assert len(rows) == 1
    assert rows[0]["target"] == pytest.approx(0.08 / intensity.TARGET_CAP)
    assert rows[0]["extreme"] == 1
    qfq[("000001", "d2")]["close"] = 1.0
    after = intensity.make_examples(("d1", "d1"), calendar, ["000001"], qfq,
                                    bags, {"shock": 0}, counts)
    assert after == rows
    qfq[("000001", "d1")]["close"] = 92.0
    opposite = intensity.make_examples(("d1", "d1"), calendar, ["000001"], qfq,
                                       bags, {"shock": 0}, counts)
    assert opposite[0]["target"] == pytest.approx(rows[0]["target"])


def test_shift_placebo_moves_only_fit_bags_within_stock():
    rows = []
    for code in ("a", "b"):
        for index in range(10):
            rows.append({"code": code, "day": f"d{index:02d}", "ids": [code, index],
                         "target": index / 10, "raw_count": 1})
    shifted = intensity.shift_fit_titles(rows)
    for before, after in zip(rows, shifted):
        assert after["target"] == before["target"]
        assert after["raw_count"] == before["raw_count"]
        assert after["ids"][0] == before["code"]
        assert after["ids"] != before["ids"]
    assert rows[0]["ids"] == ["a", 0]


def test_count_baseline_uses_only_fit_target_and_count():
    fit = [{"code": "a", "target": 0.1, "raw_count": 1, "unique_count": 1},
           {"code": "a", "target": 0.2, "raw_count": 2, "unique_count": 2},
           {"code": "b", "target": 0.3, "raw_count": 3, "unique_count": 3}]
    gate = [{"code": "a", "target": 0.0, "raw_count": 2, "unique_count": 2}]
    late = [{"code": "b", "target": 1.0, "raw_count": 3, "unique_count": 3}]
    before, _ = intensity.fit_count_baseline(fit, gate, late, ["a", "b"])
    gate[0]["target"] = 1.0
    late[0]["target"] = 0.0
    after, _ = intensity.fit_count_baseline(fit, gate, late, ["a", "b"])
    for split in before:
        assert np.array_equal(before[split], after[split])


def test_diagnostic_metric_does_not_infer_direction():
    rows = [{"target": 0.1, "extreme": 0}, {"target": 0.8, "extreme": 1}]
    metrics = intensity.diagnostic_metrics(rows, [0.2, 0.7])
    assert metrics["scaled_target_mse"] == pytest.approx(0.01)
    assert metrics["extreme_auc"] == 1.0
    assert metrics["extreme_average_precision"] == 1.0
