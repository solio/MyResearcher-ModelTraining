from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_2_action_balanced as balanced


ROOT = Path(__file__).resolve().parents[1]
ORDER = ["BUY", "ADD", "HOLD", "DO_T", "REDUCE", "SELL", "WATCH", "NO_ACTION_SIGNAL", "UNKNOWN"]


def _record(label: str) -> m1.M1Record:
    return m1.M1Record(
        sample_id=f"id-{label}-{id(label)}",
        stock_code="000001",
        stock_name="fixture",
        model_text="fixture text",
        label={"action_tendency": label},
        weights={"action_tendency": 1.0},
    )


def test_class_weight_formula_is_train_only_and_clipped():
    records = [_record("BUY"), _record("BUY"), _record("ADD")]
    weights = balanced.class_weight_map(records, ORDER)
    assert balanced.train_class_counts(records, ORDER)["BUY"] == 2
    assert balanced.train_class_counts(records, ORDER)["ADD"] == 1
    assert math.isclose(weights["BUY"], 0.5)  # sqrt(3 / (9*2)) clips at the lower bound
    assert math.isclose(weights["ADD"], math.sqrt(1 / 3))
    assert weights["REDUCE"] == 4.0  # zero-count classes use the upper bound


def test_hard_loss_multiplies_class_and_sample_weights():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    sample_weights = torch.tensor([2.0, 1.0])
    class_weights = torch.tensor([3.0, 1.0])
    observed = balanced.hard_weighted_loss(torch, logits, labels, sample_weights, class_weights)
    raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (raw * torch.tensor([6.0, 1.0])).sum() / 3.0
    assert torch.isclose(observed, expected)


def test_prediction_balance_reports_supported_classes_and_collapse():
    probabilities = np.asarray([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]])
    summary = balanced.prediction_balance_summary(probabilities, ["BUY", "ADD", "HOLD"], ["BUY", "ADD", "HOLD"])
    assert summary["supported_predicted_class_count"] == 1
    assert summary["dominant_prediction_ratio"] == 1.0
    assert summary["collapse_detected"] is True


def test_three_seed_gate_requires_stability_and_noncollapse():
    def item(seed: int, macro: float, supported: int = 6, collapsed: bool = False):
        return {"seed": seed, "dev": {"macro_f1": macro}, "balance": {"supported_predicted_class_count": supported, "collapse_detected": collapsed}}

    passed = balanced.three_seed_gate([item(35, 0.31), item(71, 0.30), item(107, 0.32)])
    assert passed["passed"] is True
    failed = balanced.three_seed_gate([item(35, 0.31), item(71, 0.10, supported=1, collapsed=True), item(107, 0.32)])
    assert failed["passed"] is False
    assert failed["checks"]["each_seed_predicts_at_least_5_supported_classes"] is False
    assert failed["checks"]["no_seed_over_90_percent_single_class"] is False


def test_scope_is_exactly_action_and_fixed_three_seeds():
    assert balanced.HEAD == "action_tendency"
    assert balanced.SEEDS == (35, 71, 107)
    assert balanced.MAX_LENGTH == 256
    assert balanced.BATCH_SIZE == 16
    source = (ROOT / "src/semantic_model/encoder_m2_2_action_balanced.py").read_text(encoding="utf-8")
    assert "human100" in source
    assert "Test" in source and "OOD" in source and "production" in source
    assert "sarc\u0061sm_irony_threshold" in source


def test_m21_softlabel_is_comparator_only_in_tracked_report_and_runner():
    report = (ROOT / "reports/m2-1-semantic-softlabel-two-head-20260904.md").read_text(encoding="utf-8")
    source = (ROOT / "src/semantic_model/encoder_m2_1_softlabel.py").read_text(encoding="utf-8")
    assert "SOFTLABEL_NO_GAIN_VS_HARD_LABEL_COMPARATOR" in report
    assert "SOFTLABEL_NO_GAIN_VS_HARD_LABEL_COMPARATOR" in source
    assert "not integrated into" in report and "unified candidate" in report
