from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from semantic_model.encoder_m3f_fundamental_hard_negative import (
    HARD_NEGATIVE_MULTIPLIER,
    MIN_MEAN_F1,
    MIN_MEAN_RECALL,
    OLD_DEV_F1,
    OLD_DEV_PRECISION,
    TARGET_LABEL,
    THRESHOLD,
    _aggregate,
    _gate,
    effective_sample_weight,
    fundamental_binary_metrics,
    is_hard_negative,
    make_fundamental_output_row,
)


def _record(text: str, tags: list[str], weight: float = 2.0, sample_id: str = "s") -> SimpleNamespace:
    return SimpleNamespace(sample_id=sample_id, model_text=text, label={"reasoning_tags": tags}, weights={"reasoning_tags": weight})


def test_hard_negative_definition_is_train_only_and_multiplies_only_negative_examples():
    negative = _record("机构关注股价走势", [])
    positive = _record("机构披露基本面", [TARGET_LABEL], sample_id="positive")
    unrelated = _record("市场情绪观察", [], sample_id="unrelated")
    assert is_hard_negative(negative)
    assert not is_hard_negative(positive)
    assert not is_hard_negative(unrelated)
    assert effective_sample_weight(negative) == pytest.approx(2.0 * HARD_NEGATIVE_MULTIPLIER)
    assert effective_sample_weight(positive) == pytest.approx(2.0)
    assert effective_sample_weight(unrelated) == pytest.approx(2.0)


def test_fundamental_metrics_and_boundary_false_positive_count_are_explicit():
    result = fundamental_binary_metrics(
        [0.9, 0.8, 0.8, 0.2],
        [1, 0, 0, 1],
        threshold=THRESHOLD,
        boundary_ids={"b1", "b2"},
        sample_ids=["positive", "b1", "b2", "other"],
    )
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["boundary_false_positives"] == 2
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(0.5)


def test_only_fundamental_output_row_updates_and_other_rows_stay_frozen():
    base = torch.nn.Linear(4, 3)
    row = make_fundamental_output_row(torch, base, 1)
    before = row(torch.ones(1, 4)).detach().clone()
    assert {name for name, parameter in row.named_parameters() if parameter.requires_grad} == {"weight", "bias"}
    optimizer = torch.optim.SGD([row.weight, row.bias], lr=0.1)
    loss = row(torch.ones(1, 4))[:, 1].sum()
    loss.backward()
    optimizer.step()
    after = row(torch.ones(1, 4)).detach()
    assert torch.equal(before[:, 0], after[:, 0])
    assert torch.equal(before[:, 2], after[:, 2])
    assert not torch.equal(before[:, 1], after[:, 1])


def _result(seed: int, precision: float, recall: float, f1: float, boundary_fp: int = 0, device: str = "cpu") -> dict:
    return {
        "seed": seed,
        "resource": {"actual_device": device},
        "metrics": {
            "dev": {TARGET_LABEL: {"precision": precision, "recall": recall, "f1": f1, "boundary_false_positives": boundary_fp}},
            "cpu_reload_inference_smoke": {"finite": True},
            "non_target_reasoning_rows_unchanged": True,
        },
        "checkpoint_sha256": "a" * 64,
    }


def test_gate_keeps_zero_old_boundary_false_positive_condition_visible():
    results = [_result(35, 0.8, 0.8, 0.8), _result(71, 0.8, 0.8, 0.8), _result(107, 0.8, 0.8, 0.8)]
    aggregate = _aggregate(results, old_boundary_fp=0)
    gate = _gate(aggregate)
    assert aggregate["old_dev"]["boundary_false_positives"] == 0
    assert gate["checks"]["mean_precision_above_old"] is True
    assert gate["checks"]["mean_f1_at_least_floor"] is True
    assert gate["checks"]["mean_recall_at_least_floor"] is True
    assert gate["checks"]["at_least_two_boundary_reduction_seeds"] is False
    assert gate["passed"] is False


def test_aggregate_requires_exact_seed_order():
    with pytest.raises(Exception, match="exact seeds"):
        _aggregate([_result(35, 0.7, 0.7, 0.7)], old_boundary_fp=0)
