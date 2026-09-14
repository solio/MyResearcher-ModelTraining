from __future__ import annotations

import torch

from semantic_model.auxiliary_heads import AuxiliaryHeadModel, AuxiliaryTargetBatch
from semantic_model.encoder_m10a_multilabel_ablation import (
    MULTI_FACTORS,
    apply_multi_label_policy,
    fold_pos_weight,
    multilabel_metrics,
    run_policy_oof,
    select_policy,
)
from semantic_model.encoder_m9b_effect_gate import _clone_state
from semantic_model.schema import SemanticSchemaCandidate


SCHEMA = SemanticSchemaCandidate.load("schema/semantic-schema-candidate-v0.3.json")


def _batch() -> AuxiliaryTargetBatch:
    n = 6
    market = torch.zeros((n, 4), dtype=torch.float32)
    market[[0, 1], 0] = 1
    market[[2, 3], 1] = 1
    market[[4], 2] = 1
    discourse = torch.zeros((n, 7), dtype=torch.float32)
    discourse[[0, 1], 1] = 1
    discourse[[2, 3], 2] = 1
    discourse[[4], 3] = 1
    return AuxiliaryTargetBatch(
        targets={
            "market_scope": market,
            "discourse_focus": discourse,
        },
        loss_masks={
            "market_scope": torch.ones(n),
            "discourse_focus": torch.ones(n),
        },
        sample_ids=tuple(f"s{index}" for index in range(n)),
    )


def _representations(n: int = 6, width: int = 3) -> dict[str, torch.Tensor]:
    return {
        "emotion_target": torch.arange(n * width, dtype=torch.float32).reshape(n, width),
        "target_mode": torch.arange(n * width, dtype=torch.float32).reshape(n, width) / 10,
    }


def _initial_state(width: int = 3):
    torch.manual_seed(20260913)
    return _clone_state(AuxiliaryHeadModel(SCHEMA, width).state_dict())


def test_unknown_is_exclusive_and_top1_fallback_only_fills_empty_rows():
    order = SCHEMA.auxiliary_factors["market_scope"].class_order
    logits = torch.tensor([[5.0, 4.0, -4.0, 6.0], [-4.0, -3.0, -2.0, -1.0]])
    _, ordinary = apply_multi_label_policy(logits, order, fallback=False)
    _, fallback = apply_multi_label_policy(logits, order, fallback=True)
    assert ordinary[0].tolist() == [0, 0, 0, 1]
    assert ordinary[1].sum() == 0
    assert fallback[1].sum() == 1
    assert fallback[1, 3] == 1  # highest sigmoid probability is UNKNOWN


def test_pos_weight_uses_only_training_fold_labels():
    batch = _batch()
    weights, report = fold_pos_weight(batch.targets["market_scope"], batch.loss_masks["market_scope"], [0, 1, 2])
    # Training rows have positives [2, 1, 0, 0], so only the first two ratios
    # are finite standard negative/positive values; zero-positive classes are
    # explicitly recorded and kept at 1.0.
    assert weights.tolist() == [0.5, 2.0, 1.0, 1.0]
    assert report["positive"] == [2, 1, 0, 0]
    assert report["negative"] == [1, 2, 3, 3]


def test_all_four_policies_share_exact_sample_grouped_folds_and_report_full_metrics():
    reps = _representations()
    batch = _batch()
    initial = _initial_state()
    ids = list(batch.sample_ids)
    ordinary = run_policy_oof("A_BCE_THRESHOLD", reps, batch, ids, SCHEMA, initial_state=initial, epochs=1)
    weighted = run_policy_oof("B_POS_WEIGHT_BCE", reps, batch, ids, SCHEMA, initial_state=initial, epochs=1)
    assert ordinary["fold_assignment"] == weighted["fold_assignment"]
    for policy in ("A_BCE_THRESHOLD", "B_POS_WEIGHT_BCE"):
        result = run_policy_oof(policy, reps, batch, ids, SCHEMA, initial_state=initial, epochs=1)
        for factor in MULTI_FACTORS:
            metrics = result["aggregate"][factor]
            assert {"micro_f1", "macro_f1", "exact_match", "per_label", "empty_prediction_count", "empty_prediction_rate", "mean_prediction_cardinality", "mean_true_cardinality"} <= set(metrics)
            assert len(result["folds"]) == 3


def _minimal_policy_result(micro: float, macro: float, market_micro: float, market_macro: float, market_exact: float, *, non_author: bool) -> dict:
    labels = {label: {"f1": 0.0} for label in SCHEMA.auxiliary_factors["discourse_focus"].class_order}
    if non_author:
        labels["MARKET_OR_INDUSTRY"]["f1"] = 0.2
    return {
        "aggregate": {
            "discourse_focus": {"micro_f1": micro, "macro_f1": macro, "per_label": labels},
            "market_scope": {"micro_f1": market_micro, "macro_f1": market_macro, "exact_match": market_exact},
        }
    }


def test_selection_requires_both_discourse_metrics_and_market_no_regression():
    results = {
        "A_BCE_THRESHOLD": _minimal_policy_result(0.10, 0.10, 0.40, 0.30, 0.30, non_author=False),
        # Better discourse, but market regresses: reject.
        "B_POS_WEIGHT_BCE": _minimal_policy_result(0.20, 0.20, 0.39, 0.30, 0.30, non_author=True),
        # Empty-fallback-like aggregate is not enough if discourse does not improve both metrics.
        "C_BCE_TOP1_FALLBACK": _minimal_policy_result(0.20, 0.09, 0.40, 0.30, 0.30, non_author=True),
        # Better discourse and market: this is the only eligible policy.
        "D_POS_WEIGHT_BCE_TOP1_FALLBACK": _minimal_policy_result(0.21, 0.21, 0.40, 0.30, 0.30, non_author=True),
    }
    selection = select_policy(results)
    assert selection["selected_policy"] == "D_POS_WEIGHT_BCE_TOP1_FALLBACK"
    assert selection["verdict"] == "DIAGNOSTIC_MECHANISM_SELECTED"

