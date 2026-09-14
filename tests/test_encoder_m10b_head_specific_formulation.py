from __future__ import annotations

import torch

from semantic_model.auxiliary_heads import AuxiliaryHeadModel, AuxiliaryTargetBatch
from semantic_model.encoder_m10a_multilabel_ablation import _clone_state
from semantic_model.encoder_m10b_head_specific_formulation import (
    discourse_policy_passes,
    market_softmax_passes,
    run_bce_oof,
    run_market_softmax_oof,
    _softmax_metrics,
)
from semantic_model.schema import SemanticSchemaCandidate


SCHEMA = SemanticSchemaCandidate.load("schema/semantic-schema-candidate-v0.3.json")


def _batch(n: int = 6) -> AuxiliaryTargetBatch:
    market = torch.zeros((n, 4), dtype=torch.float32)
    market[0, 0] = 1
    market[1, 0] = 1
    market[2, 1] = 1
    market[3, 1] = 1
    market[4, 2] = 1
    market[5, 2] = 1
    discourse = torch.zeros((n, 7), dtype=torch.float32)
    discourse[0, 1] = 1
    discourse[1, 1] = 1
    discourse[2, 2] = 1
    discourse[3, 2] = 1
    discourse[4, 3] = 1
    discourse[5, 1] = 1
    return AuxiliaryTargetBatch(
        targets={"market_scope": market, "discourse_focus": discourse},
        loss_masks={"market_scope": torch.ones(n), "discourse_focus": torch.ones(n)},
        sample_ids=tuple(f"s{index}" for index in range(n)),
    )


def _reps(n: int = 6, width: int = 3) -> dict[str, torch.Tensor]:
    values = torch.arange(n * width, dtype=torch.float32).reshape(n, width)
    return {"emotion_target": values, "target_mode": values / 10}


def _state(width: int = 3):
    torch.manual_seed(20260913)
    return _clone_state(AuxiliaryHeadModel(SCHEMA, width).state_dict())


def test_market_softmax_emits_single_class_and_confusion_matrix():
    target = torch.eye(4, dtype=torch.float32)
    logits = torch.eye(4, dtype=torch.float32) * 10
    metrics = _softmax_metrics(logits, target, torch.ones(4), SCHEMA.auxiliary_factors["market_scope"].class_order)
    assert metrics["accuracy"] == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 4
    assert all(sum(row) == 1 for row in metrics["confusion_matrix"])


def test_market_and_discourse_keep_the_same_three_fold_assignment():
    batch = _batch()
    reps = _reps()
    state = _state()
    market_a = run_bce_oof("market_scope", reps, batch, list(batch.sample_ids), SCHEMA, initial_state=state, use_pos_weight=False, fallback=False)
    market_softmax = run_market_softmax_oof(reps, batch, list(batch.sample_ids), SCHEMA, initial_state=state)
    discourse_a = run_bce_oof("discourse_focus", reps, batch, list(batch.sample_ids), SCHEMA, initial_state=state, use_pos_weight=False, fallback=False)
    discourse_d = run_bce_oof("discourse_focus", reps, batch, list(batch.sample_ids), SCHEMA, initial_state=state, use_pos_weight=True, fallback=True)
    assert market_a["fold_assignment"] == market_softmax["fold_assignment"] == discourse_a["fold_assignment"] == discourse_d["fold_assignment"]
    assert all("confusion_matrix" in market_softmax["folds"][fold]["metrics"] for fold in range(3))


def test_head_specific_gates_are_independent():
    market_a = {"aggregate": {"accuracy": 0.5, "macro_f1": 0.5}}
    market_s = {"aggregate": {"accuracy": 0.51, "macro_f1": 0.51}}
    discourse_a = {"aggregate": {"micro_f1": 0.1, "macro_f1": 0.1}}
    discourse_d = {"aggregate": {"micro_f1": 0.2, "macro_f1": 0.2, "per_label": {"MARKET_OR_INDUSTRY": {"f1": 0.2}, "AUTHOR_VIEWPOINT": {"f1": 0.2}}}}
    assert market_softmax_passes(market_a, market_s)
    assert discourse_policy_passes(discourse_a, discourse_d)

