from __future__ import annotations

import numpy as np

from semantic_model import encoder_m13_frozen_lert_linear as m13


def test_normalise_scalar_probabilities_handles_empty_row() -> None:
    observed = m13.normalise_scalar_probabilities(np.asarray([[0.2, 0.3], [0.0, 0.0]]))

    assert np.allclose(observed.sum(axis=1), 1.0)
    assert np.allclose(observed[1], [0.5, 0.5])


def test_binary_models_support_constant_and_fitted_columns() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([[0, 1], [0, 1], [1, 1], [1, 1]])

    models = m13.fit_binary_models(features, targets, 0.3)
    observed = m13.predict_binary_models(models, features)

    assert observed.shape == (4, 2)
    assert np.allclose(observed[:, 1], 1.0)
    assert observed[0, 0] < observed[-1, 0]


def test_promotion_requires_two_recent_scalar_hits() -> None:
    heads = m13.V1_HEADS
    current_recent = {head: {"accuracy": 0.4, "macro_f1": 0.2} for head in heads}
    candidate_recent = {head: {"accuracy": 0.4, "macro_f1": 0.2} for head in heads}
    current_recent["reasoning_tags"] = {"micro_f1": 0.4, "macro_f1": 0.2}
    candidate_recent["reasoning_tags"] = {"micro_f1": 0.4, "macro_f1": 0.2}
    current_check = {head: dict(value) for head, value in current_recent.items()}
    candidate_check = {head: dict(value) for head, value in candidate_recent.items()}
    h48_heads = m13.terminal.HUMAN48_HEADS
    current_h48 = {head: {"accuracy": 0.4, "macro_f1": 0.2} for head in h48_heads}
    candidate_h48 = {head: {"accuracy": 0.4, "macro_f1": 0.2} for head in h48_heads}
    candidate_recent["stance"]["accuracy"] += 1.0 / 19.0

    decisions = m13.promotion_decisions(current_recent, candidate_recent, current_check, candidate_check, current_h48, candidate_h48)

    assert decisions["stance"]["promote"] is False
    candidate_recent["stance"]["accuracy"] += 1.0 / 19.0
    decisions = m13.promotion_decisions(current_recent, candidate_recent, current_check, candidate_check, current_h48, candidate_h48)
    assert decisions["stance"]["promote"] is True
