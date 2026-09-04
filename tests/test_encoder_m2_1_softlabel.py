from __future__ import annotations

import numpy as np

from semantic_model import encoder_m2_1_softlabel as soft


def test_adjacent_soft_label_preserves_fixed_masses():
    order = ["FEAR", "ANXIETY", "ANGER", "FRUSTRATION", "UNKNOWN"]
    target = soft.soft_label_distribution("emotion_primary", "FEAR", order)
    assert target[0] == 0.8
    assert np.isclose(target[1], 0.2)
    assert sum(target) == 1.0


def test_multi_neighbor_soft_label_normalizes_similarity_weights():
    order = ["HOPE", "EXCITEMENT", "FOMO", "UNKNOWN"]
    target = soft.soft_label_distribution("emotion_primary", "EXCITEMENT", order)
    assert target[1] == 0.8
    assert np.isclose(target[0], 0.2 * 0.5 / 1.2)
    assert np.isclose(target[2], 0.2 * 0.7 / 1.2)
    assert np.isclose(sum(target), 1.0)


def test_no_declared_neighbor_remains_one_hot():
    order = ["CALM", "UNKNOWN"]
    target = soft.soft_label_distribution("emotion_primary", "CALM", order)
    assert target == [1.0, 0.0]


def test_action_relations_are_limited_to_declared_pairs():
    order = ["BUY", "ADD", "HOLD", "WATCH", "UNKNOWN"]
    target = soft.soft_label_distribution("action_tendency", "BUY", order)
    assert target[0] == 0.8
    assert np.isclose(target[1], 0.2)
    assert target[2] == 0.0


def test_expected_semantic_score_uses_probabilities_not_argmax_only():
    order = ["BUY", "ADD", "SELL"]
    probabilities = np.asarray([[0.0, 1.0, 0.0]])
    assert soft.expected_semantic_score(probabilities, ["BUY"], order) == 0.8


def test_selection_gate_requires_both_numeric_conditions():
    current = {"expected_semantic_score": 0.40, "macro_f1": 0.50}
    assert soft.selection_gate({"expected_semantic_score": 0.42, "macro_f1": 0.49}, current)["passed"]
    assert not soft.selection_gate({"expected_semantic_score": 0.419, "macro_f1": 0.50}, current)["passed"]
    assert not soft.selection_gate({"expected_semantic_score": 0.43, "macro_f1": 0.48}, current)["passed"]


def test_experiment_is_exactly_two_heads_and_human100_is_post_selection():
    assert soft.HEADS == ("emotion_primary", "action_tendency")
    assert soft.SEED == 35
    assert soft.M3B_WORKBOOK.name == "m3b-gold100-review-pack.xlsx"
    source = soft.__file__
    assert "human100_role" in open(source, encoding="utf-8").read()
