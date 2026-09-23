from __future__ import annotations

import copy

from semantic_model.output_conditioning import add_conditioned_outputs


def _row(primary: str, target: str = "PRICE") -> dict:
    return {
        "sample_id": "s1",
        "predictions": {
            "emotion_primary": {"prediction": primary, "probabilities": {primary: 1.0}},
            "emotion_target": {"prediction": target, "probabilities": {target: 1.0}},
        },
    }


def test_none_explicit_conditions_only_effective_target_and_preserves_raw_output():
    row = _row("NONE_EXPLICIT")
    before = copy.deepcopy(row["predictions"])
    observed = add_conditioned_outputs(row)
    assert set(observed["predictions"]) == {"emotion_primary", "emotion_target"}
    assert observed["conditioned_outputs"]["emotion_target"] == {
        "raw_prediction": "PRICE",
        "effective_prediction": "NOT_APPLICABLE",
        "conditioning_applied": True,
        "conditioning_rule": "emotion_primary:NONE_EXPLICIT=>emotion_target:NOT_APPLICABLE",
    }
    assert observed["predictions"]["emotion_primary"] == before["emotion_primary"]
    assert observed["predictions"]["emotion_target"] == before["emotion_target"]
    assert row["predictions"] == before


def test_other_primary_leaves_effective_target_unchanged():
    observed = add_conditioned_outputs(_row("FEAR", "COMPANY"))
    assert set(observed["predictions"]) == {"emotion_primary", "emotion_target"}
    effective = observed["conditioned_outputs"]["emotion_target"]
    assert effective["effective_prediction"] == "COMPANY"
    assert effective["conditioning_applied"] is False


def test_conditioning_is_idempotent_and_stable():
    row = _row("NONE_EXPLICIT")
    first = add_conditioned_outputs(row)
    second = add_conditioned_outputs(first)
    assert first == second


def test_legacy_in_prediction_condition_is_migrated_to_sibling_field():
    row = _row("NONE_EXPLICIT")
    row["predictions"]["emotion_target_effective"] = {"effective_prediction": "PRICE"}
    observed = add_conditioned_outputs(row)
    assert set(observed["predictions"]) == {"emotion_primary", "emotion_target"}
    assert observed["conditioned_outputs"]["emotion_target"]["effective_prediction"] == "NOT_APPLICABLE"
