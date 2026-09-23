from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_model import encoder_m12_hybrid_runtime as hybrid
from semantic_model.schema import FROZEN_CLASS_ORDER


def _legacy_row(sample_id: str = "sample-1") -> dict:
    predictions = {}
    for head in hybrid.selected.HEADS:
        order = FROZEN_CLASS_ORDER[head]
        if head == "reasoning_tags":
            probabilities = {label: 0.1 for label in order}
            probabilities[order[0]] = 0.3
            predictions[head] = {
                "probabilities": probabilities,
                "prediction": [order[0]],
                "predicted_labels": [order[0]],
                "thresholds": {label: 0.5 for label in order},
            }
        else:
            probabilities = {label: 0.0 for label in order}
            probabilities[order[0]] = 1.0
            predictions[head] = {"probabilities": probabilities, "prediction": order[0]}
    return {"sample_id": sample_id, "predictions": predictions}


def test_merge_replaces_only_stance_and_emotion_target() -> None:
    row = _legacy_row()
    passthrough_before = {
        head: json.loads(json.dumps(row["predictions"][head]))
        for head in hybrid.PASSTHROUGH_HEADS
    }
    candidate = {
        "stance": np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        "emotion_target": np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    }

    observed = hybrid.merge_replaced_heads(
        [row],
        candidate,
        type("Schema", (), {"class_order": FROZEN_CLASS_ORDER})(),
    )[0]

    assert set(observed["predictions"]) == set(hybrid.selected.HEADS)
    assert observed["predictions"]["stance"]["prediction"] == "BEAR"
    assert observed["predictions"]["emotion_target"]["prediction"] == "COMPANY"
    assert observed["conditioned_outputs"]["emotion_target"]["effective_prediction"] == "COMPANY"
    for head in hybrid.PASSTHROUGH_HEADS:
        assert observed["predictions"][head] == passthrough_before[head]


def test_merge_rejects_non_normalized_candidate_probability() -> None:
    with pytest.raises(Exception) as error:
        hybrid.merge_replaced_heads(
            [_legacy_row()],
            {
                "stance": np.asarray([[0.1, 0.1, 0.1, 0.1, 0.1]]),
                "emotion_target": np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            type("Schema", (), {"class_order": FROZEN_CLASS_ORDER})(),
        )
    assert getattr(error.value, "code", None) == "M12_HYBRID_PROBABILITY_NOT_NORMALIZED"


def test_merge_rejects_conditioned_field_inside_predictions() -> None:
    row = _legacy_row()
    row["predictions"]["emotion_target_effective"] = {}
    with pytest.raises(Exception) as error:
        hybrid.merge_replaced_heads(
            [row],
            {
                "stance": np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]]),
                "emotion_target": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            },
            type("Schema", (), {"class_order": FROZEN_CLASS_ORDER})(),
        )
    assert getattr(error.value, "code", None) == "M12_HYBRID_LEGACY_HEAD_SET_INVALID"


def test_hybrid_contract_names_only_two_replacements() -> None:
    assert hybrid.REPLACED_HEADS == ("stance", "emotion_target")
    assert set(hybrid.PASSTHROUGH_HEADS) | set(hybrid.REPLACED_HEADS) == set(hybrid.selected.HEADS)
    assert set(hybrid.PASSTHROUGH_HEADS).isdisjoint(hybrid.REPLACED_HEADS)
