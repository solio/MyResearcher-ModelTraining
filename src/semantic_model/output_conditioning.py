"""Deterministic downstream conditioning for cross-head semantic outputs.

Model predictions are observations and are never rewritten here.  The
conditioned field is an explicitly separate downstream view used by feed and
aggregation consumers.  Keeping this rule in a small shared module prevents
individual reports from silently applying different semantics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


EFFECTIVE_EMOTION_TARGET = "emotion_target_effective"
NONE_EXPLICIT = "NONE_EXPLICIT"
NOT_APPLICABLE = "NOT_APPLICABLE"
CONDITIONING_RULE = "emotion_primary:NONE_EXPLICIT=>emotion_target:NOT_APPLICABLE"


def effective_emotion_target(predictions: Mapping[str, Any]) -> dict[str, Any]:
    """Return the downstream target view without changing raw head values."""

    primary = predictions.get("emotion_primary")
    target = predictions.get("emotion_target")
    if not isinstance(primary, Mapping) or not isinstance(target, Mapping):
        raise ValueError("emotion_primary and emotion_target raw outputs are required")
    raw_prediction = target.get("prediction")
    effective_prediction = NOT_APPLICABLE if primary.get("prediction") == NONE_EXPLICIT else raw_prediction
    return {
        "raw_prediction": raw_prediction,
        "effective_prediction": effective_prediction,
        "conditioning_applied": primary.get("prediction") == NONE_EXPLICIT,
        "conditioning_rule": CONDITIONING_RULE,
    }


def add_conditioned_outputs(row: Mapping[str, Any]) -> dict[str, Any]:
    """Attach effective downstream fields while preserving raw predictions byte-for-byte."""

    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("prediction row must contain a predictions mapping")
    output = deepcopy(dict(row))
    output_predictions = deepcopy(dict(predictions))
    # Some historical market-feature unit fixtures intentionally carry only
    # the three grouped heads.  They cannot be conditioned, but remain valid
    # for those legacy features; complete public model rows always include
    # both emotion heads and receive the effective field below.
    if isinstance(output_predictions.get("emotion_primary"), Mapping) and isinstance(output_predictions.get("emotion_target"), Mapping):
        output_predictions[EFFECTIVE_EMOTION_TARGET] = effective_emotion_target(output_predictions)
    output["predictions"] = output_predictions
    return output
