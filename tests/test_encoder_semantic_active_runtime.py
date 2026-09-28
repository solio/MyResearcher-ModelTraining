from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_model import encoder_semantic_active_runtime as active
from semantic_model.schema import FROZEN_CLASS_ORDER


class _Movable:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = device
        return self


def _row() -> dict:
    predictions = {}
    for head, order in FROZEN_CLASS_ORDER.items():
        if head == "reasoning_tags":
            predictions[head] = {"prediction": [order[0]], "predicted_labels": [order[0]],
                                 "probabilities": {label: 0.1 for label in order},
                                 "thresholds": {label: 0.5 for label in order}}
        else:
            predictions[head] = {"prediction": order[0],
                                 "probabilities": {label: float(label == order[0]) for label in order}}
    return {"sample_id": "one", "predictions": predictions,
            "conditioned_outputs": {"emotion_target": {"effective_prediction": "NOT_APPLICABLE"}}}


def _legacy() -> SimpleNamespace:
    row = _row()
    selected_runtime = SimpleNamespace(schema=SimpleNamespace(class_order=FROZEN_CLASS_ORDER),
                                       model=_Movable(), artifact_dir=Path("/selected"))
    return SimpleNamespace(selected_runtime=selected_runtime, m12_model=_Movable(),
                           artifact=Path("/m12"), schema=selected_runtime.schema,
                           config={}, torch=SimpleNamespace(device=lambda value: value), tokenizer=None,
                           predict_records=lambda records: [deepcopy(row) for _ in records])


def _record() -> SimpleNamespace:
    return SimpleNamespace(sample_id="one", stock_code="600000", stock_name="甲", model_text="我买了")


def test_a0_keeps_existing_seven_head_row_and_names_version() -> None:
    runtime = active.SemanticActiveInferenceRuntime(version=active.VERSION_A0, legacy_runtime=_legacy())
    observed = runtime.predict_records([_record()])[0]
    assert observed.pop("model_version") == active.VERSION_A0
    assert observed == _row()
    assert runtime.action_model is None
    assert runtime.artifact == Path("/m12")


def test_a5_replaces_only_action_and_keeps_conditioned_output(monkeypatch) -> None:
    action_model = _Movable()
    runtime = active.SemanticActiveInferenceRuntime(version=active.VERSION_A5, legacy_runtime=_legacy(),
                                                      action_model=action_model, action_artifact=Path("/a5"))
    order = list(FROZEN_CLASS_ORDER[active.ACTION_HEAD])
    vector = np.zeros((1, len(order)), dtype=float)
    vector[0, order.index("BUY")] = 1.0
    monkeypatch.setattr(runtime, "_action_probabilities", lambda records: vector)
    observed = runtime.predict_records([_record()])[0]
    baseline = _row()
    assert observed["model_version"] == active.VERSION_A5
    assert observed["predictions"][active.ACTION_HEAD]["prediction"] == "BUY"
    assert observed["predictions"][active.ACTION_HEAD]["probabilities"]["BUY"] == 1.0
    assert set(observed["predictions"]) == set(baseline["predictions"])
    for head in set(baseline["predictions"]) - {active.ACTION_HEAD}:
        assert observed["predictions"][head] == baseline["predictions"][head]
    assert observed["conditioned_outputs"] == baseline["conditioned_outputs"]
    assert runtime.artifact == Path("/a5")
    assert runtime.to("cpu") is runtime
    assert runtime.selected_runtime.model.device == "cpu"
    assert runtime.m12_model.device == "cpu"
    assert runtime.action_model.device == "cpu"


def test_version_selection_defaults_to_a5_and_allows_explicit_a0(monkeypatch) -> None:
    legacy = _legacy()
    action_model = _Movable()
    loaded = []
    monkeypatch.setattr(active.hybrid.M12HybridInferenceRuntime, "from_artifacts", classmethod(lambda cls, **kwargs: legacy))
    monkeypatch.setattr(active, "_load_a5_action_model", lambda current, path: (loaded.append(path), action_model)[1])
    newest = active.SemanticActiveInferenceRuntime.from_artifacts()
    old = active.SemanticActiveInferenceRuntime.from_artifacts(version=active.VERSION_A0)
    assert newest.version == active.DEFAULT_VERSION == active.VERSION_A5
    assert newest.action_model is action_model
    assert loaded == [active.DEFAULT_A5_ARTIFACT.resolve()]
    assert old.version == active.VERSION_A0 and old.action_model is None
    with pytest.raises(Exception) as error:
        active.SemanticActiveInferenceRuntime.from_artifacts(version="M12")
    assert getattr(error.value, "code", None) == "SEM7_VERSION_INVALID"


def test_a5_rejects_invalid_action_distribution(monkeypatch) -> None:
    runtime = active.SemanticActiveInferenceRuntime(version=active.VERSION_A5, legacy_runtime=_legacy(),
                                                      action_model=_Movable(), action_artifact=Path("/a5"))
    monkeypatch.setattr(runtime, "_action_probabilities", lambda records: np.zeros((1, len(FROZEN_CLASS_ORDER[active.ACTION_HEAD]))))
    with pytest.raises(Exception) as error:
        runtime.predict_records([_record()])
    assert getattr(error.value, "code", None) == "SEM7_ACTION_PROBABILITY_INVALID"
