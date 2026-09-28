"""Versioned seven-head research inference with an optional Action replacement.

The A0 version is the existing M12 hybrid.  A5 preserves its other six heads
and replaces only ``action_tendency`` with the frozen 2026-09-29 checkpoint.
Neither version changes the source artifacts or earlier prediction snapshots.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import encoder_m12_hybrid_runtime as hybrid
from . import encoder_m2_selected_runtime as selected
from .errors import ContractError


VERSION_A0 = "SEM7-20260915-A0"
VERSION_A5 = "SEM7-20260929-A5"
DEFAULT_VERSION = VERSION_A5
ACTION_HEAD = "action_tendency"
DEFAULT_A5_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "semantic-action-contrast-role-negative-v1-20260929"


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _load_a5_action_model(legacy: hybrid.M12HybridInferenceRuntime, artifact: Path) -> Any:
    checkpoint_path = artifact / "action-checkpoint.pt"
    _require(checkpoint_path.is_file(), "SEM7_A5_CHECKPOINT_MISSING", "A5 Action checkpoint is missing", path=str(checkpoint_path))
    torch = legacy.torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require(isinstance(checkpoint, Mapping), "SEM7_A5_CHECKPOINT_INVALID", "A5 checkpoint must be a mapping")
    _require(checkpoint.get("head") == ACTION_HEAD and checkpoint.get("two_block") is True,
             "SEM7_A5_CHECKPOINT_INVALID", "checkpoint is not a two-block Action head")
    _require(isinstance(checkpoint.get("specialist_block_state_dict"), Mapping) and isinstance(checkpoint.get("head_state_dict"), Mapping),
             "SEM7_A5_CHECKPOINT_INVALID", "A5 checkpoint lacks Action weights")
    # M12's base encoder is the same fixed official LERT.  Its two-block
    # architecture matches A5; replacing both Action blocks and the head
    # reconstructs the saved candidate without changing the M12 model.
    model = deepcopy(legacy.m12_model)
    model.specialist_blocks[ACTION_HEAD].load_state_dict(checkpoint["specialist_block_state_dict"], strict=True)
    model.heads[ACTION_HEAD].load_state_dict(checkpoint["head_state_dict"], strict=True)
    model.set_active_head(ACTION_HEAD)
    model.eval()
    return model


class SemanticActiveInferenceRuntime:
    """The named A0 or A5 seven-head inference configuration."""

    def __init__(self, *, version: str, legacy_runtime: hybrid.M12HybridInferenceRuntime, action_model: Any = None, action_artifact: Path | None = None) -> None:
        _require(version in (VERSION_A0, VERSION_A5), "SEM7_VERSION_INVALID", "unknown semantic model version", version=version)
        _require((action_model is None) == (version == VERSION_A0), "SEM7_ACTION_MODEL_INVALID", "Action model presence does not match version", version=version)
        _require((action_artifact is None) == (version == VERSION_A0), "SEM7_ACTION_ARTIFACT_INVALID", "Action artifact presence does not match version", version=version)
        self.version = version
        self.model_version = version
        self.legacy_runtime = legacy_runtime
        self.selected_runtime = legacy_runtime.selected_runtime
        self.m12_model = legacy_runtime.m12_model
        self.action_model = action_model
        self.action_artifact = action_artifact
        self.m12_artifact = legacy_runtime.artifact
        self.selected_artifact = legacy_runtime.selected_runtime.artifact_dir
        self.artifact = action_artifact if action_artifact is not None else legacy_runtime.artifact
        self.schema = legacy_runtime.schema
        self.config = legacy_runtime.config
        self.torch = legacy_runtime.torch

    @classmethod
    def from_artifacts(
        cls,
        *,
        version: str = DEFAULT_VERSION,
        m12_artifact: str | Path = hybrid.DEFAULT_M12_ARTIFACT,
        selected_artifact: str | Path = selected.DEFAULT_SELECTED_ARTIFACT,
        action_artifact: str | Path = DEFAULT_A5_ARTIFACT,
        config_path: str | Path = selected.CONFIG_PATH,
        cache_dir: str | Path = selected.DEFAULT_CACHE,
    ) -> "SemanticActiveInferenceRuntime":
        _require(version in (VERSION_A0, VERSION_A5), "SEM7_VERSION_INVALID", "unknown semantic model version", version=version)
        legacy = hybrid.M12HybridInferenceRuntime.from_artifacts(
            m12_artifact=m12_artifact,
            selected_artifact=selected_artifact,
            config_path=config_path,
            cache_dir=cache_dir,
        )
        if version == VERSION_A0:
            return cls(version=version, legacy_runtime=legacy)
        action_path = Path(action_artifact).resolve()
        action_model = _load_a5_action_model(legacy, action_path)
        return cls(version=version, legacy_runtime=legacy, action_model=action_model, action_artifact=action_path)

    def to(self, device: Any) -> "SemanticActiveInferenceRuntime":
        target = self.torch.device(device)
        self.selected_runtime.model.to(target)
        self.m12_model.to(target)
        if self.action_model is not None:
            self.action_model.to(target)
        return self

    def _action_probabilities(self, records: Sequence[selected.InferenceRecord]) -> np.ndarray:
        _require(self.action_model is not None, "SEM7_ACTION_MODEL_MISSING", "A5 Action model is not loaded")
        pieces: list[np.ndarray] = []
        self.action_model.eval()
        with self.torch.no_grad():
            for offset in range(0, len(records), 16):
                batch = selected.build_inference_batch(
                    self.torch, self.legacy_runtime.tokenizer, records[offset:offset + 16],
                    self.config, next(self.action_model.parameters()).device,
                )
                logits = self.action_model(batch["input_ids"], batch["attention_mask"])[ACTION_HEAD]
                _require(bool(self.torch.isfinite(logits).all().item()), "SEM7_ACTION_NONFINITE_LOGITS", "A5 Action logits are non-finite")
                pieces.append(self.torch.softmax(logits, dim=-1).detach().cpu().numpy())
        return np.concatenate(pieces, axis=0)

    def predict_records(self, records: Sequence[selected.InferenceRecord]) -> list[dict[str, Any]]:
        if not records:
            return []
        legacy_rows = self.legacy_runtime.predict_records(records)
        _require(len(legacy_rows) == len(records) and [row.get("sample_id") for row in legacy_rows] == [record.sample_id for record in records],
                 "SEM7_ROW_ORDER_INVALID", "legacy predictions differ from input order")
        if self.version == VERSION_A0:
            return [{**row, "model_version": self.version} for row in legacy_rows]
        values = self._action_probabilities(records)
        order = list(self.schema.class_order[ACTION_HEAD])
        _require(values.shape == (len(records), len(order)), "SEM7_ACTION_SHAPE_INVALID", "A5 Action probability shape differs from schema")
        result: list[dict[str, Any]] = []
        for row, vector in zip(legacy_rows, values):
            vector = np.asarray(vector, dtype=float)
            _require(bool(np.isfinite(vector).all()) and np.isclose(float(vector.sum()), 1.0, rtol=1e-6, atol=1e-6),
                     "SEM7_ACTION_PROBABILITY_INVALID", "A5 Action probabilities must be finite and normalized", sample_id=row["sample_id"])
            predictions = row.get("predictions")
            _require(isinstance(predictions, Mapping) and set(predictions) == set(selected.HEADS),
                     "SEM7_HEAD_SET_INVALID", "legacy prediction row must contain exactly seven heads", sample_id=row["sample_id"])
            action = {
                "probabilities": {label: float(vector[index]) for index, label in enumerate(order)},
                "prediction": order[int(np.argmax(vector))],
            }
            result.append({**row, "predictions": {**predictions, ACTION_HEAD: action}, "model_version": self.version})
        return result
