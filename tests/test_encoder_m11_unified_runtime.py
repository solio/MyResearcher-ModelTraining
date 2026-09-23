from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from semantic_model.auxiliary_heads import AuxiliaryHeadModel
from semantic_model.encoder_m11_unified_runtime import (
    M10C_POLICY_VERSION,
    Unified7Plus5OfflineRuntime,
    _artifact_manifest,
    _market_prediction,
    load_m10c_model,
    verify_m10c_artifact,
)
from semantic_model.encoder_m2_selected_runtime import InferenceRecord
from semantic_model.schema import SemanticSchemaCandidate


SCHEMA = SemanticSchemaCandidate.load("schema/semantic-schema-candidate-v0.3.json")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _make_m10c_artifact(tmp_path: Path, width: int = 4) -> tuple[Path, AuxiliaryHeadModel]:
    root = tmp_path / "m10c"
    root.mkdir()
    model = AuxiliaryHeadModel(SCHEMA, width)
    checkpoint = root / "head-specific-auxiliary-checkpoint.pt"
    torch.save({"component": "m10c_v03_head_specific_diagnostic_freeze", "auxiliary_state_dict": model.state_dict()}, checkpoint)
    policy = root / "policy.json"
    _write_json(policy, {"component": M10C_POLICY_VERSION})
    files = {}
    for path in (checkpoint, policy):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"status": "DIAGNOSTIC_ONLY", "selected_candidate": False, "files": files}
    _write_json(root / "manifest.json", manifest)
    return root, model


def test_m10c_manifest_and_checkpoint_reload(tmp_path: Path):
    root, original = _make_m10c_artifact(tmp_path)
    manifest = verify_m10c_artifact(root)
    loaded, _, _ = load_m10c_model(root, SCHEMA, 4)
    reps = {"emotion_target": torch.ones(2, 4), "emotion_primary": torch.ones(2, 4), "target_mode": torch.ones(2, 4)}
    with torch.no_grad():
        before = original(reps)
        after = loaded(reps)
    assert manifest["status"] == "DIAGNOSTIC_ONLY"
    assert all(torch.equal(before[factor], after[factor]) for factor in before)


def test_market_policy_is_ordinary_bce_and_empty_is_allowed():
    order = SCHEMA.auxiliary_factors["market_scope"].class_order
    result = _market_prediction(torch.full((len(order),), -8.0), order)
    assert result["prediction"] == []
    assert result["threshold"] == 0.5


def test_unified_output_separates_legacy_and_auxiliary_heads(tmp_path: Path):
    root, model = _make_m10c_artifact(tmp_path)
    records = [InferenceRecord(sample_id="s1", stock_code="000001", stock_name="A", model_text="主力资金流入大盘，期待上涨")]

    class AttachedModel:
        def __call__(self, input_ids, attention_mask):
            reps = {"emotion_target": torch.ones(1, 4), "emotion_primary": torch.ones(1, 4), "target_mode": torch.ones(1, 4)}
            return {"owner_representations": reps}

        def eval(self):
            return self

    class AttachedRuntime:
        schema = SCHEMA
        model = AttachedModel()

        def build_inputs(self, current, device):
            return {"input_ids": torch.ones(len(current), 2, dtype=torch.long), "attention_mask": torch.ones(len(current), 2, dtype=torch.long)}

    legacy_stub = SimpleNamespace(
        predict_records=lambda current: [{"sample_id": current[0].sample_id, "predictions": {head: {"probabilities": {"X": 1.0}, "prediction": "X"} for head in ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "context_dependency", "reasoning_tags")}}],
        artifact_dir=Path("/selected"),
        manifest={"content_address": "legacy-address", "status": "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3"},
        schema=SimpleNamespace(schema_version="semantic-schema-calibrated-v0.2.1"),
    )
    legacy_bundle = tmp_path / "legacy-bundle.pt"
    legacy_bundle.write_bytes(b"legacy")
    runtime = Unified7Plus5OfflineRuntime(legacy_stub, AttachedRuntime(), model, legacy_checkpoint=legacy_bundle, auxiliary_checkpoint=root / "head-specific-auxiliary-checkpoint.pt", auxiliary_manifest={"content_address": "aux-address"})
    rows = runtime.predict_records(records)
    assert len(rows) == 1
    assert set(rows[0]) >= {"sample_id", "legacy_heads", "auxiliary_factors", "model_bundle"}
    assert set(rows[0]["legacy_heads"]) == {"target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "context_dependency", "reasoning_tags"}
    assert "conditioned_outputs" in rows[0]
    assert rows[0]["model_bundle"]["auxiliary_policy_version"] == M10C_POLICY_VERSION


def test_artifact_manifest_includes_all_snapshot_files(tmp_path: Path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "unified-predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "bundle-manifest.json").write_text("{}\n", encoding="utf-8")
    manifest = _artifact_manifest(root, bundle={"schema_version": "v0.3"}, input_rows=1, output_rows=1, auxiliary_consistent=True)
    assert manifest["files"]["unified-predictions.jsonl"]["bytes"] > 0
    assert len(manifest["content_address"]) == 64
