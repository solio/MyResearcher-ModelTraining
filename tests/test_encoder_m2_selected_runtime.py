from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_selected_runtime as selected
from semantic_model.schema import FROZEN_CLASS_ORDER


def _record() -> m1.M1Record:
    return m1.M1Record(
        sample_id="dev-smoke-1",
        stock_code="000001",
        stock_name="fixture",
        model_text="fixture text",
        label={},
        weights={},
    )


def test_context_mapping_uses_estimator_classes_and_fills_schema_unknown_with_zero():
    native = np.asarray([[0.2, 0.3, 0.5]])
    classes = ["EXTERNAL_CONTEXT_REQUIRED", "SELF_CONTAINED", "PARTIAL_CONTEXT"]
    observed = selected.map_classical_scalar_probabilities(native, classes, FROZEN_CLASS_ORDER["context_dependency"])
    assert observed.shape == (1, 4)
    assert observed.tolist() == [[0.3, 0.5, 0.2, 0.0]]
    assert observed.sum(axis=1).tolist() == [1.0]


def test_context_mapping_rejects_probability_column_mismatch():
    with pytest.raises(Exception) as error:
        selected.map_classical_scalar_probabilities([[0.2, 0.8]], ["SELF_CONTAINED"], FROZEN_CLASS_ORDER["context_dependency"])
    assert getattr(error.value, "code", None) == "M2_CLASSICAL_PROBABILITY_SHAPE_INVALID"


def test_reasoning_classical_positive_probability_does_not_assume_column_order():
    observed = selected.classical_positive_probability([[0.75, 0.25]], [1, 0])
    assert observed.tolist() == [0.75]


def test_prediction_payload_has_all_seven_heads_and_frozen_sarcasm_threshold():
    records = [_record()]
    scalar = {
        head: np.eye(len(FROZEN_CLASS_ORDER[head]), dtype=float)[[0]]
        for head in selected.SCALAR_HEADS
    }
    scalar["emotion_primary"] = np.eye(len(FROZEN_CLASS_ORDER["emotion_primary"]), dtype=float)[[9]]
    reasoning = np.full((1, len(FROZEN_CLASS_ORDER["reasoning_tags"])), 0.25, dtype=float)
    payload = selected._prediction_payload(records, type("Schema", (), {"class_order": FROZEN_CLASS_ORDER})(), scalar, reasoning, {"SARCASM_IRONY": 0.45})
    predictions = payload[0]["predictions"]
    assert set(predictions) == set(selected.HEADS)
    assert predictions["emotion_primary"]["prediction"] == "NONE_EXPLICIT"
    assert predictions["emotion_target"]["prediction"] == "PRICE"
    assert payload[0]["conditioned_outputs"]["emotion_target"]["effective_prediction"] == "NOT_APPLICABLE"
    assert {head: len(predictions[head]["probabilities"]) for head in selected.SCALAR_HEADS} == {
        "target_mode": 4,
        "stance": 5,
        "emotion_primary": 11,
        "emotion_target": 7,
        "action_tendency": 9,
        "context_dependency": 4,
    }
    assert len(predictions["reasoning_tags"]["probabilities"]) == 15
    assert predictions["reasoning_tags"]["thresholds"]["SARCASM_IRONY"] == 0.45


def test_runtime_contract_is_exactly_selected_candidate_and_no_training_path():
    assert selected.STAGE_ID == "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3"
    assert selected.PARENT_CONTENT_ADDRESS == "ca2d1399ab252a14d1fb33f4e6ee2581f2253b0d5c63d84ef0ebb5fdad3caf92"
    assert selected.LERT_HEADS == ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "reasoning_tags")
    source = (selected.Path(__file__).resolve().parents[1] / "src/semantic_model/encoder_m2_selected_runtime.py").read_text(encoding="utf-8")
    assert "snapshot_download" not in source
    assert "fit(" not in source


def test_manifest_verifier_rejects_parent_path_escape(tmp_path):
    manifest = {"files": [{"path": "../outside", "bytes": 0, "sha256": "0" * 64}]}
    from semantic_model.hashes import content_addressed_id

    manifest["content_address"] = content_addressed_id(manifest)
    (tmp_path / "content-addressed-manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception) as error:
        selected._verify_manifest(tmp_path)
    assert getattr(error.value, "code", None) == "M2_SELECTED_MANIFEST_INVALID"


def test_unlabelled_record_accepts_only_input_fields_without_fabricating_labels():
    record = selected._validate_inference_record({"sample_id": "x", "stock_code": "000001", "stock_name": "fixture", "model_text": "text", "expected_scope": "CLEAR_OOD"})
    assert record.sample_id == "x"
    assert record.metadata == {"expected_scope": "CLEAR_OOD"}
    assert not hasattr(record, "label")


def test_inference_batch_builder_returns_only_input_ids_and_attention_mask():
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0

        def __call__(self, value, **_kwargs):
            return {"input_ids": [11, 12] if value == "text" else [13]}

    records = [selected.InferenceRecord("x", "000001", "fixture", "text")]
    config = {"stock_code_token_cap": 8, "stock_name_token_cap": 16, "max_length": 16, "truncation": "HEAD_TAIL"}
    batch = selected.build_inference_batch(torch, Tokenizer(), records, config, torch.device("cpu"))
    assert set(batch) == {"input_ids", "attention_mask"}
    assert tuple(batch["input_ids"].shape) == (1, 8)
    assert tuple(batch["attention_mask"].shape) == (1, 8)


def test_complete_synthetic_challenge_loads_as_100_unlabelled_records():
    fixture = selected.Path(__file__).resolve().parent / "fixtures" / "m3_synthetic_challenge_v1.jsonl"
    records = selected.load_unlabelled_jsonl(fixture)
    assert len(records) == 100
    assert all(record.metadata and "expected_scope" in record.metadata for record in records)
    assert all(not hasattr(record, "label") and not hasattr(record, "weights") for record in records)
