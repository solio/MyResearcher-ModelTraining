import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "inference-output.schema.json"


def test_inference_json_schema_is_valid_draft_2020_12():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_inference_schema_rejects_missing_abstention():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    head = {
        "task_type": "single_label",
        "label": "UNKNOWN",
        "confidence": 0.4,
        "threshold": 0.5,
        "probabilities": {"UNKNOWN": 0.4},
    }
    predictions = {
        name: dict(head)
        for name in (
            "target_mode",
            "stance",
            "emotion_primary",
            "emotion_target",
            "action_tendency",
            "context_dependency",
        )
    }
    predictions["reasoning_tags"] = {
        "task_type": "multi_label",
        "labels": [],
        "abstained": True,
        "thresholds": {"UNKNOWN": 0.5},
        "probabilities": {"UNKNOWN": 0.4},
    }
    record = {
        "sample_id": "P001",
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "model_version": "semantic-student-v0.1.0",
        "preprocessing_contract_version": "test-v1",
        "predictions": predictions,
    }
    errors = list(Draft202012Validator(schema).iter_errors(record))
    assert errors
    assert any("abstained" in error.message for error in errors)


def test_head_schema_rejects_label_outside_frozen_class_order():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    stance_schema = {"$ref": "#/$defs/stanceHead", "$defs": schema["$defs"]}
    head = {
        "task_type": "single_label",
        "label": "CALM",
        "confidence": 0.4,
        "abstained": True,
        "threshold": 0.5,
        "probabilities": {
            "BULL": 0.1,
            "BEAR": 0.1,
            "NEUTRAL": 0.4,
            "MIXED": 0.1,
            "UNKNOWN": 0.3,
        },
    }
    errors = list(Draft202012Validator(stance_schema).iter_errors(head))
    assert any("CALM" in error.message for error in errors)


def test_conditioned_outputs_are_sibling_to_strict_seven_head_predictions():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    scalar_probability_defs = {
        "target_mode": "targetModeProbabilities",
        "stance": "stanceProbabilities",
        "emotion_primary": "emotionPrimaryProbabilities",
        "emotion_target": "emotionTargetProbabilities",
        "action_tendency": "actionTendencyProbabilities",
        "context_dependency": "contextDependencyProbabilities",
    }
    record = {
        "sample_id": "P001",
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "model_version": "semantic-student-v0.1.0",
        "preprocessing_contract_version": "test-v1",
        "predictions": {
            name: {
                "task_type": "single_label",
                "label": "UNKNOWN",
                "confidence": 0.4,
                "abstained": True,
                "threshold": 0.5,
                "probabilities": {label: 0.1 for label in schema["$defs"][definition]["required"]},
            }
            for name, definition in scalar_probability_defs.items()
        },
        "conditioned_outputs": {
            "emotion_target": {
                "raw_prediction": "PRICE",
                "effective_prediction": "NOT_APPLICABLE",
                "conditioning_applied": True,
                "conditioning_rule": "emotion_primary:NONE_EXPLICIT=>emotion_target:NOT_APPLICABLE",
            }
        },
    }
    record["predictions"]["reasoning_tags"] = {
        "task_type": "multi_label",
        "labels": [],
        "abstained": True,
        "thresholds": {label: 0.5 for label in schema["$defs"]["reasoningTagProbabilities"]["required"]},
        "probabilities": {label: 0.1 for label in schema["$defs"]["reasoningTagProbabilities"]["required"]},
    }
    assert list(Draft202012Validator(schema).iter_errors(record)) == []
