from pathlib import Path

import pytest

from semantic_model.schema import FROZEN_CLASS_ORDER, LabelSchema
from semantic_model.validation import ContractError, validate_label_record


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "schema"
    / "semantic-schema-calibrated-v0.2.1.json"
)


def test_frozen_class_order_and_semantic_independence():
    schema = LabelSchema.load(SCHEMA_PATH)
    assert schema.class_order == FROZEN_CLASS_ORDER
    assert (
        schema.raw["source"]["workbook_sha256"]
        == "22a82a30aa1f08ad48554cd1ee054e522fda30e292cd386bda30f8e019d8eee8"
    )
    assert schema.raw["source"]["sheet_name"] == "04_Schema"
    assert schema.class_order["stance"].index("NEUTRAL") != schema.class_order[
        "stance"
    ].index("UNKNOWN")
    assert "CALM" in schema.class_order["emotion_primary"]
    assert "NONE_EXPLICIT" in schema.class_order["emotion_primary"]
    assert "WATCH" in schema.class_order["action_tendency"]
    assert "NO_ACTION_SIGNAL" in schema.class_order["action_tendency"]
    assert len(schema.class_order["reasoning_tags"]) == 15


@pytest.mark.parametrize(
    "schema_version",
    [
        "semantic-schema-candidate-v0.1",
        "semantic-schema-calibrated-v0.2",
        "semantic-schema-calibrated-v0.2.2-remediation",
    ],
)
def test_non_frozen_schema_versions_fail(valid_label, clone, schema_version):
    label = clone(valid_label)
    label["schema_version"] = schema_version
    schema = LabelSchema.load(SCHEMA_PATH)
    with pytest.raises(ContractError, match="SCHEMA_VERSION_MISMATCH"):
        validate_label_record(label, schema)


def test_reasoning_tags_are_multi_label(valid_label, clone):
    label = clone(valid_label)
    label["reasoning_tags"] = "FUNDAMENTAL"
    schema = LabelSchema.load(SCHEMA_PATH)
    with pytest.raises(ContractError, match="MULTI_LABEL_REQUIRED"):
        validate_label_record(label, schema)


def test_reasoning_tags_encode_to_frozen_multi_hot_order():
    schema = LabelSchema.load(SCHEMA_PATH)
    encoded = schema.encode_reasoning_tags(["FUNDAMENTAL", "WORDPLAY"])
    assert len(encoded) == 15
    assert encoded[schema.class_order["reasoning_tags"].index("FUNDAMENTAL")] == 1
    assert encoded[schema.class_order["reasoning_tags"].index("WORDPLAY")] == 1
    assert sum(encoded) == 2


def test_no_reason_given_cannot_mix_with_positive_reason(valid_label, clone):
    label = clone(valid_label)
    label["reasoning_tags"] = ["NO_REASON_GIVEN", "FUNDAMENTAL"]
    schema = LabelSchema.load(SCHEMA_PATH)
    with pytest.raises(ContractError, match="LABEL_DEPENDENCY_VIOLATION"):
        validate_label_record(label, schema)
