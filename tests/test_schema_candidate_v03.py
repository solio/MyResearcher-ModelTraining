import json
from pathlib import Path

import pytest

from semantic_model.data import (
    AuxiliaryFactorLabels,
    join_inputs_and_labels,
    parse_optional_auxiliary_factors,
)
from semantic_model.errors import ContractError
from semantic_model.validation import validate_label_record
from semantic_model.schema import (
    ACTION_PARENT_CLASS_ORDER,
    FROZEN_CLASS_ORDER,
    SemanticSchemaCandidate,
    action_parent_from_label,
    action_parent_labels,
    action_parent_metric_labels,
    action_parent_probabilities,
)


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "semantic-schema-candidate-v0.3.json"


@pytest.fixture
def candidate_schema() -> SemanticSchemaCandidate:
    return SemanticSchemaCandidate.load(SCHEMA_PATH)


def test_candidate_schema_preserves_v021_heads_and_declares_five_sparse_factors(candidate_schema):
    assert candidate_schema.base_schema_version == "semantic-schema-calibrated-v0.2.1"
    assert candidate_schema.class_order == FROZEN_CLASS_ORDER
    assert tuple(candidate_schema.auxiliary_factors) == (
        "expectation_phase",
        "arousal_level",
        "desired_price_direction",
        "market_scope",
        "discourse_focus",
    )
    assert candidate_schema.auxiliary_factors["arousal_level"].task_type == "ordered_single_label"
    assert candidate_schema.auxiliary_factors["discourse_focus"].allow_multiple


def test_frozen_v021_record_is_compatible_and_all_auxiliary_masks_are_zero(candidate_schema, valid_label, clone):
    legacy = clone(valid_label)
    values, masks = candidate_schema.parse_auxiliary_factors(legacy)
    assert values == {
        "expectation_phase": None,
        "arousal_level": None,
        "desired_price_direction": None,
        "market_scope": None,
        "discourse_focus": None,
    }
    assert masks == {factor: 0 for factor in values}
    parsed = parse_optional_auxiliary_factors(legacy, candidate_schema)
    assert isinstance(parsed, AuxiliaryFactorLabels)
    assert parsed.loss_mask == masks
    candidate_schema.validate_record(legacy)
    validate_label_record(legacy, candidate_schema)


def test_join_loader_can_attach_optional_factors_without_changing_legacy_join(candidate_schema, canonical_input, valid_label):
    joined = join_inputs_and_labels([canonical_input], [valid_label])
    assert joined[0].auxiliary is None
    sparse_label = dict(valid_label)
    sparse_label["schema_version"] = "semantic-schema-candidate-v0.3"
    sparse_label["auxiliary_factors"] = {"arousal_level": "UNKNOWN"}
    joined_v03 = join_inputs_and_labels(
        [canonical_input], [sparse_label], auxiliary_schema=candidate_schema
    )
    assert joined_v03[0].auxiliary is not None
    assert joined_v03[0].auxiliary.values["arousal_level"] == "UNKNOWN"
    assert joined_v03[0].auxiliary.loss_mask["arousal_level"] == 1


def test_sparse_v03_record_distinguishes_missing_from_explicit_unknown(candidate_schema):
    sparse = {
        "schema_version": "semantic-schema-candidate-v0.3",
        "auxiliary_factors": {
            "expectation_phase": "UNKNOWN",
            "market_scope": ["TRADING_FLOW", "INDUSTRY_PRODUCT_REGION"],
        },
    }
    values, masks = candidate_schema.parse_auxiliary_factors(sparse)
    assert values["expectation_phase"] == "UNKNOWN"
    assert masks["expectation_phase"] == 1
    assert values["market_scope"] == ["TRADING_FLOW", "INDUSTRY_PRODUCT_REGION"]
    assert masks["market_scope"] == 1
    for factor in ("arousal_level", "desired_price_direction", "discourse_focus"):
        assert values[factor] is None
        assert masks[factor] == 0
    encoded, encoded_masks = candidate_schema.encode_auxiliary_factors(sparse)
    assert encoded_masks == masks
    assert encoded["expectation_phase"] == [0, 0, 1]
    assert encoded["market_scope"] == [1, 1, 0, 0]
    assert encoded["arousal_level"] == [0, 0, 0, 0]


def test_unknown_is_not_a_missing_marker(candidate_schema):
    with pytest.raises(ContractError, match="AUXILIARY_FACTOR_NULL"):
        candidate_schema.parse_auxiliary_factors({"auxiliary_factors": {"arousal_level": None}})
    values, masks = candidate_schema.parse_auxiliary_factors(
        {"auxiliary_factors": {"discourse_focus": ["UNKNOWN", "AUTHOR_VIEWPOINT"]}}
    )
    assert values["discourse_focus"] == ["UNKNOWN", "AUTHOR_VIEWPOINT"]
    assert masks["discourse_focus"] == 1


def test_action_parent_probability_and_label_mapping(candidate_schema):
    child = {
        "BUY": 0.10,
        "ADD": 0.20,
        "HOLD": 0.15,
        "DO_T": 0.05,
        "REDUCE": 0.10,
        "SELL": 0.10,
        "WATCH": 0.10,
        "NO_ACTION_SIGNAL": 0.10,
        "UNKNOWN": 0.10,
    }
    parent = action_parent_probabilities(child)
    assert tuple(parent) == tuple(ACTION_PARENT_CLASS_ORDER)
    assert parent["INCREASE_EXPOSURE"] == pytest.approx(0.30)
    assert parent["DECREASE_EXPOSURE"] == pytest.approx(0.20)
    assert sum(parent.values()) == pytest.approx(sum(child.values()))
    assert action_parent_from_label("DO_T") == "ACTIVE_REBALANCE"
    assert action_parent_labels(["BUY", "HOLD", "NO_ACTION_SIGNAL"]) == [
        "INCREASE_EXPOSURE",
        "MAINTAIN_EXPOSURE",
        "NO_EXECUTION",
    ]
    truth, prediction = action_parent_metric_labels(
        ["BUY", "SELL"], ["ADD", "WATCH"]
    )
    assert truth == ["INCREASE_EXPOSURE", "DECREASE_EXPOSURE"]
    assert prediction == ["INCREASE_EXPOSURE", "NO_EXECUTION"]


def test_action_parent_mapping_rejects_unknown_child_or_changed_order():
    with pytest.raises(ContractError, match="UNKNOWN_LABEL_VALUE"):
        action_parent_from_label("NOT_A_CHILD")
    with pytest.raises(ContractError, match="ACTION_CLASS_ORDER_MISMATCH"):
        action_parent_probabilities([1.0] * 9, child_order=list(reversed(FROZEN_CLASS_ORDER["action_tendency"])))
    with pytest.raises(ContractError, match="ACTION_METRIC_LENGTH_MISMATCH"):
        action_parent_metric_labels(["BUY"], [])


def test_candidate_json_is_machine_readable_and_does_not_change_frozen_schema():
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert raw["status"] == "CANDIDATE"
    assert raw["record_contract"]["missing_loss_mask"] == 0
    assert raw["record_contract"]["unknown_is_explicit"] is True
    assert raw["migration_rules"]["retain_v021_heads"] is True
