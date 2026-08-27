import pytest

from semantic_model.schema import V1_HEADS
from semantic_model.validation import ContractError
from semantic_model.weighting import validate_field_weights


def full_weight(sample_id="P001"):
    return {"sample_id": sample_id, "weights": {head: 1.0 for head in V1_HEADS}}


def test_sample_by_head_weight_matrix_passes():
    result = validate_field_weights([full_weight()], expected_ids={"P001"})
    assert result["P001"]["emotion_primary"] == 1.0


def test_global_sample_weight_substitute_fails():
    with pytest.raises(ContractError, match="FIELD_WEIGHT_CONTRACT_VIOLATION"):
        validate_field_weights(
            [{"sample_id": "P001", "sample_weight": 1.0}], expected_ids={"P001"}
        )


def test_missing_head_weight_fails():
    row = full_weight()
    del row["weights"]["action_tendency"]
    with pytest.raises(ContractError, match="FIELD_WEIGHT_CONTRACT_VIOLATION"):
        validate_field_weights([row], expected_ids={"P001"})


def test_missing_sample_weight_row_fails():
    with pytest.raises(ContractError, match="FIELD_WEIGHT_CONTRACT_VIOLATION"):
        validate_field_weights([], expected_ids={"P001"})


def test_quarantine_zeroes_every_head():
    row = full_weight()
    for head in V1_HEADS:
        row["weights"][head] = 0.0
    validate_field_weights(
        [row], expected_ids={"P001"}, quarantine_ids={"P001"}
    )

