import pytest

from semantic_model.data import join_inputs_and_labels
from semantic_model.validation import ContractError


def test_duplicate_input_id_blocks(canonical_input, valid_label, clone):
    with pytest.raises(ContractError, match="DUPLICATE_SAMPLE_ID"):
        join_inputs_and_labels(
            [canonical_input, clone(canonical_input)], [valid_label]
        )


def test_label_id_not_in_input_blocks(canonical_input, valid_label, clone):
    label = clone(valid_label)
    label["sample_id"] = "P999"
    with pytest.raises(ContractError, match="LABEL_SAMPLE_ID_NOT_FOUND"):
        join_inputs_and_labels([canonical_input], [label])


def test_missing_label_fails_in_strict_mode(canonical_input):
    with pytest.raises(ContractError, match="MISSING_LABEL_SAMPLE_ID"):
        join_inputs_and_labels([canonical_input], [], require_complete=True)


def test_repeated_metadata_mismatch_fails(canonical_input, valid_label, clone):
    label = clone(valid_label)
    label["stock_name"] = "错误名称"
    with pytest.raises(ContractError, match="CANONICAL_METADATA_MISMATCH"):
        join_inputs_and_labels([canonical_input], [label])


def test_excel_serial_time_fails_and_never_replaces_input_time(
    canonical_input, valid_label, clone
):
    label = clone(valid_label)
    label["published_at"] = 46216.35383101852
    with pytest.raises(ContractError, match="NON_CANONICAL_LABEL_TIMESTAMP"):
        join_inputs_and_labels([canonical_input], [label])
    assert canonical_input["published_at"] == "2026-07-13T16:29:31+08:00"


def test_timezone_representation_mismatch_fails(canonical_input, valid_label, clone):
    label = clone(valid_label)
    label["published_at"] = "2026-07-13T08:29:31"
    with pytest.raises(ContractError, match="CANONICAL_METADATA_MISMATCH"):
        join_inputs_and_labels([canonical_input], [label])

