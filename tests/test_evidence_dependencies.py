import pytest

from semantic_model.validation import ContractError, validate_evidence_dependencies


@pytest.mark.parametrize(
    ("field", "label_value", "evidence_value"),
    [
        ("stance", "UNKNOWN", ["订单增长"]),
        ("emotion_primary", "NONE_EXPLICIT", ["继续持有"]),
        ("emotion_target", "NOT_APPLICABLE", ["订单增长"]),
        ("action_tendency", "NO_ACTION_SIGNAL", ["继续持有"]),
    ],
)
def test_sentinel_labels_forbid_evidence(
    canonical_input, valid_label, clone, field, label_value, evidence_value
):
    label = clone(valid_label)
    label[field] = label_value
    label["evidence_spans"][field] = evidence_value
    with pytest.raises(ContractError, match="EVIDENCE_DEPENDENCY_VIOLATION"):
        validate_evidence_dependencies(canonical_input, label)


def test_no_reason_given_forbids_reason_evidence(canonical_input, valid_label, clone):
    label = clone(valid_label)
    label["reasoning_tags"] = ["NO_REASON_GIVEN"]
    label["evidence_spans"]["reasoning_tags"] = {
        "NO_REASON_GIVEN": ["订单增长"]
    }
    with pytest.raises(ContractError, match="EVIDENCE_DEPENDENCY_VIOLATION"):
        validate_evidence_dependencies(canonical_input, label)


def test_every_evidence_span_must_be_canonical_text_substring(
    canonical_input, valid_label, clone
):
    label = clone(valid_label)
    label["evidence_spans"]["stance"] = ["联网补充的外部事实"]
    with pytest.raises(ContractError, match="EVIDENCE_NOT_SUBSTRING"):
        validate_evidence_dependencies(canonical_input, label)


def test_valid_evidence_dependencies_pass(canonical_input, valid_label):
    validate_evidence_dependencies(canonical_input, valid_label)


def test_native_evidence_object_list_passes(canonical_input, valid_label, clone):
    label = clone(valid_label)
    label["evidence_spans"] = [
        {"field": "stance", "label": "BULL", "span": "订单增长"},
        {
            "field": "reasoning_tags",
            "label": "FUNDAMENTAL",
            "span": "订单增长",
        },
    ]
    validate_evidence_dependencies(canonical_input, label)


def test_native_sentinel_evidence_object_is_rejected(
    canonical_input, valid_label, clone
):
    label = clone(valid_label)
    label["reasoning_tags"] = ["NO_REASON_GIVEN"]
    label["evidence_spans"] = [
        {
            "field": "reasoning_tags",
            "label": "NO_REASON_GIVEN",
            "span": "订单增长",
        }
    ]
    with pytest.raises(ContractError, match="EVIDENCE_DEPENDENCY_VIOLATION"):
        validate_evidence_dependencies(canonical_input, label)
