import pytest

from semantic_model.data import DataRole, assert_role_permission
from semantic_model.validation import ContractError


@pytest.mark.parametrize(
    "role",
    [
        DataRole.TEACHER_CANDIDATE,
        DataRole.GOLD_CANDIDATE,
        DataRole.MODEL_PREDICTION,
        DataRole.QUARANTINE,
        DataRole.EMBARGO,
    ],
)
def test_non_trainable_roles_cannot_be_training_truth(role):
    with pytest.raises(ContractError, match="DATA_ROLE_PERMISSION_DENIED"):
        assert_role_permission(role, purpose="train_label")


def test_prediction_cannot_be_annotation():
    with pytest.raises(ContractError, match="PREDICTION_IS_NOT_ANNOTATION"):
        assert_role_permission(DataRole.MODEL_PREDICTION, purpose="annotation")


def test_anchor_is_evaluation_only():
    assert_role_permission(DataRole.ANCHOR, purpose="evaluation")
    with pytest.raises(ContractError, match="DATA_ROLE_PERMISSION_DENIED"):
        assert_role_permission(DataRole.ANCHOR, purpose="train_label")

