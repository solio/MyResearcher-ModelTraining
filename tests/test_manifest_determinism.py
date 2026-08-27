import pytest

from semantic_model.hashes import content_addressed_id, verify_content_addressed_id
from semantic_model.validation import ContractError


def test_manifest_identity_is_key_order_independent():
    left = {"seed": 7, "input": {"sha256": "abc", "rows": 3}}
    right = {"input": {"rows": 3, "sha256": "abc"}, "seed": 7}
    assert content_addressed_id(left) == content_addressed_id(right)


def test_self_id_is_omitted_from_identity():
    left = {"manifest_id": "old", "seed": 7}
    right = {"manifest_id": "new", "seed": 7}
    assert content_addressed_id(left, omit_keys={"manifest_id"}) == content_addressed_id(
        right, omit_keys={"manifest_id"}
    )


def test_content_address_verification_fails_closed():
    manifest = {"manifest_id": "wrong", "seed": 7}
    with pytest.raises(ContractError, match="CONTENT_ADDRESS_MISMATCH"):
        verify_content_addressed_id(manifest, id_key="manifest_id")
