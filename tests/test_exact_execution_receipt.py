from __future__ import annotations

from copy import deepcopy

import pytest

from semantic_model import audit_exact_environment
from semantic_model import exact_execution_receipt as receipts
from semantic_model import prepare, train
from semantic_model.errors import ContractError
from semantic_model.hashes import content_addressed_id
from semantic_model.reference_package import (
    _reproduction_status,
    _strict_receipt_claim_eligibility,
)


def matching_preflight() -> dict:
    return {
        "status": "EXACT_REFERENCE_ENVIRONMENT_READY",
        "exact_environment_ready": True,
        "environment_policy_eligible_for_exact_reproduction": True,
        "source_identity": {
            "accepted_source_commit": "a" * 40,
            "source_worktree_clean": True,
        },
        "canonical_data_audit_id": "b" * 64,
        "data_package_manifest_id": "c" * 64,
        "reference_package_audit_id": "d" * 64,
        "reference_package_manifest_id": "e" * 64,
        "original_model_sha256": "f" * 64,
        "reference_environment_evidence_sha256": "1" * 64,
        "current_environment_capture": {
            "operating_system": {"system": "Linux", "machine": "x86_64"},
            "cpu": {"model_name": "AMD EPYC 9V74 80-Core Processor"},
        },
        "exact_environment_audit_id": "2" * 64,
    }


def owner_receipt(strict_receipt: dict, *, scope: str) -> dict:
    receipt = {
        "receipt_schema_version": receipts.OWNER_AUTHORIZATION_RECEIPT_SCHEMA,
        "scope": scope,
        "approved": True,
        "authorized_by": "owner-recorded-decision",
        "external_record_id": f"owner-ticket-{scope.lower()}",
        "strict_preflight_receipt_id": strict_receipt[
            "strict_preflight_receipt_id"
        ],
        "production_approval": False,
    }
    receipt["owner_authorization_receipt_id"] = content_addressed_id(
        receipt, omit_keys={"owner_authorization_receipt_id", "observation"}
    )
    return receipt


def test_strict_receipt_is_content_addressed_and_keeps_timestamp_observational():
    left = receipts.build_strict_preflight_receipt(matching_preflight())
    right = receipts.build_strict_preflight_receipt(matching_preflight())
    assert left["strict_preflight_receipt_id"] == right[
        "strict_preflight_receipt_id"
    ]
    assert "observed_at_utc" in left["observation"]
    assert left["owner_prepare_authorized"] is False
    assert left["owner_train_authorized"] is False
    assert receipts.verify_strict_preflight_receipt(left) == left


def test_tampered_strict_receipt_blocks_before_execution():
    receipt = receipts.build_strict_preflight_receipt(matching_preflight())
    receipt["data_package_manifest_id"] = "tampered"
    with pytest.raises(ContractError, match="CONTENT_ADDRESS_MISMATCH"):
        receipts.verify_strict_preflight_receipt(receipt)


def test_strict_receipt_rejects_local_path_content_even_with_a_recomputed_id():
    receipt = receipts.build_strict_preflight_receipt(matching_preflight())
    receipt["current_strict_runtime_capture"]["unsafe_path"] = "/local/runtime"
    receipt["strict_preflight_receipt_id"] = content_addressed_id(
        receipt, omit_keys={"strict_preflight_receipt_id", "observation"}
    )
    with pytest.raises(ContractError, match="EXACT_EXECUTION_RECEIPT_INVALID"):
        receipts.verify_strict_preflight_receipt(receipt)


@pytest.mark.parametrize(
    "drift",
    [
        lambda value: value["source_identity"].update(accepted_source_commit="9" * 40),
        lambda value: value.update(canonical_data_audit_id="9" * 64),
        lambda value: value.update(data_package_manifest_id="9" * 64),
        lambda value: value.update(reference_package_audit_id="9" * 64),
        lambda value: value.update(reference_package_manifest_id="9" * 64),
        lambda value: value["current_environment_capture"]["cpu"].update(
            model_name="other CPU"
        ),
        lambda value: value.update(reference_environment_evidence_sha256="9" * 64),
    ],
)
def test_current_preflight_rejects_source_package_runtime_or_evidence_drift(
    monkeypatch, drift
):
    original = receipts.build_strict_preflight_receipt(matching_preflight())
    current_preflight = deepcopy(matching_preflight())
    drift(current_preflight)
    current_receipt = receipts.build_strict_preflight_receipt(current_preflight)
    monkeypatch.setattr(
        audit_exact_environment,
        "run_exact_environment_audit",
        lambda config_path, reference_archive: (
            {
                "status": "EXACT_REFERENCE_ENVIRONMENT_READY",
                "strict_preflight_receipt": current_receipt,
            },
            0,
        ),
    )
    with pytest.raises(ContractError, match="EXACT_EXECUTION_RECEIPT_STALE"):
        receipts.verify_current_strict_preflight_receipt(
            original,
            config_path="config.yaml",
            reference_archive="reference.zip",
        )


def test_owner_receipts_are_scope_bound_and_never_grant_production():
    strict = receipts.build_strict_preflight_receipt(matching_preflight())
    prepare_owner = owner_receipt(strict, scope="PREPARE")
    assert receipts.verify_owner_authorization_receipt(
        prepare_owner,
        required_scope="PREPARE",
        strict_preflight_receipt=strict,
    ) == prepare_owner
    with pytest.raises(ContractError, match="OWNER_AUTHORIZATION_SCOPE_DENIED"):
        receipts.verify_owner_authorization_receipt(
            prepare_owner,
            required_scope="TRAIN",
            strict_preflight_receipt=strict,
        )


def test_exact_mode_prepare_and_train_block_without_strict_receipt(tmp_path):
    prepared, prepare_exit = prepare.run_prepare(tmp_path / "missing.yaml", exact_mode=True)
    trained, train_exit = train.run_train(tmp_path / "missing.yaml", exact_mode=True)
    assert prepare_exit == train_exit == 3
    assert prepared["blocker_codes"] == ["BLOCKED_EXACT_EXECUTION_RECEIPT_MISSING"]
    assert trained["blocker_codes"] == ["BLOCKED_EXACT_EXECUTION_RECEIPT_MISSING"]
    assert list(tmp_path.iterdir()) == []


def test_exact_mode_prepare_blocks_without_owner_prepare_receipt(
    monkeypatch, tmp_path
):
    def missing_owner(**kwargs):
        raise ContractError(
            "BLOCKED_OWNER_AUTHORIZATION_RECEIPT_MISSING",
            "prepare owner authorization is absent",
        )

    monkeypatch.setattr(prepare, "validate_exact_write_authorization", missing_owner)
    monkeypatch.setattr(
        prepare,
        "prepare_dataset",
        lambda *args, **kwargs: pytest.fail("prepare must not start without owner receipt"),
    )
    result, exit_code = prepare.run_prepare(
        tmp_path / "config.yaml",
        exact_mode=True,
        strict_preflight_receipt="strict.json",
        reference_archive="reference.zip",
    )
    assert exit_code == 3
    assert result["blocker_codes"] == ["BLOCKED_OWNER_AUTHORIZATION_RECEIPT_MISSING"]
    assert list(tmp_path.iterdir()) == []


def test_exact_mode_train_requires_independent_prepare_and_train_owner_receipts(
    monkeypatch, tmp_path
):
    evidence = {
        "strict_preflight_receipt_id": "receipt-id",
        "strict_preflight_receipt": {"strict_preflight_receipt_id": "receipt-id"},
    }
    calls: list[str] = []

    def validate(**kwargs):
        calls.append(kwargs["required_scope"])
        if kwargs["required_scope"] == "TRAIN":
            raise ContractError(
                "BLOCKED_OWNER_AUTHORIZATION_RECEIPT_MISSING",
                "train owner authorization is absent",
            )
        return evidence

    monkeypatch.setattr(train, "validate_exact_write_authorization", validate)
    monkeypatch.setattr(
        train,
        "prepare_dataset",
        lambda *args, **kwargs: pytest.fail("prepare must not start before train authorization"),
    )
    result, exit_code = train.run_train(
        tmp_path / "config.yaml",
        exact_mode=True,
        strict_preflight_receipt="strict.json",
        reference_archive="reference.zip",
        owner_prepare_authorization_receipt="prepare-owner.json",
    )
    assert exit_code == 3
    assert calls == ["PREPARE", "TRAIN"]
    assert result["blocker_codes"] == ["BLOCKED_OWNER_AUTHORIZATION_RECEIPT_MISSING"]
    assert list(tmp_path.iterdir()) == []


def test_legacy_match_and_perfect_oracles_cannot_reach_exact_without_receipt():
    # A legacy package audit can say true, but the final classifier deliberately
    # accepts only a valid strict receipt as same-environment evidence.
    legacy_reference_audit = {"exact_reproduction_environment_match": True}
    eligible, blockers = _strict_receipt_claim_eligibility(
        object(), object(), legacy_reference_audit, None
    )
    status, status_blockers = _reproduction_status(
        strict_receipt_eligible=eligible,
        labels_exact=True,
        probabilities_exact=True,
        metrics_exact=True,
        receipt_blocker_codes=blockers,
    )
    assert status == "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
    assert status_blockers == ["BLOCKED_EXACT_EXECUTION_RECEIPT_MISSING"]
