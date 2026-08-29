from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import semantic_model.audit_encoder_readiness as readiness


SOURCE_ROOT = Path("/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining")
SOURCE_CONFIG = SOURCE_ROOT / "configs/baseline_v0.3.5.yaml"
SOURCE_PACKAGE_MANIFEST = SOURCE_ROOT / (
    "data/local/MyResearcher_Semantic_Immutable_Data_v0.3.5/CONTENT_MANIFEST.json"
)


def _canonical_success() -> dict:
    """Synthetic canonical audit output; the wrapper must not recreate this gate."""
    return {
        "status": "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN",
        "audit_id": "a" * 64,
        "training_allowed": True,
        "blocker_codes": [],
        "validation_summary": {
            "package_manifest_id": readiness.EXPECTED_DATA_PACKAGE_MANIFEST_ID,
            "reference_package": {
                "available": True,
                "package_manifest_id": readiness.EXPECTED_REFERENCE_PACKAGE_MANIFEST_ID,
                "binding_data_package_content_address": readiness.EXPECTED_REFERENCE_DATA_BINDING,
                "allowed_current_status": "COMPARABLE_DIAGNOSTIC_RUN_ONLY",
            },
        },
    }


def _synthetic_statistics() -> dict:
    """Synthetic data/reference planning observations, never a second audit."""
    return {
        "data_roles": {"trainable_labels": "WEAK_LABEL_ONLY"},
        "split_counts": {"train": 1, "dev": 1, "test": 1, "embargo_total": 0},
        "per_class_train_support": {},
        "field_weight_summary": {},
        "raw_text_audit": {
            "token_length_audit": {
                "status": "OWNER_AUTHORIZED_PENDING_TOKENIZER_RETRIEVAL"
            }
        },
        "anchor50": {"role": "FIXED_DIAGNOSTIC_ANCHOR_NOT_INDEPENDENT_FINAL_GOLD"},
        "gold_and_ood_evidence": {
            "independent_adjudicated_gold": {"status": "BLOCKED"},
            "ood_set": {"status": "BLOCKED"},
        },
        "hardware": {},
    }


def _install_synthetic_success(monkeypatch) -> dict:
    canonical = _canonical_success()
    monkeypatch.setattr(readiness, "run_audit", lambda _path: (canonical, 0))
    monkeypatch.setattr(readiness, "_statistics", lambda _path: _synthetic_statistics())
    return canonical


def _assert_blocked(monkeypatch, canonical: dict, expected_code: str) -> None:
    monkeypatch.setattr(readiness, "run_audit", lambda _path: (canonical, 3))
    result, exit_code = readiness.run_encoder_readiness("synthetic.yaml")
    assert exit_code == 2
    assert result["status"] == readiness.BLOCKED_DATA_STATUS
    assert result["selection_or_training_authorized"] is False
    assert expected_code in result["blocker_codes"]


def test_valid_synthetic_data_and_reference_package_success(monkeypatch):
    canonical = _install_synthetic_success(monkeypatch)
    result, exit_code = readiness.run_encoder_readiness("synthetic.yaml")
    assert exit_code == 0
    assert result["status"] == readiness.MILESTONE_STATUS
    assert result["selection_or_training_authorized"] is True
    assert result["canonical_data_audit"]["audit_id"] == canonical["audit_id"]
    assert result["canonical_data_audit"]["training_allowed"] is True
    assert result["canonical_data_audit"]["data_package_manifest_id"] == readiness.EXPECTED_DATA_PACKAGE_MANIFEST_ID
    assert result["canonical_data_audit"]["reference_package_manifest_id"] == readiness.EXPECTED_REFERENCE_PACKAGE_MANIFEST_ID


def test_data_manifest_hash_tamper_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical["validation_summary"]["package_manifest_id"] = "0" * 64
    _assert_blocked(monkeypatch, canonical, "BLOCKED_CANONICAL_DATA_MANIFEST_PIN")


def test_data_payload_tamper_canonical_failure_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical.update(status="BLOCKED_INVALID_CANONICAL_ARTIFACTS", training_allowed=False, blocker_codes=["CANONICAL_PACKAGE_HASH_MISMATCH"])
    _assert_blocked(monkeypatch, canonical, "CANONICAL_PACKAGE_HASH_MISMATCH")


def test_reference_manifest_or_payload_tamper_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical["validation_summary"]["reference_package"]["package_manifest_id"] = "0" * 64
    _assert_blocked(monkeypatch, canonical, "BLOCKED_CANONICAL_REFERENCE_MANIFEST_PIN")


def test_contract_static_pin_vs_canonical_audit_mismatch(monkeypatch):
    canonical = _canonical_success()
    canonical["validation_summary"]["reference_package"]["binding_data_package_content_address"] = "sha256:" + "0" * 64
    _assert_blocked(monkeypatch, canonical, "BLOCKED_CANONICAL_REFERENCE_DATA_BINDING")


def test_split_label_content_change_with_same_sample_id_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical.update(status="BLOCKED_INVALID_CANONICAL_ARTIFACTS", training_allowed=False, blocker_codes=["SPLIT_LABEL_CONTENT_MISMATCH"])
    _assert_blocked(monkeypatch, canonical, "SPLIT_LABEL_CONTENT_MISMATCH")


def test_split_timestamp_or_policy_change_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical.update(status="BLOCKED_INVALID_CANONICAL_ARTIFACTS", training_allowed=False, blocker_codes=["SPLIT_TIME_POLICY_MISMATCH"])
    _assert_blocked(monkeypatch, canonical, "SPLIT_TIME_POLICY_MISMATCH")


def test_nonnegative_field_weight_value_change_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical.update(status="BLOCKED_INVALID_CANONICAL_ARTIFACTS", training_allowed=False, blocker_codes=["CANONICAL_PACKAGE_HASH_MISMATCH"])
    _assert_blocked(monkeypatch, canonical, "CANONICAL_PACKAGE_HASH_MISMATCH")


def test_anchor_provenance_or_reference_binding_change_fails_closed(monkeypatch):
    canonical = _canonical_success()
    canonical.update(status="BLOCKED_INVALID_CANONICAL_ARTIFACTS", training_allowed=False, blocker_codes=["ANCHOR_PROVENANCE_MISMATCH"])
    _assert_blocked(monkeypatch, canonical, "ANCHOR_PROVENANCE_MISMATCH")


def test_repeat_success_audit_id_is_stable(monkeypatch):
    _install_synthetic_success(monkeypatch)
    left, left_code = readiness.run_encoder_readiness("synthetic.yaml")
    right, right_code = readiness.run_encoder_readiness("synthetic.yaml")
    assert left_code == right_code == 0
    assert left["audit_id"] == right["audit_id"]
    assert left["canonical_data_audit"]["audit_id"] == right["canonical_data_audit"]["audit_id"]


def _identity_fixture() -> dict:
    return {
        "status": readiness.MILESTONE_STATUS,
        "canonical_data_audit": {"audit_id": "a" * 64},
        "hardware": {
            "operating_system": {"system": "Darwin"},
            "cpu": {"architecture": "arm64", "logical_cpu_count": 12},
            "python": {
                "implementation": "CPython",
                "version": "3.12.13",
                "executable": "/absolute/project/.venv/bin/python",
            },
            "encoder_runtime_packages": {
                "torch": {"installed": False, "version": None}
            },
            "disk_at_audit_worktree": {"free_bytes": 1, "free_gib": 0.0},
        },
    }


def test_identity_ignores_executable_spelling_but_preserves_runtime_facts():
    absolute = _identity_fixture()
    relative = copy.deepcopy(absolute)
    relative["hardware"]["python"]["executable"] = "../../project/.venv/bin/python"
    assert readiness._readiness_id(absolute) == readiness._readiness_id(relative)

    for mutation in (
        ("python", "version", "3.12.14"),
        ("operating_system", "system", "Linux"),
        ("encoder_runtime_packages", "torch", {"installed": True, "version": "2.0.0"}),
    ):
        changed = copy.deepcopy(absolute)
        changed["hardware"][mutation[0]][mutation[1]] = mutation[2]
        assert readiness._readiness_id(changed) != readiness._readiness_id(absolute)


def test_blocked_identity_ignores_local_error_path_spelling():
    absolute = readiness._blocked(
        ["CONFIG_NOT_FOUND"],
        error={"code": "CONFIG_NOT_FOUND", "message": "/absolute/local/config.yaml"},
    )
    relative = readiness._blocked(
        ["CONFIG_NOT_FOUND"],
        error={"code": "CONFIG_NOT_FOUND", "message": "../../local/config.yaml"},
    )
    assert absolute["audit_id"] == relative["audit_id"]


def test_cli_creates_no_runs_model_data_or_report_artifacts(monkeypatch, tmp_path: Path, capsys):
    _install_synthetic_success(monkeypatch)
    config_path = tmp_path / "synthetic.yaml"
    before = list(tmp_path.iterdir())
    assert readiness.main(["--config", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == readiness.MILESTONE_STATUS
    assert list(tmp_path.iterdir()) == before
    assert not any((tmp_path / name).exists() for name in ("runs", "models", "data", "reports"))


def test_missing_canonical_config_fails_closed_and_is_deterministic(tmp_path: Path):
    config_path = tmp_path / "missing-config.yaml"
    left, left_exit_code = readiness.run_encoder_readiness(config_path)
    right, right_exit_code = readiness.run_encoder_readiness(config_path)
    assert left_exit_code == right_exit_code == 2
    assert left == right
    assert left["status"] == readiness.BLOCKED_DATA_STATUS
    assert "CONFIG_NOT_FOUND" in left["blocker_codes"]


@pytest.mark.real_data
@pytest.mark.skipif(
    not SOURCE_CONFIG.is_file() or not SOURCE_PACKAGE_MANIFEST.is_file(),
    reason="local immutable data/reference packages unavailable",
)
def test_true_local_package_readiness_uses_canonical_audit_read_only():
    before = SOURCE_PACKAGE_MANIFEST.read_bytes()
    result, exit_code = readiness.run_encoder_readiness(SOURCE_CONFIG)
    assert SOURCE_PACKAGE_MANIFEST.read_bytes() == before
    assert exit_code == 0
    assert result["canonical_data_audit"]["status"] == "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN"
    assert result["canonical_data_audit"]["audit_id"] == "5fab05d633c509122bb8bbddd95b5d79f8d76a660b284f5cb20120df2865e414"
    assert result["selection_or_training_authorized"] is True
