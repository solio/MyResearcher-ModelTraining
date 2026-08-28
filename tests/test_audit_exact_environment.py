from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_model import audit_exact_environment as gate
from semantic_model.errors import ContractError
from semantic_model.reference_package import compare_runtime_to_reference


PROJECT_ROOT = Path(__file__).parents[1]
EXECUTION_CONTRACT_PATH = (
    PROJECT_ROOT / "manifests/exact-reproduction-execution-contract-v0.3.5.json"
)


def reference_environment() -> dict:
    return {
        "environment_manifest_schema": "semantic-baseline-reference-environment-v0.3.5",
        "capture_scope": (
            "Retrospective capture from the same persistent runtime bundle currently "
            "holding and loading the original artifact; the original training script "
            "did not emit an environment manifest at fit time."
        ),
        "python": {
            "version_info": [3, 12, 13, "final", 0],
            "implementation": "CPython",
            "compiler": "Clang 22.1.3 ",
            "platform_tag": "linux-x86_64",
        },
        "operating_system": {
            "platform": "Linux-6.18.35-x86_64-with-glibc2.39",
            "system": "Linux",
            "release": "6.18.35",
            "version": "#1 SMP Fri Aug 21 00:36:21 UTC 2026",
            "machine": "x86_64",
            "processor": "x86_64",
            "glibc": ["glibc", "2.39"],
            "os_release": "ID=ubuntu\nVERSION_ID=24.04\n",
        },
        "cpu": {"logical_count": 9, "model_name": "AMD EPYC 9V74 80-Core Processor"},
        "packages": {
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "scikit_learn": "1.8.0",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        },
        "threadpools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "num_threads": 9,
                "prefix": "libscipy_openblas",
                "version": "0.3.30",
                "threading_layer": "pthreads",
                "architecture": "SkylakeX",
                "filepath": "/reference/runtime/numpy.libs/openblas.so",
            },
            {
                "user_api": "openmp",
                "internal_api": "openmp",
                "num_threads": 9,
                "prefix": "libgomp",
                "version": None,
                "threading_layer": None,
                "architecture": None,
                "filepath": "/reference/runtime/sklearn.libs/libgomp.so",
            },
        ],
        "thread_environment": {
            "OMP_NUM_THREADS": None,
            "OPENBLAS_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "VECLIB_MAXIMUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        "runtime_bundle": {"CODEX_PRIMARY_RUNTIME_BUNDLE_VERSION": "26.812.11052"},
    }


def matching_current_environment() -> dict:
    reference = reference_environment()
    return {
        "capture_schema_version": "semantic-exact-runtime-capture-v0.3.5",
        "operating_system": dict(reference["operating_system"]),
        "cpu": dict(reference["cpu"]),
        "python": dict(reference["python"]),
        "packages": {
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "scikit-learn": "1.8.0",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        },
        "threadpools": [
            {
                key: value
                for key, value in record.items()
                if key != "filepath"
            }
            for record in reference["threadpools"]
        ],
        "thread_environment": dict(reference["thread_environment"]),
        "runtime_bundle": dict(reference["runtime_bundle"]),
    }


def install_passing_package_audits(monkeypatch, *, current: dict | None = None) -> None:
    config = object()
    data_audit = {
        "audit_id": "data-audit-id",
        "status": "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN",
        "training_allowed": True,
        "validation_summary": {
            "package_manifest_id": "data-package-id",
            "reference_package": {
                "available": True,
                # This deliberately stays comparable-only: the strict gate must
                # make its own decision from the full capture, not this exit-0 fact.
                "exact_reproduction_environment_match": False,
                "exact_reproduction_authorized_for_current_environment": False,
                "reproduction_blocker_codes": [
                    "BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"
                ],
                "package_manifest_id": "reference-package-id",
                "original_model_sha256": "a" * 64,
                "environment_capture_scope": reference_environment()["capture_scope"],
            },
        },
    }
    archive = {
        "zip_sha256": "b" * 64,
        "package_manifest_id": "reference-package-id",
        "payload_file_count": 17,
        "payload_total_bytes": 11_439_730,
    }
    monkeypatch.setattr(gate.ProjectConfig, "load", lambda path: config)
    monkeypatch.setattr(gate, "audit_config", lambda observed: data_audit)
    monkeypatch.setattr(gate, "audit_reference_archive", lambda observed, path: archive)
    monkeypatch.setattr(
        gate, "load_reference_environment", lambda observed: reference_environment()
    )
    monkeypatch.setattr(
        gate, "reference_environment_evidence_sha256", lambda observed: "c" * 64
    )
    monkeypatch.setattr(
        gate,
        "capture_runtime_environment",
        lambda: current if current is not None else matching_current_environment(),
    )


def test_strict_comparison_accepts_only_a_complete_matching_environment():
    comparison = compare_runtime_to_reference(
        matching_current_environment(), reference_environment(), strict=True
    )
    assert comparison["matches_reference"] is True
    assert comparison["mismatches"] == []


@pytest.mark.parametrize(
    ("mutate", "expected_field"),
    [
        (lambda value: value["operating_system"].update(system="Darwin"), "system"),
        (lambda value: value["operating_system"].update(machine="arm64"), "machine"),
        (lambda value: value["cpu"].update(model_name="Apple M2 Pro"), "cpu.model_name"),
        (lambda value: value["python"].update(version_info=[3, 13, 0, "final", 0]), "python"),
        (lambda value: value["packages"].update(numpy="2.3.3"), "package:numpy"),
        (lambda value: value["packages"].update(scipy="1.16.2"), "package:scipy"),
        (
            lambda value: value["packages"].update(**{"scikit-learn": "1.7.2"}),
            "package:scikit-learn",
        ),
        (lambda value: value["packages"].update(joblib="1.5.2"), "package:joblib"),
        (
            lambda value: value["packages"].update(threadpoolctl="3.5.0"),
            "package:threadpoolctl",
        ),
        (
            lambda value: value["threadpools"][0].update(internal_api="accelerate"),
            "blas.implementation",
        ),
        (
            lambda value: value["threadpools"][0].update(version="0.3.29"),
            "blas.version",
        ),
        (
            lambda value: value["threadpools"][0].update(threading_layer="openmp"),
            "blas.threading_layer",
        ),
        (
            lambda value: value["threadpools"][1].update(prefix="libomp-other"),
            "openmp.runtime",
        ),
        (
            lambda value: value["threadpools"][0].update(num_threads=8),
            "threadpools",
        ),
        (
            lambda value: value["thread_environment"].update(OMP_NUM_THREADS="1"),
            "thread_environment",
        ),
    ],
)
def test_strict_comparison_reports_each_frozen_identity_mismatch(
    mutate, expected_field
):
    current = matching_current_environment()
    mutate(current)
    comparison = compare_runtime_to_reference(current, reference_environment(), strict=True)
    assert comparison["matches_reference"] is False
    assert expected_field in {item["field"] for item in comparison["mismatches"]}


def test_missing_config_fails_closed(monkeypatch):
    def missing_config(path):
        raise ContractError("CONFIG_NOT_FOUND", str(path))

    monkeypatch.setattr(gate.ProjectConfig, "load", missing_config)
    result, exit_code = gate.run_exact_environment_audit("missing.yaml", "reference.zip")
    assert exit_code != 0
    assert result["exact_environment_ready"] is False
    assert result["training_invoked"] is False
    assert result["blocker_codes"][-1] == "CONFIG_NOT_FOUND"


def test_missing_data_package_fails_closed(monkeypatch):
    config = object()
    monkeypatch.setattr(gate.ProjectConfig, "load", lambda path: config)
    monkeypatch.setattr(
        gate,
        "audit_config",
        lambda observed: {
            "audit_id": "blocked-data-audit",
            "status": "BLOCKED_MISSING_CANONICAL_ARTIFACTS",
            "training_allowed": False,
            "blocker_codes": ["BLOCKED_MISSING_CANONICAL_PACKAGE_ARTIFACT"],
        },
    )
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 2
    assert result["status"] == "BLOCKED_CANONICAL_DATA_AUDIT"
    assert result["package_audit_success"] is False
    assert result["training_invoked"] is False


def test_missing_extracted_reference_package_fails_closed(monkeypatch):
    install_passing_package_audits(monkeypatch)
    monkeypatch.setattr(
        gate,
        "audit_config",
        lambda observed: {
            "audit_id": "data-audit-id",
            "status": "READY_FOR_DIAGNOSTIC_BASELINE_RUN",
            "training_allowed": True,
            "validation_summary": {
                "package_manifest_id": "data-package-id",
                "reference_package": {"available": False},
            },
        },
    )
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 3
    assert result["status"] == "BLOCKED_INVALID_REFERENCE_PACKAGE"
    assert result["blocker_codes"][-1] == "REFERENCE_PACKAGE_NOT_FOUND"
    assert result["package_audit_success"] is False


@pytest.mark.parametrize(
    "error_code",
    ["REFERENCE_ARCHIVE_NOT_FOUND", "REFERENCE_ARCHIVE_HASH_MISMATCH"],
)
def test_missing_or_hash_mismatched_reference_archive_fails_closed(
    monkeypatch, error_code
):
    install_passing_package_audits(monkeypatch)

    def invalid_archive(config, path):
        raise ContractError(error_code, "reference archive rejected", path=str(path))

    monkeypatch.setattr(gate, "audit_reference_archive", invalid_archive)
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 3
    assert result["status"] == "BLOCKED_INVALID_REFERENCE_PACKAGE"
    assert result["blocker_codes"][-1] == error_code
    assert result["exact_reproduction_authorized"] is False


def test_reference_to_data_binding_mismatch_fails_closed(monkeypatch):
    config = object()
    monkeypatch.setattr(gate.ProjectConfig, "load", lambda path: config)

    def binding_failure(observed):
        raise ContractError("REFERENCE_DATA_BINDING_MISMATCH", "different package IDs")

    monkeypatch.setattr(gate, "audit_config", binding_failure)
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 3
    assert result["status"] == "BLOCKED_INVALID_REFERENCE_PACKAGE"
    assert result["blocker_codes"][-1] == "REFERENCE_DATA_BINDING_MISMATCH"


def test_comparable_package_audit_exit_zero_cannot_bypass_strict_gate(monkeypatch):
    mac_current = matching_current_environment()
    mac_current["operating_system"].update(
        system="Darwin", machine="arm64", platform="macOS-26.5-arm64"
    )
    install_passing_package_audits(monkeypatch, current=mac_current)
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 2
    assert result["status"] == "BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"
    assert result["reference_package_audit_status"] == (
        "REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY"
    )
    assert result["package_audit_success"] is True
    assert result["exact_environment_ready"] is False


def test_matching_synthetic_environment_only_authorizes_next_step(monkeypatch, tmp_path):
    install_passing_package_audits(monkeypatch)
    result, exit_code = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert exit_code == 0
    assert result["status"] == "EXACT_REFERENCE_ENVIRONMENT_READY"
    assert result["exact_environment_ready"] is True
    assert result["exact_reproduction_authorized"] is True
    assert result["training_invoked"] is False
    assert result["production_approval"] is False
    assert result["production_inference_49054_allowed"] is False
    assert result["next_authorized_action"] == "AWAIT_OWNER_AUTHORIZATION_TO_RUN_PREPARE"
    assert not (tmp_path / "runs").exists()


def test_audit_id_is_stable_and_cli_is_sorted_read_only(monkeypatch, tmp_path, capsys):
    install_passing_package_audits(monkeypatch)
    left, left_exit = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    right, right_exit = gate.run_exact_environment_audit("config.yaml", "reference.zip")
    assert left_exit == right_exit == 0
    assert left["exact_environment_audit_id"] == right["exact_environment_audit_id"]
    assert not (tmp_path / "runs").exists()

    assert gate.main(["--config", "config.yaml", "--reference-archive", "reference.zip"]) == 0
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert payload["training_invoked"] is False
    assert rendered.index('"audit_schema_version"') < rendered.index('"blocker_codes"')


def test_execution_contract_freezes_all_terminal_gates_and_success_conditions():
    contract = json.loads(EXECUTION_CONTRACT_PATH.read_text(encoding="utf-8"))
    expected_states = [
        "VERIFY_SOURCE_COMMIT",
        "VERIFY_DATA_ARCHIVE",
        "VERIFY_REFERENCE_ARCHIVE",
        "AUDIT_CANONICAL_DATA",
        "AUDIT_REFERENCE_PACKAGE",
        "AUDIT_EXACT_ENVIRONMENT",
        "PREPARE",
        "VERIFY_FEATURE_PARITY",
        "TRAIN",
        "CAPTURE_CONVERGENCE_AND_WARNINGS",
        "COMPARE_2787_REFERENCE_PREDICTIONS",
        "COMPARE_METRICS",
        "COMPARE_PROBABILITIES",
        "EVALUATE",
        "EXPORT",
        "TWO_ROW_INFERENCE_SMOKE_TEST",
        "IMMUTABLE_REPLAY",
        "FINALIZE_STATUS",
    ]
    states = contract["state_machine"]
    assert [state["state"] for state in states] == expected_states
    required_fields = {
        "entry_conditions",
        "command_template",
        "expected_machine_status",
        "expected_artifact",
        "artifact_content_id_or_hash",
        "failure_status",
        "may_write",
        "may_fit",
        "owner_authorization_required",
        "terminal_failure",
    }
    assert all(required_fields <= set(state) for state in states)
    assert all(state["terminal_failure"] is True for state in states[:6])
    assert contract["global_guardrails"]["first_six_gates_must_pass_before_train_reachable"] is True
    assert contract["global_guardrails"]["production_approval"] is False
    assert contract["global_guardrails"]["production_inference_49054_allowed"] is False
    assert contract["exact_success_conditions"]["reference_prediction_rows"] == {
        "train": 1822,
        "dev": 448,
        "test": 467,
        "anchor50": 50,
        "total": 2787,
    }
    assert contract["exact_success_conditions"]["metrics_maximum_absolute_delta"] == 1e-12
    assert contract["exact_success_conditions"]["probabilities_maximum_absolute_delta"] == 1e-10
