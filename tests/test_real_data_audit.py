from pathlib import Path

import pytest

from semantic_model.audit_data import run_audit
from semantic_model.hashes import sha256_file


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "baseline_v0.3.5.yaml"
DATA_PATH = Path(
    __file__
).parents[1] / (
    "data/local/MyResearcher_Semantic_Immutable_Data_v0.3.5/"
    "data/teacher_inputs_3000_v0.3.jsonl"
)
REFERENCE_MANIFEST_PATH = Path(__file__).parents[1] / (
    "data/local/MyResearcher_Semantic_Baseline_Reference_v0.3.5/"
    "CONTENT_MANIFEST.json"
)


@pytest.mark.real_data
@pytest.mark.skipif(
    not DATA_PATH.is_file() or not REFERENCE_MANIFEST_PATH.is_file(),
    reason="local immutable data/reference packages unavailable",
)
def test_local_immutable_package_audit_is_read_only_and_ready():
    before = sha256_file(DATA_PATH)
    reference_before = sha256_file(REFERENCE_MANIFEST_PATH)
    result, exit_code = run_audit(CONFIG_PATH)
    after = sha256_file(DATA_PATH)
    reference_after = sha256_file(REFERENCE_MANIFEST_PATH)
    assert before == after
    assert reference_before == reference_after
    assert exit_code == 0
    assert result["status"] == "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN"
    assert result["training_allowed"] is True
    assert result["reproduction_blocker_codes"] == [
        "BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"
    ]
    assert result["observed"]["canonical_inputs"]["rows"] == 3000
    assert result["observed"]["canonical_inputs"]["sample_id_unique"] == 3000
    assert result["validation_summary"]["evidence_violation_count"] == 21
    assert result["validation_summary"]["trainable_count"] == 2979
    assert result["validation_summary"]["package_manifest_id"] == (
        "cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b"
    )
    reference = result["validation_summary"]["reference_package"]
    assert reference["package_manifest_id"] == (
        "828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85"
    )
    assert reference["original_model_sha256"] == (
        "4e1dbe0fe1d4d37be728cebe849630ffd75a1fb6d66988bd15112375e6476b5a"
    )
    assert reference["metrics_recomputation_maximum_absolute_difference"] == 0.0
    assert reference["diagnostics"] == {
        "scalar_estimators": 6,
        "scalar_converged": 0,
        "scalar_not_converged": 6,
        "reason_estimators": 15,
        "reason_converged": 15,
        "reason_not_converged": 0,
    }
    assert reference["predictions"]["rows"] == {
        "train": 1822,
        "dev": 448,
        "test": 467,
        "anchor50": 50,
    }
    assert reference["predictions"]["total_rows"] == 2787
    assert reference["environment_capture_is_retrospective"] is True
    assert reference["exact_reproduction_environment_match"] is False
    assert reference["allowed_current_status"] == "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
    assert reference["production_approval"] is False
