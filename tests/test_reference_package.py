from pathlib import Path
import json

import pytest

from semantic_model.audit_reference import run_reference_audit
from semantic_model.hashes import content_addressed_id, sha256_file
from semantic_model.reference_package import compare_runtime_to_reference


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PROJECT_ROOT = Path(
    "/Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining"
)
# A worktree does not carry large immutable packages.  When the maintained
# local package is available, audit it read-only through its own config rather
# than treating a worktree-relative missing package as a package failure.
CONFIG_PATH = (
    SOURCE_PROJECT_ROOT / "configs/baseline_v0.3.5.yaml"
    if (SOURCE_PROJECT_ROOT / "configs/baseline_v0.3.5.yaml").is_file()
    else PROJECT_ROOT / "configs/baseline_v0.3.5.yaml"
)
REFERENCE_ZIP = Path(
    "/Users/mac/Documents/Codex/2026-08-27/"
    "MyResearcher_Semantic_Baseline_Reference_v0.3.5_828944580b96d872.zip"
)
AUDIT_MANIFEST = PROJECT_ROOT / "manifests/baseline-reference-audit-v0.3.5.json"


def reference_environment() -> dict:
    return {
        "python": {
            "version_info": [3, 12, 13, "final", 0],
            "implementation": "CPython",
        },
        "operating_system": {"system": "Linux", "machine": "x86_64"},
        "packages": {
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "scikit_learn": "1.8.0",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        },
    }


def exact_observed_runtime() -> dict:
    return {
        "python_version": "3.12.13",
        "implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "packages": {
            "numpy": "2.3.5",
            "scipy": "1.17.0",
            "scikit-learn": "1.8.0",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        },
        "threadpools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "version": "0.3.30",
                "threading_layer": "pthreads",
                "architecture": "SkylakeX",
            }
        ],
    }


def test_exact_reference_runtime_is_eligible_for_prediction_comparison():
    result = compare_runtime_to_reference(
        exact_observed_runtime(), reference_environment()
    )
    assert result["matches_reference"] is True
    assert result["mismatches"] == []


def test_dependency_or_platform_drift_blocks_exact_reproduction():
    observed = exact_observed_runtime()
    observed["system"] = "Darwin"
    observed["machine"] = "arm64"
    observed["packages"]["scikit-learn"] = "1.7.2"
    result = compare_runtime_to_reference(observed, reference_environment())
    assert result["matches_reference"] is False
    assert {item["field"] for item in result["mismatches"]} == {
        "system",
        "machine",
        "package:scikit-learn",
    }


def test_tracked_reference_audit_manifest_is_content_addressed():
    value = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))
    assert value["manifest_id"] == content_addressed_id(
        value, omit_keys={"manifest_id"}
    )


@pytest.mark.real_data
@pytest.mark.skipif(
    not REFERENCE_ZIP.is_file()
    or not (SOURCE_PROJECT_ROOT / "data/local/MyResearcher_Semantic_Baseline_Reference_v0.3.5").is_dir(),
    reason="reference ZIP or maintained local reference package unavailable",
)
def test_reference_zip_and_extracted_package_audit_are_read_only():
    before = sha256_file(REFERENCE_ZIP)
    result, exit_code = run_reference_audit(CONFIG_PATH, REFERENCE_ZIP)
    after = sha256_file(REFERENCE_ZIP)
    assert exit_code == 0
    assert before == after
    assert result["status"] == (
        "REFERENCE_PACKAGE_VALIDATED_COMPARABLE_ENVIRONMENT_ONLY"
    )
    assert result["archive"]["zip_sha256"] == (
        "78064a4fe739920491d70ff1888d9233b02b6ac3ac38db8e82080e3549857410"
    )
    assert result["archive"]["package_manifest_id"] == (
        "828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85"
    )
    assert result["archive"]["payload_file_count"] == 17
    assert result["archive"]["payload_total_bytes"] == 11_439_730
    assert result["exact_reproduction_authorized_for_current_environment"] is False
    assert result["production_approval"] is False
