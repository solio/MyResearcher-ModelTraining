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


@pytest.mark.real_data
@pytest.mark.skipif(not DATA_PATH.is_file(), reason="local upstream snapshot unavailable")
def test_local_immutable_package_audit_is_read_only_and_ready():
    before = sha256_file(DATA_PATH)
    result, exit_code = run_audit(CONFIG_PATH)
    after = sha256_file(DATA_PATH)
    assert before == after
    assert exit_code == 0
    assert result["status"] == "READY_FOR_DIAGNOSTIC_BASELINE_RUN"
    assert result["training_allowed"] is True
    assert result["reproduction_blocker_codes"] == [
        "BLOCKED_MISSING_REFERENCE_ENVIRONMENT"
    ]
    assert result["observed"]["canonical_inputs"]["rows"] == 3000
    assert result["observed"]["canonical_inputs"]["sample_id_unique"] == 3000
    assert result["validation_summary"]["evidence_violation_count"] == 21
    assert result["validation_summary"]["trainable_count"] == 2979
    assert result["validation_summary"]["package_manifest_id"] == (
        "cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b"
    )
