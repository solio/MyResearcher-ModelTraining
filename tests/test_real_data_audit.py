from pathlib import Path

import pytest

from semantic_model.audit_data import run_audit
from semantic_model.hashes import sha256_file


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "baseline_v0.3.5.yaml"
DATA_PATH = Path(
    "/Users/mac/Documents/trae_projects/MyResearcher/produce-docs/"
    "MyResearcher_Semantic_Sampling_Local_Pipeline_Formal_v0.1/formal_run/"
    "semantic_pilot_inputs.jsonl"
)


@pytest.mark.real_data
@pytest.mark.skipif(not DATA_PATH.is_file(), reason="local upstream snapshot unavailable")
def test_local_real_data_audit_is_read_only_and_explicitly_blocked():
    before = sha256_file(DATA_PATH)
    result, exit_code = run_audit(CONFIG_PATH)
    after = sha256_file(DATA_PATH)
    assert before == after
    assert exit_code == 2
    assert result["status"] == "BLOCKED_MISSING_CANONICAL_ARTIFACTS"
    assert result["observed"]["canonical_inputs"]["rows"] == 3000
    assert result["observed"]["canonical_inputs"]["sample_id_unique"] == 3000
    assert "BLOCKED_MISSING_SPLIT_V0_3_5" in result["blocker_codes"]

