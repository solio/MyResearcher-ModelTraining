from __future__ import annotations

import json
from pathlib import Path

from semantic_model.audit_encoder_readiness import (
    BLOCKED_DATA_STATUS,
    READINESS_SCHEMA_VERSION,
    main,
    run_encoder_readiness,
)


def test_missing_canonical_config_fails_closed_and_is_deterministic(tmp_path: Path):
    config_path = tmp_path / "missing-config.yaml"
    left, left_exit_code = run_encoder_readiness(config_path)
    right, right_exit_code = run_encoder_readiness(config_path)

    assert left_exit_code == right_exit_code == 2
    assert left == right
    assert left["audit_schema_version"] == READINESS_SCHEMA_VERSION
    assert left["status"] == BLOCKED_DATA_STATUS
    assert left["selection_or_training_authorized"] is False
    assert left["blocker_codes"][0] == "BLOCKED_CANONICAL_DATA_AUDIT"
    assert "CONFIG_NOT_FOUND" in left["blocker_codes"]
    assert not (tmp_path / "runs").exists()


def test_cli_emits_stable_json_without_creating_an_artifact(tmp_path: Path, capsys):
    config_path = tmp_path / "missing-config.yaml"

    assert main(["--config", str(config_path)]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_DATA_STATUS
    assert payload["selection_or_training_authorized"] is False
    assert not list(tmp_path.iterdir())
