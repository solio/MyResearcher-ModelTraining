from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from semantic_model.audit_data import main, run_audit


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "schema"
    / "semantic-schema-calibrated-v0.2.1.json"
)


def write_blocked_config(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir()
    input_record = {
        "sample_id": "P001",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "published_at": "2026-07-13T16:29:31+08:00",
        "model_text": "继续持有",
    }
    (data_root / "inputs.jsonl").write_text(
        f"{json.dumps(input_record, ensure_ascii=False)}\n", encoding="utf-8"
    )
    with (data_root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(input_record))
        writer.writeheader()
        writer.writerow(input_record)
    config = {
        "config_schema_version": "myresearcher.semantic-baseline-config.v1",
        "project_root": str(SCHEMA_PATH.parents[1]),
        "schema_path": str(SCHEMA_PATH),
        "data": {
            "root": str(data_root),
            "canonical_inputs": "inputs.jsonl",
            "canonical_metadata": "metadata.csv",
            "frozen_teacher_labels": "missing-labels.jsonl",
            "quarantine_manifest": "missing-quarantine.json",
            "split_manifest": "missing-split.json",
            "field_weights": "missing-weights.jsonl",
            "anchor_labels": "missing-anchor.jsonl",
            "anchor_manifest": "missing-anchor-manifest.json",
            "baseline_report": "missing-baseline.json",
            "preprocessing_contract": "missing-preprocessing.json",
            "package_manifest": "missing-package.json",
            "expected": {"inputs": 1},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_missing_canonical_artifacts_return_stable_blocker(tmp_path):
    config_path = write_blocked_config(tmp_path)
    result, exit_code = run_audit(config_path)
    assert exit_code == 2
    assert result["status"] == "BLOCKED_MISSING_CANONICAL_ARTIFACTS"
    assert result["blocker_codes"][0] == "BLOCKED_MISSING_CANONICAL_ARTIFACTS"
    assert "BLOCKED_MISSING_FROZEN_TEACHER_LABELS" in result["blocker_codes"]
    assert result["training_allowed"] is False
    assert not (tmp_path / "runs").exists()


def test_audit_cli_emits_json_and_nonzero_without_writing_training_artifacts(
    tmp_path, capsys
):
    config_path = write_blocked_config(tmp_path)
    assert main(["--config", str(config_path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED_MISSING_CANONICAL_ARTIFACTS"
    assert not (tmp_path / "runs").exists()


def test_blocked_audit_is_deterministic(tmp_path):
    config_path = write_blocked_config(tmp_path)
    left, _ = run_audit(config_path)
    right, _ = run_audit(config_path)
    assert left == right
    assert left["audit_id"] == right["audit_id"]
