from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_reasoning_r2 as r2
from semantic_model.errors import ContractError


def _labels() -> list[str]:
    return ["NO_REASON_GIVEN", "TECHNICAL_PRICE", "FUNDAMENTAL", "SARCASM_IRONY", "OTHER"]


def _metric(value: float = 0.50, *, critical: float | None = None) -> dict:
    critical = value if critical is None else critical
    return {"macro_f1": value, "micro_f1": value, "exact_set_accuracy": value, "per_label": {name: {"f1": critical if name in r2.CRITICAL_LABELS else value, "support": 25 if name in r2.CRITICAL_LABELS else 10} for name in _labels()}}


def _rows(value: float = 0.50, *, critical: float | None = None) -> list[dict]:
    return [{"seed": seed, "resource": {"actual_device": "cpu"}, "metrics": {"dev_raw": _metric(value), "dev_calibrated": _metric(value, critical=critical), "train_raw": _metric(value), "train_calibrated": _metric(value), "thresholds": {}}} for seed in r2.SEEDS]


def test_threshold_grid_and_tie_break_are_frozen():
    assert len(r2.THRESHOLD_GRID) == 91
    assert 0.50 in r2.THRESHOLD_GRID
    labels = [[0] * 15 for _ in range(8)]
    probabilities = [[0.5] * 15 for _ in range(8)]
    thresholds, details = r2.choose_thresholds(labels, probabilities, ["ONLY"] * 15)
    assert thresholds["ONLY"] == 0.50
    assert details["ONLY"]["tie_break"] == "closest_to_0_50_then_higher_threshold"


def test_train_selection_is_independent_of_dev_and_metrics_use_frozen_thresholds():
    labels = [[1] + [0] * 14, [0] * 15, [1] + [0] * 14, [0] * 15]
    probabilities = [[0.90] + [0.1] * 14, [0.80] + [0.1] * 14, [0.40] + [0.1] * 14, [0.20] + [0.1] * 14]
    thresholds, _ = r2.choose_thresholds(labels, probabilities, ["NO_REASON_GIVEN", *(["x"] * 14)])
    assert thresholds["NO_REASON_GIVEN"] == 0.40
    raw = r2.metrics_from_probabilities(labels, probabilities, ["NO_REASON_GIVEN", *(["x"] * 14)], {name: 0.50 for name in ["NO_REASON_GIVEN", *(["x"] * 14)]})
    calibrated = r2.metrics_from_probabilities(labels, probabilities, ["NO_REASON_GIVEN", *(["x"] * 14)], thresholds)
    assert calibrated["per_label"]["NO_REASON_GIVEN"]["f1"] >= raw["per_label"]["NO_REASON_GIVEN"]["f1"]


def test_classical_gate_checks_all_numeric_bounds_and_critical_labels():
    classical = {"per_label": {name: {"f1": 0.50, "support": 25} for name in r2.CRITICAL_LABELS}}
    aggregate = {"macro_f1": {"mean": 0.41, "worst_seed": 0.39, "sample_standard_deviation": 0.01}, "micro_f1": {"mean": 0.48, "worst_seed": 0.46}, "exact_set_accuracy": {"mean": 0.12, "worst_seed": 0.10}}
    assert r2.classical_gate(aggregate, _rows(0.50), classical)["passed"] is True
    rejected = r2.classical_gate(aggregate, _rows(0.50, critical=0.44), classical)
    assert rejected["passed"] is False
    assert "NO_REASON_GIVEN" in rejected["critical_failures"]


def test_checkpoint_identity_rejects_seed_or_revision_mismatch():
    checkpoint = {"stage_id": r2.corrective.STAGE_ID, "seed": 35, "frozen_config": {"model_id": r2.MODEL_ID, "revision": r2.REVISION, "max_length": 256, "batch_size": 16, "truncation": "HEAD_TAIL", "local_files_only": True, "trust_remote_code": False}, "last_transformer_block_prefix": "encoder.encoder.layer.2", "last_transformer_block_state_dict": {}, "reasoning_head_state_dict": {}}
    r2._validate_checkpoint(checkpoint, 35, checkpoint["frozen_config"])
    checkpoint["seed"] = 71
    with pytest.raises(ContractError) as error:
        r2._validate_checkpoint(checkpoint, 35, checkpoint["frozen_config"])
    assert error.value.code == "M2_R2_CHECKPOINT_SEED_MISMATCH"


def test_preflight_only_loads_train_dev_and_fixed_cache(monkeypatch, tmp_path: Path):
    contract = json.loads((Path(__file__).resolve().parents[1] / "manifests/encoder-m2-experiment-contract-v1.json").read_text())
    calls: list[str] = []
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: _labels() for head in r2.V1_HEADS})
    monkeypatch.setattr(r2.s1, "_contract_requirements", lambda _path: {"contract": contract, "contract_sha256": "c" * 64})
    monkeypatch.setattr(m1.ProjectConfig, "load", lambda _path: object())
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: calls.append("train_dev") or (schema, [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(r2.s1, "validate_fixed_cache_snapshot", lambda *_args: calls.append("cache") or (tmp_path / "snapshot", {"content_address": "cache"}))
    monkeypatch.setattr(r2.corrective, "_load_comparator", lambda *_args, **_kwargs: {"manifest": {"content_address": r2.PARENT_CONTENT_ADDRESS}, "seed_metrics": {}})
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda *_args: pytest.fail("R2 must not run canonical audit"), raising=False)
    result = r2.validate_r2_preflight(Path("configs/baseline_v0.3.5.yaml"), tmp_path / "cache", worktree=Path.cwd(), contract_path=Path("contract.json"), parent_artifact=tmp_path / "parent")
    assert calls == ["train_dev", "cache"]
    assert len(result["train"]) == 1822 and len(result["dev"]) == 448


def test_success_path_is_calibration_only_and_writes_small_manifest(monkeypatch, tmp_path: Path):
    contract = json.loads((Path(__file__).resolve().parents[1] / "manifests/encoder-m2-experiment-contract-v1.json").read_text())
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: _labels() for head in r2.V1_HEADS})
    preflight = {"frozen_contract": {"contract": contract, "contract_sha256": "c" * 64}, "snapshot": tmp_path / "snapshot", "snapshot_identity": {"content_address": "cache"}, "schema": schema, "train": [object()] * 1822, "dev": [object()] * 448, "parent": {"manifest": {"content_address": r2.PARENT_CONTENT_ADDRESS}}, "classical_reasoning": {"primary_macro_f1": 0.41025605014132455, "micro_f1": 0.48633879781420764, "exact_set_accuracy": 0.12723214285714285, "per_label": {name: {"f1": 0.40, "support": 25} for name in r2.CRITICAL_LABELS}}, "identity": {"schema_version": "synthetic"}}
    root = tmp_path / "out"
    monkeypatch.setattr(r2, "validate_r2_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(r2.s1, "validate_runtime_identity", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(r2.s1, "validate_output_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(r2.s1, "_limits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r2.corrective, "_config", lambda *_args: {"head_dropout": 0.1, "max_length": 256, "batch_size": 16, "class_order": {head: _labels() for head in r2.V1_HEADS}, "stopping": {"minimum_delta": 0.0}, "optimizer": {"betas": [0.9, 0.999], "epsilon": 1e-8}})

    def fake_seed(**kwargs):
        seed = kwargs["seed"]
        return {"seed": seed, "resource": {"actual_device": "cpu"}, "metrics": {"dev_raw": _metric(0.50), "dev_calibrated": _metric(0.50), "train_raw": _metric(0.50), "train_calibrated": _metric(0.50), "thresholds": {}}, "checkpoint_sha256": "a" * 64, "thresholds_sha256": "b" * 64, "train_summary_sha256": "c" * 64, "dev_metrics_sha256": "d" * 64}

    result = r2.run_r2(tmp_path / "config", root, tmp_path / "cache", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_seed)
    assert result["status"] == "CALIBRATED_REASONING_DIAGNOSTIC_PASSED"
    assert result["training_invoked"] is False
    assert result["selected_candidate"] is False
    assert (root / "content-addressed-manifest.json").is_file()
