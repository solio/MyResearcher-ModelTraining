from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_s1 as s1
from semantic_model.errors import ContractError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "manifests/encoder-m2-experiment-contract-v1.json"


def test_contract_is_direct_train_dev_execution_without_owner_artifacts():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["m2_direct_owner_execution"] == {
        "training_allowed": True,
        "scope": "M2_S1_TRAIN_DEV_ONLY",
        "direct_instruction_is_sufficient": True,
        "no_additional_execution_identity_gate": True,
    }
    assert "m2_execution_authorization" not in contract
    assert "m2_s1_pre_training_execution_contract" not in contract
    assert contract["m2_s1_train_dev_technical_preflight"]["forbidden_data_roles"] == ["Test", "Anchor", "Gold", "OOD", "reference_predictions"]
    assert not (ROOT / "manifests" / "owner-decisions").exists()


def test_preflight_uses_only_selected_train_dev_loader_and_never_canonical_audit(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: ["x"] for head in s1.V1_HEADS})
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: calls.append("train_dev") or (schema, [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda *_args: pytest.fail("canonical audit must not run"))
    monkeypatch.setattr(s1, "validate_fixed_cache_snapshot", lambda _cache, frozen: calls.append("cache") or (tmp_path / "snapshot", {"content_address": "synthetic"}))
    preflight = s1.validate_m2_s1_preflight(ROOT / "configs/baseline_v0.3.5.yaml", tmp_path / "cache", worktree=ROOT, contract_path=CONTRACT)
    assert calls == ["train_dev", "cache"]
    assert len(preflight["train"]) == 1822 and len(preflight["dev"]) == 448


def test_preflight_contract_rejects_non_train_dev_role_change(tmp_path: Path):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["m2_s1_train_dev_technical_preflight"]["forbidden_data_roles"].append("anything")
    path = tmp_path / "contract.json"; path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ContractError) as error:
        s1._contract_requirements(path)
    assert error.value.code == "M2_S1_TECHNICAL_PREFLIGHT_CONTRACT_INVALID"


def test_cache_verifier_checks_every_one_of_eight_files(monkeypatch, tmp_path: Path):
    frozen = s1._contract_requirements(CONTRACT)
    base = tmp_path / "cache" / frozen["snapshot"]["required_relative_directory"]
    base.mkdir(parents=True)
    for row in frozen["snapshot"]["files"]:
        (base / row["path"]).write_bytes(b"x")
    with pytest.raises(ContractError) as error:
        s1.validate_fixed_cache_snapshot(tmp_path / "cache", frozen)
    assert error.value.code == "M2_S1_CACHE_SNAPSHOT_FILE_MISMATCH"


def test_output_cannot_replay_or_enter_protected_paths(tmp_path: Path):
    frozen = s1._contract_requirements(CONTRACT)
    existing = tmp_path / "existing"; existing.mkdir()
    with pytest.raises(ContractError) as replay:
        s1.validate_output_dir(existing, tmp_path / "cache", worktree=tmp_path, frozen=frozen)
    assert replay.value.code == "M2_S1_OUTPUT_REPLAY_OR_EXISTS"
    with pytest.raises(ContractError) as protected:
        s1.validate_output_dir(tmp_path / "runs" / "x", tmp_path / "cache", worktree=tmp_path, frozen=frozen)
    assert protected.value.code == "M2_S1_OUTPUT_PROTECTED_PATH"


def _seed(seed: int, device: str = "mps") -> dict:
    return {"seed": seed, "resource": {"actual_device": device}, "metrics": {"dev": {head: {"macro_f1": 0.5} for head in s1.V1_HEADS}}}


def test_aggregate_requires_exact_three_seeds_and_one_device():
    with pytest.raises(ContractError) as incomplete:
        s1.aggregate_s1_seed_results([_seed(35)])
    assert incomplete.value.code == "M2_S1_INCOMPLETE_SEEDS"
    with pytest.raises(ContractError) as mixed:
        s1.aggregate_s1_seed_results([_seed(35), _seed(71, "cpu"), _seed(107)])
    assert mixed.value.code == "M2_S1_MIXED_DEVICE"
    assert s1.aggregate_s1_seed_results([_seed(35), _seed(71), _seed(107)])["selected_candidate"] is False


def test_preflight_failure_never_imports_runtime_or_creates_output(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr(s1, "validate_m2_s1_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(ContractError("M2_S1_DATA_CONTRACT_INVALID", "synthetic")))
    result = s1.run_m2_s1(tmp_path / "config", tmp_path / "output", tmp_path / "cache", runtime_loader=lambda: calls.append("runtime"))
    assert result["blocker_codes"] == ["M2_S1_DATA_CONTRACT_INVALID"]
    assert result["training_invoked"] is result["model_loaded"] is result["cache_accessed"] is result["output_created"] is False
    assert calls == [] and not (tmp_path / "output").exists()


def test_mixed_device_failure_is_device_stratified_and_not_selected(tmp_path: Path):
    preflight = {"frozen_contract": {"contract": {"new_model_lineage": {}}, "m1_controls": {}}, "snapshot_identity": {}, "identity": {}}
    result = s1._failure(None, ContractError("M2_S1_MIXED_DEVICE", "mixed", device_stratified_seed_devices={"35": "mps", "71": "cpu"}), preflight, True)
    assert result["selected_candidate"] is False
    assert result["device_stratified_rejected_evidence"] == {"35": "mps", "71": "cpu"}


def test_success_path_records_every_seed_critical_boundary_report(monkeypatch, tmp_path: Path):
    root = tmp_path / "fresh-output"
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: ["x"] for head in s1.V1_HEADS})
    preflight = {
        "frozen_contract": {"contract": {"new_model_lineage": {"lineage_id": "synthetic"}}, "m1_controls": {}},
        "snapshot": tmp_path / "snapshot",
        "snapshot_identity": {},
        "schema": schema,
        "train": [object()] * 1822,
        "dev": [object()] * 448,
        "identity": {},
    }
    captured: dict = {}
    monkeypatch.setattr(s1, "validate_m2_s1_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(s1, "validate_runtime_identity", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(s1, "validate_output_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(s1, "_config", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(s1, "_limits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m1, "_write_content_manifest", lambda _root, payload: captured.update(payload) or {"content_address": "synthetic"})

    reports = {35: "a" * 64, 71: "b" * 64, 107: "c" * 64}
    def fake_seed(**kwargs):
        seed = kwargs["seed"]
        return {"seed": seed, "resource": {"actual_device": "mps"}, "metrics": {"dev": {head: {"macro_f1": 0.5} for head in s1.V1_HEADS}}, "checkpoint_sha256": "d" * 64, "critical_boundary_report_sha256": reports[seed]}

    result = s1.run_m2_s1(tmp_path / "config", root, tmp_path / "cache", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_seed)
    assert result["status"] == "M2_S1_CONTROL_COMPLETED"
    assert captured["critical_boundary_report"] == {str(seed): report for seed, report in reports.items()}
    assert captured["selected_candidate"] is False
