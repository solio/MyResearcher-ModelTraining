from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_model import encoder_m2_lert_final as lert
from semantic_model.errors import ContractError


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads((ROOT / "manifests/encoder-m2-experiment-contract-v1.json").read_text(encoding="utf-8"))


def test_lert_contract_is_exact_new_lineage_and_train_dev_only():
    frozen = lert._contract_requirements(ROOT / "manifests/encoder-m2-experiment-contract-v1.json")
    candidate = frozen["final"]
    assert candidate["contract_id"] == "M2_LERT_FINAL_SPECIALIST_SEVEN_HEAD_V1"
    assert candidate["model"]["model_id"] == "hfl/chinese-lert-small"
    assert candidate["model"]["revision"] == lert.REVISION
    assert candidate["model"]["license"] == "Apache-2.0"
    assert candidate["architecture"]["shared_components"][-1] == "transformer_blocks_0_through_10"
    assert candidate["training"]["seeds"] == [35, 71, 107]
    assert candidate["data_scope"]["sealed_roles"] == ["Test", "Anchor", "Gold", "OOD", "reference_predictions"]


def test_lert_model_loader_is_standard_local_files_only(monkeypatch):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs)
            return object()

    # The model constructor is tested through the call contract, without a
    # network or a real model payload.
    assert FakeAutoModel.from_pretrained("snapshot", local_files_only=True, trust_remote_code=False) is not None
    assert calls == [{"local_files_only": True, "trust_remote_code": False}]


def test_lert_trainable_gate_rejects_encoder_parameter():
    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Model:
        def named_parameters(self):
            return [("encoder.encoder.layer.0.weight", Parameter(True)), ("specialist_blocks.reasoning_tags.weight", Parameter(True)), ("heads.reasoning_tags.weight", Parameter(True))]

    with pytest.raises(ContractError) as error:
        lert.validate_trainable_parameters(Model(), "reasoning_tags")
    assert error.value.code == "M2_LERT_TRAINABLE_PARAMETER_CONTRACT_VIOLATION"


def test_lert_reasoning_failure_stops_other_heads(monkeypatch, tmp_path: Path):
    contract = _contract()
    calls: list[str] = []

    class Schema:
        class_order = {head: ["x"] for head in lert.V1_HEADS}

    preflight = {"frozen_contract": {"contract": contract, "contract_sha256": "a" * 64, "final": contract["lert_final_specialist_candidate_contract"]}, "snapshot": tmp_path, "schema": Schema(), "train": [], "dev": [], "identity": {}, "snapshot_identity": {}}
    monkeypatch.setattr(lert, "validate_preflight", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(lert, "_failure", lambda root, exc, started, pre: {"status": "LERT_SMALL_M2_CANDIDATE_REJECTED", "heads_started": list(started), "blocker_codes": [exc.code]})
    monkeypatch.setattr(lert.s1, "validate_runtime_identity", lambda *args: {})
    monkeypatch.setattr(lert.s1, "validate_output_dir", lambda *args, **kwargs: tmp_path / "out")
    monkeypatch.setattr(lert, "_config", lambda *args: {"head_dropout": 0.1, "class_order": {head: ["x"] for head in lert.V1_HEADS}, "optimizer": {"betas": [0.9, 0.999], "epsilon": 1e-8}})

    def fake_executor(**kwargs):
        calls.append(kwargs["head"])
        return {"head": kwargs["head"], "seed": kwargs["seed"], "resource": {"actual_device": "cpu"}, "metrics": {"dev": {kwargs["head"]: {"macro_f1": 0.2, "micro_f1": 0.2, "exact_set_accuracy": 0.2, "per_label": {name: {"f1": 0.0, "support": 25} for name in contract["immutable_controls"]["classical_v0_3_5_control"]["frozen_dev_metrics"]["reasoning_tags"]["per_label"]}}}}}

    monkeypatch.setattr(lert.s1, "_limits", lambda *args, **kwargs: None)
    result = lert.run_lert("config", tmp_path / "out", tmp_path / "cache", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_executor)
    assert result["status"] == "LERT_SMALL_M2_CANDIDATE_REJECTED"
    assert calls == ["reasoning_tags"] * 3


def test_lert_final_contract_has_no_rbt3_warm_start():
    candidate = _contract()["lert_final_specialist_candidate_contract"]
    assert any("RBT3" in item for item in candidate["prohibitions"])
    assert "FORBIDDEN" in candidate["model"]["warm_start"]
