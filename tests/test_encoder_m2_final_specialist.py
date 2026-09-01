from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_model import encoder_m2_final_specialist as final
from semantic_model.errors import ContractError
from semantic_model.schema import V1_HEADS


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads((ROOT / "manifests/encoder-m2-experiment-contract-v1.json").read_text(encoding="utf-8"))


def _metric(value: float, *, reasoning: bool = False) -> dict:
    if reasoning:
        labels = {name: {"f1": value, "support": 25} for name in _contract()["immutable_controls"]["classical_v0_3_5_control"]["frozen_dev_metrics"]["reasoning_tags"]["per_label"]}
        return {"macro_f1": value, "micro_f1": value, "exact_set_accuracy": value, "per_label": labels}
    classes = {name: {"f1": value, "support": 25} for name in ["ON_TARGET", "CROSS_TARGET", "MARKET_GENERAL", "UNKNOWN", "BULL", "BEAR", "NEUTRAL", "MIXED", "ANGER", "ANXIETY", "CALM", "EXCITEMENT", "FEAR", "FOMO", "FRUSTRATION", "HOPE", "NONE_EXPLICIT", "REGRET", "COMPANY", "MARKET", "NOT_APPLICABLE", "OTHER", "POSITION", "PRICE", "ADD", "BUY", "DO_T", "HOLD", "NO_ACTION_SIGNAL", "REDUCE", "SELL", "WATCH", "SELF_CONTAINED", "PARTIAL_CONTEXT", "EXTERNAL_CONTEXT_REQUIRED"]}
    return {"macro_f1": value, "per_class": classes}


def _rows(value: float = 0.8) -> list[dict]:
    result = []
    for head in V1_HEADS:
        for seed in final.SEEDS:
            metric = _metric(value, reasoning=head == final.REASONING_HEAD)
            result.append({"head": head, "seed": seed, "metrics": {"dev": {head: metric}}, "resource": {"actual_device": "cpu"}})
    return result


def test_final_contract_is_fixed_and_train_dev_only():
    frozen = final._contract_requirements(ROOT / "manifests/encoder-m2-experiment-contract-v1.json")
    candidate = frozen["final"]
    assert candidate["model"]["model_id"] == "hfl/rbt3"
    assert candidate["model"]["revision"] == final.REVISION
    assert candidate["training"]["seeds"] == [35, 71, 107]
    assert candidate["data_scope"]["train_rows"] == 1822
    assert candidate["data_scope"]["dev_rows"] == 448
    assert candidate["data_scope"]["sealed_roles"] == ["Test", "Anchor", "Gold", "OOD", "reference_predictions"]
    assert candidate["selection"]["success_status"] == "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3"


def test_trainable_parameter_gate_allows_only_selected_block_and_head():
    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Model:
        def named_parameters(self):
            return [("encoder.embeddings.weight", Parameter(False)), ("specialist_blocks.stance.weight", Parameter(True)), ("heads.stance.weight", Parameter(True)), ("heads.reasoning_tags.weight", Parameter(False))]

    identity = final.validate_trainable_parameters(Model(), "stance")
    assert identity["active_head"] == "stance"
    assert identity["shared_encoder_frozen"] is True


def test_trainable_parameter_gate_rejects_shared_or_other_head_gradient():
    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Model:
        def named_parameters(self):
            return [("encoder.encoder.layer.0.weight", Parameter(True)), ("specialist_blocks.stance.weight", Parameter(True)), ("heads.stance.weight", Parameter(True))]

    with pytest.raises(ContractError) as error:
        final.validate_trainable_parameters(Model(), "stance")
    assert error.value.code == "M2_FINAL_TRAINABLE_PARAMETER_CONTRACT_VIOLATION"


def test_reasoning_fail_fast_gate_rejects_before_other_heads():
    contract = _contract()
    rows = []
    for seed in final.SEEDS:
        metric = _metric(0.20, reasoning=True)
        rows.append({"head": final.REASONING_HEAD, "seed": seed, "metrics": {"dev": {final.REASONING_HEAD: metric}}, "resource": {"actual_device": "cpu"}})
    gate = final._reasoning_gate(contract, rows)
    assert gate["passed"] is False
    assert gate["action"] == "STOP_BEFORE_OTHER_HEADS"
    assert gate["selected_candidate"] is False


def test_reasoning_gate_success_is_not_selection():
    contract = _contract()
    classical = contract["immutable_controls"]["classical_v0_3_5_control"]["frozen_dev_metrics"]["reasoning_tags"]
    rows = []
    for seed in final.SEEDS:
        value = {"macro_f1": 0.45, "micro_f1": 0.55, "exact_set_accuracy": 0.20, "per_label": {name: {"f1": max(0.0, float(item["f1"]) + 0.01), "support": item["support"]} for name, item in classical["per_label"].items()}}
        rows.append({"head": final.REASONING_HEAD, "seed": seed, "metrics": {"dev": {final.REASONING_HEAD: value}}, "resource": {"actual_device": "cpu"}})
    gate = final._reasoning_gate(contract, rows)
    assert gate["passed"] is True
    assert gate["action"] == "CONTINUE_SIX_SPECIALISTS"
    assert gate["selected_candidate"] is False


def test_final_classical_gate_requires_four_mean_improvements_and_critical_labels():
    contract = _contract()
    rows = _rows(0.9)
    gate = final._classical_gate(contract, rows)
    assert gate["passed"] is True
    assert gate["checks"]["minimum_four_head_improvements"] is True
    # A candidate below the Classical reference must fail at least one head.
    bad = _rows(0.0)
    rejected = final._classical_gate(contract, bad)
    assert rejected["passed"] is False
    assert rejected["failures"]


def test_sealed_roles_are_not_part_of_preflight(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: ["x"] for head in V1_HEADS})
    contract = _contract()
    monkeypatch.setattr(final.s1, "_contract_requirements", lambda _path: {"contract": contract, "contract_sha256": "a" * 64, "snapshot": {"required_relative_directory": "official-snapshot/x", "files": [], "content_address": "x"}, "runtime_identity": {}, "m1_controls": {}, "protected_output_roots": []})
    monkeypatch.setattr(final.m1.ProjectConfig, "load", lambda _path: object())
    monkeypatch.setattr(final.m1, "load_m1_partitions", lambda _config: calls.append("train_dev") or (schema, [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(final.s1, "validate_fixed_cache_snapshot", lambda *_args: calls.append("cache") or (tmp_path / "snapshot", {"content_address": "cache"}))
    monkeypatch.setattr(final.m1, "validate_canonical_audit", lambda *_args: pytest.fail("final preflight must not run canonical audit"), raising=False)
    result = final.validate_preflight("config", tmp_path / "cache", worktree=tmp_path, contract_path="contract.json")
    assert calls == ["train_dev", "cache"]
    assert len(result["train"]) == 1822 and len(result["dev"]) == 448
