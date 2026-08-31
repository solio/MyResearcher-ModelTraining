from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_model import encoder_m2_s2 as s2
from semantic_model.errors import ContractError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "manifests" / "encoder-m2-experiment-contract-v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _fake_model(extra_encoder_parameter: bool = False):
    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Heads(dict):
        pass

    class Model:
        last_transformer_block_prefix = "encoder.layer.2"
        heads = Heads({head: object() for head in range(7)})

        def named_parameters(self):
            values = [
                ("encoder.embeddings.word_embeddings.weight", Parameter(False)),
                ("encoder.layer.0.attention.weight", Parameter(False)),
                ("encoder.layer.1.attention.weight", Parameter(False)),
                ("encoder.layer.2.attention.weight", Parameter(True)),
                ("encoder.layer.2.output.weight", Parameter(True)),
            ]
            values.extend((f"heads.{index}.weight", Parameter(True)) for index in range(7))
            if extra_encoder_parameter:
                values.append(("encoder.layer.1.output.bias", Parameter(True)))
            return values

    return Model()


def _metric_rows(contract: dict, value: float, *, support: int = 25) -> dict:
    dev: dict = {}
    for head in s2.V1_HEADS:
        dev[head] = {"macro_f1": value}
        if head == "reasoning_tags":
            dev[head]["per_label"] = {}
        else:
            dev[head]["per_class"] = {}
    for item in contract["dev_metrics_and_no_regression"]["critical_boundary_proxies"]:
        key = "per_label" if item["head"] == "reasoning_tags" else "per_class"
        dev[item["head"]][key] = {label: {"f1": value, "support": support} for label in item["labels"]}
    return {"dev": dev}


def _synthetic_control(contract: dict, value: float = 0.5, *, support: int = 25) -> dict:
    return {"seed_metrics": {seed: _metric_rows(contract, value, support=support) for seed in s2.SEEDS}}


def _synthetic_results(contract: dict, value: float = 0.52, *, support: int = 25) -> list[dict]:
    return [{"seed": seed, "metrics": _metric_rows(contract, value, support=support), "resource": {"actual_device": "mps"}} for seed in s2.SEEDS]


def test_s2_trainable_parameter_gate_allows_only_final_block_and_heads():
    identity = s2.validate_s2_trainable_parameters(_fake_model())
    assert identity["last_transformer_block_prefix"] == "encoder.layer.2"
    with pytest.raises(ContractError) as error:
        s2.validate_s2_trainable_parameters(_fake_model(extra_encoder_parameter=True))
    assert error.value.code == "M2_S2_TRAINABLE_PARAMETER_CONTRACT_VIOLATION"


def test_s2_contract_freezes_partial_unfreeze_and_three_seeds():
    frozen = s2._contract_requirements(CONTRACT_PATH)
    assert frozen["stage"]["stage_id"] == s2.STAGE_ID
    assert frozen["stage"]["unfrozen_transformer_blocks"] == 1
    assert frozen["stage"]["seeds"] == [35, 71, 107]


def test_s2_aggregate_requires_three_seeds_and_one_device():
    contract = _contract()
    complete = _synthetic_results(contract)
    assert s2.aggregate_s2_seed_results(complete)["selected_candidate"] is False
    with pytest.raises(ContractError) as incomplete:
        s2.aggregate_s2_seed_results(complete[:2])
    assert incomplete.value.code == "M2_S2_INCOMPLETE_SEEDS"
    complete[1]["resource"]["actual_device"] = "cpu"
    with pytest.raises(ContractError) as mixed:
        s2.aggregate_s2_seed_results(complete)
    assert mixed.value.code == "M2_S2_MIXED_DEVICE"


def test_matching_seed_report_passes_explicit_promotion_predicates():
    contract = _contract()
    control = _synthetic_control(contract, value=0.50)
    results = _synthetic_results(contract, value=0.52)
    aggregate = s2.aggregate_s2_seed_results(results)
    report = s2._matching_seed_report(contract, control, results, aggregate)
    assert report["promotion"]["passed"] is True
    assert len(report["mean_improvement_heads"]) == 7
    assert report["failed_heads"] == []


def test_matching_seed_report_rejects_mean_regression_and_marks_s3_trigger():
    contract = _contract()
    control = _synthetic_control(contract, value=0.50)
    results = _synthetic_results(contract, value=0.52)
    for item in results:
        item["metrics"]["dev"]["target_mode"]["macro_f1"] = 0.47
    report = s2._matching_seed_report(contract, control, results, s2.aggregate_s2_seed_results(results))
    assert "target_mode" in report["failed_heads"]
    assert "target_mode" in report["s3_triggered_heads"]
    assert report["promotion"]["passed"] is False


def test_support_below_twenty_is_not_evaluable_and_cannot_claim_pass():
    contract = _contract()
    control = _synthetic_control(contract, value=0.50, support=10)
    results = _synthetic_results(contract, value=0.52, support=10)
    report = s2._matching_seed_report(contract, control, results, s2.aggregate_s2_seed_results(results))
    entry = report["critical_labels"]["target_mode"]["ON_TARGET"]
    assert entry["status"] == "NOT_EVALUABLE_FOR_NUMERICAL_NO_REGRESSION"
    assert entry["no_regression_passed"] is None


def test_s2_success_path_writes_matching_report_and_never_selects(monkeypatch, tmp_path: Path):
    contract = _contract()
    root = tmp_path / "s2-output"
    preflight = {
        "frozen_contract": {"contract": contract, "snapshot": {}, "m1_controls": {}},
        "snapshot": tmp_path / "snapshot",
        "snapshot_identity": {},
        "schema": type("Schema", (), {"class_order": {head: ["x"] for head in s2.V1_HEADS}})(),
        "train": [object()] * 1822,
        "dev": [object()] * 448,
        "identity": {},
    }
    control = _synthetic_control(contract)
    captured: dict = {}
    monkeypatch.setattr(s2, "validate_s2_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(s2, "load_s1_control", lambda *_args, **_kwargs: {"root": tmp_path / "s1", "manifest": {"content_address": s2.S1_ARTIFACT_CONTENT_ADDRESS}, **control})
    monkeypatch.setattr(s2, "validate_output_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(s2, "validate_runtime_identity", lambda *_args, **_kwargs: {"synthetic": True})
    monkeypatch.setattr(s2, "_config", lambda *_args, **_kwargs: {"head_dropout": 0.1, "class_order": {head: ["x"] for head in s2.V1_HEADS}, "stopping": {"minimum_delta": 0.0}, "optimizer": {"betas": [0.9, 0.999], "epsilon": 1e-8}})
    monkeypatch.setattr(s2.s1, "_limits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(s2.m1, "_write_content_manifest", lambda _root, payload: captured.update(payload) or {"content_address": "synthetic"})

    def fake_seed(**kwargs):
        seed = kwargs["seed"]
        return {"seed": seed, "metrics": _metric_rows(contract, 0.52), "resource": {"actual_device": "mps"}, "checkpoint_sha256": "a" * 64, "critical_boundary_report_sha256": "b" * 64}

    result = s2.run_m2_s2(tmp_path / "config", root, tmp_path / "cache", s1_artifact=tmp_path / "s1", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_seed)
    assert result["status"] == "M2_S2_CONTROL_COMPLETED"
    assert captured["selected_candidate"] is False
    assert captured["s2_vs_s1_report_sha256"]
