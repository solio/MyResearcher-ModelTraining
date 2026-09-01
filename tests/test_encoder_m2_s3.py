from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_s1 as s1
from semantic_model import encoder_m2_s3 as s3
from semantic_model.errors import ContractError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "manifests/encoder-m2-experiment-contract-v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


class _Parameter:
    def __init__(self, requires_grad: bool = False):
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self, *, unexpected_encoder: bool = False):
        self.heads = {head: object() for head in s3.V1_HEADS}
        self.parameters_by_name = {
            "encoder.embeddings.word_embeddings.weight": _Parameter(),
            "encoder.layer.0.weight": _Parameter(),
            "encoder.layer.2.weight": _Parameter(),
        }
        self.parameters_by_name.update({f"heads.{head}.weight": _Parameter() for head in s3.V1_HEADS})
        if unexpected_encoder:
            self.parameters_by_name["encoder.layer.0.bias"] = _Parameter(True)

    def named_parameters(self):
        return list(self.parameters_by_name.items())


def _metric_rows(contract: dict, value: float, *, support: int = 25) -> dict:
    dev: dict = {}
    for head in s3.V1_HEADS:
        dev[head] = {"macro_f1": value}
        key = "per_label" if head == "reasoning_tags" else "per_class"
        dev[head][key] = {}
    for item in contract["dev_metrics_and_no_regression"]["critical_boundary_proxies"]:
        key = "per_label" if item["head"] == "reasoning_tags" else "per_class"
        dev[item["head"]][key] = {label: {"f1": value, "support": support} for label in item["labels"]}
    dev["reasoning_tags"].update({"micro_f1": value, "exact_set_accuracy": value})
    return {"dev": dev}


def _control(contract: dict, value: float = 0.20, *, support: int = 25) -> dict:
    return {"seed_metrics": {seed: _metric_rows(contract, value, support=support) for seed in s3.SEEDS}}


def _results(contract: dict, value: float = 0.21) -> list[dict]:
    return [
        {
            "target_head": head,
            "seed": seed,
            "metrics": {**_metric_rows(contract, value), "target_head": head},
            "resource": {"actual_device": "mps"},
        }
        for head in s3.TRIGGERED_HEADS
        for seed in s3.SEEDS
    ]


def test_s3_contract_freezes_two_heads_and_three_seeds_without_parent_change():
    frozen = s3._contract_requirements(CONTRACT)
    assert frozen["stage"]["stage_id"] == s3.STAGE_ID
    assert frozen["stage"]["seeds"] == [35, 71, 107]
    assert frozen["stage"]["encoder_state"] == "FROZEN"
    assert frozen["stage"]["unfrozen_transformer_blocks"] == 0
    assert s3.TRIGGERED_HEADS == ("emotion_primary", "reasoning_tags")


def test_only_selected_head_is_trainable_and_encoder_is_rejected_if_trainable():
    model = _FakeModel()
    s3._set_single_head_trainable(model, "emotion_primary")
    identity = s3.validate_s3_trainable_parameters(model, "emotion_primary")
    assert identity["target_head"] == "emotion_primary"
    assert identity["encoder_trainable"] is False
    with pytest.raises(ContractError) as error:
        s3.validate_s3_trainable_parameters(_FakeModel(unexpected_encoder=True), "emotion_primary")
    assert error.value.code == "M2_S3_TRAINABLE_PARAMETER_CONTRACT_VIOLATION"


def test_s3_aggregate_requires_all_six_runs_and_one_device():
    contract = _contract()
    complete = _results(contract)
    aggregate = s3.aggregate_s3_results(complete)
    assert aggregate["all_six_runs_complete"] is True
    assert aggregate["selected_candidate"] is False
    assert aggregate["promotion"] == "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC"
    with pytest.raises(ContractError) as incomplete:
        s3.aggregate_s3_results(complete[:-1])
    assert incomplete.value.code == "M2_S3_INCOMPLETE_RUNS"
    complete[1]["resource"]["actual_device"] = "cpu"
    with pytest.raises(ContractError) as mixed:
        s3.aggregate_s3_results(complete)
    assert mixed.value.code == "M2_S3_MIXED_DEVICE"


def test_matching_seed_report_is_diagnostic_and_includes_reasoning_secondary_metrics():
    contract = _contract()
    control = _control(contract, value=0.20)
    results = _results(contract, value=0.21)
    aggregate = s3.aggregate_s3_results(results)
    report = s3.matching_s1_report(contract, control, results, aggregate)
    assert report["selected_candidate"] is False
    assert report["promotion"] == "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC"
    assert report["per_head"]["emotion_primary"]["mean_delta"] == pytest.approx(0.01)
    assert "micro_f1" in report["per_head"]["reasoning_tags"]
    assert "exact_set_accuracy" in report["per_head"]["reasoning_tags"]
    assert report["critical_labels"]["emotion_primary"]["CALM"]["status"] == "REPORTED_S3_DIAGNOSTIC_ONLY"


def test_support_below_twenty_remains_report_only_not_a_pass():
    contract = _contract()
    control = _control(contract, support=10)
    results = _results(contract)
    report = s3.matching_s1_report(contract, control, results, s3.aggregate_s3_results(results))
    assert report["critical_labels"]["emotion_primary"]["CALM"]["status"] == "REPORTED_S3_DIAGNOSTIC_ONLY"
    assert report["critical_labels"]["emotion_primary"]["CALM"]["support_per_seed"] == [10, 10, 10]


def test_preflight_reads_only_train_dev_and_fixed_cache(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: ["x"] for head in s3.V1_HEADS})
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: calls.append("train_dev") or (schema, [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda *_args: pytest.fail("canonical audit must not run"))
    monkeypatch.setattr(s1, "validate_fixed_cache_snapshot", lambda *_args: calls.append("cache") or (tmp_path / "snapshot", {"content_address": "synthetic"}))
    preflight = s3.validate_s3_preflight(ROOT / "configs/baseline_v0.3.5.yaml", tmp_path / "cache", worktree=ROOT, contract_path=CONTRACT)
    assert calls == ["train_dev", "cache"]
    assert len(preflight["train"]) == 1822 and len(preflight["dev"]) == 448


def test_success_path_writes_six_run_evidence_and_never_promotes(monkeypatch, tmp_path: Path):
    contract = _contract()
    root = tmp_path / "s3-output"
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: ["x"] for head in s3.V1_HEADS})
    preflight = {
        "frozen_contract": {"contract": contract, "contract_sha256": "c" * 64},
        "snapshot": tmp_path / "snapshot",
        "snapshot_identity": {"content_address": "snapshot"},
        "schema": schema,
        "train": [object()] * 1822,
        "dev": [object()] * 448,
        "identity": {"schema_version": "synthetic"},
    }
    captured: dict = {}
    monkeypatch.setattr(s3, "validate_s3_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(s3.s2, "load_s1_control", lambda *_args, **_kwargs: {"root": tmp_path / "s1", "manifest": {"content_address": "s1"}, **_control(contract)})
    monkeypatch.setattr(s3.s1, "validate_runtime_identity", lambda *_args, **_kwargs: {"synthetic": True})
    monkeypatch.setattr(s3.s1, "validate_output_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(s3.s2, "_config", lambda *_args, **_kwargs: {"head_dropout": 0.1, "class_order": {head: ["x"] for head in s3.V1_HEADS}, "stopping": {"minimum_delta": 0.0}, "optimizer": {"betas": [0.9, 0.999], "epsilon": 1e-8}})
    monkeypatch.setattr(s3.s1, "_limits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(s3.m1, "_write_content_manifest", lambda _root, payload: captured.update(payload) or {"content_address": "synthetic"})

    def fake_seed(**kwargs):
        head, seed = kwargs["target_head"], kwargs["seed"]
        return {"target_head": head, "seed": seed, "metrics": {**_metric_rows(contract, 0.21), "target_head": head}, "resource": {"actual_device": "mps"}, "checkpoint_sha256": "a" * 64}

    result = s3.run_m2_s3(tmp_path / "config", root, tmp_path / "cache", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_seed)
    assert result["status"] == "M2_S3_DIAGNOSTIC_COMPLETED"
    assert result["selected_candidate"] is False
    assert captured["selected_candidate"] is False
    assert captured["promotion"] == "NOT_APPLICABLE_S3_SINGLE_TASK_DIAGNOSTIC"
    assert set(captured["seed_checkpoints"]) == {f"{head}:seed-{seed}" for head in s3.TRIGGERED_HEADS for seed in s3.SEEDS}
