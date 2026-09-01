from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_reasoning_corrective as corrective
from semantic_model.errors import ContractError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "manifests/encoder-m2-experiment-contract-v1.json"
LABELS = ["NO_REASON_GIVEN", "OTHER"]


def _reasoning_metrics(value: float, *, no_reason: float | None = None) -> dict:
    no_reason = value if no_reason is None else no_reason
    return {
        "macro_f1": value,
        "micro_f1": value,
        "exact_set_accuracy": value,
        "per_label": {
            "NO_REASON_GIVEN": {"f1": no_reason, "support": 86},
            "OTHER": {"f1": value, "support": 10},
        },
    }


def _control(value: float = 0.15, *, no_reason: float | None = None) -> dict:
    return {"seed_metrics": {seed: {"dev": {"reasoning_tags": _reasoning_metrics(value, no_reason=no_reason)}} for seed in corrective.SEEDS}}


def _result(value: float = 0.17, *, device: str = "cpu", no_reason: float | None = None) -> list[dict]:
    return [{"seed": seed, "metrics": {"dev": {"reasoning_tags": _reasoning_metrics(value, no_reason=no_reason)}, "initial_reasoning_head_sha256": "a" * 64}, "resource": {"actual_device": device}} for seed in corrective.SEEDS]


def test_initial_reasoning_head_is_deterministic_for_s3_matching_seed():
    class FakeTensor:
        def __init__(self, value: bytes):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return self.value

    class FakeHead:
        def __init__(self, value: bytes):
            self.value = value

        def state_dict(self):
            return {"weight": FakeTensor(self.value)}

    class FakeModel:
        def __init__(self):
            # The order mirrors M1/S3: encoder, dropout, then V1_HEADS.
            self.heads = {head: FakeHead(f"{fake_torch.seed}:{head}".encode()) for head in corrective.V1_HEADS}

    class FakeTorch:
        seed = 0

        @classmethod
        def manual_seed(cls, seed):
            cls.seed = seed

    class FakeNumpy:
        class random:
            @staticmethod
            def seed(seed):
                FakeTorch.seed = seed

    fake_torch = FakeTorch
    fake_numpy = FakeNumpy

    def factory():
        return FakeModel()

    first = corrective.initial_reasoning_head_snapshot(fake_torch, fake_numpy, 35, factory)
    second = corrective.initial_reasoning_head_snapshot(fake_torch, fake_numpy, 35, factory)
    other_seed = corrective.initial_reasoning_head_snapshot(fake_torch, fake_numpy, 71, factory)
    assert first == second
    assert first["reasoning_head_sha256"] != other_seed["reasoning_head_sha256"]


def test_trainable_scope_is_exactly_last_block_and_reasoning_head():
    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Head:
        def __init__(self, trainable: bool):
            self._parameters = [Parameter(trainable)]

        def parameters(self):
            return self._parameters

    class Model:
        heads = {head: Head(head == corrective.TARGET_HEAD) for head in corrective.V1_HEADS}

        def named_parameters(self):
            return [("encoder.layer.2.weight", Parameter(True)), ("encoder.layer.1.weight", Parameter(False)), *[(f"heads.{head}.weight", Parameter(head == corrective.TARGET_HEAD)) for head in corrective.V1_HEADS]]

    identity = corrective.validate_trainable_parameters(Model(), object(), "encoder.layer.2")
    assert identity["last_transformer_block_prefix"] == "encoder.layer.2"

    class Bad(Model):
        def named_parameters(self):
            rows = super().named_parameters()
            rows.append(("encoder.embeddings.weight", Parameter(True)))
            return rows

    with pytest.raises(ContractError) as error:
        corrective.validate_trainable_parameters(Bad(), object(), "encoder.layer.2")
    assert error.value.code == "M2_CORRECTIVE_TRAINABLE_PARAMETER_CONTRACT_VIOLATION"


def test_aggregate_requires_three_matching_device_seeds():
    complete = _result()
    aggregate = corrective.aggregate_results(complete)
    assert aggregate["all_seeds_complete"] is True
    with pytest.raises(ContractError) as incomplete:
        corrective.aggregate_results(complete[:2])
    assert incomplete.value.code == "M2_CORRECTIVE_INCOMPLETE_SEEDS"
    complete[1]["resource"]["actual_device"] = "mps"
    with pytest.raises(ContractError) as mixed:
        corrective.aggregate_results(complete)
    assert mixed.value.code == "M2_CORRECTIVE_MIXED_DEVICE"


def test_gate_requires_macro_improvement_and_no_reason_safety():
    control = _control(0.15, no_reason=0.30)
    passed = corrective._gate(control, _result(0.16, no_reason=0.30))
    assert passed["passed"] is True
    failed = corrective._gate(control, _result(0.16, no_reason=0.25))
    assert failed["passed"] is False
    assert failed["checks"]["no_reason_mean_not_below_s1"] is False


def test_preflight_loads_only_train_dev_and_fixed_cache(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    common = contract["frozen_input_and_common_training_configuration"]
    fake_frozen = {"contract": contract, "contract_sha256": "c" * 64}
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: LABELS for head in corrective.V1_HEADS})
    monkeypatch.setattr(corrective.s1, "_contract_requirements", lambda _path: fake_frozen)
    monkeypatch.setattr(m1.ProjectConfig, "load", lambda _path: object())
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: calls.append("train_dev") or (schema, [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(corrective.s1, "validate_fixed_cache_snapshot", lambda *_args: calls.append("cache") or (tmp_path / "snapshot", {"content_address": "snapshot"}))
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda *_args: pytest.fail("canonical audit must not run"), raising=False)
    result = corrective.validate_corrective_preflight(ROOT / "configs/baseline_v0.3.5.yaml", tmp_path / "cache", worktree=ROOT, contract_path=CONTRACT)
    assert calls == ["train_dev", "cache"]
    assert len(result["train"]) == 1822 and len(result["dev"]) == 448
    assert common["truncation"] == "HEAD_TAIL"


def test_success_path_writes_reasoning_only_manifest(monkeypatch, tmp_path: Path):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schema = SimpleNamespace(schema_version="synthetic", class_order={head: LABELS for head in corrective.V1_HEADS})
    preflight = {"frozen_contract": {"contract": contract, "contract_sha256": "c" * 64}, "snapshot": tmp_path / "snapshot", "snapshot_identity": {"content_address": "snapshot"}, "schema": schema, "train": [object()] * 1822, "dev": [object()] * 448, "identity": {"schema_version": "synthetic"}}
    comparator = {"root": tmp_path / "baseline", "manifest": {"content_address": "baseline"}, "seed_metrics": {seed: {"dev": {"reasoning_tags": _reasoning_metrics(0.15, no_reason=0.30)}} for seed in corrective.SEEDS}}
    root = tmp_path / "out"
    monkeypatch.setattr(corrective, "validate_corrective_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(corrective, "_load_comparator", lambda *_args, **_kwargs: comparator)
    monkeypatch.setattr(corrective.s1, "validate_runtime_identity", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(corrective.s1, "validate_output_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(corrective.s1, "_limits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(corrective, "_config", lambda *_args, **_kwargs: {"head_dropout": 0.1, "class_order": {head: LABELS for head in corrective.V1_HEADS}, "stopping": {"minimum_delta": 0.0}, "optimizer": {"betas": [0.9, 0.999], "epsilon": 1e-8}})

    def fake_seed(**kwargs):
        seed = kwargs["seed"]
        metrics = {"dev": {"reasoning_tags": _reasoning_metrics(0.17, no_reason=0.30)}, "initial_reasoning_head_sha256": "a" * 64}
        return {"seed": seed, "metrics": metrics, "resource": {"actual_device": "cpu"}, "checkpoint_sha256": "b" * 64}

    result = corrective.run_corrective(tmp_path / "config", root, tmp_path / "cache", runtime_loader=lambda: (None, None, None, None), seed_executor=fake_seed)
    assert result["status"] == "PASSED_DIAGNOSTIC_ONLY"
    assert result["selected_candidate"] is False
    assert (root / "stage-aggregate.json").is_file()
    assert (root / "content-addressed-manifest.json").is_file()
