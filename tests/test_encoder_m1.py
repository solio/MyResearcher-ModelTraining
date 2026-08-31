from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest

from semantic_model import encoder_m1 as m1
from semantic_model.errors import ContractError
from semantic_model.hashes import sha256_file
from semantic_model.schema import SINGLE_LABEL_HEADS, V1_HEADS


def _valid_contract() -> dict:
    return {
        "scope_guardrails": {
            "encoder_weight_download_allowed": True,
            "tokenizer_artifact_download_allowed": True,
            "encoder_training_allowed": True,
            "new_encoder_dependency_install_allowed": True,
            "external_llm_call_allowed": False,
            "external_project_data_transfer_allowed": False,
            "new_gold_creation_allowed": False,
            "new_ood_creation_allowed": False,
            "test_based_selection_allowed": False,
            "production_inference_49054_allowed": False,
            "baseline_v0_3_5_mutation_allowed": False,
            "production_approval": False,
        },
        "current_owner_authorization": {
            "authorization_granted": True,
            "model_id": m1.MODEL_ID,
            "revision": m1.REVISION,
            "license": m1.LICENSE,
            "official_model_and_tokenizer_download_authorized": True,
            "isolated_encoder_runtime_dependency_installation_authorized": True,
            "training_policy": "MPS_FIRST_WITH_CPU_FALLBACK",
            "cpu_checkpoint_reload_and_inference_verification_required": True,
            "first_run": {
                "encoder_state": "FROZEN",
                "trainable_heads": 7,
                "seeds": 1,
                "train_rows": 1822,
                "dev_role": "EARLY_STOPPING_AND_DIAGNOSTIC_REPORTING_ONLY",
                "test_based_selection_allowed": False,
                "local_additional_disk_limit_gib": 10,
                "single_run_wall_time_limit_minutes": 120,
            },
        },
        "selected_model": {
            "official_model_id": m1.MODEL_ID,
            "required_revision": m1.REVISION,
            "license": m1.LICENSE,
            "trust_remote_code": False,
            "artifact_hashes": {
                "pytorch_model_bin_sha256": "a" * 64,
                "tokenizer_json_sha256": "b" * 64,
                "vocab_txt_sha256": "c" * 64,
            },
        },
        "milestone_execution_contract": {
            "M1_FIRST_RUN": {
                "encoder_state": "FROZEN",
                "encoder_count": 1,
                "seeds": 1,
                "fit_population": "TRAIN_1822",
                "dev_role": "EARLY_STOPPING_AND_DIAGNOSTIC_ONLY_FOR_FROZEN_CONFIGURATION",
                "production_claim_allowed": False,
            }
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True)


def _clean_git_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    entry = repo / "src/semantic_model/encoder_m1.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# tracked test entry point\n", encoding="utf-8")
    _write_json(repo / "manifests/encoder-experiment-contract-v1.json", _valid_contract())
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Encoder M1 Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "tracked M1 entry")
    return repo, entry


def _success_audit(config_path: Path, schema_path: Path) -> dict:
    return {
        "status": "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN",
        "training_allowed": True,
        "blocker_codes": [],
        "audit_id": "1" * 64,
        "config": {"sha256": sha256_file(config_path)},
        "observed": {
            "schema": {
                "path": str(schema_path),
                "schema_version": "semantic-schema-calibrated-v0.2.1",
                "sha256": sha256_file(schema_path),
            }
        },
        "validation_summary": {
            "package_manifest_id": "2" * 64,
            "reference_package": {
                "package_manifest_id": "3" * 64,
                "binding_data_package_content_address": "sha256:" + "2" * 64,
            },
        },
    }


def test_canonical_audit_blocker_prevents_runtime_download_and_fit(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    blocked = {"training_allowed": False, "blocker_codes": ["BLOCKED_INPUT"], "audit_id": "0" * 64}
    monkeypatch.setattr(m1, "run_audit", lambda _path: (blocked, 2))
    monkeypatch.setattr(m1, "_load_runtime_dependencies", lambda: calls.append("runtime") or None)
    monkeypatch.setattr(m1, "_run_m1_after_preflight", lambda *_args, **_kwargs: calls.append("fit") or {})

    with pytest.raises(ContractError, match="M1_CANONICAL_AUDIT_EXIT_NONZERO"):
        m1.run_m1(tmp_path / "blocked.yaml", tmp_path / "new-output", tmp_path / "cache")

    assert calls == []
    assert not (tmp_path / "new-output").exists()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda value: value["current_owner_authorization"].update(authorization_granted=False), "M1_OWNER_AUTHORIZATION_NOT_GRANTED"),
        (lambda value: value["current_owner_authorization"].update(revision="f" * 64), "M1_OWNER_MODEL_REVISION_MISMATCH"),
        (lambda value: value["selected_model"].update(official_model_id="other/model"), "M1_OWNER_MODEL_ID_MISMATCH"),
    ],
)
def test_missing_revoked_or_mismatched_owner_contract_fails_closed(tmp_path: Path, mutation, expected_code: str):
    missing = tmp_path / "missing.json"
    with pytest.raises(ContractError) as missing_error:
        m1.validate_owner_contract(missing)
    assert missing_error.value.code == "M1_OWNER_CONTRACT_MISSING"

    contract = _valid_contract()
    mutation(contract)
    path = tmp_path / "contract.json"
    _write_json(path, contract)
    with pytest.raises(ContractError) as error:
        m1.validate_owner_contract(path)
    assert error.value.code == expected_code


def test_dirty_or_untracked_training_source_fails_closed(tmp_path: Path):
    repo, entry = _clean_git_repo(tmp_path)
    assert m1.validate_source_provenance(repo, critical_sources=("src/semantic_model/encoder_m1.py",))["git_head"]

    entry.write_text("# dirty tracked entry point\n", encoding="utf-8")
    with pytest.raises(ContractError) as dirty:
        m1.validate_source_provenance(repo, critical_sources=("src/semantic_model/encoder_m1.py",))
    assert dirty.value.code == "M1_GIT_WORKTREE_NOT_CLEAN"

    _git(repo, "checkout", "--", "src/semantic_model/encoder_m1.py")
    (repo / "src/semantic_model/untracked_training_helper.py").write_text("# untracked\n", encoding="utf-8")
    with pytest.raises(ContractError) as untracked:
        m1.validate_source_provenance(repo, critical_sources=("src/semantic_model/encoder_m1.py",))
    assert untracked.value.code == "M1_GIT_WORKTREE_NOT_CLEAN"


def test_successful_gate_records_complete_identity_in_content_manifest(monkeypatch, tmp_path: Path):
    repo, _entry = _clean_git_repo(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    config_path = external / "config.yaml"
    schema_path = external / "schema.json"
    config_path.write_text("config: fixed\n", encoding="utf-8")
    schema_path.write_text("{\"schema\": \"fixed\"}\n", encoding="utf-8")
    audit = _success_audit(config_path, schema_path)
    monkeypatch.setattr(m1, "run_audit", lambda _path: (audit, 0))
    monkeypatch.setattr(m1, "CRITICAL_SOURCE_PATHS", ("src/semantic_model/encoder_m1.py",))

    preflight = m1.validate_m1_preflight(
        config_path,
        worktree=repo,
        contract_path=repo / "manifests/encoder-experiment-contract-v1.json",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "evidence.txt").write_text("immutable\n", encoding="utf-8")
    manifest = m1._write_content_manifest(artifact, {"provenance": preflight["identity"]})

    identity = manifest["provenance"]
    assert identity["git_head"] == _git_head(repo)
    assert identity["canonical_audit_id"] == "1" * 64
    assert identity["data_package_content_id"] == "2" * 64
    assert identity["reference_package_content_id"] == "3" * 64
    assert identity["reference_binding_data_package_content_address"] == "sha256:" + "2" * 64
    assert identity["config_sha256"] == sha256_file(config_path)
    assert identity["schema_sha256"] == sha256_file(schema_path)
    assert identity["contract_sha256"] == sha256_file(repo / "manifests/encoder-experiment-contract-v1.json")
    assert set(identity["critical_source_sha256"]) == {"src/semantic_model/encoder_m1.py"}


def _git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


class _Tokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._ids = {"code": [11, 12, 13], "name": [21, 22, 23], "body": list(range(31, 41))}

    def __call__(self, value: str, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_token_type_ids"] is False
        return {"input_ids": list(self._ids[value])}


class _BatchTorch:
    long = "long"
    float32 = "float32"

    def tensor(self, value, *, dtype, device):
        return {"value": value, "dtype": dtype, "device": device}


def _record() -> m1.M1Record:
    return m1.M1Record(
        sample_id="sample-1",
        stock_code="code",
        stock_name="name",
        model_text="body",
        label={**{head: "class-0" for head in SINGLE_LABEL_HEADS}, "reasoning_tags": ["reason-0"]},
        weights={head: (index + 1) / 10 for index, head in enumerate(V1_HEADS)},
    )


def _input_config() -> dict:
    return {
        "stock_code_token_cap": 2,
        "stock_name_token_cap": 2,
        "max_length": 12,
        "truncation": "HEAD_TAIL",
        "class_order": {**{head: ["class-0"] for head in SINGLE_LABEL_HEADS}, "reasoning_tags": ["reason-0"]},
    }


def test_input_builder_head_tail_four_special_tokens_no_token_type_ids_and_per_head_weights():
    tokenizer = _Tokenizer()
    record = _record()
    config = _input_config()

    input_ids = m1.build_input_ids(tokenizer, record, config)
    assert input_ids == [101, 11, 12, 102, 21, 22, 102, 31, 32, 39, 40, 102]
    assert input_ids.count(tokenizer.cls_token_id) == 1
    assert input_ids.count(tokenizer.sep_token_id) == 3

    batch = m1._as_batch(_BatchTorch(), tokenizer, [record], config, "cpu")
    assert set(batch) == {"input_ids", "attention_mask", "labels", "weights"}
    assert "token_type_ids" not in batch
    assert set(batch["weights"]) == set(V1_HEADS)
    for head in V1_HEADS:
        assert batch["weights"][head]["value"] == [record.weights[head]]
    assert all(call["return_token_type_ids"] is False for call in tokenizer.calls)


class _Finite:
    def all(self):
        return self

    def item(self):
        return True


class _Output:
    shape = (1, 1)


class _CpuTorch(_BatchTorch):
    def device(self, name: str):
        return name

    def no_grad(self):
        return nullcontext()

    def isfinite(self, _value):
        return _Finite()


class _Heads:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, value) -> None:
        self.loaded = value


class _CpuModel:
    def __init__(self) -> None:
        self.heads = _Heads()
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True

    def __call__(self, _input_ids, _attention_mask):
        return {head: _Output() for head in V1_HEADS}


def test_cpu_checkpoint_reload_and_seven_head_inference_smoke_minimum():
    model = _CpuModel()
    smoke = m1.cpu_reload_and_inference_smoke(
        _CpuTorch(),
        lambda: model,
        {"heads_state_dict": {"heads.weight": "saved"}},
        _Tokenizer(),
        _record(),
        _input_config(),
    )
    assert model.device == "cpu"
    assert model.evaluated is True
    assert model.heads.loaded == {"heads.weight": "saved"}
    assert smoke["device"] == "cpu"
    assert smoke["all_logits_finite"] is True
    assert set(smoke["output_shapes"]) == set(V1_HEADS)
