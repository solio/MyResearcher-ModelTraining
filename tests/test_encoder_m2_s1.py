from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m2_s1 as s1
from semantic_model.errors import ContractError
from semantic_model.hashes import content_addressed_id, sha256_file
from semantic_model.schema import V1_HEADS


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "manifests" / "encoder-m2-experiment-contract-v1.json"
SCHEMA_PATH = ROOT / "manifests" / "encoder-m2-s1-owner-authorization-receipt-schema.json"
TEMPLATE_PATH = ROOT / "manifests" / "encoder-m2-s1-owner-authorization-receipt-template.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _receipt() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    model = contract["recommended_first_execution"]["recommended_model"]
    value = {
        "receipt_schema_version": s1.RECEIPT_SCHEMA_VERSION,
        "authorization_granted": True,
        "owner_decision_id": "D-FUTURE-M2-S1-SYNTHETIC-TEST-ONLY",
        "expires_at_utc": (datetime.now(UTC) + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        "frozen_contract_commit": s1.FROZEN_CONTRACT_COMMIT,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "stage_id": s1.STAGE_ID,
        "model": {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "license": model["license"],
            "model_weight_sha256": model["model_weight_sha256"],
            "tokenizer_json_sha256": model["tokenizer_json_sha256"],
            "vocab_txt_sha256": model["vocab_txt_sha256"],
        },
        "cache_policy": {"local_files_only": True, "no_download": True},
        "execution": {
            "seeds": [35, 71, 107],
            "train_rows": 1822,
            "dev_rows": 448,
            "train_role": "TRAIN_ONLY_FIT",
            "dev_role": "EARLY_STOPPING_AND_DIAGNOSTIC_ONLY",
            "device_policy": "MPS_FIRST_CPU_FALLBACK",
            "per_run_wall_time_minutes": 120,
            "total_new_local_disk_gib": 10,
            "prohibitions": {
                "test": False,
                "anchor": False,
                "gold": False,
                "ood": False,
                "llm": False,
                "cloud_or_external_api": False,
                "production": False,
                "model_download": False,
                "dependency_install": False,
                "full_unfreeze": False,
                "s2_or_s3": False,
            },
        },
    }
    value["receipt_content_address"] = content_addressed_id(value)
    return value


def _receipt_path(tmp_path: Path, value: dict | None = None) -> Path:
    path = tmp_path / "receipt.json"
    _write_json(path, value or _receipt())
    return path


def _restamp(value: dict) -> dict:
    value.pop("receipt_content_address", None)
    value["receipt_content_address"] = content_addressed_id(value)
    return value


def test_tracked_schema_and_template_are_not_an_authorization():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

    assert schema["$id"] == s1.RECEIPT_SCHEMA_VERSION
    assert schema["properties"]["stage_id"]["const"] == s1.STAGE_ID
    assert schema["properties"]["execution"]["properties"]["seeds"]["const"] == [35, 71, 107]
    assert template["authorization_granted"] is False
    assert template["model"]["revision"] == m1.REVISION
    assert template["cache_policy"] == {"local_files_only": True, "no_download": True}


def test_missing_or_ungranted_receipt_fails_before_runtime_cache_or_output(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr(m1, "_load_runtime_dependencies", lambda: calls.append("runtime") or None)
    missing = s1.run_m2_s1(tmp_path / "config.yaml", tmp_path / "output", tmp_path / "cache", tmp_path / "missing.json", contract_path=CONTRACT_PATH)
    assert missing["blocker_codes"] == ["M2_S1_OWNER_RECEIPT_MISSING"]
    assert missing["training_invoked"] is False
    assert not (tmp_path / "output").exists()

    denied = _receipt()
    denied["authorization_granted"] = False
    _restamp(denied)
    result = s1.run_m2_s1(tmp_path / "config.yaml", tmp_path / "denied-output", tmp_path / "cache", _receipt_path(tmp_path, denied), contract_path=CONTRACT_PATH)
    assert result["blocker_codes"] == ["M2_S1_OWNER_AUTHORIZATION_NOT_GRANTED"]
    assert result["model_loaded"] is False
    assert result["cache_accessed"] is False
    assert result["output_created"] is False
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(receipt_content_address="0" * 64), "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH"),
        (lambda value: value.update(frozen_contract_commit="f" * 40), "M2_S1_FROZEN_CONTRACT_COMMIT_MISMATCH"),
        (lambda value: value.update(contract_sha256="f" * 64), "M2_S1_CONTRACT_SHA256_MISMATCH"),
        (lambda value: value.update(expires_at_utc="2000-01-01T00:00:00Z"), "M2_S1_OWNER_RECEIPT_EXPIRED"),
        (lambda value: value.update(stage_id="M2-S2-PARTIAL-LAST-ONE-SHARED-SEVEN-HEAD"), "M2_S1_STAGE_MISMATCH"),
        (lambda value: value["model"].update(model_id="other/model"), "M2_S1_MODEL_ID_MISMATCH"),
        (lambda value: value["model"].update(revision="f" * 40), "M2_S1_MODEL_REVISION_MISMATCH"),
        (lambda value: value["execution"].update(seeds=[35]), "M2_S1_SEEDS_MISMATCH"),
        (lambda value: value["execution"].update(per_run_wall_time_minutes=121), "M2_S1_RESOURCE_LIMIT_MISMATCH"),
    ],
)
def test_tampered_or_scope_mismatched_receipt_fails_closed(tmp_path: Path, mutation, expected: str):
    value = _receipt()
    mutation(value)
    if expected != "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH":
        _restamp(value)
    with pytest.raises(ContractError) as error:
        s1.validate_owner_authorization_receipt(_receipt_path(tmp_path, value), contract_path=CONTRACT_PATH)
    assert error.value.code == expected


def test_canonical_audit_blocker_after_valid_receipt_never_reaches_cache_runtime_or_fit(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda _config: (_ for _ in ()).throw(ContractError("M1_CANONICAL_AUDIT_BLOCKERS_PRESENT", "blocked")))
    monkeypatch.setattr(m1, "_validated_fixed_snapshot", lambda *_args: calls.append("cache") or None)
    monkeypatch.setattr(m1, "_load_runtime_dependencies", lambda: calls.append("runtime") or None)
    monkeypatch.setattr(s1, "_run_authorized_s1", lambda *_args, **_kwargs: calls.append("fit") or {})

    result = s1.run_m2_s1(tmp_path / "config.yaml", tmp_path / "output", tmp_path / "cache", _receipt_path(tmp_path), contract_path=CONTRACT_PATH)
    assert result["blocker_codes"] == ["M1_CANONICAL_AUDIT_BLOCKERS_PRESENT"]
    assert result["training_invoked"] is False
    assert calls == []
    assert not (tmp_path / "output").exists()


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True)


def _success_audit(config_path: Path, schema_path: Path) -> dict:
    return {
        "canonical_audit_id": "a" * 64,
        "config_sha256": sha256_file(config_path),
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "schema_sha256": sha256_file(schema_path),
        "data_package_content_id": "b" * 64,
        "reference_package_content_id": "c" * 64,
        "reference_binding_data_package_content_address": "sha256:" + "b" * 64,
    }


def test_dirty_or_untracked_m2_training_source_fails_closed(monkeypatch, tmp_path: Path):
    def clean_repo(name: str) -> Path:
        repo = tmp_path / name
        for relative in ("src/semantic_model/encoder_m1.py", "src/semantic_model/encoder_m2_s1.py"):
            candidate = repo / relative
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("# tracked test source\n", encoding="utf-8")
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "tests@example.invalid")
        _git(repo, "config", "user.name", "M2 S1 Tests")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "tracked M2 entry")
        return repo

    repo = clean_repo("dirty-repo")
    config = tmp_path / "config.yaml"
    schema = tmp_path / "schema.json"
    config.write_text("fixed: config\n", encoding="utf-8")
    schema.write_text("{}\n", encoding="utf-8")
    frozen = s1._contract_requirements(CONTRACT_PATH)
    auth = {"receipt": _receipt(), "frozen_contract": frozen}
    monkeypatch.setattr(s1, "CRITICAL_SOURCE_PATHS", ("src/semantic_model/encoder_m1.py", "src/semantic_model/encoder_m2_s1.py"))
    monkeypatch.setattr(m1, "validate_canonical_audit", lambda _config: _success_audit(config, schema))
    monkeypatch.setattr(s1, "_git_has_ancestor", lambda *_args: None)
    monkeypatch.setattr(m1, "_validated_fixed_snapshot", lambda *_args: tmp_path / "fixed-cache")

    (repo / "src/semantic_model/encoder_m2_s1.py").write_text("# dirty\n", encoding="utf-8")
    with pytest.raises(ContractError) as dirty:
        s1.validate_m2_s1_preflight(config, tmp_path / "cache", worktree=repo, receipt_authorization=auth)
    assert dirty.value.code == "M1_GIT_WORKTREE_NOT_CLEAN"

    untracked_repo = clean_repo("untracked-repo")
    (untracked_repo / "src/semantic_model/untracked_m2_helper.py").write_text("# untracked\n", encoding="utf-8")
    with pytest.raises(ContractError) as untracked:
        s1.validate_m2_s1_preflight(config, tmp_path / "cache", worktree=untracked_repo, receipt_authorization=auth)
    assert untracked.value.code == "M1_GIT_WORKTREE_NOT_CLEAN"


class _Schema:
    schema_version = "semantic-schema-calibrated-v0.2.1"
    class_order = {head: [f"{head}-class"] for head in V1_HEADS}


def _seed_result(seed: int) -> dict:
    return {
        "seed": seed,
        "metrics": {"dev": {head: {"macro_f1": 0.2 + seed / 100000} for head in V1_HEADS}},
        "checkpoint_sha256": f"{seed:064x}",
    }


def _materialize_synthetic_seed_evidence(kwargs: dict) -> None:
    """Synthetic test evidence only; it never creates a model checkpoint."""

    seed_root = kwargs["root"] / f"seed-{kwargs['seed']}"
    seed_root.mkdir()
    (seed_root / "seed-metrics.json").write_text("{}\n", encoding="utf-8")
    (seed_root / "resource-log.json").write_text("{}\n", encoding="utf-8")


def _authorized_preflight() -> dict:
    frozen = s1._contract_requirements(CONTRACT_PATH)
    return {
        "frozen_contract": frozen,
        "snapshot": Path("/synthetic/fixed-cache/official-snapshot") / m1.REVISION,
        "identity": {
            "git_head": "d" * 40,
            "critical_source_sha256": {"src/semantic_model/encoder_m2_s1.py": "e" * 64},
            "contract_sha256": frozen["contract_sha256"],
            "config_sha256": "f" * 64,
            "canonical_audit_id": "1" * 64,
            "data_package_content_id": "2" * 64,
            "reference_package_content_id": "3" * 64,
            "reference_binding_data_package_content_address": "sha256:" + "2" * 64,
            "schema_sha256": "4" * 64,
        },
    }


def _fake_runner_setup(monkeypatch):
    monkeypatch.setattr(s1.ProjectConfig, "load", lambda _path: object())
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: (_Schema(), [object()] * 1822, [object()] * 448))


def test_authorized_synthetic_runtime_runs_exact_three_seeds_and_records_complete_identity(monkeypatch, tmp_path: Path):
    _fake_runner_setup(monkeypatch)
    seen: list[int] = []

    def fake_seed_executor(**kwargs):
        seen.append(kwargs["seed"])
        assert kwargs["config"]["max_length"] == 256
        assert kwargs["config"]["batch_size"] == 16
        assert kwargs["config"]["encoder_state"] == "FROZEN"
        assert kwargs["config"]["reasoning_probability_threshold"] == 0.5
        _materialize_synthetic_seed_evidence(kwargs)
        return _seed_result(kwargs["seed"])

    result = s1._run_authorized_s1("unused.yaml", tmp_path / "new-immutable", tmp_path / "cache", preflight=_authorized_preflight(), runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=fake_seed_executor)
    assert seen == [35, 71, 107]
    assert result["aggregate_created"] is True
    assert result["selected_candidate"] is False
    assert result["allowed_output"] == "MAY_REQUEST_S2_OWNER_AUTHORIZATION"
    manifest = json.loads((tmp_path / "new-immutable" / "content-addressed-manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_address"] == content_addressed_id(manifest, omit_keys={"content_address"})
    assert manifest["provenance"]["git_head"] == "d" * 40
    assert manifest["provenance"]["contract_sha256"] == sha256_file(CONTRACT_PATH)
    assert manifest["provenance"]["canonical_audit_id"] == "1" * 64
    assert manifest["provenance"]["data_package_content_id"] == "2" * 64
    assert manifest["provenance"]["reference_package_content_id"] == "3" * 64
    assert manifest["provenance"]["schema_sha256"] == "4" * 64


@pytest.mark.parametrize(
    "code",
    [
        "M2_S1_NONFINITE_LOSS",
        "M2_S1_WALL_TIME_LIMIT_EXCEEDED",
        "M2_S1_DISK_LIMIT_EXCEEDED",
        "M2_S1_CHECKPOINT_MISSING",
        "M2_S1_CPU_RELOAD_SMOKE_FAILED",
    ],
)
def test_seed_failure_never_creates_aggregate_or_selected_candidate(monkeypatch, tmp_path: Path, code: str):
    _fake_runner_setup(monkeypatch)

    def fail_on_second_seed(**kwargs):
        if kwargs["seed"] == 71:
            raise ContractError(code, "synthetic failure")
        _materialize_synthetic_seed_evidence(kwargs)
        return _seed_result(kwargs["seed"])

    output = tmp_path / code
    result = s1._run_authorized_s1("unused.yaml", output, tmp_path / "cache", preflight=_authorized_preflight(), runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=fail_on_second_seed)
    assert result["status"] == "S1_REJECTED_OR_BLOCKED_EVIDENCE"
    assert result["blocker_codes"] == [code]
    assert result["aggregate_created"] is False
    assert result["selected_candidate"] is False
    assert not (output / "stage-aggregate.json").exists()


def test_incomplete_or_out_of_order_seed_set_cannot_be_aggregated():
    with pytest.raises(ContractError) as incomplete:
        s1.aggregate_s1_seed_results([_seed_result(35), _seed_result(71)])
    assert incomplete.value.code == "M2_S1_INCOMPLETE_SEEDS"
    with pytest.raises(ContractError) as unordered:
        s1.aggregate_s1_seed_results([_seed_result(71), _seed_result(35), _seed_result(107)])
    assert unordered.value.code == "M2_S1_INCOMPLETE_SEEDS"


def test_shared_m1_input_metric_weight_and_cpu_smoke_interfaces_are_the_only_runner_interfaces():
    assert s1.m1.build_input_ids is m1.build_input_ids
    assert s1.m1._as_batch is m1._as_batch
    assert s1.m1.diagnostic_metrics is m1.diagnostic_metrics
    assert s1.m1.cpu_reload_and_inference_smoke is m1.cpu_reload_and_inference_smoke
    assert set(s1.V1_HEADS) == set(V1_HEADS)
    assert "torch" not in s1.__dict__
    assert "transformers" not in s1.__dict__
    assert "snapshot_download" not in s1.__dict__
    assert "torch" not in sys.modules or sys.modules["torch"] is not None
