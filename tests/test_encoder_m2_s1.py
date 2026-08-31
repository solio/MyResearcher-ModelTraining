from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _restamp_receipt(receipt: dict) -> dict:
    receipt["receipt_content_address"] = content_addressed_id(receipt, omit_keys={"receipt_content_address"})
    return receipt


def _restamp_record(record: dict) -> dict:
    record["record_content_address"] = content_addressed_id(record, omit_keys={"record_content_address"})
    return record


def _valid_receipt(output_dir: Path) -> dict:
    frozen = s1._contract_requirements(CONTRACT_PATH)
    receipt = {
        "receipt_schema_version": s1.RECEIPT_SCHEMA_VERSION,
        "authorization_granted": True,
        "owner_decision_id": "D-SYNTHETIC-M2-S1-TEST",
        "expires_at_utc": (datetime.now(UTC) + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        "frozen_contract_commit": s1.FROZEN_CONTRACT_COMMIT,
        "contract_sha256": frozen["contract_sha256"],
        "stage_id": s1.STAGE_ID,
        "unified_repository": {"branch": "feat/m2-s1-runner", "commit": "a" * 40, "remote_branch": "origin/feat/m2-s1-runner"},
        "canonical_config_sha256": "92436eff13c4c67a6dec8a3d645c4e40a4ce8c927d6cb522720709d980cff617",
        "model": {
            "model_id": m1.MODEL_ID,
            "revision": m1.REVISION,
            "license": m1.LICENSE,
            "model_weight_sha256": frozen["model"]["model_weight_sha256"],
            "tokenizer_json_sha256": frozen["model"]["tokenizer_json_sha256"],
            "vocab_txt_sha256": frozen["model"]["vocab_txt_sha256"],
        },
        "cache_snapshot_content_address": frozen["snapshot"]["content_address"],
        "runtime_environment_sha256": frozen["runtime_identity"]["environment_sha256"],
        "cache_policy": {"local_files_only": True, "no_download": True},
        "execution": {
            "seeds": [35, 71, 107], "train_rows": 1822, "dev_rows": 448,
            "train_role": "TRAIN_ONLY_FIT", "dev_role": "EARLY_STOPPING_AND_DIAGNOSTIC_ONLY",
            "device_policy": "MPS_FIRST_CPU_FALLBACK", "per_run_wall_time_minutes": 120,
            "total_new_local_disk_gib": 10,
            "prohibitions": {"test": False, "anchor": False, "gold": False, "ood": False, "llm": False, "cloud_or_external_api": False, "production": False, "model_download": False, "dependency_install": False, "full_unfreeze": False, "s2_or_s3": False},
        },
        "output_dir": str(output_dir.resolve()),
    }
    return _restamp_receipt(receipt)


def _valid_record(receipt: dict) -> dict:
    return _restamp_record({
        "record_schema_version": s1.OWNER_DECISION_RECORD_SCHEMA_VERSION,
        "authorization_granted": True,
        "owner_decision_id": receipt["owner_decision_id"],
        "authorized_receipt_content_addresses": [receipt["receipt_content_address"]],
    })


def _tracked_owner_tree(tmp_path: Path, receipt: dict | None = None, record: dict | None = None) -> tuple[Path, dict, dict]:
    """Create only a temporary tracked owner-decision fixture repository."""

    repo = tmp_path / "owner-tree"
    owner = repo / "manifests/owner-decisions"
    owner.mkdir(parents=True)
    for name in ("m2-s1-owner-authorization-receipt-schema.json", "m2-s1-owner-decision-record-schema.json"):
        shutil.copy2(ROOT / "manifests/owner-decisions" / name, owner / name)
    receipt = receipt or _valid_receipt(tmp_path / "new-output")
    record = record or _valid_record(receipt)
    _write_json(owner / "m2-s1-owner-authorization-receipt.json", receipt)
    _write_json(owner / "m2-s1-owner-decision-record.json", record)
    _git(repo, "init", "-q", "-b", "feat/m2-s1-runner")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "M2 S1 Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "tracked owner files")
    return repo, receipt, record


def test_current_tracked_template_and_record_remain_ungranted():
    receipt = json.loads((ROOT / s1.OWNER_RECEIPT_RELATIVE_PATH).read_text(encoding="utf-8"))
    record = json.loads((ROOT / s1.OWNER_DECISION_RECORD_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert receipt["authorization_granted"] is False
    assert record["authorization_granted"] is False
    assert record["authorized_receipt_content_addresses"] == []
    assert receipt["receipt_content_address"] == content_addressed_id(receipt, omit_keys={"receipt_content_address"})
    assert record["record_content_address"] == content_addressed_id(record, omit_keys={"record_content_address"})


def test_arbitrary_self_hashed_file_is_not_an_authorization_and_current_runner_touches_no_cache_or_output(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    _write_json(tmp_path / "forged.json", _valid_receipt(tmp_path / "forged-output"))
    denied = _valid_receipt(tmp_path / "denied-output")
    denied["authorization_granted"] = False
    _restamp_receipt(denied)
    denied_record = _restamp_record({"record_schema_version": s1.OWNER_DECISION_RECORD_SCHEMA_VERSION, "authorization_granted": False, "owner_decision_id": denied["owner_decision_id"], "authorized_receipt_content_addresses": []})
    repo, _receipt, _record = _tracked_owner_tree(tmp_path, receipt=denied, record=denied_record)
    monkeypatch.setattr(m1, "_load_runtime_dependencies", lambda: calls.append("runtime") or None)
    monkeypatch.setattr(m1, "_validated_fixed_snapshot", lambda *_args: calls.append("cache") or None)

    result = s1.run_m2_s1(tmp_path / "config.yaml", tmp_path / "output", tmp_path / "cache", worktree=repo, contract_path=CONTRACT_PATH)
    assert result["blocker_codes"] == ["M2_S1_OWNER_AUTHORIZATION_NOT_GRANTED"]
    assert result["training_invoked"] is False
    assert result["model_loaded"] is False
    assert result["cache_accessed"] is False
    assert result["output_created"] is False
    assert not (tmp_path / "output").exists()
    assert calls == []


def test_receipt_extra_field_and_missing_fixed_tracked_path_are_rejected(tmp_path: Path):
    receipt = _valid_receipt(tmp_path / "out")
    receipt["forged_extra_field"] = True
    _restamp_receipt(receipt)
    repo, _receipt, _record = _tracked_owner_tree(tmp_path, receipt=receipt, record=_valid_record(receipt))
    with pytest.raises(ContractError) as extra:
        s1.validate_owner_authorization_receipt(worktree=repo, contract_path=CONTRACT_PATH)
    assert extra.value.code == "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"

    missing = tmp_path / "missing-owner-tree"
    missing.mkdir()
    _git(missing, "init", "-q", "-b", "feat/m2-s1-runner")
    with pytest.raises(ContractError) as absent:
        s1.validate_owner_authorization_receipt(worktree=missing, contract_path=CONTRACT_PATH)
    assert absent.value.code == "M2_S1_OWNER_RECEIPT_PATH_INVALID"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(receipt_content_address="0" * 64), "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH"),
        (lambda value: value.update(frozen_contract_commit="f" * 40), "M2_S1_FROZEN_CONTRACT_COMMIT_MISMATCH"),
        (lambda value: value.update(contract_sha256="f" * 64), "M2_S1_CONTRACT_SHA256_MISMATCH"),
        (lambda value: value.update(expires_at_utc="2000-01-01T00:00:00Z"), "M2_S1_OWNER_RECEIPT_EXPIRED"),
        (lambda value: value["unified_repository"].update(branch="other"), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
        (lambda value: value.update(canonical_config_sha256="f" * 64), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
        (lambda value: value.update(cache_snapshot_content_address="f" * 64), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
        (lambda value: value.update(runtime_environment_sha256="f" * 64), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
        (lambda value: value["execution"].update(seeds=[35]), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
        (lambda value: value["execution"].update(per_run_wall_time_minutes=121), "M2_S1_OWNER_RECEIPT_SCHEMA_INVALID"),
    ],
)
def test_tracked_allowlisted_receipt_scope_tampering_fails_closed(tmp_path: Path, mutation, expected: str):
    receipt = _valid_receipt(tmp_path / "out")
    mutation(receipt)
    if expected != "M2_S1_OWNER_RECEIPT_CONTENT_ADDRESS_MISMATCH":
        _restamp_receipt(receipt)
    repo, _receipt, _record = _tracked_owner_tree(tmp_path, receipt=receipt, record=_valid_record(receipt))
    with pytest.raises(ContractError) as error:
        s1.validate_owner_authorization_receipt(worktree=repo, contract_path=CONTRACT_PATH)
    assert error.value.code == expected


def _repository_authorization(commit: str) -> dict:
    frozen = s1._contract_requirements(CONTRACT_PATH)
    return {"receipt": {"unified_repository": {"branch": "feat/m2-s1-runner", "commit": commit, "remote_branch": "origin/feat/m2-s1-runner"}}, "frozen_contract": frozen}


def _consolidated_repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    repo = tmp_path / "consolidated"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feat/m2-s1-runner")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "M2 S1 Tests")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "single unified branch")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-qu", "origin", "feat/m2-s1-runner")
    return repo


def test_d026_repository_gate_accepts_only_the_single_synced_unified_fixture_and_fails_closed_on_drift(monkeypatch, tmp_path: Path):
    repo = _consolidated_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    result = s1.validate_repository_consolidation(repo, _repository_authorization(head))
    assert result["status"] == "D026_REPOSITORY_CONSOLIDATED"
    assert result["local_branch_count"] == result["matching_remote_branch_count"] == result["worktree_count"] == 1

    bad = _repository_authorization("f" * 40)
    with pytest.raises(ContractError) as mismatch:
        s1.validate_repository_consolidation(repo, bad)
    assert mismatch.value.code == "BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED"
    assert mismatch.value.details["cause"] == "M2_S1_REPOSITORY_HEAD_MISMATCH"

    monkeypatch.setattr(s1, "_git_output", lambda _root, args, **_kwargs: "M changed" if args[:2] == ["status", "--porcelain=v1"] else _git(repo, *args))
    with pytest.raises(ContractError) as dirty:
        s1.validate_repository_consolidation(repo, _repository_authorization(head))
    assert dirty.value.code == "BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED"
    assert dirty.value.details["cause"] == "M2_S1_REPOSITORY_WORKTREE_NOT_CLEAN"


@pytest.mark.parametrize(
    ("arguments_prefix", "replacement", "cause"),
    [
        (("branch", "--show-current"), "wrong-branch", "M2_S1_REPOSITORY_BRANCH_MISMATCH"),
        (("for-each-ref", "--format=%(refname:short)", "refs/heads"), "feat/m2-s1-runner\nextra", "M2_S1_REPOSITORY_LOCAL_BRANCHES_NOT_CONSOLIDATED"),
        (("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"), "origin/feat/m2-s1-runner\norigin/extra", "M2_S1_REPOSITORY_REMOTE_BRANCHES_NOT_CONSOLIDATED"),
        (("worktree", "list", "--porcelain"), "worktree /synthetic/one\nworktree /synthetic/two", "M2_S1_REPOSITORY_WORKTREES_NOT_CONSOLIDATED"),
        (("rev-list", "--left-right"), "0 1", "M2_S1_REPOSITORY_UPSTREAM_NOT_SYNCED"),
    ],
)
def test_d026_branch_remote_worktree_and_upstream_drift_all_fail_closed(monkeypatch, tmp_path: Path, arguments_prefix, replacement: str, cause: str):
    repo = _consolidated_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    original = s1._git_output

    def synthetic_output(root, arguments, **kwargs):
        if tuple(arguments[:len(arguments_prefix)]) == arguments_prefix:
            return replacement
        return original(root, arguments, **kwargs)

    monkeypatch.setattr(s1, "_git_output", synthetic_output)
    with pytest.raises(ContractError) as error:
        s1.validate_repository_consolidation(repo, _repository_authorization(head))
    assert error.value.code == "BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED"
    assert error.value.details["cause"] == cause


def test_d026_failure_after_authorization_prevents_canonical_cache_runtime_and_output(monkeypatch, tmp_path: Path):
    frozen = s1._contract_requirements(CONTRACT_PATH)
    authorization = {"receipt": _valid_receipt(tmp_path / "out"), "owner_decision_record": _valid_record(_valid_receipt(tmp_path / "other")), "frozen_contract": frozen}
    calls: list[str] = []
    monkeypatch.setattr(s1, "validate_owner_authorization_receipt", lambda **_kwargs: authorization)
    monkeypatch.setattr(s1, "validate_repository_consolidation", lambda *_args: (_ for _ in ()).throw(ContractError("BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED", "synthetic D-026")))
    monkeypatch.setattr(s1, "validate_m2_s1_preflight", lambda *_args, **_kwargs: calls.append("canonical") or {})
    monkeypatch.setattr(m1, "_load_runtime_dependencies", lambda: calls.append("runtime") or None)

    result = s1.run_m2_s1(tmp_path / "config.yaml", tmp_path / "out", tmp_path / "cache")
    assert result["blocker_codes"] == ["BLOCKED_PRE_TRAINING_REPOSITORY_NOT_CONSOLIDATED"]
    assert result["training_invoked"] is False
    assert result["cache_accessed"] is False
    assert result["output_created"] is False
    assert calls == []


def _synthetic_snapshot(tmp_path: Path) -> tuple[Path, dict]:
    cache = tmp_path / "cache"
    snapshot = cache / "official-snapshot" / m1.REVISION
    snapshot.mkdir(parents=True)
    rows = []
    for index, name in enumerate(("README.md", "added_tokens.json", "config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"), 1):
        item = snapshot / name
        item.write_bytes(bytes([index]) * index)
        rows.append({"path": name, "bytes": index, "sha256": sha256_file(item)})
    frozen = deepcopy(s1._contract_requirements(CONTRACT_PATH))
    frozen["snapshot"] = {"required_relative_directory": f"official-snapshot/{m1.REVISION}", "files": rows, "content_address": content_addressed_id({"files": rows})}
    return cache, frozen


@pytest.mark.parametrize("tampered_index", range(8))
def test_all_eight_cache_snapshot_files_are_verified_with_synthetic_fixtures(tmp_path: Path, tampered_index: int):
    cache, frozen = _synthetic_snapshot(tmp_path)
    snapshot, identity = s1.validate_fixed_cache_snapshot(cache, frozen)
    assert snapshot.name == m1.REVISION
    assert identity["content_address"] == frozen["snapshot"]["content_address"]
    target = snapshot / frozen["snapshot"]["files"][tampered_index]["path"]
    target.write_bytes(b"tampered")
    with pytest.raises(ContractError) as error:
        s1.validate_fixed_cache_snapshot(cache, frozen)
    assert error.value.code == "M2_S1_CACHE_SNAPSHOT_FILE_MISMATCH"


def test_runtime_identity_mismatch_fails_before_model_load(monkeypatch):
    frozen = s1._contract_requirements(CONTRACT_PATH)
    monkeypatch.setattr(s1, "observe_runtime_identity", lambda _runtime: {"python": {"implementation": "CPython", "version": "3.12.13"}, "packages": {**frozen["runtime_identity"]["packages"], "torch": "9.9.9"}, "torch_runtime_version": "9.9.9", "numpy_runtime_version": "2.5.2"})
    with pytest.raises(ContractError) as error:
        s1.validate_runtime_identity((object(), object(), object(), object()), frozen)
    assert error.value.code == "M2_S1_RUNTIME_IDENTITY_MISMATCH"


def test_output_dir_must_be_unique_receipt_bound_and_outside_protected_roots(tmp_path: Path):
    frozen = s1._contract_requirements(CONTRACT_PATH)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    receipt = _valid_receipt(tmp_path / "authorized")
    assert s1.validate_authorized_output_dir(receipt["output_dir"], cache, worktree=worktree, receipt=receipt, frozen=frozen) == Path(receipt["output_dir"])
    with pytest.raises(ContractError) as replay:
        s1.validate_authorized_output_dir(tmp_path / "other", cache, worktree=worktree, receipt=receipt, frozen=frozen)
    assert replay.value.code == "M2_S1_OUTPUT_DIR_NOT_AUTHORIZED"
    protected = worktree / ".encoder-artifacts" / "m1-history"
    receipt["output_dir"] = str(protected)
    with pytest.raises(ContractError) as protected_error:
        s1.validate_authorized_output_dir(protected, cache, worktree=worktree, receipt=receipt, frozen=frozen)
    assert protected_error.value.code == "M2_S1_OUTPUT_PROTECTED_PATH"
    receipt["output_dir"] = str(tmp_path / "exists")
    Path(receipt["output_dir"]).mkdir()
    with pytest.raises(ContractError) as exists:
        s1.validate_authorized_output_dir(receipt["output_dir"], cache, worktree=worktree, receipt=receipt, frozen=frozen)
    assert exists.value.code == "M2_S1_OUTPUT_REPLAY_OR_EXISTS"


class _Schema:
    schema_version = "semantic-schema-calibrated-v0.2.1"
    class_order = {head: [f"{head}-class"] for head in V1_HEADS}


def _seed_result(seed: int, *, device: str = "mps") -> dict:
    return {"seed": seed, "metrics": {"dev": {head: {"macro_f1": 0.2 + seed / 100000} for head in V1_HEADS}}, "resource": {"actual_device": device}, "checkpoint_sha256": f"{seed:064x}", "critical_boundary_report_sha256": f"{seed + 1000:064x}"}


def _materialize_synthetic_seed_evidence(kwargs: dict) -> None:
    seed_root = kwargs["root"] / f"seed-{kwargs['seed']}"
    seed_root.mkdir()
    for name in ("seed-metrics.json", "resource-log.json", "critical-boundary-report.json"):
        (seed_root / name).write_text("{}\n", encoding="utf-8")


def _authorized_preflight(tmp_path: Path) -> dict:
    frozen = s1._contract_requirements(CONTRACT_PATH)
    receipt = _valid_receipt(tmp_path / "new-immutable")
    record = _valid_record(receipt)
    return {
        "frozen_contract": frozen,
        "snapshot": tmp_path / "synthetic-snapshot",
        "snapshot_identity": frozen["snapshot"],
        "repository_consolidation": {"branch": "feat/m2-s1-runner", "head": "d" * 40, "worktree": str(tmp_path / "safe-worktree")},
        "owner_receipt": receipt,
        "owner_decision_record": record,
        "identity": {"git_head": "d" * 40, "critical_source_sha256": {"src/semantic_model/encoder_m2_s1.py": "e" * 64}, "contract_sha256": frozen["contract_sha256"], "config_sha256": "f" * 64, "canonical_audit_id": "1" * 64, "data_package_content_id": "2" * 64, "reference_package_content_id": "3" * 64, "reference_binding_data_package_content_address": "sha256:" + "2" * 64, "schema_sha256": "4" * 64},
    }


def _fake_runner_setup(monkeypatch):
    monkeypatch.setattr(s1.ProjectConfig, "load", lambda _path: object())
    monkeypatch.setattr(m1, "load_m1_partitions", lambda _config: (_Schema(), [object()] * 1822, [object()] * 448))
    monkeypatch.setattr(s1, "validate_runtime_identity", lambda _runtime, frozen: {"environment_sha256": frozen["runtime_identity"]["environment_sha256"], "observed": "synthetic", "device_policy": "MPS_FIRST_CPU_FALLBACK"})


def test_success_manifest_has_required_governance_cache_runtime_and_critical_boundary_fields(monkeypatch, tmp_path: Path):
    _fake_runner_setup(monkeypatch)
    preflight = _authorized_preflight(tmp_path)
    Path(preflight["repository_consolidation"]["worktree"]).mkdir()

    def execute(**kwargs):
        _materialize_synthetic_seed_evidence(kwargs)
        return _seed_result(kwargs["seed"])

    result = s1._run_authorized_s1("unused.yaml", preflight["owner_receipt"]["output_dir"], tmp_path / "cache", preflight=preflight, runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=execute)
    assert result["aggregate_created"] is True
    assert result["selected_candidate"] is False
    manifest = json.loads((Path(result["output_dir"]) / "content-addressed-manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_address"] == content_addressed_id(manifest, omit_keys={"content_address"})
    assert set(("m2_lineage", "m1_controls", "unified_branch_and_commit", "repository_consolidation_receipt", "complete_cache_snapshot", "runtime_identity", "device_identity", "owner_receipt", "critical_boundary_report")) <= set(manifest)


@pytest.mark.parametrize("failure", ["M2_S1_NONFINITE_LOSS", "M2_S1_WALL_TIME_LIMIT_EXCEEDED", "M2_S1_DISK_LIMIT_EXCEEDED", "M2_S1_CHECKPOINT_MISSING", "M2_S1_CPU_RELOAD_SMOKE_FAILED", "runtime"])
def test_contract_or_runtime_failure_after_output_creation_writes_content_addressed_rejected_manifest(monkeypatch, tmp_path: Path, failure: str):
    _fake_runner_setup(monkeypatch)
    preflight = _authorized_preflight(tmp_path)
    Path(preflight["repository_consolidation"]["worktree"]).mkdir()

    def execute(**kwargs):
        _materialize_synthetic_seed_evidence(kwargs)
        if failure == "runtime":
            raise RuntimeError("synthetic OOM")
        raise ContractError(failure, "synthetic failure")

    result = s1._run_authorized_s1("unused.yaml", preflight["owner_receipt"]["output_dir"], tmp_path / "cache", preflight=preflight, runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=execute)
    assert result["status"] == "S1_REJECTED_OR_BLOCKED_EVIDENCE"
    assert result["aggregate_created"] is False
    assert result["selected_candidate"] is False
    assert (Path(preflight["owner_receipt"]["output_dir"]) / "content-addressed-manifest.json").is_file()
    assert result["blocker_codes"] == (["M2_S1_RUNTIME_EXCEPTION"] if failure == "runtime" else [failure])


def test_mixed_device_seeds_create_only_device_stratified_rejected_evidence(monkeypatch, tmp_path: Path):
    _fake_runner_setup(monkeypatch)
    preflight = _authorized_preflight(tmp_path)
    Path(preflight["repository_consolidation"]["worktree"]).mkdir()

    def execute(**kwargs):
        _materialize_synthetic_seed_evidence(kwargs)
        return _seed_result(kwargs["seed"], device="mps" if kwargs["seed"] != 107 else "cpu")

    result = s1._run_authorized_s1("unused.yaml", preflight["owner_receipt"]["output_dir"], tmp_path / "cache", preflight=preflight, runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=execute)
    assert result["blocker_codes"] == ["M2_S1_MIXED_DEVICE"]
    assert result["aggregate_created"] is False
    assert result["selected_candidate"] is False
    assert result["device_stratified_rejected_evidence"] == {"35": "mps", "71": "mps", "107": "cpu"}
    assert not (Path(preflight["owner_receipt"]["output_dir"]) / "stage-aggregate.json").exists()


@pytest.mark.parametrize("final_code", ["M2_S1_WALL_TIME_LIMIT_EXCEEDED", "M2_S1_DISK_LIMIT_EXCEEDED"])
def test_final_wall_or_disk_check_cannot_finish_a_successful_manifest(monkeypatch, tmp_path: Path, final_code: str):
    _fake_runner_setup(monkeypatch)
    preflight = _authorized_preflight(tmp_path)
    Path(preflight["repository_consolidation"]["worktree"]).mkdir()
    original = s1._enforce_resource_limits

    def final_gate(start, root, *, phase, seed=None):
        if phase == "after_final_manifest":
            raise ContractError(final_code, "synthetic final resource limit")
        return original(start, root, phase=phase, seed=seed)

    monkeypatch.setattr(s1, "_enforce_resource_limits", final_gate)

    def execute(**kwargs):
        _materialize_synthetic_seed_evidence(kwargs)
        return _seed_result(kwargs["seed"])

    result = s1._run_authorized_s1("unused.yaml", preflight["owner_receipt"]["output_dir"], tmp_path / "cache", preflight=preflight, runtime_loader=lambda: (object(), object(), object(), object()), seed_executor=execute)
    assert result["status"] == "S1_REJECTED_OR_BLOCKED_EVIDENCE"
    assert result["aggregate_created"] is False
    assert result["blocker_codes"] == [final_code]
    assert (Path(preflight["owner_receipt"]["output_dir"]) / "content-addressed-manifest.json").is_file()
    invalidated = json.loads((Path(preflight["owner_receipt"]["output_dir"]) / "stage-aggregate.json").read_text(encoding="utf-8"))
    assert invalidated["status"] == "INVALIDATED_NOT_AN_M2_S1_AGGREGATE"


def test_incomplete_seeds_and_shared_m1_interfaces_remain_fail_closed():
    with pytest.raises(ContractError) as incomplete:
        s1.aggregate_s1_seed_results([_seed_result(35), _seed_result(71)])
    assert incomplete.value.code == "M2_S1_INCOMPLETE_SEEDS"
    assert s1.m1.build_input_ids is m1.build_input_ids
    assert s1.m1._as_batch is m1._as_batch
    assert s1.m1.diagnostic_metrics is m1.diagnostic_metrics
    assert s1.m1.cpu_reload_and_inference_smoke is m1.cpu_reload_and_inference_smoke
