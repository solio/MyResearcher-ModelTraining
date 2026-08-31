"""Direct-owner M2-S1 frozen shared-seven-head runner, Train/Dev only."""
from __future__ import annotations
import argparse, json, math, os, platform, random, statistics, sys, time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from . import encoder_m1 as m1
from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .schema import V1_HEADS

STAGE_ID = "M2-S1-FROZEN-SHARED-SEVEN-HEAD-CONTROL"
RUN_SCHEMA_VERSION = "myresearcher.encoder-m2-s1-control-run.v2"
CONTRACT_RELATIVE_PATH = Path("manifests/encoder-m2-experiment-contract-v1.json")
SEEDS = (35, 71, 107)
MAX_WALL_TIME_SECONDS = 120 * 60
MAX_NEW_DISK_GIB = 10

def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok: raise ContractError(code, message, **details)
def _mapping(value: Any, code: str, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object"); return value
def _read_json(path: str | Path, *, missing_code: str, invalid_code: str) -> dict[str, Any]:
    try: value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc: raise ContractError(missing_code, "required JSON file is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc: raise ContractError(invalid_code, "required JSON file is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), invalid_code, "JSON root must be an object"); return value
def _sha(value: Any, code: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), code, "expected lowercase SHA-256"); return value

def _contract_requirements(contract_path: str | Path) -> dict[str, Any]:
    path = Path(contract_path).resolve(); contract = _read_json(path, missing_code="M2_S1_CONTRACT_MISSING", invalid_code="M2_S1_CONTRACT_INVALID")
    _require(contract.get("manifest_schema_version") == "myresearcher.encoder-m2-experiment-contract.v1", "M2_S1_CONTRACT_INVALID", "requires frozen M2 contract")
    direct = _mapping(contract.get("m2_direct_owner_execution"), "M2_S1_CONTRACT_INVALID", "m2_direct_owner_execution")
    _require(direct.get("training_allowed") is True and direct.get("scope") == "M2_S1_TRAIN_DEV_ONLY", "M2_S1_DIRECT_OWNER_SCOPE_INVALID", "direct owner instruction must be M2-S1 Train/Dev only")
    model = _mapping(_mapping(contract.get("recommended_first_execution"), "M2_S1_CONTRACT_INVALID", "recommended execution").get("recommended_model"), "M2_S1_CONTRACT_INVALID", "recommended model")
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S1_CONTRACT_INVALID", "training configuration")
    roles = _mapping(contract.get("data_role_and_seal"), "M2_S1_CONTRACT_INVALID", "data roles")
    _require(_mapping(roles.get("train"), "M2_S1_CONTRACT_INVALID", "train").get("rows") == 1822 and _mapping(roles.get("dev"), "M2_S1_CONTRACT_INVALID", "dev").get("rows") == 448, "M2_S1_DATA_CONTRACT_INVALID", "requires Train 1822 and Dev 448")
    stages = contract.get("minimal_experiment_gradient"); stage = next((v for v in stages if isinstance(v, Mapping) and v.get("stage_id") == STAGE_ID), None) if isinstance(stages, list) else None
    _require(stage is not None and stage.get("encoder_state") == "FROZEN" and stage.get("architecture") == "ONE_SHARED_ENCODER_WITH_SEVEN_HEADS" and stage.get("seeds") == list(SEEDS), "M2_S1_STAGE_CONTRACT_INVALID", "requires frozen three-seed shared seven-head stage")
    _require(model.get("model_id") == m1.MODEL_ID and model.get("revision") == m1.REVISION and model.get("license") == m1.LICENSE and model.get("trust_remote_code") is False and model.get("new_download_allowed_by_this_contract") is False, "M2_S1_MODEL_CONTRACT_INVALID", "requires fixed local RBT3")
    _require(common.get("max_length") == 256 and common.get("batch_size") == 16 and common.get("truncation") == "HEAD_TAIL" and common.get("token_type_ids") == "NOT_EMITTED", "M2_S1_TRAINING_CONTRACT_INVALID", "fixed input configuration changed")
    early, optimizer, gradient = _mapping(common.get("early_stopping"), "M2_S1_CONTRACT_INVALID", "early stopping"), _mapping(common.get("optimizer"), "M2_S1_CONTRACT_INVALID", "optimizer"), _mapping(common.get("gradient_controls"), "M2_S1_CONTRACT_INVALID", "gradient controls")
    _require(early.get("max_epochs") == 12 and early.get("patience_epochs") == 3 and optimizer.get("name") == "AdamW" and optimizer.get("head_learning_rate") == .0005 and optimizer.get("weight_decay") == .01 and gradient.get("gradient_clipping_max_norm") == 1.0, "M2_S1_OPTIMIZATION_CONTRACT_INVALID", "fixed optimization changed")
    technical = _mapping(contract.get("m2_s1_train_dev_technical_preflight"), "M2_S1_CONTRACT_INVALID", "technical preflight")
    _require(technical.get("forbidden_data_roles") == ["Test", "Anchor", "Gold", "OOD", "reference_predictions"], "M2_S1_TECHNICAL_PREFLIGHT_CONTRACT_INVALID", "technical preflight must exclude non-Train/Dev data")
    snapshot = _mapping(technical.get("fixed_local_cache_snapshot"), "M2_S1_CONTRACT_INVALID", "cache snapshot"); files = snapshot.get("files")
    _require(isinstance(files, list) and len(files) == 8, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "eight cache files required")
    for row in files:
        item = _mapping(row, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "snapshot file"); _require(isinstance(item.get("path"), str) and isinstance(item.get("bytes"), int), "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "bad cache file entry"); _sha(item.get("sha256"), "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID")
    runtime = _mapping(technical.get("frozen_runtime_identity_from_accepted_m1_environment"), "M2_S1_CONTRACT_INVALID", "runtime identity")
    controls = _mapping(technical.get("accepted_m1_control_evidence"), "M2_S1_CONTRACT_INVALID", "M1 controls")
    _require(controls.get("artifact_content_address") == "b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58", "M2_S1_M1_CONTROL_CONTRACT_INVALID", "wrong accepted M1 control")
    return {"contract": contract, "contract_path": path, "contract_sha256": sha256_file(path), "snapshot": {"required_relative_directory": snapshot.get("required_relative_directory"), "files": files, "content_address": content_addressed_id({"files": files})}, "runtime_identity": runtime, "m1_controls": controls, "protected_output_roots": technical.get("protected_output_roots", [])}

def validate_fixed_cache_snapshot(cache_dir: Path, frozen: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    relative = Path(str(frozen["snapshot"]["required_relative_directory"])); _require(not relative.is_absolute() and ".." not in relative.parts, "M2_S1_CACHE_SNAPSHOT_CONTRACT_INVALID", "cache path must be relative")
    snapshot = (cache_dir / relative).resolve(); _require(snapshot.is_dir() and snapshot.name == m1.REVISION, "M2_S1_FIXED_REVISION_CACHE_MISSING", "fixed RBT3 cache missing")
    rows = []
    for expected in frozen["snapshot"]["files"]:
        name = Path(str(expected["path"])); candidate = snapshot / name; _require(candidate.is_file(), "M2_S1_CACHE_SNAPSHOT_FILE_MISSING", "cache file missing", path=name.as_posix())
        observed = {"path": name.as_posix(), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}; _require(observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"], "M2_S1_CACHE_SNAPSHOT_FILE_MISMATCH", "cache file differs", path=name.as_posix()); rows.append(observed)
    identity = {"required_relative_directory": relative.as_posix(), "files": rows, "content_address": content_addressed_id({"files": rows})}; _require(identity["content_address"] == frozen["snapshot"]["content_address"], "M2_S1_CACHE_SNAPSHOT_IDENTITY_MISMATCH", "cache identity differs"); return snapshot, identity

def validate_runtime_identity(runtime: tuple[Any, Any, Any, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    np, torch, _model, _tokenizer = runtime; packages = {}
    for package in ("torch", "transformers", "tokenizers", "numpy", "huggingface-hub", "safetensors"):
        try: packages[package] = version(package)
        except PackageNotFoundError: packages[package] = None
    observed = {"python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "packages": packages, "torch_runtime_version": getattr(torch, "__version__", None), "numpy_runtime_version": getattr(np, "__version__", None)}; expected = frozen["runtime_identity"]
    _require(observed["python"] == expected["python"] and observed["packages"] == expected["packages"] and observed["torch_runtime_version"] == expected["packages"]["torch"] and observed["numpy_runtime_version"] == expected["packages"]["numpy"], "M2_S1_RUNTIME_IDENTITY_MISMATCH", "Encoder runtime differs from accepted M1 runtime", observed=observed, expected=expected)
    return {"environment_sha256": expected["environment_sha256"], "observed": observed, "device_policy": expected["device_policy"]}

def _within(child: Path, parent: Path) -> bool:
    try: child.relative_to(parent); return True
    except ValueError: return False
def validate_output_dir(output_dir: str | Path, cache_dir: str | Path, *, worktree: Path, frozen: Mapping[str, Any]) -> Path:
    root, cache = Path(output_dir).expanduser().resolve(), Path(cache_dir).expanduser().resolve(); _require(not root.exists(), "M2_S1_OUTPUT_REPLAY_OR_EXISTS", "output directory must be new")
    _require(not _within(root, cache), "M2_S1_OUTPUT_PROTECTED_PATH", "output cannot be in cache")
    for item in frozen["protected_output_roots"]: _require(not _within(root, (worktree / str(item)).resolve()), "M2_S1_OUTPUT_PROTECTED_PATH", "output cannot be in protected path", protected_path=str(item))
    _require(not any(item.name in {".encoder-artifacts", ".encoder-venv"} for item in (root, *root.parents)), "M2_S1_OUTPUT_PROTECTED_PATH", "output cannot be in historical Encoder assets"); return root
def _limits(start: float, root: Path, phase: str, seed: int | None = None) -> None:
    _require(time.monotonic() - start <= MAX_WALL_TIME_SECONDS, "M2_S1_WALL_TIME_LIMIT_EXCEEDED", "120-minute limit exceeded", phase=phase, seed=seed); _require(m1._directory_size(root) <= MAX_NEW_DISK_GIB * 1024**3, "M2_S1_DISK_LIMIT_EXCEEDED", "10 GiB output limit exceeded", phase=phase, seed=seed)

def validate_m2_s1_preflight(config_path: str | Path, cache_dir: str | Path, *, worktree: str | Path, contract_path: str | Path) -> dict[str, Any]:
    """The only data preflight: M1's selected Train/Dev loader, never audit_data."""
    frozen = _contract_requirements(contract_path); config = ProjectConfig.load(config_path); schema, train, dev = m1.load_m1_partitions(config)
    _require(len(train) == 1822 and len(dev) == 448, "M2_S1_DATA_CONTRACT_INVALID", "expected Train 1822/Dev 448")
    snapshot, identity = validate_fixed_cache_snapshot(Path(cache_dir), frozen)
    return {"frozen_contract": frozen, "snapshot": snapshot, "snapshot_identity": identity, "schema": schema, "train": train, "dev": dev, "identity": {"contract_sha256": frozen["contract_sha256"], "model_id": m1.MODEL_ID, "revision": m1.REVISION, "train_rows": len(train), "dev_rows": len(dev), "schema_version": schema.schema_version}, "worktree": str(Path(worktree).resolve())}

def _config(contract: Mapping[str, Any], order: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    common = _mapping(contract.get("frozen_input_and_common_training_configuration"), "M2_S1_CONTRACT_INVALID", "training configuration"); optimizer, early = _mapping(common.get("optimizer"), "M2_S1_CONTRACT_INVALID", "optimizer"), _mapping(common.get("early_stopping"), "M2_S1_CONTRACT_INVALID", "early stopping")
    return {"stage_id": STAGE_ID, "model_id": m1.MODEL_ID, "revision": m1.REVISION, "trust_remote_code": False, "local_files_only": True, "input_builder_version": common["input_builder_version"], "stock_code_token_cap": common["stock_code_token_cap"], "stock_name_token_cap": common["stock_name_token_cap"], "max_length": 256, "truncation": "HEAD_TAIL", "padding": common["padding"], "token_type_ids": "NOT_EMITTED", "batch_size": 16, "head_dropout": common["head_dropout"], "class_order": {h: list(order[h]) for h in V1_HEADS}, "optimizer": {"name": "AdamW", "learning_rate": .0005, "weight_decay": .01, "betas": optimizer["betas"], "epsilon": optimizer["epsilon"]}, "stopping": {"max_epochs": 12, "patience": 3, "minimum_delta": early["minimum_delta"]}, "gradient_clipping_max_norm": 1.0, "reasoning_probability_threshold": .5, "encoder_state": "FROZEN", "fit_population": "Train_1822_only", "dev_role": "early_stopping_and_diagnostic_only", "test_role": "not_loaded_not_used"}
def _finite(torch: Any, value: Any, code: str) -> None: _require(bool(torch.isfinite(value).all().item()), code, "non-finite training values")
def _critical(contract: Mapping[str, Any], dev: Mapping[str, Any], seed: int) -> dict[str, Any]:
    proxies = _mapping(contract.get("dev_metrics_and_no_regression"), "M2_S1_CONTRACT_INVALID", "metric contract").get("critical_boundary_proxies"); _require(isinstance(proxies, list) and len(proxies) == 7, "M2_S1_CONTRACT_INVALID", "seven critical boundaries required"); result = {}
    for item in proxies:
        head = item["head"]; labels = _mapping(dev[head].get("per_label") if head == "reasoning_tags" else dev[head].get("per_class"), "M2_S1_METRICS_MISSING", head); result[head] = {label: {"support": labels.get(label, {}).get("support"), "f1": labels.get(label, {}).get("f1"), "status": "REPORTED_ONLY_S1_CONTROL_NOT_A_SELECTION_OR_PRODUCTION_GATE"} for label in item["labels"]}
    return {"stage_id": STAGE_ID, "seed": seed, "selected_candidate": False, "critical_boundaries": result}

def _seed(*, runtime: tuple[Any, Any, Any, Any], seed: int, root: Path, snapshot: Path, schema: Any, train: Sequence[m1.M1Record], dev: Sequence[m1.M1Record], config: Mapping[str, Any], provenance: Mapping[str, Any], m2_contract: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    np, torch, AutoModel, AutoTokenizer = runtime; folder = root / f"seed-{seed}"; folder.mkdir(); start = time.monotonic(); _limits(start, root, "before_model_load", seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "mps"): torch.mps.manual_seed(seed)
    device, device_name, mps = m1._choose_device(torch); tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True); model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device); _limits(start, root, "after_model_load", seed)
    trainable = [p for p in model.parameters() if p.requires_grad]; _require(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) == 0 and len(model.heads) == 7, "M2_S1_FREEZE_OR_HEAD_CONTRACT_VIOLATION", "encoder must remain frozen")
    opt = torch.optim.AdamW(trainable, lr=.0005, weight_decay=.01, betas=tuple(config["optimizer"]["betas"]), eps=float(config["optimizer"]["epsilon"])); best, epoch_best, stale = -1., 0, 0; log = folder / "training-log.jsonl"
    for epoch in range(1, 13):
        model.train(); shuffled = list(train); random.Random(seed + epoch).shuffle(shuffled); losses = []
        for off in range(0, len(shuffled), 16):
            _limits(start, root, "during_epoch", seed); batch = m1._as_batch(torch, tokenizer, shuffled[off:off + 16], config, device); opt.zero_grad(set_to_none=True); logits = model(batch["input_ids"], batch["attention_mask"])
            for h in V1_HEADS: _finite(torch, logits[h], "M2_S1_NONFINITE_LOGITS")
            loss = m1._weighted_loss(torch, logits, batch); _finite(torch, loss, "M2_S1_NONFINITE_LOSS"); loss.backward()
            for p in trainable:
                if p.grad is not None: _finite(torch, p.grad, "M2_S1_NONFINITE_GRADIENT")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        dev_metrics = m1.diagnostic_metrics(torch, model, tokenizer, dev, config, device); score = float(dev_metrics["diagnostic_score"]); improved = score > best + float(config["stopping"]["minimum_delta"])
        if improved: best, epoch_best, stale = score, epoch, 0; torch.save({"run_schema_version": RUN_SCHEMA_VERSION, "stage_id": STAGE_ID, "seed": seed, "frozen_config": config, "provenance": provenance, "heads_state_dict": {k: v.detach().cpu() for k, v in model.heads.state_dict().items()}}, folder / "heads-checkpoint.pt")
        else: stale += 1
        m1._jsonl_append(log, {"seed": seed, "epoch": epoch, "train_weighted_loss": round(sum(losses)/len(losses), 6), "dev_diagnostic_score": score, "improved": improved, "stale_epochs": stale}); _limits(start, root, "after_epoch", seed)
        if stale >= 3: break
    checkpoint_path = folder / "heads-checkpoint.pt"; _require(checkpoint_path.is_file(), "M2_S1_CHECKPOINT_MISSING", "checkpoint missing", seed=seed); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model = m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])).to(device); best_model.heads.load_state_dict(checkpoint["heads_state_dict"]); train_metrics, dev_metrics = m1.diagnostic_metrics(torch, best_model, tokenizer, train, config, device), m1.diagnostic_metrics(torch, best_model, tokenizer, dev, config, device); _limits(start, root, "after_final_metrics", seed)
    try: smoke = m1.cpu_reload_and_inference_smoke(torch, lambda: m1._make_model(torch, AutoModel, snapshot, schema, float(config["head_dropout"])), checkpoint, tokenizer, dev[0], config)
    except ContractError as exc: raise ContractError("M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU reload failed", cause=exc.code) from exc
    _require(smoke.get("all_logits_finite") is True, "M2_S1_CPU_RELOAD_SMOKE_FAILED", "CPU logits not finite"); metrics = {"metric_scope": "WEAK_LABEL_DEV_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_PRODUCTION", "seed": seed, "best_epoch": epoch_best, "early_stopping_score": best, "sample_counts": {"train": len(train), "dev": len(dev)}, "train": train_metrics, "dev": dev_metrics, "cpu_reload_inference_smoke": smoke}; m1._json_dump(folder / "seed-metrics.json", metrics); critical = _critical(m2_contract, dev_metrics, seed); m1._json_dump(folder / "critical-boundary-report.json", critical)
    resource = {"seed": seed, "actual_device": device_name, "mps_available": mps, "elapsed_seconds": round(time.monotonic()-start,3), "cache_bytes": m1._directory_size(cache_dir), "artifact_bytes": m1._directory_size(folder), "checkpoint_bytes": checkpoint_path.stat().st_size, "cpu_reload_result": smoke, "python": {"implementation": platform.python_implementation(), "version": platform.python_version()}, "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}; m1._json_dump(folder / "resource-log.json", resource); _limits(start, root, "after_seed_evidence", seed)
    return {"seed": seed, "metrics": metrics, "resource": resource, "checkpoint_sha256": sha256_file(checkpoint_path), "critical_boundary_report_sha256": sha256_file(folder / "critical-boundary-report.json")}

def aggregate_s1_seed_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require([x.get("seed") for x in results] == list(SEEDS), "M2_S1_INCOMPLETE_SEEDS", "all seeds required"); devices = [x["resource"]["actual_device"] for x in results]; _require(len(set(devices)) == 1, "M2_S1_MIXED_DEVICE", "mixed device seeds rejected", device_stratified_seed_devices={str(x["seed"]): d for x,d in zip(results,devices,strict=True)})
    heads = {}
    for h in V1_HEADS:
        values = [float(x["metrics"]["dev"][h]["macro_f1"]) for x in results]; _require(all(math.isfinite(v) for v in values), "M2_S1_METRICS_MISSING", "nonfinite metric", head=h); heads[h] = {"per_seed_values": values, "mean": sum(values)/3, "sample_standard_deviation": statistics.stdev(values), "minimum_worst_seed": min(values), "maximum": max(values)}
    stable = all(v["sample_standard_deviation"] <= .05 for v in heads.values()); return {"stage_id": STAGE_ID, "seeds": list(SEEDS), "actual_device": devices[0], "all_seeds_complete": True, "per_head_primary_macro_f1": heads, "seed_stability_gate_passed": stable, "allowed_output": "S1_CONTROL_EVIDENCE_ONLY" if stable else "S1_REJECTED_OR_BLOCKED_EVIDENCE", "selected_candidate": False}

def _failure(root: Path | None, exc: ContractError, preflight: Mapping[str, Any], entered: bool) -> dict[str, Any]:
    result = {"status": "S1_REJECTED_OR_BLOCKED_EVIDENCE", "stage_id": STAGE_ID, "blocker_codes": [exc.code], "details": exc.details, "training_invoked": entered, "model_loaded": entered, "cache_accessed": True, "output_created": bool(root and root.exists()), "aggregate_created": False, "selected_candidate": False}
    if exc.code == "M2_S1_MIXED_DEVICE": result["device_stratified_rejected_evidence"] = exc.details.get("device_stratified_seed_devices", {})
    if root and root.exists(): m1._json_dump(root / "blocked-evidence.json", result); result["rejected_content_address"] = m1._write_content_manifest(root, {"manifest_schema_version":"myresearcher.encoder-m2-s1-rejected-artifact-manifest.v2", "status":result["status"], "stage_id":STAGE_ID, "failure":result, "m2_lineage":preflight["frozen_contract"]["contract"]["new_model_lineage"], "m1_controls":preflight["frozen_contract"]["m1_controls"], "complete_cache_snapshot":preflight["snapshot_identity"]})["content_address"]
    return result

def run_m2_s1(config_path: str | Path, output_dir: str | Path, cache_dir: str | Path, *, worktree: str | Path | None = None, contract_path: str | Path | None = None, runtime_loader: Callable[[],tuple[Any,Any,Any,Any]] = m1._load_runtime_dependencies, seed_executor: Callable[...,Mapping[str,Any]] | None = None) -> dict[str, Any]:
    worktree_path = Path(worktree).resolve() if worktree else Path(__file__).resolve().parents[2]; contract = Path(contract_path).resolve() if contract_path else worktree_path / CONTRACT_RELATIVE_PATH
    try: preflight = validate_m2_s1_preflight(config_path, cache_dir, worktree=worktree_path, contract_path=contract)
    except ContractError as exc: return {"status":"M2_S1_BLOCKED_FAIL_CLOSED", "phase":"TRAIN_DEV_TECHNICAL_PREFLIGHT", "blocker_codes":[exc.code], "training_invoked":False, "model_loaded":False, "cache_accessed":False, "output_created":False, "aggregate_created":False, "selected_candidate":False}
    started, root, results, entered = time.monotonic(), None, [], False
    try:
        runtime = runtime_loader(); runtime_id = validate_runtime_identity(runtime, preflight["frozen_contract"]); root = validate_output_dir(output_dir, cache_dir, worktree=worktree_path, frozen=preflight["frozen_contract"]); root.mkdir(parents=True); _limits(started, root, "before_fit")
        config = _config(preflight["frozen_contract"]["contract"], preflight["schema"].class_order); m1._json_dump(root / "training-config.json", config); execute = seed_executor or _seed
        for seed in SEEDS: entered = True; results.append(execute(runtime=runtime, seed=seed, root=root, snapshot=preflight["snapshot"], schema=preflight["schema"], train=preflight["train"], dev=preflight["dev"], config=config, provenance=preflight["identity"], m2_contract=preflight["frozen_contract"]["contract"], cache_dir=Path(cache_dir)))
        aggregate = aggregate_s1_seed_results(results); m1._json_dump(root / "stage-aggregate.json", aggregate); manifest = m1._write_content_manifest(root, {"manifest_schema_version":"myresearcher.encoder-m2-s1-artifact-manifest.v3", "stage_id":STAGE_ID, "diagnostic_only":True, "selected_candidate":False, "allowed_output":aggregate["allowed_output"], "m2_lineage":preflight["frozen_contract"]["contract"]["new_model_lineage"], "m1_controls":preflight["frozen_contract"]["m1_controls"], "complete_cache_snapshot":preflight["snapshot_identity"], "runtime_identity":runtime_id, "device_identity":{"actual_device":aggregate["actual_device"], "policy":"MPS_FIRST_CPU_FALLBACK"}, "training_config_sha256":sha256_file(root / "training-config.json"), "stage_aggregate_sha256":sha256_file(root / "stage-aggregate.json"), "seed_checkpoints":{str(x["seed"]):x["checkpoint_sha256"] for x in results}, "critical_boundary_report":{str(x["seed"]):x["critical_boundary_report_sha256"] for x in results}}); _limits(started,root,"after_final_manifest")
        return {"status":"M2_S1_CONTROL_COMPLETED", "stage_id":STAGE_ID, "training_invoked":True, "model_loaded":True, "cache_accessed":True, "output_created":True, "aggregate_created":True, "selected_candidate":False, "allowed_output":aggregate["allowed_output"], "output_dir":str(root), "content_address":manifest["content_address"], "aggregate":aggregate}
    except ContractError as exc: return _failure(root, exc, preflight, entered)
    except Exception as exc: return _failure(root, ContractError("M2_S1_RUNTIME_EXCEPTION", "runtime/OOM exception", exception_type=type(exc).__name__, detail=str(exc)), preflight, entered)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--cache-dir",required=True); args=parser.parse_args(argv); result=run_m2_s1(args.config,args.output_dir,args.cache_dir); (sys.stdout if result.get("status")=="M2_S1_CONTROL_COMPLETED" else sys.stderr).write(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); return 0 if result.get("status")=="M2_S1_CONTROL_COMPLETED" else 2
if __name__ == "__main__": raise SystemExit(main())
