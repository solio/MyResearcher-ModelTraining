from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.exceptions import ConvergenceWarning

from .audit_data import run_audit
from .config import ProjectConfig
from .data import read_json
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file
from .metrics import calibrate_model_thresholds, evaluate_model
from .models.classical import ClassicalMultiHeadModel
from .prepare import PreparedDataset, _run_root, prepare_dataset
from .preprocessing import build_model_inputs
from .reference_package import (
    audit_reference_package,
    compare_trained_model_to_reference,
)
from .schema import SINGLE_LABEL_HEADS


MODEL_VERSION = "semantic-student-v0.1.0"


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": status not in ("", "UNAVAILABLE"),
    }


def _code_manifest(project_root: Path, config_path: Path, schema_path: Path) -> dict[str, str]:
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.extend([config_path, schema_path])
    weighting = project_root / "configs" / "weighting_v0.3.5.yaml"
    if weighting.is_file():
        paths.append(weighting)
    result = {}
    for path in sorted(set(paths)):
        try:
            logical_path = str(path.relative_to(project_root))
        except ValueError:
            logical_path = str(path)
        result[logical_path] = sha256_file(path)
    return result


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("joblib", "jsonschema", "numpy", "PyYAML", "scikit-learn", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "accelerators": {
            "execution_device": "CPU",
            "torch": "NOT_INSTALLED_OR_UNUSED",
            "mps": "UNUSED",
            "cuda": "UNUSED",
        },
    }


def _reference_comparison(
    baseline_report: Mapping[str, Any],
    anchor_metrics: Mapping[str, Any],
    *,
    declared_tolerance: float | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    missing_reference_environment = False
    if baseline_report.get("status") == "DIAGNOSTIC_BASELINE_COMPLETE":
        scalar = baseline_report.get("scalar_fields")
        reasoning = baseline_report.get("reasoning_tags")
        if not isinstance(scalar, Mapping) or not isinstance(reasoning, Mapping):
            raise ContractError(
                "BASELINE_REPORT_INVALID", "native reference metrics are incomplete"
            )
        references = {
            head: scalar.get(head, {}).get("anchor50", {}).get("macro_f1")
            for head in SINGLE_LABEL_HEADS
        }
        references["reasoning_tags"] = reasoning.get("anchor50", {}).get("micro_f1")
        tolerance = declared_tolerance
        missing_reference_environment = not isinstance(
            baseline_report.get("reference_environment"), Mapping
        )
    elif baseline_report.get("reproduction_contract_complete"):
        references = baseline_report.get("reference_metrics")
        tolerance = baseline_report.get("tolerance")
    else:
        return {
            "claim_allowed": False,
            "within_tolerance": False,
            "reason": "reference report does not declare a complete reproduction contract",
        }
    if not isinstance(references, Mapping) or not isinstance(tolerance, (int, float, Mapping)):
        raise ContractError(
            "BASELINE_REPORT_INVALID",
            "complete reproduction report requires reference_metrics and tolerance",
        )
    observed: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    for head in SINGLE_LABEL_HEADS:
        observed[head] = float(anchor_metrics[head]["macro_f1"])
    observed["reasoning_tags"] = float(anchor_metrics["reasoning_tags"]["micro_f1"])
    for head, value in observed.items():
        if head not in references:
            raise ContractError(
                "BASELINE_REPORT_INVALID", "reference metric is missing", head=head
            )
        allowed_delta = (
            float(tolerance.get(head))
            if isinstance(tolerance, Mapping)
            else float(tolerance)
        )
        reference = float(references[head])
        delta = abs(value - reference)
        comparisons[head] = {
            "reference": reference,
            "observed": value,
            "absolute_delta": delta,
            "tolerance": allowed_delta,
            "within_tolerance": delta <= allowed_delta,
        }
    result = {
        "claim_allowed": True,
        "within_tolerance": all(
            comparison["within_tolerance"] for comparison in comparisons.values()
        ),
        "heads": comparisons,
    }
    if missing_reference_environment:
        result.update(
            {
                "claim_allowed": False,
                "within_tolerance": False,
                "reason": (
                    "reference metrics omit Python, dependency, platform, and "
                    "solver-convergence provenance"
                ),
                "blocker_codes": ["BLOCKED_MISSING_REFERENCE_ENVIRONMENT"],
                "observed_environment_is_comparable": False,
            }
        )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                f"{json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            )


def _model_card(run_manifest: Mapping[str, Any]) -> str:
    return f"""# Model Card — {run_manifest['model_version']}

## Status

`{run_manifest['status']}`. Capability maturity: `{run_manifest['capability_maturity']}`.

## Intended use

Transparent TF-IDF + Logistic Regression diagnostic learning of post-level
belief–emotion–action fields. This artifact is not a production model, price
forecast, investment recommendation, or group-state model.

## Contracts

- Schema: `{run_manifest['schema_version']}`
- Preprocessing: `{run_manifest['preprocessing_contract_version']}`
- Prepared manifest: `{run_manifest['prepare_manifest_id']}`
- Split manifest: `{run_manifest['split_manifest_id']}`
- Quarantine manifest: `{run_manifest['quarantine_manifest_id']}`
- Anchor manifest: `{run_manifest['anchor_manifest_id']}`

## Limitations

The observed upstream population contains one platform (`eastmoney_guba`) and
16 stocks. This diagnostic baseline must not be run over all 49,054 records or
connected to production in Milestone 1. Abstention thresholds are calibrated on
Dev and must be honored by every caller.
"""


def train_prepared(
    prepared: PreparedDataset,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    config = prepared.config
    start = time.perf_counter()
    model_config = config.raw.get("model")
    if not isinstance(model_config, Mapping):
        raise ContractError("MODEL_CONFIG_INVALID", "model config must be an object")
    train = prepared.partition("train")
    dev = prepared.partition("dev")
    test = prepared.partition("test")
    if not train.sample_ids or not dev.sample_ids or not test.sample_ids:
        raise ContractError(
            "TRAINING_SPLIT_EMPTY", "train/dev/test must all contain records"
        )
    code_manifest = _code_manifest(
        config.project_root, config.path, prepared.schema.path
    )
    weighting_config = config.raw.get("weighting", {})
    weighting_value = str(
        weighting_config.get("config_path", "configs/weighting_v0.3.5.yaml")
    )
    weighting_path = Path(weighting_value)
    if not weighting_path.is_absolute():
        weighting_path = config.project_root / weighting_path
    if not weighting_path.is_file():
        raise ContractError(
            "WEIGHTING_CONFIG_NOT_FOUND", "versioned weighting config is required"
        )
    weighting_raw = yaml.safe_load(weighting_path.read_text(encoding="utf-8"))
    if not isinstance(weighting_raw, Mapping) or not weighting_raw.get("config_version"):
        raise ContractError(
            "WEIGHTING_CONFIG_INVALID", "weighting config_version is required"
        )
    git_state = _git_state(config.project_root)
    identity_payload = {
        "run_manifest_schema_version": "myresearcher.classical-run.v1",
        "model_version": MODEL_VERSION,
        "model_family": "tfidf-logistic-regression",
        "prepare_manifest_id": prepared.manifest["prepare_manifest_id"],
        "schema_version": prepared.schema.schema_version,
        "preprocessing_contract_id": prepared.preprocessing.contract_id,
        "seed": int(config.raw.get("seed", 0)),
        "model_config": model_config,
        "weighting_config_version": weighting_raw["config_version"],
        "weighting_config_sha256": sha256_file(weighting_path),
        "code_manifest": code_manifest,
        "git_commit": git_state["commit"],
    }
    run_id = content_addressed_id(identity_payload)
    root = run_root or _run_root(config)
    final_dir = root / run_id
    if final_dir.exists():
        manifest_path = final_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise ContractError(
                "IMMUTABLE_RUN_CONFLICT", "run directory exists without manifest"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_id") != run_id:
            raise ContractError("IMMUTABLE_RUN_CONFLICT", "run identity mismatch")
        return {
            "status": "EXISTING_IMMUTABLE_RUN",
            "run_id": run_id,
            "run_dir": str(final_dir),
        }

    model = ClassicalMultiHeadModel.create(prepared.schema, model_config)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train.texts, train.labels, train.weights)
    training_diagnostics = {
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in caught_warnings
        ],
        "convergence": model.convergence_diagnostics(),
    }
    calibration_config = config.raw.get("calibration", {})
    expected_features = dict(prepared.preprocessing.expected_feature_counts)
    observed_features = model.feature_counts()
    if expected_features and observed_features != expected_features:
        raise ContractError(
            "FITTED_FEATURE_COUNT_MISMATCH",
            "fitted TF-IDF vocabulary differs from the immutable contract",
            observed=observed_features,
            expected=expected_features,
        )
    thresholds = calibrate_model_thresholds(
        model,
        dev.texts,
        dev.labels,
        minimum_coverage=float(calibration_config.get("minimum_coverage", 0.8)),
        method=str(calibration_config.get("method", "dev-threshold-v0.1")),
    )
    all_metrics: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for split_name, partition in (("dev", dev), ("test", test)):
        split_metrics, split_errors = evaluate_model(
            model,
            partition.texts,
            partition.labels,
            partition.sample_ids,
            thresholds,
        )
        all_metrics[split_name] = split_metrics
        all_errors.extend({"split": split_name, **error} for error in split_errors)
    anchor_metrics: Mapping[str, Any] = {}
    if prepared.anchors:
        anchor_texts = build_model_inputs(prepared.anchors, prepared.preprocessing)
        anchor_ids = [str(anchor["sample_id"]) for anchor in prepared.anchors]
        anchor_metrics, anchor_errors = evaluate_model(
            model,
            anchor_texts,
            prepared.anchors,
            anchor_ids,
            thresholds,
        )
        all_metrics["anchor"] = anchor_metrics
        all_errors.extend({"split": "anchor", **error} for error in anchor_errors)
    baseline_report = read_json(config.data_path("baseline_report"))
    if not isinstance(baseline_report, Mapping):
        raise ContractError("BASELINE_REPORT_INVALID", "report must be an object")
    canonical_ids = prepared.manifest.get("canonical_contract_ids", {})
    reference_audit = audit_reference_package(
        config,
        data_package_manifest_id=str(canonical_ids.get("package_manifest_id", "")),
        schema=prepared.schema,
        preprocessing=prepared.preprocessing,
        input_index=prepared.records_by_id,
        trainable_index=prepared.labels_by_id,
        split_ids=prepared.split_ids,
        anchors=prepared.anchors,
    )
    if reference_audit.get("available"):
        comparison = compare_trained_model_to_reference(
            config, prepared, model, thresholds, reference_audit
        )
    else:
        reproduction = config.raw.get("baseline_reproduction", {})
        declared_tolerance = (
            reproduction.get("tolerance")
            if isinstance(reproduction, Mapping)
            else None
        )
        comparison = _reference_comparison(
            baseline_report,
            anchor_metrics,
            declared_tolerance=declared_tolerance,
        )
    comparison_blockers = list(comparison.get("blocker_codes", []))
    if comparison.get("status") == "BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY":
        status = "BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY"
        maturity = "DATA_VALIDATED_REPRODUCED_DIAGNOSTIC_ONLY"
    elif comparison.get("status") == "COMPARABLE_DIAGNOSTIC_RUN_ONLY":
        status = "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
        maturity = "DATA_AND_REFERENCE_VALIDATED_COMPARABLE_ONLY"
    elif comparison.get("status") == "BASELINE_V0_3_5_REPRODUCTION_MISMATCH":
        status = "BASELINE_V0_3_5_REPRODUCTION_MISMATCH"
        maturity = "DATA_VALIDATED_DIAGNOSTIC_MISMATCH"
    elif comparison_blockers:
        status = "BASELINE_V0_3_5_REPRODUCTION_BLOCKED_REFERENCE_ENVIRONMENT"
        maturity = "DATA_VALIDATED_REFERENCE_ENVIRONMENT_MISSING"
    elif comparison["claim_allowed"] and comparison["within_tolerance"]:
        status = "BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY"
        maturity = "DATA_VALIDATED_REPRODUCED_DIAGNOSTIC_ONLY"
    elif comparison["claim_allowed"]:
        status = "BASELINE_V0_3_5_REPRODUCTION_MISMATCH"
        maturity = "DATA_VALIDATED_DIAGNOSTIC_MISMATCH"
    else:
        status = "BASELINE_HARNESS_TESTED"
        maturity = "TESTED"

    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-run-", dir=root))
    try:
        model_path = temp_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        _write_json(temp_dir / "thresholds.json", thresholds)
        _write_json(temp_dir / "metrics.json", all_metrics)
        _write_json(temp_dir / "baseline_comparison.json", comparison)
        _write_json(temp_dir / "training_diagnostics.json", training_diagnostics)
        _write_json(temp_dir / "schema.json", prepared.schema.raw)
        _write_json(
            temp_dir / "preprocessing_contract.json", prepared.preprocessing.raw
        )
        _write_json(temp_dir / "config.json", dict(config.raw))
        _write_json(temp_dir / "weighting_config.json", weighting_raw)
        inference_schema_path = config.project_root / "schema" / "inference-output.schema.json"
        inference_schema = read_json(inference_schema_path)
        _write_json(temp_dir / "inference-output.schema.json", inference_schema)
        _write_jsonl(temp_dir / "errors.jsonl", all_errors)
        model_manifest: dict[str, Any] = {
            "model_manifest_schema_version": "myresearcher.classical-model.v1",
            "model_version": MODEL_VERSION,
            "schema_version": prepared.schema.schema_version,
            "class_order": model.class_order,
            "preprocessing_contract_version": prepared.preprocessing.contract_version,
            "preprocessing_contract_id": prepared.preprocessing.contract_id,
            "feature_count": model.feature_count(),
            "feature_counts": observed_features,
            "model_sha256": sha256_file(model_path),
            "thresholds_sha256": sha256_file(temp_dir / "thresholds.json"),
            "schema_sha256": sha256_file(temp_dir / "schema.json"),
            "preprocessing_contract_sha256": sha256_file(
                temp_dir / "preprocessing_contract.json"
            ),
            "inference_schema_sha256": sha256_file(
                temp_dir / "inference-output.schema.json"
            ),
            "training_diagnostics_sha256": sha256_file(
                temp_dir / "training_diagnostics.json"
            ),
        }
        model_manifest["model_manifest_id"] = content_addressed_id(
            model_manifest, omit_keys={"model_manifest_id"}
        )
        _write_json(temp_dir / "model_manifest.json", model_manifest)
        run_manifest: dict[str, Any] = {
            **identity_payload,
            "run_id": run_id,
            "status": status,
            "capability_maturity": maturity,
            "blocker_codes": comparison_blockers,
            "schema_sha256": sha256_file(prepared.schema.path),
            "preprocessing_contract_version": prepared.preprocessing.contract_version,
            "weighting_config_version": weighting_raw["config_version"],
            "weighting_config_sha256": sha256_file(temp_dir / "weighting_config.json"),
            "split_manifest_id": prepared.manifest.get(
                "canonical_contract_ids", {}
            ).get("split_manifest_id")
            or read_json(config.data_path("split_manifest")).get("split_manifest_id"),
            "quarantine_manifest_id": prepared.manifest.get(
                "canonical_contract_ids", {}
            ).get("quarantine_manifest_id")
            or read_json(config.data_path("quarantine_manifest")).get(
                "quarantine_manifest_id"
            ),
            "anchor_manifest_id": prepared.manifest.get(
                "canonical_contract_ids", {}
            ).get("anchor_manifest_id")
            or read_json(config.data_path("anchor_manifest")).get("anchor_manifest_id"),
            "canonical_package_manifest_id": prepared.manifest.get(
                "canonical_contract_ids", {}
            ).get("package_manifest_id")
            or read_json(config.data_path("package_manifest")).get("package_manifest_id"),
            "reference_package_manifest_id": reference_audit.get(
                "package_manifest_id"
            ),
            "reference_model_sha256": reference_audit.get(
                "original_model_sha256"
            ),
            "config_path": str(config.path),
            "config_sha256": sha256_file(config.path),
            "input_artifacts": prepared.manifest["input_artifacts"],
            "git": git_state,
            "environment": _environment_manifest(),
            "model_manifest_id": model_manifest["model_manifest_id"],
            "model_manifest_sha256": sha256_file(temp_dir / "model_manifest.json"),
            "metrics_sha256": sha256_file(temp_dir / "metrics.json"),
            "errors_sha256": sha256_file(temp_dir / "errors.jsonl"),
            "baseline_comparison_sha256": sha256_file(
                temp_dir / "baseline_comparison.json"
            ),
            "training_diagnostics_sha256": sha256_file(
                temp_dir / "training_diagnostics.json"
            ),
            "elapsed_seconds": time.perf_counter() - start,
        }
        run_manifest["run_manifest_id"] = content_addressed_id(
            run_manifest, omit_keys={"run_manifest_id", "elapsed_seconds", "environment"}
        )
        _write_json(temp_dir / "run_manifest.json", run_manifest)
        (temp_dir / "MODEL_CARD.md").write_text(
            _model_card(run_manifest), encoding="utf-8"
        )
        os.replace(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "status": status,
        "run_id": run_id,
        "run_dir": str(final_dir),
        "blocker_codes": comparison_blockers,
    }


def run_train(path: str | Path) -> tuple[dict[str, Any], int]:
    audit_result, exit_code = run_audit(path)
    if exit_code:
        return audit_result, exit_code
    try:
        config = ProjectConfig.load(path)
        prepared = prepare_dataset(config, audit_result=audit_result)
        result = train_prepared(prepared)
    except ContractError as exc:
        return {
            "status": "BLOCKED_TRAIN_CONTRACT_ERROR",
            "training_allowed": False,
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }, 3
    return result, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train classical diagnostic baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result, exit_code = run_train(args.config)
    sys.stdout.write(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
