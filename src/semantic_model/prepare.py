from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit_data import audit_config, run_audit
from .config import ProjectConfig
from .data import index_by_sample_id, join_inputs_and_labels, read_json, read_jsonl
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, without_local_paths
from .immutable_package import (
    is_native_package,
    load_native_quarantine,
    load_native_split_ids,
    load_native_trainable_labels,
)
from .preprocessing import PreprocessingContract, build_model_inputs
from .schema import LabelSchema
from .weighting import validate_field_weights


@dataclass(frozen=True)
class DatasetPartition:
    sample_ids: list[str]
    records: list[Mapping[str, Any]]
    texts: list[str]
    labels: list[Mapping[str, Any]]
    weights: list[Mapping[str, float]]


@dataclass(frozen=True)
class PreparedDataset:
    config: ProjectConfig
    schema: LabelSchema
    preprocessing: PreprocessingContract
    manifest: Mapping[str, Any]
    records_by_id: Mapping[str, Mapping[str, Any]]
    labels_by_id: Mapping[str, Mapping[str, Any]]
    texts_by_id: Mapping[str, str]
    weights_by_id: Mapping[str, Mapping[str, float]]
    split_ids: Mapping[str, list[str]]
    anchors: list[Mapping[str, Any]]
    artifact_dir: Path | None

    def partition(self, split: str) -> DatasetPartition:
        sample_ids = list(self.split_ids.get(split, []))
        return DatasetPartition(
            sample_ids=sample_ids,
            records=[self.records_by_id[sample_id] for sample_id in sample_ids],
            texts=[self.texts_by_id[sample_id] for sample_id in sample_ids],
            labels=[self.labels_by_id[sample_id] for sample_id in sample_ids],
            weights=[self.weights_by_id[sample_id] for sample_id in sample_ids],
        )


def _run_root(config: ProjectConfig) -> Path:
    runtime = config.raw.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ContractError("CONFIG_INVALID", "runtime must be an object")
    path = Path(str(runtime.get("run_root", "runs")))
    return path if path.is_absolute() else config.project_root / path


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ContractError(
                "IMMUTABLE_ARTIFACT_CONFLICT", "existing artifact bytes differ", path=str(path)
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def prepare_dataset(
    config: ProjectConfig,
    *,
    audit_result: Mapping[str, Any] | None = None,
    write_artifacts: bool = True,
) -> PreparedDataset:
    audit_result = dict(audit_result or audit_config(config))
    if not audit_result.get("training_allowed"):
        raise ContractError(
            "BLOCKED_MISSING_CANONICAL_ARTIFACTS",
            "prepare requires a passing canonical data audit",
            blocker_codes=audit_result.get("blocker_codes", []),
        )
    schema = LabelSchema.load(config.repo_path("schema_path"))
    inputs = read_jsonl(config.data_path("canonical_inputs"))
    native_package = is_native_package(config)
    labels = (
        load_native_trainable_labels(config)
        if native_package
        else read_jsonl(config.data_path("frozen_teacher_labels"))
    )
    joined = join_inputs_and_labels(
        inputs, labels, require_complete=not native_package
    )
    records_by_id = {
        str(record.input["sample_id"]): record.input for record in joined
    }
    labels_by_id = {
        str(record.label["sample_id"]): record.label for record in joined
    }
    if native_package:
        quarantine_ids = set(
            index_by_sample_id(load_native_quarantine(config), role="quarantine")
        )
        split_ids = load_native_split_ids(config)
        split_by_id = {
            sample_id: split
            for split, sample_ids in split_ids.items()
            for sample_id in sample_ids
        }
    else:
        quarantine_manifest = read_json(config.data_path("quarantine_manifest"))
        if not isinstance(quarantine_manifest, Mapping):
            raise ContractError("MANIFEST_INVALID", "quarantine manifest must be an object")
        quarantine_ids = set(
            index_by_sample_id(
                quarantine_manifest.get("records", []), role="quarantine"
            )
        )
        split_manifest = read_json(config.data_path("split_manifest"))
        if not isinstance(split_manifest, Mapping):
            raise ContractError("MANIFEST_INVALID", "split manifest must be an object")
        split_ids = {
            split: [] for split in ("train", "dev", "test", "embargo")
        }
        split_by_id = {}
        for assignment in split_manifest["assignments"]:
            split = str(assignment["split"])
            sample_id = str(assignment["sample_id"])
            split_ids[split].append(sample_id)
            split_by_id[sample_id] = split
    trainable_ids = set().union(*map(set, split_ids.values()))
    if trainable_ids & quarantine_ids:
        raise ContractError(
            "QUARANTINE_SPLIT_LEAKAGE", "quarantine identity reached prepare"
        )
    weights_by_id = validate_field_weights(
        read_jsonl(config.data_path("field_weights")), expected_ids=trainable_ids
    )
    preprocessing = PreprocessingContract.load(
        config.data_path("preprocessing_contract")
    )
    ordered_ids = sorted(trainable_ids)
    ordered_records = [records_by_id[sample_id] for sample_id in ordered_ids]
    ordered_texts = build_model_inputs(ordered_records, preprocessing)
    texts_by_id = dict(zip(ordered_ids, ordered_texts, strict=True))
    sample_contract = [
        {
            "sample_id": sample_id,
            "split": split_by_id[sample_id],
            "model_input_sha256": content_addressed_id(
                {"text": texts_by_id[sample_id]}
            ),
            "label_sha256": content_addressed_id(dict(labels_by_id[sample_id])),
            "field_weights_sha256": content_addressed_id(
                dict(weights_by_id[sample_id])
            ),
        }
        for sample_id in ordered_ids
    ]
    manifest: dict[str, Any] = {
        "prepare_manifest_schema_version": "myresearcher.prepared-dataset.v1",
        "schema_version": schema.schema_version,
        "schema_sha256": sha256_file(schema.path),
        "preprocessing_contract_version": preprocessing.contract_version,
        "preprocessing_contract_id": preprocessing.contract_id,
        "config_sha256": sha256_file(config.path),
        "seed": int(config.raw.get("seed", 0)),
        "audit_id": audit_result["audit_id"],
        "canonical_contract_ids": {
            key: value
            for key, value in audit_result.get("validation_summary", {}).items()
            if key.endswith("_id")
        },
        "input_artifacts": without_local_paths(audit_result["observed"]),
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "quarantine_count": len(quarantine_ids),
        "samples": sample_contract,
    }
    manifest["prepare_manifest_id"] = content_addressed_id(
        manifest, omit_keys={"prepare_manifest_id"}
    )
    artifact_dir = None
    if write_artifacts:
        artifact_dir = (
            _run_root(config) / "prepared" / str(manifest["prepare_manifest_id"])
        )
        _write_immutable_json(artifact_dir / "manifest.json", manifest)
    anchors = read_jsonl(config.data_path("anchor_labels"))
    return PreparedDataset(
        config=config,
        schema=schema,
        preprocessing=preprocessing,
        manifest=manifest,
        records_by_id=records_by_id,
        labels_by_id=labels_by_id,
        texts_by_id=texts_by_id,
        weights_by_id=weights_by_id,
        split_ids=split_ids,
        anchors=anchors,
        artifact_dir=artifact_dir,
    )


def run_prepare(path: str | Path) -> tuple[dict[str, Any], int]:
    audit_result, exit_code = run_audit(path)
    if exit_code:
        return audit_result, exit_code
    try:
        config = ProjectConfig.load(path)
        prepared = prepare_dataset(config, audit_result=audit_result)
    except ContractError as exc:
        return {
            "status": "BLOCKED_PREPARE_CONTRACT_ERROR",
            "training_allowed": False,
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }, 3
    return {
        "status": "PREPARED",
        "prepare_manifest_id": prepared.manifest["prepare_manifest_id"],
        "artifact_dir": str(prepared.artifact_dir),
        "split_counts": prepared.manifest["split_counts"],
    }, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic dataset preparation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result, exit_code = run_prepare(args.config)
    sys.stdout.write(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
