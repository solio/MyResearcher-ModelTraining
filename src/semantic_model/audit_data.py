from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .data import (
    index_by_sample_id,
    join_inputs_and_labels,
    read_json,
    read_jsonl,
)
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, verify_content_addressed_id
from .schema import LabelSchema
from .split import validate_split_manifest
from .validation import validate_evidence_dependencies, validate_label_record
from .weighting import validate_field_weights


MISSING_REQUIREMENTS = (
    (
        "frozen_teacher_labels",
        "BLOCKED_MISSING_FROZEN_TEACHER_LABELS",
        "3,000 frozen teacher labels",
    ),
    (
        "quarantine_manifest",
        "BLOCKED_MISSING_QUARANTINE_MANIFEST",
        "21-row Evidence quarantine manifest",
    ),
    (
        "split_manifest",
        "BLOCKED_MISSING_SPLIT_V0_3_5",
        "frozen v0.3.5 split manifest",
    ),
    (
        "field_weights",
        "BLOCKED_MISSING_DRIFT_WEIGHT_MAP",
        "2,979-row field-weight map",
    ),
    ("anchor_labels", "BLOCKED_MISSING_ANCHOR_50", "Anchor50 labels"),
    ("anchor_manifest", "BLOCKED_MISSING_ANCHOR_50", "Anchor50 provenance"),
    (
        "baseline_report",
        "BLOCKED_MISSING_BASELINE_REPORT",
        "immutable v0.3.5 baseline report bundle",
    ),
    (
        "preprocessing_contract",
        "BLOCKED_MISSING_PREPROCESSING_CONTRACT_V0_3_5",
        "exact v0.3.5 preprocessing contract",
    ),
    (
        "package_manifest",
        "BLOCKED_MISSING_CANONICAL_PACKAGE_MANIFEST",
        "content-addressed canonical package manifest",
    ),
)


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _load_metadata(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise ContractError("CANONICAL_METADATA_NOT_FOUND", str(path)) from exc


def _validate_inputs(
    config: ProjectConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = config.data_path("canonical_inputs")
    metadata_path = config.data_path("canonical_metadata")
    if not input_path.exists():
        raise ContractError("CANONICAL_INPUTS_NOT_FOUND", str(input_path))
    if not metadata_path.exists():
        raise ContractError("CANONICAL_METADATA_NOT_FOUND", str(metadata_path))
    inputs = read_jsonl(input_path)
    input_index = index_by_sample_id(inputs, role="canonical-input")
    required_fields = {
        "sample_id",
        "stock_code",
        "stock_name",
        "published_at",
        "model_text",
    }
    for sample_id, record in input_index.items():
        missing = sorted(
            field
            for field in required_fields
            if not isinstance(record.get(field), str) or not record.get(field)
        )
        if missing:
            raise ContractError(
                "CANONICAL_INPUT_INVALID",
                "canonical input is missing required string fields",
                sample_id=sample_id,
                fields=missing,
            )
    expected_inputs = config.expected.get("inputs")
    if expected_inputs is not None and len(inputs) != int(expected_inputs):
        raise ContractError(
            "INPUT_COUNT_MISMATCH",
            "canonical input count differs from config",
            actual=len(inputs),
            expected=int(expected_inputs),
        )
    metadata = _load_metadata(metadata_path)
    metadata_index = index_by_sample_id(metadata, role="canonical-metadata")
    if set(metadata_index) != set(input_index):
        raise ContractError(
            "CANONICAL_METADATA_COVERAGE_MISMATCH",
            "canonical metadata identities differ from canonical inputs",
            missing=sorted(set(input_index) - set(metadata_index)),
            extra=sorted(set(metadata_index) - set(input_index)),
        )
    for sample_id, record in input_index.items():
        metadata_record = metadata_index[sample_id]
        mismatch = {
            field: {
                "input": record.get(field),
                "metadata": metadata_record.get(field),
            }
            for field in ("stock_code", "stock_name", "published_at", "model_text")
            if metadata_record.get(field) != record.get(field)
        }
        if mismatch:
            raise ContractError(
                "CANONICAL_METADATA_MISMATCH",
                "canonical JSONL and metadata CSV disagree",
                sample_id=sample_id,
                mismatches=mismatch,
            )
    inventory = {
        "canonical_inputs": {
            "path": str(input_path),
            "rows": len(inputs),
            "sample_id_unique": len(input_index),
            "sha256": sha256_file(input_path),
        },
        "canonical_metadata": {
            "path": str(metadata_path),
            "rows": len(metadata),
            "sample_id_unique": len(metadata_index),
            "sha256": sha256_file(metadata_path),
        },
    }
    return inputs, inventory


def _extract_records(
    manifest: Mapping[str, Any], *, artifact: str
) -> list[Mapping[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ContractError(
            "MANIFEST_INVALID", f"{artifact}.records must be a list of objects"
        )
    return records


def _validate_complete_package(
    config: ProjectConfig,
    inputs: list[dict[str, Any]],
    inventory: dict[str, Any],
    schema: LabelSchema,
) -> dict[str, Any]:
    labels_path = config.data_path("frozen_teacher_labels")
    labels = read_jsonl(labels_path)
    expected_labels = int(config.expected.get("frozen_labels", 3000))
    if len(labels) != expected_labels:
        raise ContractError(
            "LABEL_COVERAGE_MISMATCH",
            "frozen label count differs from contract",
            actual=len(labels),
            expected=expected_labels,
        )
    for label in labels:
        validate_label_record(label, schema)
    joined = join_inputs_and_labels(inputs, labels, require_complete=True)

    evidence_violations: dict[str, dict[str, Any]] = {}
    for record in joined:
        try:
            validate_evidence_dependencies(record.input, record.label)
        except ContractError as exc:
            if exc.code not in {
                "EVIDENCE_DEPENDENCY_VIOLATION",
                "EVIDENCE_NOT_SUBSTRING",
                "EVIDENCE_SHAPE_INVALID",
            }:
                raise
            evidence_violations[str(record.input["sample_id"])] = exc.as_dict()

    quarantine_path = config.data_path("quarantine_manifest")
    quarantine_manifest = read_json(quarantine_path)
    if not isinstance(quarantine_manifest, Mapping):
        raise ContractError("MANIFEST_INVALID", "quarantine manifest must be an object")
    verify_content_addressed_id(
        quarantine_manifest, id_key="quarantine_manifest_id"
    )
    quarantine_records = _extract_records(
        quarantine_manifest, artifact="quarantine_manifest"
    )
    quarantine_index = index_by_sample_id(quarantine_records, role="quarantine")
    expected_quarantine = int(config.expected.get("quarantine", 21))
    if len(quarantine_index) != expected_quarantine:
        raise ContractError(
            "QUARANTINE_COUNT_MISMATCH",
            "quarantine count differs from contract",
            actual=len(quarantine_index),
            expected=expected_quarantine,
        )
    if set(quarantine_index) != set(evidence_violations):
        raise ContractError(
            "QUARANTINE_EVIDENCE_MISMATCH",
            "quarantine identities must exactly match Evidence violations",
            missing=sorted(set(evidence_violations) - set(quarantine_index)),
            extra=sorted(set(quarantine_index) - set(evidence_violations)),
        )
    label_ids = set(index_by_sample_id(labels, role="frozen-label"))
    trainable_ids = label_ids - set(quarantine_index)
    expected_trainable = int(config.expected.get("trainable", 2979))
    if len(trainable_ids) != expected_trainable:
        raise ContractError(
            "TRAINABLE_COUNT_MISMATCH",
            "non-quarantined label count differs from contract",
            actual=len(trainable_ids),
            expected=expected_trainable,
        )

    anchor_path = config.data_path("anchor_labels")
    anchors = read_jsonl(anchor_path)
    for anchor in anchors:
        validate_label_record(anchor, schema)
    anchor_index = index_by_sample_id(anchors, role="anchor")
    expected_anchor = int(config.expected.get("anchor", 50))
    if len(anchor_index) != expected_anchor:
        raise ContractError(
            "ANCHOR_COUNT_MISMATCH",
            "Anchor count differs from contract",
            actual=len(anchor_index),
            expected=expected_anchor,
        )
    anchor_manifest = read_json(config.data_path("anchor_manifest"))
    if not isinstance(anchor_manifest, Mapping):
        raise ContractError("MANIFEST_INVALID", "Anchor manifest must be an object")
    verify_content_addressed_id(anchor_manifest, id_key="anchor_manifest_id")
    anchor_records = _extract_records(anchor_manifest, artifact="anchor_manifest")
    anchor_manifest_index = index_by_sample_id(anchor_records, role="anchor-manifest")
    if set(anchor_manifest_index) != set(anchor_index):
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor labels and provenance manifest identities differ",
        )
    provenance_counts = anchor_manifest.get("provenance_counts")
    expected_anchor_provenance = config.expected.get(
        "anchor_provenance", {"human_confirmed": 11, "expert_weak_gold": 39}
    )
    if provenance_counts != expected_anchor_provenance:
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor provenance counts differ from the frozen config contract",
            observed=provenance_counts,
            expected=expected_anchor_provenance,
        )

    split_manifest = read_json(config.data_path("split_manifest"))
    if not isinstance(split_manifest, Mapping):
        raise ContractError("MANIFEST_INVALID", "split manifest must be an object")
    verify_content_addressed_id(split_manifest, id_key="split_manifest_id")
    split_expected = config.expected.get("split", {})
    by_split = validate_split_manifest(
        split_manifest,
        trainable_ids=trainable_ids,
        quarantine_ids=set(quarantine_index),
        expected_counts=split_expected,
        anchor_ids=set(anchor_index),
        canonical_inputs=index_by_sample_id(inputs, role="canonical-input"),
    )

    weight_path = config.data_path("field_weights")
    weights = validate_field_weights(
        read_jsonl(weight_path), expected_ids=trainable_ids
    )
    baseline_report = read_json(config.data_path("baseline_report"))
    preprocessing_contract = read_json(config.data_path("preprocessing_contract"))
    if not isinstance(baseline_report, Mapping):
        raise ContractError("BASELINE_REPORT_INVALID", "report must be an object")
    if not isinstance(preprocessing_contract, Mapping):
        raise ContractError(
            "PREPROCESSING_CONTRACT_INVALID", "contract must be an object"
        )
    verify_content_addressed_id(baseline_report, id_key="baseline_report_id")
    verify_content_addressed_id(
        preprocessing_contract, id_key="preprocessing_contract_id"
    )

    package_path = config.data_path("package_manifest")
    package_manifest = read_json(package_path)
    if not isinstance(package_manifest, Mapping):
        raise ContractError("MANIFEST_INVALID", "package manifest must be an object")
    verify_content_addressed_id(package_manifest, id_key="package_manifest_id")
    package_artifacts = package_manifest.get("artifacts")
    if not isinstance(package_artifacts, Mapping):
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID", "package artifacts must be an object"
        )
    packaged_paths = {
        "canonical_inputs": config.data_path("canonical_inputs"),
        "canonical_metadata": config.data_path("canonical_metadata"),
        "frozen_teacher_labels": labels_path,
        "quarantine_manifest": quarantine_path,
        "split_manifest": config.data_path("split_manifest"),
        "field_weights": weight_path,
        "anchor_labels": anchor_path,
        "anchor_manifest": config.data_path("anchor_manifest"),
        "baseline_report": config.data_path("baseline_report"),
        "preprocessing_contract": config.data_path("preprocessing_contract"),
    }
    for logical_name, artifact_path in packaged_paths.items():
        package_entry = package_artifacts.get(logical_name)
        if not isinstance(package_entry, Mapping) or package_entry.get(
            "sha256"
        ) != sha256_file(artifact_path):
            raise ContractError(
                "CANONICAL_PACKAGE_HASH_MISMATCH",
                "package manifest does not match canonical artifact bytes",
                logical_name=logical_name,
                path=str(artifact_path),
            )

    for logical_name, path in (
        ("frozen_teacher_labels", labels_path),
        ("quarantine_manifest", quarantine_path),
        ("split_manifest", config.data_path("split_manifest")),
        ("field_weights", weight_path),
        ("anchor_labels", anchor_path),
        ("anchor_manifest", config.data_path("anchor_manifest")),
        ("baseline_report", config.data_path("baseline_report")),
        ("preprocessing_contract", config.data_path("preprocessing_contract")),
        ("package_manifest", package_path),
    ):
        inventory[logical_name] = {"path": str(path), "sha256": sha256_file(path)}
    inventory["frozen_teacher_labels"].update(
        {"rows": len(labels), "sample_id_unique": len(label_ids)}
    )
    inventory["quarantine_manifest"].update(
        {"rows": len(quarantine_index), "sample_id_unique": len(quarantine_index)}
    )
    inventory["field_weights"].update(
        {"rows": len(weights), "sample_id_unique": len(weights)}
    )
    inventory["anchor_labels"].update(
        {"rows": len(anchor_index), "sample_id_unique": len(anchor_index)}
    )
    return {
        "split_counts": {name: len(ids) for name, ids in by_split.items()},
        "evidence_violation_count": len(evidence_violations),
        "trainable_count": len(trainable_ids),
        "anchor_count": len(anchor_index),
    }


def audit_config(config: ProjectConfig) -> dict[str, Any]:
    inputs, inventory = _validate_inputs(config)
    schema_path = config.repo_path("schema_path")
    schema = LabelSchema.load(schema_path)
    inventory["schema"] = {
        "path": str(schema_path),
        "schema_version": schema.schema_version,
        "sha256": sha256_file(schema_path),
    }
    missing_artifacts = []
    blocker_codes = []
    for key, code, description in MISSING_REQUIREMENTS:
        path = config.data_path(key)
        if not path.is_file():
            missing_artifacts.append(
                {
                    "logical_name": key,
                    "description": description,
                    "path": str(path),
                    "blocker_code": code,
                }
            )
            blocker_codes.append(code)
    if missing_artifacts:
        result: dict[str, Any] = {
            "audit_schema_version": "myresearcher.semantic-data-audit.v1",
            "status": "BLOCKED_MISSING_CANONICAL_ARTIFACTS",
            "capability_maturity": "TESTED_WITH_SYNTHETIC_FIXTURES_ONLY",
            "training_allowed": False,
            "baseline_v0_3_5_reproduced": False,
            "blocker_codes": [
                "BLOCKED_MISSING_CANONICAL_ARTIFACTS",
                *_unique_in_order(blocker_codes),
            ],
            "missing_artifacts": missing_artifacts,
            "observed": inventory,
            "config": {
                "path": str(config.path),
                "sha256": sha256_file(config.path),
            },
        }
    else:
        validation_summary = _validate_complete_package(
            config, inputs, inventory, schema
        )
        result = {
            "audit_schema_version": "myresearcher.semantic-data-audit.v1",
            "status": "READY_FOR_BASELINE_REPRODUCTION",
            "capability_maturity": "TESTED_PENDING_REVIEWED_REAL_RUN",
            "training_allowed": True,
            "baseline_v0_3_5_reproduced": False,
            "blocker_codes": [],
            "missing_artifacts": [],
            "observed": inventory,
            "validation_summary": validation_summary,
            "config": {
                "path": str(config.path),
                "sha256": sha256_file(config.path),
            },
        }
    result["audit_id"] = content_addressed_id(result, omit_keys={"audit_id"})
    return result


def run_audit(path: str | Path) -> tuple[dict[str, Any], int]:
    try:
        result = audit_config(ProjectConfig.load(path))
    except ContractError as exc:
        result = {
            "audit_schema_version": "myresearcher.semantic-data-audit.v1",
            "status": "BLOCKED_INVALID_CANONICAL_ARTIFACTS",
            "capability_maturity": "TESTED_WITH_SYNTHETIC_FIXTURES_ONLY",
            "training_allowed": False,
            "baseline_v0_3_5_reproduced": False,
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }
        result["audit_id"] = content_addressed_id(result, omit_keys={"audit_id"})
        return result, 3
    return result, 0 if result["training_allowed"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only canonical data audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args(argv)
    result, exit_code = run_audit(args.config)
    serialized = f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
