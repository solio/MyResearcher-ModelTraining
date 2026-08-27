from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ProjectConfig
from .data import index_by_sample_id, join_inputs_and_labels, read_json, read_jsonl
from .errors import ContractError
from .hashes import sha256_file
from .preprocessing import PreprocessingContract
from .schema import SCHEMA_VERSION, V1_HEADS, LabelSchema
from .validation import validate_evidence_dependencies, validate_label_record
from .weighting import validate_field_weights


PACKAGE_FORMAT = "myresearcher.semantic-immutable-data.v0.3.5"
PACKAGE_MANIFEST_SCHEMA = "content-addressed-package-manifest-v1"
NATIVE_SPLITS = ("train", "embargo_1", "dev", "embargo_2", "test")
FORBIDDEN_EVIDENCE_LABELS = {
    "UNKNOWN",
    "NONE_EXPLICIT",
    "NO_ACTION_SIGNAL",
    "NO_REASON_GIVEN",
}


NATIVE_REQUIRED_ARTIFACTS = (
    "canonical_inputs",
    "frozen_teacher_labels",
    "repaired_teacher_labels",
    "trainable_teacher_labels",
    "quarantine_manifest",
    "split_manifest",
    "split_summary",
    "split_labels_train",
    "split_labels_embargo_1",
    "split_labels_dev",
    "split_labels_embargo_2",
    "split_labels_test",
    "field_weights",
    "anchor_labels",
    "anchor_manifest",
    "anchor_decision_manifest",
    "baseline_report",
    "baseline_report_markdown",
    "preprocessing_contract",
    "weighting_contract",
    "audit_expectations",
    "training_gate_manifest",
    "package_schema",
    "package_manifest",
    "package_manifest_sha256",
)


def is_native_package(config: ProjectConfig) -> bool:
    return config.raw["data"].get("package_format") == PACKAGE_FORMAT


def native_artifact_id(config: ProjectConfig, key: str) -> str:
    return sha256_file(config.data_path(key))


def _relative_payload_path(config: ProjectConfig, key: str) -> str:
    path = config.data_path(key).resolve()
    root = config.data_root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError(
            "PACKAGE_PATH_ESCAPE",
            "native package artifacts must remain under data.root",
            logical_name=key,
            path=str(path),
            root=str(root),
        ) from exc


def verify_content_manifest(config: ProjectConfig) -> dict[str, Any]:
    """Verify the extracted immutable package without mutating any payload byte."""

    root = config.data_root.resolve()
    if not root.is_dir():
        raise ContractError("PACKAGE_ROOT_NOT_FOUND", str(root))
    symlinks = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise ContractError(
            "PACKAGE_SYMLINK_FORBIDDEN",
            "immutable packages must not contain symbolic links",
            paths=symlinks,
        )
    manifest_path = config.data_path("package_manifest")
    checksum_path = config.data_path("package_manifest_sha256")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ContractError("CANONICAL_PACKAGE_INVALID", "manifest must be an object")
    if manifest.get("manifest_schema_version") != PACKAGE_MANIFEST_SCHEMA:
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID",
            "unsupported content manifest schema",
            observed=manifest.get("manifest_schema_version"),
        )
    if manifest.get("package_version") != "v0.3.5":
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID",
            "only the frozen v0.3.5 package is accepted",
            observed=manifest.get("package_version"),
        )
    manifest_sha256 = sha256_file(manifest_path)
    expected_pinned = config.raw["data"].get("expected_package_manifest_sha256")
    if expected_pinned != manifest_sha256:
        raise ContractError(
            "CANONICAL_PACKAGE_HASH_MISMATCH",
            "content manifest differs from the config-pinned content address",
            observed=manifest_sha256,
            expected=expected_pinned,
        )
    checksum_text = checksum_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})  CONTENT_MANIFEST\.json", checksum_text)
    if match is None or match.group(1) != manifest_sha256:
        raise ContractError(
            "CANONICAL_PACKAGE_HASH_MISMATCH",
            "CONTENT_MANIFEST.sha256 does not address CONTENT_MANIFEST.json",
            observed=checksum_text,
            expected=manifest_sha256,
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not all(
        isinstance(entry, Mapping) for entry in entries
    ):
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID", "manifest.files must be a list of objects"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    total_bytes = 0
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ContractError(
                "CANONICAL_PACKAGE_INVALID", "every payload requires a relative path"
            )
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise ContractError(
                "PACKAGE_PATH_ESCAPE", "unsafe path in content manifest", path=relative
            )
        if relative in indexed:
            raise ContractError(
                "CANONICAL_PACKAGE_INVALID",
                "duplicate path in content manifest",
                path=relative,
            )
        indexed[relative] = entry
        path = root / relative
        if not path.is_file():
            raise ContractError(
                "ARTIFACT_NOT_FOUND", "manifest payload is missing", path=str(path)
            )
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if entry.get("sha256") != actual_hash or entry.get("size_bytes") != actual_size:
            raise ContractError(
                "CANONICAL_PACKAGE_HASH_MISMATCH",
                "payload bytes differ from CONTENT_MANIFEST.json",
                path=relative,
                expected_sha256=entry.get("sha256"),
                observed_sha256=actual_hash,
                expected_size=entry.get("size_bytes"),
                observed_size=actual_size,
            )
        total_bytes += actual_size
    if manifest.get("payload_file_count") != len(indexed):
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID",
            "payload_file_count differs from manifest entries",
        )
    if manifest.get("payload_total_bytes") != total_bytes:
        raise ContractError(
            "CANONICAL_PACKAGE_INVALID",
            "payload_total_bytes differs from verified bytes",
        )
    actual_payloads = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"CONTENT_MANIFEST.json", "CONTENT_MANIFEST.sha256"}
    }
    if actual_payloads != set(indexed):
        raise ContractError(
            "CANONICAL_PACKAGE_FILE_SET_MISMATCH",
            "extracted payload file set differs from the content manifest",
            missing=sorted(set(indexed) - actual_payloads),
            extra=sorted(actual_payloads - set(indexed)),
        )
    for key in NATIVE_REQUIRED_ARTIFACTS:
        relative = _relative_payload_path(config, key)
        if key not in {"package_manifest", "package_manifest_sha256"} and relative not in indexed:
            raise ContractError(
                "CANONICAL_PACKAGE_INVALID",
                "configured canonical artifact is absent from the content manifest",
                logical_name=key,
                path=relative,
            )
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "payload_file_count": len(indexed),
        "payload_total_bytes": total_bytes,
        "package_version": manifest["package_version"],
    }


def _expected_count(config: ProjectConfig, key: str) -> int:
    value = config.expected.get(key)
    if value is None:
        raise ContractError(
            "CONFIG_INVALID", "native package expected count is required", key=key
        )
    return int(value)


def _split_for_timestamp(published_at: str) -> str:
    calendar_date = published_at[:10]
    if calendar_date <= "2026-07-30":
        return "train"
    if calendar_date == "2026-07-31":
        return "embargo_1"
    if "2026-08-01" <= calendar_date <= "2026-08-06":
        return "dev"
    if calendar_date == "2026-08-07":
        return "embargo_2"
    return "test"


def _reference_field_weight(
    label: Mapping[str, Any], weight_row: Mapping[str, Any], head: str
) -> float:
    base = {"HIGH": 1.0, "MEDIUM": 0.72, "LOW": 0.42}.get(
        label.get("label_confidence")
    )
    if base is None:
        raise ContractError(
            "FIELD_WEIGHT_CONTRACT_VIOLATION",
            "unknown label confidence in reference weighting",
            sample_id=label.get("sample_id"),
        )
    if "v0.3.3-production-harness-gate" in str(label.get("prompt_version", "")):
        return base * 1.1
    raw_batch = weight_row.get("production_batch")
    batch = (
        int(raw_batch[1:])
        if isinstance(raw_batch, str)
        and re.fullmatch(r"P\d{3}", raw_batch)
        else 0
    )
    if head in {"emotion_primary", "emotion_target"}:
        if 61 <= batch <= 96 or 133 <= batch <= 144:
            return base * 0.30
        if 25 <= batch <= 36 or 49 <= batch <= 60 or 97 <= batch <= 108:
            return base * 0.65
    if head == "action_tendency" and 49 <= batch <= 96:
        return base * 0.55
    if head == "stance" and 133 <= batch <= 144:
        return base * 0.55
    return base


def load_native_split_ids(config: ProjectConfig) -> dict[str, list[str]]:
    rows = read_jsonl(config.data_path("split_manifest"))
    result = {name: [] for name in NATIVE_SPLITS}
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        split = row.get("split")
        if not isinstance(sample_id, str) or split not in result or sample_id in seen:
            raise ContractError(
                "SPLIT_MANIFEST_INVALID",
                "native split rows require unique sample_id and a known split",
                sample_id=sample_id,
                split=split,
            )
        seen.add(sample_id)
        result[str(split)].append(sample_id)
    return result


def load_native_quarantine(config: ProjectConfig) -> list[dict[str, Any]]:
    return read_jsonl(config.data_path("quarantine_manifest"))


def load_native_trainable_labels(config: ProjectConfig) -> list[dict[str, Any]]:
    return read_jsonl(config.data_path("trainable_teacher_labels"))


def normalize_anchor_for_validation(anchor: Mapping[str, Any]) -> dict[str, Any]:
    if anchor.get("schema_version") != "semantic-schema-v0.2.1":
        raise ContractError(
            "SCHEMA_VERSION_MISMATCH",
            "Anchor50 must retain its frozen v0.2.1 source schema marker",
            sample_id=anchor.get("sample_id"),
            observed=anchor.get("schema_version"),
        )
    normalized = dict(anchor)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["label_confidence"] = anchor.get("expert_confidence")
    return normalized


def _validate_reproduction_config(
    config: ProjectConfig, preprocessing: PreprocessingContract
) -> None:
    model = config.raw.get("model")
    if not isinstance(model, Mapping):
        raise ContractError("MODEL_CONFIG_INVALID", "model config must be an object")
    expected_char = dict(preprocessing.char_tfidf)
    expected_word = dict(preprocessing.word_tfidf)
    for expected in (expected_char, expected_word):
        expected["dtype"] = str(expected["dtype"])
        expected["ngram_range"] = list(expected["ngram_range"])
    observed_char = dict(model.get("char_tfidf", {}))
    observed_word = dict(model.get("word_tfidf", {}))
    if observed_char != expected_char or observed_word != expected_word:
        raise ContractError(
            "MODEL_CONFIG_REPRODUCTION_MISMATCH",
            "TF-IDF config differs from the immutable preprocessing contract",
            observed={"char_tfidf": observed_char, "word_tfidf": observed_word},
            expected={"char_tfidf": expected_char, "word_tfidf": expected_word},
        )
    if model.get("feature_stack_order") != list(preprocessing.feature_stack_order):
        raise ContractError(
            "MODEL_CONFIG_REPRODUCTION_MISMATCH",
            "feature stack order differs from the immutable contract",
        )
    expected_scalar = {
        "C": 3.0,
        "max_iter": 2000,
        "class_weight": "balanced",
        "solver": "saga",
        "random_state": 35,
    }
    expected_reasoning = {
        "C": 3.0,
        "max_iter": 1600,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": 35,
    }
    if model.get("scalar_logistic_regression") != expected_scalar or model.get(
        "reasoning_logistic_regression"
    ) != expected_reasoning:
        raise ContractError(
            "MODEL_CONFIG_REPRODUCTION_MISMATCH",
            "Logistic Regression config differs from the reference implementation",
        )
    calibration = config.raw.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("method") != "reference-v0.3.5":
        raise ContractError(
            "MODEL_CONFIG_REPRODUCTION_MISMATCH",
            "reference-v0.3.5 threshold calibration is required",
        )


def audit_native_package(
    config: ProjectConfig,
    inputs: list[dict[str, Any]],
    inventory: dict[str, Any],
    schema: LabelSchema,
) -> dict[str, Any]:
    inventory["package_manifest"] = verify_content_manifest(config)
    input_index = index_by_sample_id(inputs, role="canonical-input")

    package_schema = LabelSchema.load(config.data_path("package_schema"))
    if package_schema.class_order != schema.class_order:
        raise ContractError(
            "SCHEMA_CLASS_ORDER_MISMATCH",
            "repository and package Schema class orders differ",
        )

    frozen = read_jsonl(config.data_path("frozen_teacher_labels"))
    if len(frozen) != _expected_count(config, "frozen_labels"):
        raise ContractError("LABEL_COVERAGE_MISMATCH", "frozen label count differs")
    for label in frozen:
        validate_label_record(label, schema)
    joined = join_inputs_and_labels(inputs, frozen, require_complete=True)
    frozen_index = index_by_sample_id(frozen, role="frozen-label")

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

    quarantine = load_native_quarantine(config)
    quarantine_index = index_by_sample_id(quarantine, role="quarantine")
    if len(quarantine) != _expected_count(config, "quarantine"):
        raise ContractError("QUARANTINE_COUNT_MISMATCH", "quarantine count differs")
    if set(quarantine_index) != set(evidence_violations):
        raise ContractError(
            "QUARANTINE_EVIDENCE_MISMATCH",
            "quarantine identities must exactly match Evidence violations",
            missing=sorted(set(evidence_violations) - set(quarantine_index)),
            extra=sorted(set(quarantine_index) - set(evidence_violations)),
        )
    for sample_id, row in quarantine_index.items():
        if row.get("original_label") != frozen_index[sample_id]:
            raise ContractError(
                "QUARANTINE_PROVENANCE_MISMATCH",
                "quarantine original_label differs from the frozen source",
                sample_id=sample_id,
            )
        if row.get("model_text") != input_index[sample_id].get("model_text"):
            raise ContractError(
                "QUARANTINE_PROVENANCE_MISMATCH",
                "quarantine text differs from canonical input",
                sample_id=sample_id,
            )

    repaired = read_jsonl(config.data_path("repaired_teacher_labels"))
    if len(repaired) != len(frozen):
        raise ContractError("PROTOCOL_REPAIR_MISMATCH", "repaired row count differs")
    removed = 0
    for source, repaired_row in zip(frozen, repaired, strict=True):
        expected = dict(source)
        evidence = source.get("evidence_spans")
        if not isinstance(evidence, list):
            raise ContractError(
                "EVIDENCE_SHAPE_INVALID", "native frozen Evidence must be an object list"
            )
        filtered = [
            item for item in evidence if item.get("label") not in FORBIDDEN_EVIDENCE_LABELS
        ]
        removed += len(evidence) - len(filtered)
        expected["evidence_spans"] = filtered
        if repaired_row != expected:
            raise ContractError(
                "PROTOCOL_REPAIR_MISMATCH",
                "protocol repair changed bytes beyond forbidden Evidence removal",
                sample_id=source.get("sample_id"),
            )
        validate_evidence_dependencies(input_index[str(source["sample_id"])], repaired_row)
    if removed != _expected_count(config, "quarantine"):
        raise ContractError(
            "PROTOCOL_REPAIR_MISMATCH",
            "protocol repair must remove exactly 21 Evidence objects",
            observed=removed,
        )

    trainable = load_native_trainable_labels(config)
    expected_trainable = [
        row for row in repaired if row["sample_id"] not in quarantine_index
    ]
    if trainable != expected_trainable or len(trainable) != _expected_count(
        config, "trainable"
    ):
        raise ContractError(
            "TRAINABLE_VIEW_MISMATCH",
            "trainable labels must be repaired labels minus quarantine, in source order",
        )
    trainable_index = index_by_sample_id(trainable, role="trainable-label")

    split_rows = read_jsonl(config.data_path("split_manifest"))
    split_ids = load_native_split_ids(config)
    split_row_index = index_by_sample_id(split_rows, role="split-row")
    if set(split_row_index) != set(trainable_index):
        raise ContractError(
            "SPLIT_COVERAGE_MISMATCH",
            "native split identities must exactly match trainable identities",
        )
    for sample_id, row in split_row_index.items():
        canonical = input_index[sample_id]
        if row.get("published_at") != canonical.get("published_at"):
            raise ContractError(
                "CANONICAL_METADATA_MISMATCH",
                "split timestamp differs from canonical input",
                sample_id=sample_id,
            )
        if row.get("calendar_date_literal") != str(canonical["published_at"])[:10]:
            raise ContractError(
                "SPLIT_TIME_SOURCE_INVALID",
                "calendar_date_literal must be published_at[:10] without timezone conversion",
                sample_id=sample_id,
            )
        if row.get("split") != _split_for_timestamp(str(canonical["published_at"])):
            raise ContractError(
                "SPLIT_POLICY_MISMATCH",
                "split assignment differs from the frozen calendar-day policy",
                sample_id=sample_id,
            )
    expected_split = config.expected.get("split")
    if not isinstance(expected_split, Mapping):
        raise ContractError("CONFIG_INVALID", "data.expected.split is required")
    actual_split = {name: len(ids) for name, ids in split_ids.items()}
    normalized_expected = {name: int(expected_split[name]) for name in NATIVE_SPLITS}
    if actual_split != normalized_expected:
        raise ContractError(
            "SPLIT_COUNT_MISMATCH",
            "native split counts differ from the frozen config",
            observed=actual_split,
            expected=normalized_expected,
        )
    split_summary = read_json(config.data_path("split_summary"))
    if not isinstance(split_summary, Mapping) or split_summary.get("counts") != {
        name: actual_split[name] for name in sorted(actual_split)
    }:
        raise ContractError(
            "SPLIT_SUMMARY_MISMATCH", "split summary counts differ from split rows"
        )
    trainable_order = [row["sample_id"] for row in trainable]
    for split in NATIVE_SPLITS:
        split_labels = read_jsonl(config.data_path(f"split_labels_{split}"))
        expected_rows = [
            trainable_index[sample_id]
            for sample_id in trainable_order
            if split_row_index[sample_id]["split"] == split
        ]
        if split_labels != expected_rows:
            raise ContractError(
                "SPLIT_LABEL_VIEW_MISMATCH",
                "split label file differs from the trainable source view",
                split=split,
            )

    weights = read_jsonl(config.data_path("field_weights"))
    weights_by_id = validate_field_weights(weights, expected_ids=set(trainable_index))
    for sample_id, row in index_by_sample_id(weights, role="field-weight").items():
        if row.get("split") != split_row_index[sample_id].get("split"):
            raise ContractError(
                "FIELD_WEIGHT_CONTRACT_VIOLATION",
                "field-weight split differs from the frozen split",
                sample_id=sample_id,
            )
        for head in V1_HEADS:
            expected_weight = _reference_field_weight(
                trainable_index[sample_id], row, head
            )
            observed_weight = weights_by_id[sample_id][head]
            if not math.isclose(
                observed_weight, expected_weight, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ContractError(
                    "FIELD_WEIGHT_CONTRACT_VIOLATION",
                    "explicit head weight differs from the reference weighting function",
                    sample_id=sample_id,
                    head=head,
                    observed=observed_weight,
                    expected=expected_weight,
                )
        if row.get("label_confidence") != trainable_index[sample_id].get(
            "label_confidence"
        ):
            raise ContractError(
                "FIELD_WEIGHT_CONTRACT_VIOLATION",
                "field-weight confidence differs from the trainable label",
                sample_id=sample_id,
            )

    anchors = read_jsonl(config.data_path("anchor_labels"))
    if len(anchors) != _expected_count(config, "anchor"):
        raise ContractError("ANCHOR_COUNT_MISMATCH", "Anchor50 count differs")
    for anchor in anchors:
        validate_label_record(
            normalize_anchor_for_validation(anchor),
            schema,
            allow_anchor_reasoning_sentinel_combinations=True,
        )
    anchor_index = index_by_sample_id(anchors, role="anchor")
    if set(anchor_index) & set(frozen_index):
        raise ContractError(
            "SPLIT_IDENTITY_LEAKAGE", "Anchor50 overlaps Teacher3000 identities"
        )
    source_counts = Counter(str(row.get("adjudication_source")) for row in anchors)
    expected_provenance = config.expected.get("anchor_provenance")
    if source_counts != Counter(expected_provenance):
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor50 row provenance differs from the frozen contract",
            observed=dict(source_counts),
            expected=expected_provenance,
        )
    anchor_manifest = read_json(config.data_path("anchor_manifest"))
    if not isinstance(anchor_manifest, Mapping) or anchor_manifest.get(
        "source_counts"
    ) != dict(source_counts):
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor50 provenance manifest differs from label rows",
        )
    if anchor_manifest.get("sample_id_overlap_with_teacher3000") != 0:
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH", "Anchor50 overlap declaration must be zero"
        )
    anchor_decision = read_json(config.data_path("anchor_decision_manifest"))
    if not isinstance(anchor_decision, Mapping) or {
        "total": anchor_decision.get("total"),
        "human_confirmed": anchor_decision.get("human_confirmed"),
        "expert_weak_gold": anchor_decision.get("expert_weak_gold"),
    } != {"total": 50, "human_confirmed": 11, "expert_weak_gold": 39}:
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor50 decision counts differ from the frozen handoff",
        )
    source_refs = anchor_manifest.get("sources")
    if not isinstance(source_refs, Mapping) or source_refs.get("labels", {}).get(
        "sha256"
    ) != sha256_file(config.data_path("anchor_labels")) or source_refs.get(
        "decision_manifest", {}
    ).get("sha256") != sha256_file(config.data_path("anchor_decision_manifest")):
        raise ContractError(
            "ANCHOR_PROVENANCE_MISMATCH",
            "Anchor50 provenance hashes do not address the embedded sources",
        )

    weighting_contract = read_json(config.data_path("weighting_contract"))
    if not isinstance(weighting_contract, Mapping) or weighting_contract.get(
        "heads"
    ) != list(V1_HEADS):
        raise ContractError(
            "FIELD_WEIGHT_CONTRACT_VIOLATION",
            "weighting contract head order differs from the frozen Schema",
        )
    audit_expectations = read_json(config.data_path("audit_expectations"))
    verified_counts = {
        "teacher_inputs": len(inputs),
        "teacher_labels_frozen": len(frozen),
        "teacher_labels_protocol_repaired": len(repaired),
        "quarantine": len(quarantine),
        "trainable": len(trainable),
        "field_weight_rows": len(weights),
        "field_weights_per_row": len(V1_HEADS),
        "split_train": actual_split["train"],
        "split_dev": actual_split["dev"],
        "split_test": actual_split["test"],
        "split_embargo_1": actual_split["embargo_1"],
        "split_embargo_2": actual_split["embargo_2"],
        "anchor50": len(anchors),
    }
    if not isinstance(audit_expectations, Mapping) or audit_expectations.get(
        "required_counts"
    ) != verified_counts:
        raise ContractError(
            "AUDIT_EXPECTATIONS_MISMATCH",
            "machine-readable audit expectations differ from verified data",
        )
    training_gate = read_json(config.data_path("training_gate_manifest"))
    if not isinstance(training_gate, Mapping):
        raise ContractError("TRAINING_GATE_INVALID", "training gate must be an object")
    declared_hashes = {
        "frozen_teacher_labels": training_gate.get("immutable_source", {}).get("sha256"),
        "repaired_teacher_labels": training_gate.get("protocol_repaired", {}).get("sha256"),
        "trainable_teacher_labels": training_gate.get("trainable", {}).get("sha256"),
    }
    for key, declared in declared_hashes.items():
        if declared != sha256_file(config.data_path(key)):
            raise ContractError(
                "TRAINING_GATE_HASH_MISMATCH",
                "training gate hash differs from canonical payload",
                logical_name=key,
            )
    if training_gate.get("immutable_source", {}).get("rows") != len(
        frozen
    ) or training_gate.get("protocol_repaired", {}) != {
        "rows": len(repaired),
        "sha256": sha256_file(config.data_path("repaired_teacher_labels")),
        "semantic_labels_changed": 0,
        "evidence_objects_removed": removed,
    } or training_gate.get("trainable", {}) != {
        "rows": len(trainable),
        "sha256": sha256_file(config.data_path("trainable_teacher_labels")),
    } or training_gate.get("quarantine", {}).get("sample_ids") != sorted(
        quarantine_index
    ):
        raise ContractError(
            "TRAINING_GATE_HASH_MISMATCH",
            "training gate counts/relations differ from verified canonical views",
        )

    preprocessing = PreprocessingContract.load(config.data_path("preprocessing_contract"))
    _validate_reproduction_config(config, preprocessing)
    baseline = read_json(config.data_path("baseline_report"))
    if not isinstance(baseline, Mapping) or baseline.get("status") != "DIAGNOSTIC_BASELINE_COMPLETE":
        raise ContractError(
            "BASELINE_REPORT_INVALID", "baseline metrics status is not complete"
        )
    if baseline.get("rows") != {
        "train": actual_split["train"],
        "dev": actual_split["dev"],
        "test": actual_split["test"],
        "anchor50": len(anchors),
    }:
        raise ContractError(
            "BASELINE_REPORT_INVALID", "baseline row counts differ from verified data"
        )
    if baseline.get("features") != dict(preprocessing.expected_feature_counts):
        raise ContractError(
            "BASELINE_REPORT_INVALID",
            "baseline feature counts differ from preprocessing contract",
        )
    reference_environment_present = isinstance(
        baseline.get("reference_environment"), Mapping
    )

    row_counts = {
        "canonical_inputs": len(inputs),
        "frozen_teacher_labels": len(frozen),
        "repaired_teacher_labels": len(repaired),
        "trainable_teacher_labels": len(trainable),
        "quarantine_manifest": len(quarantine),
        "split_manifest": len(split_rows),
        "field_weights": len(weights_by_id),
        "anchor_labels": len(anchors),
    }
    for key in NATIVE_REQUIRED_ARTIFACTS:
        if key in {"package_manifest", "package_manifest_sha256"}:
            continue
        path = config.data_path(key)
        inventory.setdefault(key, {}).update(
            {"path": str(path), "sha256": sha256_file(path)}
        )
        if key in row_counts:
            inventory[key]["rows"] = row_counts[key]
    return {
        "package_format": PACKAGE_FORMAT,
        "package_manifest_id": inventory["package_manifest"]["sha256"],
        "split_manifest_id": native_artifact_id(config, "split_manifest"),
        "quarantine_manifest_id": native_artifact_id(config, "quarantine_manifest"),
        "anchor_manifest_id": native_artifact_id(config, "anchor_manifest"),
        "preprocessing_contract_id": preprocessing.contract_id,
        "split_counts": actual_split,
        "evidence_violation_count": len(evidence_violations),
        "protocol_evidence_objects_removed": removed,
        "trainable_count": len(trainable_index),
        "anchor_count": len(anchor_index),
        "verified_payload_files": inventory["package_manifest"]["payload_file_count"],
        "reference_environment_present": reference_environment_present,
        "reproduction_claim_allowed": reference_environment_present,
        "reproduction_blocker_codes": (
            []
            if reference_environment_present
            else ["BLOCKED_MISSING_REFERENCE_ENVIRONMENT"]
        ),
    }
