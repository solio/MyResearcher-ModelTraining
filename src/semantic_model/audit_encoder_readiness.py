"""Read-only data and hardware readiness audit for the future Encoder milestone.

This module deliberately has no Encoder-runtime dependency.  It validates the
immutable v0.3.5 package first, then reports only observed source-text and
label statistics.  It never loads a tokenizer, model weight, or accelerator
runtime and never writes a training or data artifact.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ProjectConfig
from .data import index_by_sample_id, read_json, read_jsonl
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, without_local_paths
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema
from .validation import validate_label_record
from .weighting import validate_field_weights


READINESS_SCHEMA_VERSION = "myresearcher.encoder-readiness-audit.v1"
MILESTONE_STATUS = "MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL"
BLOCKED_DATA_STATUS = "MILESTONE_1B_BLOCKED_MISSING_DATA_EVIDENCE"
SUPPORT_REPORTING_FLOOR = 20

# These ranges are intentionally a deterministic observation heuristic, not an
# assertion that a code point is semantically an emoji in its particular text.
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
)

# A conservative subset of characters that are usually Traditional-only in
# modern Chinese.  It is an observation aid, never a language classifier.
TRADITIONAL_INDICATOR_CHARS = frozenset(
    "萬與專業東絲兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧雲亞產畝親億僅從侖倉儀們價儲兒兌黨蘭關興養獸內岡冊寫軍農衝凍凱擊則剛剝動勞勢勵勁勻區醫華協單賣衛卻廠厭廂壓厲參雙發變臺葉號嘆嚇呂嗎噸聽啟員啞問"
)

DATA_REQUIRED_KEYS = (
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


def _sorted_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _checked_manifest_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("CANONICAL_PACKAGE_INVALID", "manifest path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ContractError("PACKAGE_PATH_ESCAPE", "unsafe content manifest path", path=value)
    return value


def _validate_content_addressed_package(
    root: Path,
    *,
    expected_manifest_sha256: str,
    package_role: str,
) -> dict[str, Any]:
    """Verify payload bytes without importing or executing supplied artifacts."""

    if not root.is_dir():
        raise ContractError("PACKAGE_ROOT_NOT_FOUND", f"{package_role}: {root}")
    symlinks = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise ContractError(
            "PACKAGE_SYMLINK_FORBIDDEN",
            f"{package_role} contains symbolic links",
            paths=symlinks,
        )
    manifest_path = root / "CONTENT_MANIFEST.json"
    checksum_path = root / "CONTENT_MANIFEST.sha256"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ContractError("CANONICAL_PACKAGE_INVALID", "content manifest must be an object")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ContractError(
            "CANONICAL_PACKAGE_HASH_MISMATCH",
            f"{package_role} content manifest differs from its pinned hash",
            observed=manifest_sha256,
            expected=expected_manifest_sha256,
        )
    expected_checksum = f"{manifest_sha256}  CONTENT_MANIFEST.json"
    if checksum_path.read_text(encoding="utf-8").strip() != expected_checksum:
        raise ContractError(
            "CANONICAL_PACKAGE_HASH_MISMATCH",
            f"{package_role} manifest checksum sidecar is invalid",
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(row, Mapping) for row in files):
        raise ContractError("CANONICAL_PACKAGE_INVALID", "manifest.files must be object rows")
    expected_paths: set[str] = set()
    verified_bytes = 0
    for entry in files:
        relative = _checked_manifest_path(entry.get("path"))
        if relative in expected_paths:
            raise ContractError("CANONICAL_PACKAGE_INVALID", "duplicate manifest path", path=relative)
        expected_paths.add(relative)
        payload = root / relative
        if not payload.is_file():
            raise ContractError("ARTIFACT_NOT_FOUND", "manifest payload is missing", path=relative)
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_hash, str) or not isinstance(expected_size, int):
            raise ContractError("CANONICAL_PACKAGE_INVALID", "manifest payload metadata is invalid", path=relative)
        observed_hash = sha256_file(payload)
        observed_size = payload.stat().st_size
        if observed_hash != expected_hash or observed_size != expected_size:
            raise ContractError(
                "CANONICAL_PACKAGE_HASH_MISMATCH",
                f"{package_role} payload differs from manifest",
                path=relative,
            )
        verified_bytes += observed_size
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"CONTENT_MANIFEST.json", "CONTENT_MANIFEST.sha256"}
    }
    if actual_paths != expected_paths:
        raise ContractError(
            "CANONICAL_PACKAGE_FILE_SET_MISMATCH",
            f"{package_role} file set differs from content manifest",
            missing=sorted(expected_paths - actual_paths),
            extra=sorted(actual_paths - expected_paths),
        )
    if (
        manifest.get("payload_file_count") != len(expected_paths)
        or manifest.get("payload_total_bytes") != verified_bytes
    ):
        raise ContractError("CANONICAL_PACKAGE_TOTAL_MISMATCH", "manifest totals differ")
    return {
        "manifest_sha256": manifest_sha256,
        "payload_file_count": len(expected_paths),
        "payload_total_bytes": verified_bytes,
    }


def _require_count(records: Sequence[Mapping[str, Any]], expected: int, name: str) -> None:
    if len(records) != expected:
        raise ContractError(
            "ENCODER_READINESS_COUNT_MISMATCH",
            f"{name} count differs from frozen contract",
            actual=len(records),
            expected=expected,
        )


def _validate_immutable_data_roles(config: ProjectConfig) -> dict[str, Any]:
    data = config.raw.get("data")
    if not isinstance(data, Mapping) or data.get("package_format") != "myresearcher.semantic-immutable-data.v0.3.5":
        raise ContractError("CONFIG_INVALID", "Milestone 1B audit requires the native immutable v0.3.5 package")
    for key in DATA_REQUIRED_KEYS:
        if not config.data_path(key).is_file():
            raise ContractError("ARTIFACT_NOT_FOUND", "required immutable artifact is missing", logical_name=key)
    data_package = _validate_content_addressed_package(
        config.data_root,
        expected_manifest_sha256=str(data.get("expected_package_manifest_sha256", "")),
        package_role="immutable data package",
    )
    reference_config = config.raw.get("baseline_reference")
    if not isinstance(reference_config, Mapping):
        raise ContractError("REFERENCE_CONFIG_INVALID", "baseline_reference contract is required")
    reference_root_raw = reference_config.get("root")
    if not isinstance(reference_root_raw, str) or not reference_root_raw:
        raise ContractError("REFERENCE_CONFIG_INVALID", "baseline_reference.root is required")
    reference_root = Path(reference_root_raw)
    if not reference_root.is_absolute():
        reference_root = config.project_root / reference_root
    reference_package = _validate_content_addressed_package(
        reference_root,
        expected_manifest_sha256=str(reference_config.get("expected_package_manifest_sha256", "")),
        package_role="immutable baseline-reference package",
    )

    schema = LabelSchema.load(config.repo_path("schema_path"))
    inputs = read_jsonl(config.data_path("canonical_inputs"))
    frozen_labels = read_jsonl(config.data_path("frozen_teacher_labels"))
    repaired_labels = read_jsonl(config.data_path("repaired_teacher_labels"))
    trainable_labels = read_jsonl(config.data_path("trainable_teacher_labels"))
    quarantine_rows = read_jsonl(config.data_path("quarantine_manifest"))
    anchors = read_jsonl(config.data_path("anchor_labels"))
    expected = config.expected
    _require_count(inputs, int(expected["inputs"]), "canonical inputs")
    _require_count(frozen_labels, int(expected["frozen_labels"]), "frozen labels")
    _require_count(repaired_labels, int(expected["frozen_labels"]), "repaired labels")
    _require_count(trainable_labels, int(expected["trainable"]), "trainable weak labels")
    _require_count(quarantine_rows, int(expected["quarantine"]), "quarantine")
    _require_count(anchors, int(expected["anchor"]), "Anchor50")
    inputs_by_id = index_by_sample_id(inputs, role="canonical-input")
    frozen_by_id = index_by_sample_id(frozen_labels, role="frozen-teacher-label")
    repaired_by_id = index_by_sample_id(repaired_labels, role="repaired-teacher-label")
    labels_by_id = index_by_sample_id(trainable_labels, role="trainable-weak-label")
    quarantine_by_id = index_by_sample_id(quarantine_rows, role="quarantine")
    anchor_by_id = index_by_sample_id(anchors, role="anchor")
    if set(frozen_by_id) != set(inputs_by_id) or set(repaired_by_id) != set(inputs_by_id):
        raise ContractError("LABEL_COVERAGE_MISMATCH", "frozen/repaired labels must cover canonical inputs")
    if set(labels_by_id) != set(frozen_by_id) - set(quarantine_by_id):
        raise ContractError("TRAINABLE_COUNT_MISMATCH", "trainable labels must equal frozen labels minus quarantine")
    if set(anchor_by_id) & set(inputs_by_id):
        raise ContractError("SPLIT_IDENTITY_LEAKAGE", "Anchor50 overlaps Teacher3000")
    for record in frozen_labels:
        validate_label_record(record, schema)
    for record in trainable_labels:
        validate_label_record(record, schema)
    for record in inputs:
        missing = [
            field
            for field in ("sample_id", "stock_code", "stock_name", "published_at", "model_text")
            if not isinstance(record.get(field), str) or not record[field]
        ]
        if missing:
            raise ContractError("CANONICAL_INPUT_INVALID", "required canonical input field missing", fields=missing)

    split_rows = read_jsonl(config.data_path("split_manifest"))
    splits_by_id = index_by_sample_id(split_rows, role="split-manifest")
    if set(splits_by_id) != set(labels_by_id):
        raise ContractError("SPLIT_COVERAGE_MISMATCH", "frozen split does not cover trainable labels")
    expected_splits = expected["split"]
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "embargo_1", "dev", "embargo_2", "test"):
        split_ids[split] = {
            sample_id for sample_id, row in splits_by_id.items() if row.get("split") == split
        }
        if len(split_ids[split]) != int(expected_splits[split]):
            raise ContractError("SPLIT_COUNT_MISMATCH", "frozen split count differs", split=split)
        split_label_rows = read_jsonl(config.data_path(f"split_labels_{split}"))
        if set(index_by_sample_id(split_label_rows, role=f"split-label-{split}")) != split_ids[split]:
            raise ContractError("SPLIT_COVERAGE_MISMATCH", "split label artifact differs from split manifest", split=split)
    if len(set().union(*split_ids.values())) != len(labels_by_id):
        raise ContractError("SPLIT_IDENTITY_LEAKAGE", "duplicate or unknown frozen split assignment")

    weight_rows = read_jsonl(config.data_path("field_weights"))
    weight_records_by_id = index_by_sample_id(weight_rows, role="field-weight")
    validate_field_weights(weight_rows, expected_ids=set(labels_by_id))
    anchor_manifest = read_json(config.data_path("anchor_manifest"))
    if not isinstance(anchor_manifest, Mapping):
        raise ContractError("ANCHOR_PROVENANCE_MISMATCH", "anchor manifest must be an object")
    observed_source_counts = _sorted_counter(
        str(record.get("adjudication_source", "MISSING")) for record in anchors
    )
    if anchor_manifest.get("source_counts") != observed_source_counts:
        raise ContractError("ANCHOR_PROVENANCE_MISMATCH", "Anchor50 source counts disagree")
    return {
        "schema_version": schema.schema_version,
        "data_package": data_package,
        "reference_package": reference_package,
        "inputs": inputs,
        "inputs_by_id": inputs_by_id,
        "trainable_labels": trainable_labels,
        "labels_by_id": labels_by_id,
        "splits_by_id": splits_by_id,
        "split_ids": split_ids,
        "weights_by_id": weight_records_by_id,
        "anchor_source_counts": observed_source_counts,
    }


def _nearest_rank(values: Sequence[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered) / 100) - 1)
    return ordered[index]


def _length_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": _nearest_rank(values, 50),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "max": max(values),
    }


def _contains_emoji_codepoint(text: str) -> bool:
    return any(
        lower <= ord(character) <= upper
        for character in text
        for lower, upper in EMOJI_RANGES
    )


def _contains_han(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _contains_ascii_latin(text: str) -> bool:
    return any(("a" <= character <= "z") or ("A" <= character <= "Z") for character in text)


def _contains_url_marker(text: str) -> bool:
    normalized = text.casefold()
    return "http://" in normalized or "https://" in normalized or "www." in normalized


def _contains_control_or_format(text: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in text)


def _text_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    type_counts = _sorted_counter(
        type(record.get("model_text")).__name__ for record in records
    )
    texts = [record["model_text"] for record in records if isinstance(record.get("model_text"), str)]
    char_lengths = [len(text) for text in texts]
    byte_lengths = [len(text.encode("utf-8")) for text in texts]
    return {
        "population": "all_3000_canonical_inputs_including_quarantine",
        "records": len(records),
        "model_text_type_counts": type_counts,
        "empty_string_count": sum(text == "" for text in texts),
        "whitespace_only_count": sum(bool(text) and not text.strip() for text in texts),
        "non_string_model_text_count": len(records) - len(texts),
        "character_length": _length_summary(char_lengths),
        "utf8_byte_length": _length_summary(byte_lengths),
        "observations": {
            "url_marker_count": sum(_contains_url_marker(text) for text in texts),
            "emoji_codepoint_count": sum(_contains_emoji_codepoint(text) for text in texts),
            "han_ascii_latin_mixed_count": sum(
                _contains_han(text) and _contains_ascii_latin(text) for text in texts
            ),
            "traditional_character_indicator_count": sum(
                any(character in TRADITIONAL_INDICATOR_CHARS for character in text)
                for text in texts
            ),
            "control_or_format_character_count": sum(
                _contains_control_or_format(text) for text in texts
            ),
        },
        "observation_method_notes": {
            "url_marker_count": "case-folded substring markers: http://, https://, or www.",
            "emoji_codepoint_count": "fixed Unicode code-point ranges; not a semantic emoji judgement",
            "han_ascii_latin_mixed_count": "at least one CJK Unified Ideograph and one ASCII Latin letter",
            "traditional_character_indicator_count": "conservative Traditional-character indicator set; not a script classifier",
            "control_or_format_character_count": "Unicode general categories Cc or Cf",
        },
        "token_length_audit": {
            "status": "BLOCKED_TOKENIZER_ARTIFACT_NOT_AVAILABLE",
            "reason": "Milestone 1B does not download or load a tokenizer; no token-length statistic is fabricated.",
        },
    }


def _label_distribution(
    labels: Sequence[Mapping[str, Any]], schema: LabelSchema
) -> dict[str, Any]:
    scalar = {
        head: {
            label: sum(record.get(head) == label for record in labels)
            for label in schema.class_order[head]
        }
        for head in SINGLE_LABEL_HEADS
    }
    reasoning_order = schema.class_order["reasoning_tags"]
    reasoning_positive = {
        tag: sum(tag in record.get("reasoning_tags", []) for record in labels)
        for tag in reasoning_order
    }
    return {
        "records": len(labels),
        "scalar_class_counts": scalar,
        "reasoning_tag_support": {
            tag: {"positive": positive, "negative": len(labels) - positive}
            for tag, positive in reasoning_positive.items()
        },
    }


def _stock_counts(
    sample_ids: Iterable[str], inputs_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    values = []
    for sample_id in sample_ids:
        input_record = inputs_by_id[sample_id]
        values.append(
            f"{input_record['stock_code']} {input_record['stock_name']}"
        )
    return _sorted_counter(values)


def _calendar_date_counts(
    sample_ids: Iterable[str], inputs_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    return _sorted_counter(
        str(inputs_by_id[sample_id]["published_at"])[:10] for sample_id in sample_ids
    )


def _weight_summary(
    weight_records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in V1_HEADS:
        numeric_values = [float(record["weights"][head]) for record in weight_records.values()]
        confidences = [str(record.get("label_confidence", "MISSING")) for record in weight_records.values()]
        result[head] = {
            "effective_weight_total": round(sum(numeric_values), 6),
            "weight_value_counts": _sorted_counter(f"{value:.6g}" for value in numeric_values),
            "label_confidence_counts": _sorted_counter(confidences),
            "effective_weight_by_label_confidence": {
                confidence: round(
                    sum(
                        float(record["weights"][head])
                        for record in weight_records.values()
                        if str(record.get("label_confidence", "MISSING")) == confidence
                    ),
                    6,
                )
                for confidence in sorted(set(confidences))
            },
        }
    return result


def _anchor_summary(
    config: ProjectConfig, schema: LabelSchema
) -> dict[str, Any]:
    anchors = read_jsonl(config.data_path("anchor_labels"))
    provenance = read_json(config.data_path("anchor_manifest"))
    if not isinstance(provenance, Mapping):
        raise ContractError("ANCHOR_PROVENANCE_MISMATCH", "anchor manifest must be an object")
    return {
        "records": len(anchors),
        "role": "FIXED_DIAGNOSTIC_ANCHOR_NOT_INDEPENDENT_FINAL_GOLD",
        "source_distribution": _sorted_counter(
            str(record.get("adjudication_source", "MISSING")) for record in anchors
        ),
        "expert_confidence_distribution": _sorted_counter(
            str(record.get("expert_confidence", "MISSING")) for record in anchors
        ),
        "manifest_source_counts": provenance.get("source_counts"),
        "scalar_and_reasoning_distribution": _label_distribution(anchors, schema),
    }


def _support_below_floor(train_distribution: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for head, counts in train_distribution["scalar_class_counts"].items():
        for label, support in counts.items():
            if support < SUPPORT_REPORTING_FLOOR:
                findings.append({"head": head, "label": label, "support": support})
    for label, values in train_distribution["reasoning_tag_support"].items():
        if values["positive"] < SUPPORT_REPORTING_FLOOR:
            findings.append(
                {
                    "head": "reasoning_tags",
                    "label": label,
                    "support": values["positive"],
                }
            )
    return findings


def _read_sysctl_integer(name: str) -> int | None:
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        ["sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _readiness_audit_id(result: Mapping[str, Any]) -> str:
    """Hash stable contract facts while retaining live free-disk observation.

    Available free space is a required point-in-time report field, but normal
    operating-system activity makes its byte value volatile.  It therefore does
    not define the audit's reproducibility identity.
    """

    identity = without_local_paths(result)
    hardware = identity.get("hardware")
    if isinstance(hardware, dict):
        disk = hardware.get("disk_at_audit_worktree")
        if isinstance(disk, dict):
            disk.pop("free_bytes", None)
            disk.pop("free_gib", None)
    return content_addressed_id(identity, omit_keys={"audit_id"})


def _hardware_summary() -> dict[str, Any]:
    machine = platform.machine().lower()
    system = platform.system()
    torch_present = importlib.util.find_spec("torch") is not None
    transformers_present = importlib.util.find_spec("transformers") is not None
    mps_hardware_candidate = system == "Darwin" and machine in {"arm64", "aarch64"}
    disk = shutil.disk_usage(Path.cwd())
    return {
        "operating_system": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "macos_version": platform.mac_ver()[0] if system == "Darwin" else None,
        },
        "cpu": {
            "architecture": machine,
            "logical_cpu_count": os.cpu_count(),
            "apple_silicon_hardware_candidate": mps_hardware_candidate,
        },
        "memory": {
            "total_bytes": _read_sysctl_integer("hw.memsize"),
            "total_gib": (
                round(_read_sysctl_integer("hw.memsize") / 1024**3, 3)
                if _read_sysctl_integer("hw.memsize") is not None
                else None
            ),
        },
        "disk_at_audit_worktree": {
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / 1024**3, 3),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "encoder_runtime_packages": {
            "torch": {"installed": torch_present, "version": _distribution_version("torch")},
            "transformers": {
                "installed": transformers_present,
                "version": _distribution_version("transformers"),
            },
            "datasets": {
                "installed": importlib.util.find_spec("datasets") is not None,
                "version": _distribution_version("datasets"),
            },
            "peft": {"installed": importlib.util.find_spec("peft") is not None, "version": _distribution_version("peft")},
        },
        "mps": {
            "hardware_may_support": mps_hardware_candidate,
            "runtime_training_verified": False,
            "state": (
                "HARDWARE_MAY_SUPPORT_RUNTIME_UNVERIFIED_TORCH_NOT_INSTALLED"
                if mps_hardware_candidate and not torch_present
                else "RUNTIME_NOT_TESTED_BY_READ_ONLY_AUDIT"
            ),
        },
        "cuda": {
            "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None,
            "runtime_training_verified": False,
            "state": "NO_LINUX_CUDA_RUNTIME_EVIDENCE_IN_CURRENT_ENVIRONMENT",
        },
    }


def _blocked_result(error: ContractError) -> dict[str, Any]:
    result: dict[str, Any] = {
        "audit_schema_version": READINESS_SCHEMA_VERSION,
        "status": BLOCKED_DATA_STATUS,
        "selection_or_training_authorized": False,
        "blocker_codes": ["BLOCKED_CANONICAL_DATA_AUDIT", error.code],
        "canonical_data_audit": {
            "status": "BLOCKED",
            "audit_id": None,
            "training_allowed": False,
        },
        "error": error.as_dict(),
    }
    result["audit_id"] = _readiness_audit_id(result)
    return result


def audit_encoder_readiness(path: str | Path) -> dict[str, Any]:
    """Return a fail-closed, stable-key JSON-ready readiness report."""

    try:
        config = ProjectConfig.load(path)
        immutable = _validate_immutable_data_roles(config)
        schema = LabelSchema.load(config.repo_path("schema_path"))
        inputs = immutable["inputs"]
        inputs_by_id = immutable["inputs_by_id"]
        trainable_labels = immutable["trainable_labels"]
        labels_by_id = immutable["labels_by_id"]
        splits_by_id = immutable["splits_by_id"]
        weights_by_id = immutable["weights_by_id"]
        trainable_ids = set(labels_by_id)

        ids_by_split = {
            split: sorted(
                sample_id
                for sample_id, row in splits_by_id.items()
                if row.get("split") == split
            )
            for split in ("train", "dev", "test", "embargo_1", "embargo_2")
        }
        all_assigned = set().union(*map(set, ids_by_split.values()))
        if all_assigned != trainable_ids:
            raise ContractError(
                "ENCODER_READINESS_SPLIT_MISMATCH",
                "unexpected or missing frozen split assignment",
                missing=sorted(trainable_ids - all_assigned),
                extra=sorted(all_assigned - trainable_ids),
            )
        ids_by_population = {
            **ids_by_split,
            "embargo": sorted(ids_by_split["embargo_1"] + ids_by_split["embargo_2"]),
            "all_trainable": sorted(trainable_ids),
        }
        labels_by_population = {
            name: [labels_by_id[sample_id] for sample_id in sample_ids]
            for name, sample_ids in ids_by_population.items()
        }
        distributions = {
            name: _label_distribution(labels, schema)
            for name, labels in labels_by_population.items()
        }
        train_distribution = distributions["train"]
        split_summary = read_json(config.data_path("split_summary"))
        if not isinstance(split_summary, Mapping):
            raise ContractError("SPLIT_MANIFEST_INVALID", "split summary must be an object")

        result = {
            "audit_schema_version": READINESS_SCHEMA_VERSION,
            "status": MILESTONE_STATUS,
            "selection_or_training_authorized": False,
            "scope": {
                "data_role": "READ_ONLY_AUDIT_OF_IMMUTABLE_WEAK_LABEL_PACKAGE",
                "model_weights_downloaded": False,
                "tokenizer_artifacts_downloaded": False,
                "encoder_runtime_imported": False,
                "training_run_created": False,
                "production_inference_49054_run": False,
            },
            "canonical_data_audit": {
                "status": "CONFIRMED_READ_ONLY_CONTENT_AND_ROLE_VALIDATION",
                "audit_id": None,
                "package_manifest_id": immutable["data_package"]["manifest_sha256"],
                "reference_package_manifest_id": immutable["reference_package"]["manifest_sha256"],
                "data_payload_file_count": immutable["data_package"]["payload_file_count"],
                "reference_payload_file_count": immutable["reference_package"]["payload_file_count"],
            },
            "data_roles": {
                "trainable_labels": "WEAK_LABEL_ONLY",
                "anchor50": "FIXED_DIAGNOSTIC_ANCHOR_NOT_UNBIASED_FINAL_GOLD",
                "independent_adjudicated_gold": "NOT_PRESENT_IN_CURRENT_IMMUTABLE_PACKAGE",
                "teacher_prediction_or_agreement": "NOT_GOLD",
                "llm_suggestion": "NOT_GOLD",
            },
            "split_counts": {
                "train": len(ids_by_split["train"]),
                "dev": len(ids_by_split["dev"]),
                "test": len(ids_by_split["test"]),
                "embargo_1": len(ids_by_split["embargo_1"]),
                "embargo_2": len(ids_by_split["embargo_2"]),
                "embargo_total": len(ids_by_population["embargo"]),
                "anchor50": len(read_jsonl(config.data_path("anchor_labels"))),
                "trainable_weak_label_total": len(trainable_ids),
                "canonical_input_total": len(inputs),
            },
            "split_time_intervals": split_summary.get("boundaries"),
            "label_distribution_by_population": distributions,
            "per_class_train_support": train_distribution,
            "classes_below_train_support_reporting_floor": {
                "floor": SUPPORT_REPORTING_FLOOR,
                "findings": _support_below_floor(train_distribution),
                "interpretation": "A count below this floor is a reporting restriction, not a newly approved acceptance threshold.",
            },
            "field_weight_summary": _weight_summary(weights_by_id),
            "stock_counts": {
                name: _stock_counts(sample_ids, inputs_by_id)
                for name, sample_ids in ids_by_population.items()
            },
            "calendar_date_counts": {
                name: _calendar_date_counts(sample_ids, inputs_by_id)
                for name, sample_ids in ids_by_population.items()
            },
            "raw_text_audit": _text_audit(inputs),
            "anchor50": _anchor_summary(config, schema),
            "gold_and_ood_evidence": {
                "independent_adjudicated_gold": {
                    "status": "BLOCKED_NO_SEPARATE_VERSIONED_INDEPENDENT_GOLD_ARTIFACT",
                    "verified_count": 0,
                    "note": "The 11 HUMAN_CONFIRMED Anchor rows are part of a fixed diagnostic Anchor50 and do not establish an untouched independent Gold set.",
                },
                "ood_set": {
                    "status": "BLOCKED_NO_VERSIONED_OOD_ARTIFACT",
                    "verified_count": 0,
                },
                "priority_challenge_categories": [
                    "author_action_vs_other_person_or_advice",
                    "negation_and_conditional_action",
                    "wish_vs_completed_action",
                    "UNKNOWN_vs_NEUTRAL",
                    "CALM_vs_NONE_EXPLICIT",
                    "WATCH_vs_NO_ACTION_SIGNAL",
                    "BUY_ADD_REDUCE_SELL_and_FOMO",
                    "sarcasm_wordplay_new_slang_and_context_dependency",
                    "multi_label_reasoning",
                    "new_stock_sector_time_platform_and_character_perturbations",
                ],
                "ood_semantics": "OOD_IS_NOT_UNKNOWN_AND_OOD_IS_NOT_LOW_CONFIDENCE",
            },
            "hardware": _hardware_summary(),
        }
    except ContractError as exc:
        return _blocked_result(exc)
    result["audit_id"] = _readiness_audit_id(result)
    return result


def run_encoder_readiness(path: str | Path) -> tuple[dict[str, Any], int]:
    result = audit_encoder_readiness(path)
    return result, 0 if result["status"] == MILESTONE_STATUS else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Milestone 1B Encoder data and hardware readiness audit"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result, exit_code = run_encoder_readiness(args.config)
    sys.stdout.write(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
