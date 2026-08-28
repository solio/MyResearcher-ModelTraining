"""Read-only Encoder planning readiness, gated only by the canonical audit."""

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
from pathlib import Path
from typing import Any

from .audit_data import run_audit
from .config import ProjectConfig
from .data import read_json, read_jsonl
from .errors import ContractError
from .hashes import content_addressed_id
from .schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema

READINESS_SCHEMA_VERSION = "myresearcher.encoder-readiness-audit.v2"
MILESTONE_STATUS = "MILESTONE_1B_SELECTION_CONTRACT_READY_FOR_OWNER_APPROVAL"
BLOCKED_DATA_STATUS = "MILESTONE_1B_BLOCKED_MISSING_DATA_EVIDENCE"
SUPPORT_REPORTING_FLOOR = 20
EXPECTED_DATA_PACKAGE_MANIFEST_ID = "cf7a10f25d951d79607cfd80b70751f11415c2772274e275e6ee1b57f32f470b"
EXPECTED_REFERENCE_PACKAGE_MANIFEST_ID = "828944580b96d872241a6619bdb8f60dae2cd7067a0cc6741b418f1e6a7bdc85"
EXPECTED_REFERENCE_DATA_BINDING = f"sha256:{EXPECTED_DATA_PACKAGE_MANIFEST_ID}"
CANONICAL_ELIGIBLE_STATUSES = frozenset({
    "READY_FOR_COMPARABLE_DIAGNOSTIC_RUN",
    "READY_FOR_EXACT_BASELINE_REPRODUCTION",
})
EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2300, 0x23FF))
TRADITIONAL_INDICATOR_CHARS = frozenset(
    "萬與專業東絲兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧雲亞產畝親億僅從侖倉儀們價儲兒兌黨蘭關興養獸內岡冊寫軍農衝凍凱擊則剛剝動勞勢勵勁勻區醫華協單賣衛卻廠厭廂壓厲參雙發變臺葉號嘆嚇呂嗎噸聽啟員啞問"
)


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _lengths(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return dict.fromkeys(("min", "mean", "p50", "p90", "p95", "p99", "max"))
    ordered = sorted(values)
    percentile = lambda p: ordered[max(0, math.ceil(p * len(ordered) / 100) - 1)]
    return {
        "min": min(values), "mean": round(sum(values) / len(values), 3),
        "p50": percentile(50), "p90": percentile(90), "p95": percentile(95),
        "p99": percentile(99), "max": max(values),
    }


def _text_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [row["model_text"] for row in records if isinstance(row.get("model_text"), str)]
    def emoji(text: str) -> bool:
        return any(low <= ord(ch) <= high for ch in text for low, high in EMOJI_RANGES)
    def mixed(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text) and any(
            ("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text
        )
    return {
        "population": "all_canonical_inputs_including_quarantine",
        "records": len(records),
        "model_text_type_counts": _count(type(row.get("model_text")).__name__ for row in records),
        "empty_string_count": sum(text == "" for text in texts),
        "whitespace_only_count": sum(bool(text) and not text.strip() for text in texts),
        "non_string_model_text_count": len(records) - len(texts),
        "character_length": _lengths([len(text) for text in texts]),
        "utf8_byte_length": _lengths([len(text.encode("utf-8")) for text in texts]),
        "observations": {
            "url_marker_count": sum(any(marker in text.casefold() for marker in ("http://", "https://", "www.")) for text in texts),
            "emoji_codepoint_count": sum(emoji(text) for text in texts),
            "han_ascii_latin_mixed_count": sum(mixed(text) for text in texts),
            "traditional_character_indicator_count": sum(any(ch in TRADITIONAL_INDICATOR_CHARS for ch in text) for text in texts),
            "control_or_format_character_count": sum(any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in text) for text in texts),
        },
        "token_length_audit": {
            "status": "BLOCKED_TOKENIZER_ARTIFACT_NOT_AVAILABLE",
            "reason": "Milestone 1B neither downloads nor loads a tokenizer; no token-length statistic is fabricated.",
        },
    }


def _label_distribution(rows: Sequence[Mapping[str, Any]], schema: LabelSchema) -> dict[str, Any]:
    return {
        "records": len(rows),
        "scalar_class_counts": {
            head: {label: sum(row.get(head) == label for row in rows) for label in schema.class_order[head]}
            for head in SINGLE_LABEL_HEADS
        },
        "reasoning_tag_support": {
            tag: {"positive": sum(tag in row.get("reasoning_tags", []) for row in rows),
                  "negative": len(rows) - sum(tag in row.get("reasoning_tags", []) for row in rows)}
            for tag in schema.class_order["reasoning_tags"]
        },
    }


def _weight_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for head in V1_HEADS:
        values = [float(row["weights"][head]) for row in rows]
        confidence = [str(row.get("label_confidence", "MISSING")) for row in rows]
        result[head] = {
            "effective_weight_total": round(sum(values), 6),
            "weight_value_counts": _count(f"{value:.6g}" for value in values),
            "label_confidence_counts": _count(confidence),
        }
    return result


def _hardware_summary() -> dict[str, Any]:
    machine, system = platform.machine().lower(), platform.system()
    memory = _sysctl_int("hw.memsize")
    disk = shutil.disk_usage(Path.cwd())
    packages = {
        name: {"installed": importlib.util.find_spec(name) is not None, "version": _distribution_version(name)}
        for name in ("torch", "transformers", "datasets", "peft")
    }
    mps = system == "Darwin" and machine in {"arm64", "aarch64"}
    return {
        "operating_system": {"system": system, "release": platform.release(), "macos_version": platform.mac_ver()[0] if system == "Darwin" else None},
        "cpu": {"architecture": machine, "logical_cpu_count": os.cpu_count(), "apple_silicon_hardware_candidate": mps},
        "memory": {"total_bytes": memory, "total_gib": round(memory / 1024**3, 3) if memory is not None else None},
        "disk_at_audit_worktree": {"total_bytes": disk.total, "free_bytes": disk.free, "free_gib": round(disk.free / 1024**3, 3)},
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(), "executable": sys.executable},
        "encoder_runtime_packages": packages,
        "mps": {"hardware_may_support": mps, "runtime_training_verified": False},
        "cuda": {"runtime_training_verified": False, "state": "BLOCKED_NO_OWNER_APPROVED_LINUX_CUDA_RUNTIME_AUDIT"},
    }


def _sysctl_int(name: str) -> int | None:
    if sys.platform != "darwin":
        return None
    outcome = subprocess.run(["sysctl", "-n", name], check=False, capture_output=True, text=True)
    try:
        return int(outcome.stdout.strip()) if outcome.returncode == 0 else None
    except ValueError:
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_snapshot(canonical: Mapping[str, Any], exit_code: int) -> dict[str, Any]:
    summary = canonical.get("validation_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    reference = summary.get("reference_package")
    reference = reference if isinstance(reference, Mapping) else {}
    return {
        "exit_code": exit_code, "status": canonical.get("status"),
        "audit_id": canonical.get("audit_id"),
        "training_allowed": canonical.get("training_allowed"),
        "data_package_manifest_id": summary.get("package_manifest_id"),
        "reference_package_manifest_id": reference.get("package_manifest_id"),
        "reference_binding_data_package_content_address": reference.get("binding_data_package_content_address"),
        "reference_available": reference.get("available"),
        "reference_allowed_current_status": reference.get("allowed_current_status"),
        "canonical_blocker_codes": canonical.get("blocker_codes", []),
    }


def _gate_blockers(canonical: Mapping[str, Any], exit_code: int) -> list[str]:
    blockers = []
    canonical_blockers = canonical.get("blocker_codes")
    if isinstance(canonical_blockers, list) and canonical_blockers:
        blockers.append("BLOCKED_CANONICAL_AUDIT_BLOCKER_CODES")
        blockers.extend(
            str(code) for code in canonical_blockers if isinstance(code, str)
        )
    if exit_code != 0: blockers.append("BLOCKED_CANONICAL_AUDIT_NONZERO_EXIT")
    if canonical.get("status") not in CANONICAL_ELIGIBLE_STATUSES: blockers.append("BLOCKED_CANONICAL_AUDIT_STATUS")
    if canonical.get("training_allowed") is not True: blockers.append("BLOCKED_CANONICAL_AUDIT_TRAINING_SEMANTICS")
    audit_id = canonical.get("audit_id")
    if (
        not isinstance(audit_id, str)
        or len(audit_id) != 64
        or any(character not in "0123456789abcdef" for character in audit_id)
    ):
        blockers.append("BLOCKED_CANONICAL_AUDIT_ID")
    summary = canonical.get("validation_summary")
    if not isinstance(summary, Mapping): return [*blockers, "BLOCKED_CANONICAL_AUDIT_MANIFEST_BINDING"]
    if summary.get("package_manifest_id") != EXPECTED_DATA_PACKAGE_MANIFEST_ID: blockers.append("BLOCKED_CANONICAL_DATA_MANIFEST_PIN")
    reference = summary.get("reference_package")
    if not isinstance(reference, Mapping): return [*blockers, "BLOCKED_CANONICAL_REFERENCE_MANIFEST_BINDING"]
    if reference.get("available") is not True: blockers.append("BLOCKED_CANONICAL_REFERENCE_UNAVAILABLE")
    if reference.get("package_manifest_id") != EXPECTED_REFERENCE_PACKAGE_MANIFEST_ID: blockers.append("BLOCKED_CANONICAL_REFERENCE_MANIFEST_PIN")
    if reference.get("binding_data_package_content_address") != EXPECTED_REFERENCE_DATA_BINDING: blockers.append("BLOCKED_CANONICAL_REFERENCE_DATA_BINDING")
    return list(dict.fromkeys(blockers))


def _readiness_id(result: Mapping[str, Any]) -> str:
    identity = _identity_without_local_spelling(result)
    disk = identity.get("hardware", {}).get("disk_at_audit_worktree") if isinstance(identity.get("hardware"), dict) else None
    if isinstance(disk, dict):
        disk.pop("free_bytes", None)
        disk.pop("free_gib", None)
    return content_addressed_id(identity, omit_keys={"audit_id"})


def _identity_without_local_spelling(
    value: Any, *, inside_error: bool = False
) -> Any:
    """Keep runtime facts while excluding local path spelling from identity."""

    if isinstance(value, Mapping):
        return {
            key: _identity_without_local_spelling(
                item, inside_error=inside_error or key == "error"
            )
            for key, item in value.items()
            if key not in {"path", "executable"}
            and not (inside_error and key == "message")
        }
    if isinstance(value, list):
        return [
            _identity_without_local_spelling(item, inside_error=inside_error)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _identity_without_local_spelling(item, inside_error=inside_error)
            for item in value
        )
    return value


def _blocked(blockers: Sequence[str], *, snapshot: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "audit_schema_version": READINESS_SCHEMA_VERSION, "status": BLOCKED_DATA_STATUS,
        "selection_or_training_authorized": False,
        "blocker_codes": list(dict.fromkeys(["BLOCKED_CANONICAL_DATA_AUDIT", *blockers])),
    }
    if snapshot is not None: result["canonical_data_audit"] = dict(snapshot)
    if error is not None: result["error"] = dict(error)
    result["audit_id"] = _readiness_id(result)
    return result


def _statistics(path: str | Path) -> dict[str, Any]:
    """Descriptive reads occur only after the canonical gate has passed."""
    config = ProjectConfig.load(path)
    schema = LabelSchema.load(config.repo_path("schema_path"))
    inputs = read_jsonl(config.data_path("canonical_inputs"))
    labels = read_jsonl(config.data_path("trainable_teacher_labels"))
    splits = read_jsonl(config.data_path("split_manifest"))
    weights = read_jsonl(config.data_path("field_weights"))
    anchors = read_jsonl(config.data_path("anchor_labels"))
    anchor_manifest = read_json(config.data_path("anchor_manifest"))
    split_summary = read_json(config.data_path("split_summary"))
    by_id = {row["sample_id"]: row for row in labels if isinstance(row.get("sample_id"), str)}
    input_by_id = {row["sample_id"]: row for row in inputs if isinstance(row.get("sample_id"), str)}
    roles = {"train": [], "dev": [], "test": [], "embargo": []}
    for row in splits:
        name = "embargo" if row.get("split") in {"embargo", "embargo_1", "embargo_2"} else row.get("split")
        if name in roles and isinstance(row.get("sample_id"), str): roles[name].append(row["sample_id"])
    distribution = {name: _label_distribution([by_id[key] for key in keys if key in by_id], schema) for name, keys in roles.items()}
    def stock_counts(keys: Iterable[str]) -> dict[str, int]:
        return _count(f"{input_by_id[key].get('stock_code', 'MISSING')} {input_by_id[key].get('stock_name', 'MISSING')}" for key in keys if key in input_by_id)
    def date_counts(keys: Iterable[str]) -> dict[str, int]:
        return _count(str(input_by_id[key].get("published_at", "MISSING"))[:10] for key in keys if key in input_by_id)
    train_distribution = distribution["train"]
    below = [{"head": head, "label": label, "support": support} for head, labels_by_class in train_distribution["scalar_class_counts"].items() for label, support in labels_by_class.items() if support < SUPPORT_REPORTING_FLOOR]
    below += [{"head": "reasoning_tags", "label": label, "support": row["positive"]} for label, row in train_distribution["reasoning_tag_support"].items() if row["positive"] < SUPPORT_REPORTING_FLOOR]
    return {
        "data_roles": {
            "trainable_labels": "WEAK_LABEL_ONLY",
            "dev": "ONLY_CANDIDATE_STAGE_ARCHITECTURE_MAX_LENGTH_TRUNCATION_HPARAM_EARLY_STOPPING_THRESHOLD_CALIBRATION_AND_LOSS_POLICY_SELECTION",
            "test": "ONE_TIME_FORMAL_UNSEAL_ONLY_AFTER_CANDIDATE_STAGE_HPARAM_SEED_AGGREGATION_CODE_AND_EVALUATION_MANIFEST_FREEZE",
            "anchor50": "FIXED_REGRESSION_DISAGREEMENT_CHALLENGE_ANCHOR_NOT_GOLD_AND_NOT_FOR_SELECTION",
            "embargo": "RETAIN_FOR_ONE_EXPLICIT_FUTURE_TIME_VALIDATION_OR_OTHER_SINGLE_USE_NOT_REPEATED_DEFAULT",
        },
        "split_counts": {**_count(str(row.get("split", "MISSING")) for row in splits), "embargo_total": len(roles["embargo"]), "anchor50": len(anchors), "trainable_weak_label_total": len(labels), "canonical_input_total": len(inputs)},
        "split_time_intervals": split_summary.get("boundaries") if isinstance(split_summary, Mapping) else None,
        "label_distribution_by_population": distribution, "per_class_train_support": train_distribution,
        "classes_below_train_support_reporting_floor": {"floor": SUPPORT_REPORTING_FLOOR, "findings": below, "interpretation": "Reporting restriction, not a new acceptance threshold."},
        "field_weight_summary": _weight_summary(weights),
        "stock_counts": {name: stock_counts(keys) for name, keys in roles.items()},
        "calendar_date_counts": {name: date_counts(keys) for name, keys in roles.items()},
        "raw_text_audit": _text_audit(inputs),
        "anchor50": {"records": len(anchors), "role": "FIXED_DIAGNOSTIC_ANCHOR_NOT_INDEPENDENT_FINAL_GOLD", "source_distribution": _count(str(row.get("adjudication_source", "MISSING")) for row in anchors), "expert_confidence_distribution": _count(str(row.get("expert_confidence", "MISSING")) for row in anchors), "manifest_source_counts": anchor_manifest.get("source_counts") if isinstance(anchor_manifest, Mapping) else None, "scalar_and_reasoning_distribution": _label_distribution(anchors, schema)},
        "gold_and_ood_evidence": {"independent_adjudicated_gold": {"status": "BLOCKED_NO_SEPARATE_VERSIONED_INDEPENDENT_GOLD_ARTIFACT", "verified_count": 0}, "ood_set": {"status": "BLOCKED_NO_VERSIONED_OOD_ARTIFACT", "verified_count": 0}, "ood_semantics": "OOD_IS_NOT_UNKNOWN_AND_OOD_IS_NOT_LOW_CONFIDENCE"},
        "hardware": _hardware_summary(),
    }


def audit_encoder_readiness(path: str | Path) -> dict[str, Any]:
    try:
        canonical, exit_code = run_audit(path)
    except Exception as exc:
        return _blocked(["BLOCKED_CANONICAL_AUDIT_EXECUTION"], error={"code": "CANONICAL_AUDIT_EXECUTION_FAILED", "message": str(exc)})
    if not isinstance(canonical, Mapping):
        return _blocked(["BLOCKED_CANONICAL_AUDIT_RESULT_INVALID"])
    snapshot = _canonical_snapshot(canonical, exit_code)
    blockers = _gate_blockers(canonical, exit_code)
    if blockers:
        return _blocked(blockers, snapshot=snapshot)
    try:
        observed = _statistics(path)
    except ContractError as exc:
        return _blocked(["BLOCKED_READINESS_STATISTICS", exc.code], snapshot=snapshot, error=exc.as_dict())
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(["BLOCKED_READINESS_STATISTICS"], snapshot=snapshot, error={"code": "READINESS_STATISTICS_INVALID", "message": str(exc)})
    result: dict[str, Any] = {
        "audit_schema_version": READINESS_SCHEMA_VERSION, "status": MILESTONE_STATUS,
        "selection_or_training_authorized": False,
        "authorization_scope": "PLANNING_ONLY_NO_DOWNLOAD_INSTALL_TRAINING_OR_PRODUCTION_INFERENCE",
        "blocker_codes": [], "canonical_data_audit": snapshot,
        "immutable_contract_pins": {"data_package_manifest_id": EXPECTED_DATA_PACKAGE_MANIFEST_ID, "reference_package_manifest_id": EXPECTED_REFERENCE_PACKAGE_MANIFEST_ID, "reference_binding_data_package_content_address": EXPECTED_REFERENCE_DATA_BINDING},
        **observed,
    }
    result["audit_id"] = _readiness_id(result)
    return result


def run_encoder_readiness(path: str | Path) -> tuple[dict[str, Any], int]:
    result = audit_encoder_readiness(path)
    return result, 0 if result["status"] == MILESTONE_STATUS else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Milestone 1B Encoder readiness gated by semantic_model.audit_data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result, exit_code = run_encoder_readiness(args.config)
    sys.stdout.write(f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
