from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import ProjectConfig
from .data import index_by_sample_id, read_json, read_jsonl
from .errors import ContractError
from .hashes import sha256_file
from .models.classical import ClassicalMultiHeadModel
from .preprocessing import PreprocessingContract, build_model_input, build_model_inputs
from .schema import SINGLE_LABEL_HEADS, LabelSchema


REFERENCE_MANIFEST_SCHEMA = "content-addressed-package-manifest-v1"
REFERENCE_PACKAGE_FORMAT = "myresearcher.semantic-baseline-reference.v0.3.5"
REFERENCE_PACKAGE_NAME = "MyResearcher_Semantic_Baseline_Reference"
REFERENCE_PACKAGE_VERSION = "v0.3.5"
REFERENCE_PREDICTION_SCHEMA = "semantic-baseline-reference-prediction-v0.3.5"
REFERENCE_ENVIRONMENT_SCHEMA = "semantic-baseline-reference-environment-v0.3.5"
REFERENCE_POLICY_SCHEMA = "semantic-baseline-reproduction-policy-v0.3.5"
REFERENCE_SPLITS = ("train", "dev", "test", "anchor50")
REFERENCE_ARCHIVE_ROOT = "MyResearcher_Semantic_Baseline_Reference_v0.3.5"

REFERENCE_FILES = {
    "policy": "contracts/reproduction_tolerance_policy_v0.3.5.json",
    "diagnostics": "diagnostics/original_estimator_diagnostics_v0.3.5.json",
    "environment": "environment/reference_environment_v0.3.5.json",
    "historical_metrics": "metrics/original_semantic_baseline_metrics_v0.3.5.json",
    "recomputed_metrics": "metrics/recomputed_from_original_model_v0.3.5.json",
    "metrics_comparison": "metrics/original_metrics_recomputation_comparison_v0.3.5.json",
    "model": "model/original_model_v0.3.5.joblib",
    "source": "source/original_train_semantic_baseline_v035.py",
    "prediction_train": "predictions/train_reference_predictions_v0.3.5.jsonl",
    "prediction_dev": "predictions/dev_reference_predictions_v0.3.5.jsonl",
    "prediction_test": "predictions/test_reference_predictions_v0.3.5.jsonl",
    "prediction_anchor50": "predictions/anchor50_reference_predictions_v0.3.5.jsonl",
}


def reference_root(config: ProjectConfig) -> Path | None:
    raw = config.raw.get("baseline_reference")
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("root")
    if not isinstance(value, str) or not value:
        raise ContractError(
            "REFERENCE_CONFIG_INVALID", "baseline_reference.root is required"
        )
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config.project_root / path).resolve()


def _reference_config(config: ProjectConfig) -> Mapping[str, Any] | None:
    raw = config.raw.get("baseline_reference")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ContractError(
            "REFERENCE_CONFIG_INVALID", "baseline_reference must be an object"
        )
    return raw


def _checked_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID", "manifest payload path must be a string"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ContractError(
            "REFERENCE_PACKAGE_PATH_ESCAPE",
            "reference payload path escapes the package root",
            path=value,
        )
    return value


def _require_mapping(value: Any, *, artifact: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID", f"{artifact} must be an object"
        )
    return value


def _require_sha256(value: Any, *, artifact: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID", f"{artifact} must be a lowercase SHA-256"
        )
    return value


def _load_reference_json(root: Path, logical_name: str) -> Mapping[str, Any]:
    return _require_mapping(
        read_json(root / REFERENCE_FILES[logical_name]), artifact=logical_name
    )


def _verify_reference_manifest(
    config: ProjectConfig, root: Path, *, data_package_manifest_id: str
) -> dict[str, Any]:
    if not root.is_dir():
        raise ContractError("REFERENCE_PACKAGE_NOT_FOUND", str(root))
    symlinks = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise ContractError(
            "REFERENCE_PACKAGE_SYMLINK_FORBIDDEN",
            "reference packages must not contain symbolic links",
            paths=symlinks,
        )
    manifest_path = root / "CONTENT_MANIFEST.json"
    checksum_path = root / "CONTENT_MANIFEST.sha256"
    manifest = _require_mapping(
        read_json(manifest_path), artifact="reference content manifest"
    )
    if manifest.get("manifest_schema_version") != REFERENCE_MANIFEST_SCHEMA:
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID",
            "unsupported reference content manifest schema",
            observed=manifest.get("manifest_schema_version"),
        )
    if (
        manifest.get("package_name") != REFERENCE_PACKAGE_NAME
        or manifest.get("package_version") != REFERENCE_PACKAGE_VERSION
    ):
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID", "reference package identity differs"
        )
    manifest_sha256 = sha256_file(manifest_path)
    raw = _reference_config(config)
    assert raw is not None
    if (
        raw.get("package_format") != REFERENCE_PACKAGE_FORMAT
        or raw.get("exact_prediction_labels_required") is not True
        or raw.get("metrics_absolute_tolerance") != 1e-12
        or raw.get("probability_absolute_tolerance") != 1e-10
        or raw.get("cross_platform_exactness_authorized") is not False
        or raw.get("production_approval") is not False
    ):
        raise ContractError(
            "REFERENCE_CONFIG_INVALID",
            "baseline_reference config differs from the frozen policy",
        )
    expected_manifest = raw.get("expected_package_manifest_sha256")
    if expected_manifest != manifest_sha256:
        raise ContractError(
            "REFERENCE_PACKAGE_HASH_MISMATCH",
            "reference manifest differs from the config-pinned content address",
            observed=manifest_sha256,
            expected=expected_manifest,
        )
    checksum_text = checksum_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})  CONTENT_MANIFEST\.json", checksum_text)
    if match is None or match.group(1) != manifest_sha256:
        raise ContractError(
            "REFERENCE_PACKAGE_HASH_MISMATCH",
            "reference manifest checksum sidecar is invalid",
            observed=checksum_text,
        )
    expected_binding = f"sha256:{data_package_manifest_id}"
    if manifest.get("binding_data_package_content_address") != expected_binding:
        raise ContractError(
            "REFERENCE_DATA_BINDING_MISMATCH",
            "reference package is bound to a different immutable data package",
            observed=manifest.get("binding_data_package_content_address"),
            expected=expected_binding,
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContractError(
            "REFERENCE_PACKAGE_INVALID", "reference manifest files must be a list"
        )
    by_path: dict[str, Mapping[str, Any]] = {}
    verified_bytes = 0
    for entry in files:
        entry = _require_mapping(entry, artifact="reference manifest entry")
        relative = _checked_relative_path(entry.get("path"))
        if relative in by_path:
            raise ContractError(
                "REFERENCE_PACKAGE_DUPLICATE_PATH", "duplicate manifest payload path"
            )
        payload = root / relative
        try:
            payload.resolve().relative_to(root)
        except ValueError as exc:
            raise ContractError(
                "REFERENCE_PACKAGE_PATH_ESCAPE", "resolved payload escapes package root"
            ) from exc
        if not payload.is_file():
            raise ContractError(
                "REFERENCE_PACKAGE_ARTIFACT_MISSING",
                "manifest payload is missing",
                path=relative,
            )
        expected_size = entry.get("size_bytes")
        expected_hash = _require_sha256(entry.get("sha256"), artifact=relative)
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ContractError(
                "REFERENCE_PACKAGE_INVALID", "payload size must be a non-negative integer"
            )
        observed_size = payload.stat().st_size
        observed_hash = sha256_file(payload)
        if observed_size != expected_size or observed_hash != expected_hash:
            raise ContractError(
                "REFERENCE_PACKAGE_PAYLOAD_MISMATCH",
                "reference payload differs from its manifest",
                path=relative,
                observed_size=observed_size,
                expected_size=expected_size,
                observed_sha256=observed_hash,
                expected_sha256=expected_hash,
            )
        by_path[relative] = entry
        verified_bytes += observed_size
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = {"CONTENT_MANIFEST.json", "CONTENT_MANIFEST.sha256", *by_path}
    if actual_files != expected_files:
        raise ContractError(
            "REFERENCE_PACKAGE_FILE_SET_MISMATCH",
            "reference package has missing or unexpected files",
            missing=sorted(expected_files - actual_files),
            unexpected=sorted(actual_files - expected_files),
        )
    if manifest.get("payload_file_count") != len(by_path) or manifest.get(
        "payload_total_bytes"
    ) != verified_bytes:
        raise ContractError(
            "REFERENCE_PACKAGE_TOTAL_MISMATCH",
            "reference manifest totals differ from verified payloads",
        )
    missing_required = sorted(set(REFERENCE_FILES.values()) - set(by_path))
    if missing_required:
        raise ContractError(
            "REFERENCE_PACKAGE_ARTIFACT_MISSING",
            "required reference artifacts are absent",
            paths=missing_required,
        )
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "payload_file_count": len(by_path),
        "payload_total_bytes": verified_bytes,
        "files": by_path,
        "binding_data_package_content_address": expected_binding,
    }


def audit_reference_archive(
    config: ProjectConfig, archive_path: str | Path
) -> dict[str, Any]:
    """Verify ZIP structure and every payload without extracting or executing it."""

    path = Path(archive_path).resolve()
    raw = _reference_config(config)
    if raw is None:
        raise ContractError(
            "REFERENCE_CONFIG_INVALID", "baseline_reference config is required"
        )
    if not path.is_file():
        raise ContractError("REFERENCE_ARCHIVE_NOT_FOUND", str(path))
    observed_zip_sha256 = sha256_file(path)
    expected_zip_sha256 = raw.get("expected_zip_sha256")
    if observed_zip_sha256 != expected_zip_sha256:
        raise ContractError(
            "REFERENCE_ARCHIVE_HASH_MISMATCH",
            "reference ZIP differs from the pinned archive hash",
            observed=observed_zip_sha256,
            expected=expected_zip_sha256,
        )
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ContractError("REFERENCE_ARCHIVE_INVALID", str(exc)) from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ContractError(
                "REFERENCE_ARCHIVE_DUPLICATE_PATH", "ZIP contains duplicate entries"
            )
        for info in infos:
            pure = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "\\" in info.filename
                or stat.S_ISLNK(mode)
            ):
                raise ContractError(
                    "REFERENCE_ARCHIVE_UNSAFE_ENTRY",
                    "ZIP contains an unsafe path or symbolic link",
                    path=info.filename,
                )
            if not pure.parts or pure.parts[0] != REFERENCE_ARCHIVE_ROOT:
                raise ContractError(
                    "REFERENCE_ARCHIVE_UNSAFE_ENTRY",
                    "ZIP entry is outside the single frozen package root",
                    path=info.filename,
                )
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ContractError(
                "REFERENCE_ARCHIVE_CRC_MISMATCH",
                "ZIP CRC validation failed",
                path=corrupt,
            )
        manifest_member = f"{REFERENCE_ARCHIVE_ROOT}/CONTENT_MANIFEST.json"
        checksum_member = f"{REFERENCE_ARCHIVE_ROOT}/CONTENT_MANIFEST.sha256"
        try:
            manifest_bytes = archive.read(manifest_member)
            checksum_text = archive.read(checksum_member).decode("utf-8").strip()
        except (KeyError, UnicodeDecodeError) as exc:
            raise ContractError(
                "REFERENCE_ARCHIVE_INVALID", "ZIP manifest or checksum is missing/invalid"
            ) from exc
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha256 != raw.get("expected_package_manifest_sha256"):
            raise ContractError(
                "REFERENCE_PACKAGE_HASH_MISMATCH",
                "ZIP content manifest differs from the pinned content address",
            )
        match = re.fullmatch(
            r"([0-9a-f]{64})  CONTENT_MANIFEST\.json", checksum_text
        )
        if match is None or match.group(1) != manifest_sha256:
            raise ContractError(
                "REFERENCE_PACKAGE_HASH_MISMATCH",
                "ZIP manifest checksum sidecar is invalid",
            )
        try:
            import json

            manifest = _require_mapping(
                json.loads(manifest_bytes), artifact="ZIP content manifest"
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise ContractError(
                "REFERENCE_ARCHIVE_INVALID", "ZIP content manifest is invalid JSON"
            ) from exc
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ContractError(
                "REFERENCE_PACKAGE_INVALID", "ZIP manifest files must be a list"
            )
        expected_names = {manifest_member, checksum_member}
        verified_bytes = 0
        for entry in entries:
            entry = _require_mapping(entry, artifact="ZIP manifest entry")
            relative = _checked_relative_path(entry.get("path"))
            member = f"{REFERENCE_ARCHIVE_ROOT}/{relative}"
            expected_names.add(member)
            try:
                info = archive.getinfo(member)
            except KeyError as exc:
                raise ContractError(
                    "REFERENCE_PACKAGE_ARTIFACT_MISSING",
                    "ZIP payload is missing",
                    path=relative,
                ) from exc
            expected_size = entry.get("size_bytes")
            if info.file_size != expected_size:
                raise ContractError(
                    "REFERENCE_PACKAGE_PAYLOAD_MISMATCH",
                    "ZIP payload size differs",
                    path=relative,
                )
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry.get("sha256"):
                raise ContractError(
                    "REFERENCE_PACKAGE_PAYLOAD_MISMATCH",
                    "ZIP payload hash differs",
                    path=relative,
                )
            verified_bytes += info.file_size
        if set(names) != expected_names:
            raise ContractError(
                "REFERENCE_PACKAGE_FILE_SET_MISMATCH",
                "ZIP has missing or unexpected entries",
                missing=sorted(expected_names - set(names)),
                unexpected=sorted(set(names) - expected_names),
            )
        if manifest.get("payload_file_count") != len(entries) or manifest.get(
            "payload_total_bytes"
        ) != verified_bytes:
            raise ContractError(
                "REFERENCE_PACKAGE_TOTAL_MISMATCH", "ZIP payload totals differ"
            )
    return {
        "path": str(path),
        "zip_sha256": observed_zip_sha256,
        "zip_crc": "PASS",
        "unsafe_entries": 0,
        "duplicate_entries": 0,
        "symlinks": 0,
        "package_manifest_id": manifest_sha256,
        "payload_file_count": len(entries),
        "payload_total_bytes": verified_bytes,
    }


def _observed_runtime() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    threadpools: list[dict[str, Any]] = []
    try:
        from threadpoolctl import threadpool_info

        np.dot(np.ones((1, 1)), np.ones((1, 1)))
        threadpools = [dict(item) for item in threadpool_info()]
    except (ImportError, RuntimeError):
        pass
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "packages": packages,
        "threadpools": threadpools,
    }


def compare_runtime_to_reference(
    observed: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def compare(field: str, left: Any, right: Any) -> None:
        if left != right:
            mismatches.append(
                {"field": field, "observed": left, "reference": right}
            )

    version_info = reference.get("python", {}).get("version_info")
    expected_python = (
        ".".join(str(value) for value in version_info[:3])
        if isinstance(version_info, list) and len(version_info) >= 3
        else None
    )
    compare("python", observed.get("python_version"), expected_python)
    compare(
        "implementation",
        observed.get("implementation"),
        reference.get("python", {}).get("implementation"),
    )
    compare(
        "system",
        observed.get("system"),
        reference.get("operating_system", {}).get("system"),
    )
    compare(
        "machine",
        observed.get("machine"),
        reference.get("operating_system", {}).get("machine"),
    )
    reference_packages = reference.get("packages")
    if not isinstance(reference_packages, Mapping):
        raise ContractError(
            "REFERENCE_ENVIRONMENT_INVALID", "reference packages are missing"
        )
    observed_packages = observed.get("packages")
    if not isinstance(observed_packages, Mapping):
        raise ContractError(
            "REFERENCE_ENVIRONMENT_INVALID", "observed packages are missing"
        )
    aliases = {"scikit_learn": "scikit-learn"}
    for reference_name in ("numpy", "scipy", "scikit_learn", "joblib", "threadpoolctl"):
        observed_name = aliases.get(reference_name, reference_name)
        compare(
            f"package:{observed_name}",
            observed_packages.get(observed_name),
            reference_packages.get(reference_name),
        )
    observed_threadpools = observed.get("threadpools")
    if not isinstance(observed_threadpools, Sequence):
        observed_threadpools = []
    exact_blas = any(
        isinstance(item, Mapping)
        and item.get("internal_api") == "openblas"
        and item.get("version") == "0.3.30"
        and item.get("threading_layer") == "pthreads"
        for item in observed_threadpools
    )
    if not exact_blas:
        mismatches.append(
            {
                "field": "blas",
                "observed": [
                    {
                        key: item.get(key)
                        for key in (
                            "internal_api",
                            "version",
                            "threading_layer",
                            "architecture",
                        )
                    }
                    for item in observed_threadpools
                    if isinstance(item, Mapping) and item.get("user_api") == "blas"
                ],
                "reference": "OpenBLAS 0.3.30 / pthreads",
            }
        )
    return {
        "matches_reference": not mismatches,
        "mismatches": mismatches,
        "observed": dict(observed),
    }


def _validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != REFERENCE_POLICY_SCHEMA:
        raise ContractError(
            "REFERENCE_POLICY_INVALID", "unsupported reproduction policy"
        )
    exact = _require_mapping(
        policy.get("reference_environment_acceptance"),
        artifact="reference environment acceptance policy",
    )
    cross = _require_mapping(
        policy.get("cross_platform_acceptance"),
        artifact="cross-platform acceptance policy",
    )
    if (
        exact.get("prediction_labels")
        != "EXACT_MATCH on all Train/Dev/Test/Anchor50 rows"
        or exact.get("metrics_absolute_tolerance") != 1e-12
        or exact.get("probability_absolute_tolerance") != 1e-10
        or cross.get("authorized") is not False
        or cross.get("allowed_status") != "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
        or policy.get("production_gate")
        != "FAIL; v0.3.5 remains diagnostic only regardless of reproduction status."
    ):
        raise ContractError(
            "REFERENCE_POLICY_INVALID",
            "reference policy differs from the frozen diagnostic-only contract",
        )
    return {
        "prediction_labels_exact": True,
        "metrics_absolute_tolerance": 1e-12,
        "probability_absolute_tolerance": 1e-10,
        "cross_platform_exactness_authorized": False,
        "cross_platform_allowed_status": "COMPARABLE_DIAGNOSTIC_RUN_ONLY",
        "production_approval": False,
    }


def _validate_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    schema: LabelSchema,
    preprocessing: PreprocessingContract,
    model_sha256: str,
) -> dict[str, Any]:
    if diagnostics.get("model_sha256") != model_sha256:
        raise ContractError(
            "REFERENCE_DIAGNOSTICS_MISMATCH",
            "estimator diagnostics address a different model",
        )
    expected_summary = {
        "scalar_estimators": 6,
        "scalar_converged": 0,
        "scalar_not_converged": 6,
        "reason_estimators": 15,
        "reason_converged": 15,
        "reason_not_converged": 0,
    }
    if diagnostics.get("summary") != expected_summary:
        raise ContractError(
            "REFERENCE_DIAGNOSTICS_MISMATCH",
            "reference convergence summary differs from the frozen finding",
        )
    scalar = _require_mapping(
        diagnostics.get("scalar_heads"), artifact="scalar diagnostics"
    )
    reasoning = _require_mapping(
        diagnostics.get("reasoning_heads"), artifact="reasoning diagnostics"
    )
    if set(scalar) != set(SINGLE_LABEL_HEADS) or list(reasoning) != list(
        schema.class_order["reasoning_tags"]
    ):
        raise ContractError(
            "REFERENCE_DIAGNOSTICS_MISMATCH", "estimator head set/order differs"
        )
    feature_total = int(preprocessing.expected_feature_counts["total"])
    for head, raw in scalar.items():
        item = _require_mapping(raw, artifact=f"scalar diagnostics: {head}")
        classes = item.get("class_order")
        if (
            not isinstance(classes, list)
            or not set(classes) <= set(schema.class_order[head])
            or item.get("solver") != "saga"
            or item.get("max_iter") != 2000
            or item.get("n_iter") != [2000]
            or item.get("converged") is not False
            or item.get("random_state") != 35
        ):
            raise ContractError(
                "REFERENCE_DIAGNOSTICS_MISMATCH",
                "scalar estimator diagnostics differ",
                head=head,
            )
        for name in ("coef", "intercept"):
            fingerprint = _require_mapping(
                item.get(name), artifact=f"{head}.{name} fingerprint"
            )
            _require_sha256(fingerprint.get("sha256"), artifact=f"{head}.{name}")
        if item["coef"].get("shape", [None, None])[-1] != feature_total:
            raise ContractError(
                "REFERENCE_DIAGNOSTICS_MISMATCH", "scalar coefficient width differs"
            )
    for tag, raw in reasoning.items():
        item = _require_mapping(raw, artifact=f"reason diagnostics: {tag}")
        n_iter = item.get("n_iter")
        if (
            item.get("class_order") != [0, 1]
            or item.get("solver") != "liblinear"
            or item.get("max_iter") != 1600
            or not isinstance(n_iter, list)
            or not n_iter
            or not all(isinstance(value, int) and value < 1600 for value in n_iter)
            or item.get("converged") is not True
            or item.get("random_state") != 35
        ):
            raise ContractError(
                "REFERENCE_DIAGNOSTICS_MISMATCH",
                "reasoning estimator diagnostics differ",
                tag=tag,
            )
        for name in ("coef", "intercept"):
            fingerprint = _require_mapping(
                item.get(name), artifact=f"{tag}.{name} fingerprint"
            )
            _require_sha256(fingerprint.get("sha256"), artifact=f"{tag}.{name}")
        if item["coef"].get("shape", [None, None])[-1] != feature_total:
            raise ContractError(
                "REFERENCE_DIAGNOSTICS_MISMATCH", "reason coefficient width differs"
            )
    vectorizers = _require_mapping(
        diagnostics.get("vectorizers"), artifact="vectorizer diagnostics"
    )
    for name in ("char", "word"):
        item = _require_mapping(vectorizers.get(name), artifact=f"{name} vectorizer")
        if item.get("vocabulary_size") != int(
            preprocessing.expected_feature_counts[name]
        ):
            raise ContractError(
                "REFERENCE_DIAGNOSTICS_MISMATCH", "vectorizer feature count differs"
            )
        _require_sha256(item.get("vocabulary_sha256"), artifact=f"{name} vocabulary")
        _require_sha256(
            _require_mapping(item.get("idf"), artifact=f"{name} idf").get("sha256"),
            artifact=f"{name} idf",
        )
    if diagnostics.get("joblib_load_warnings_in_reference_runtime") != []:
        raise ContractError(
            "REFERENCE_DIAGNOSTICS_MISMATCH",
            "original model emitted load warnings in the claimed reference runtime",
        )
    return expected_summary


def _validate_probability(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(
            "REFERENCE_PREDICTION_INVALID", "probability must be finite", context=context
        )
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise ContractError(
            "REFERENCE_PREDICTION_INVALID",
            "probability is outside [0, 1]",
            context=context,
        )
    return result


def _validate_reference_predictions(
    root: Path,
    *,
    schema: LabelSchema,
    preprocessing: PreprocessingContract,
    diagnostics: Mapping[str, Any],
    input_index: Mapping[str, Mapping[str, Any]],
    trainable_index: Mapping[str, Mapping[str, Any]],
    split_ids: Mapping[str, list[str]],
    anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_index = index_by_sample_id(anchors, role="reference-anchor")
    expected_ids = {
        "train": list(split_ids["train"]),
        "dev": list(split_ids["dev"]),
        "test": list(split_ids["test"]),
        "anchor50": [str(row["sample_id"]) for row in anchors],
    }
    prediction_counts: dict[str, int] = {}
    thresholds: Mapping[str, Any] | None = None
    for split in REFERENCE_SPLITS:
        rows = read_jsonl(root / REFERENCE_FILES[f"prediction_{split}"])
        prediction_counts[split] = len(rows)
        row_ids = [str(row.get("sample_id")) for row in rows]
        if row_ids != expected_ids[split] or [row.get("ordinal") for row in rows] != list(
            range(1, len(rows) + 1)
        ):
            raise ContractError(
                "REFERENCE_PREDICTION_IDENTITY_MISMATCH",
                "reference prediction order/identity differs from canonical split",
                split=split,
            )
        for row in rows:
            sample_id = str(row["sample_id"])
            if (
                row.get("reference_prediction_schema") != REFERENCE_PREDICTION_SCHEMA
                or row.get("split") != split
            ):
                raise ContractError(
                    "REFERENCE_PREDICTION_INVALID",
                    "prediction schema/split marker differs",
                    sample_id=sample_id,
                )
            source = (
                anchor_index[sample_id]
                if split == "anchor50"
                else input_index[sample_id]
            )
            label = (
                anchor_index[sample_id]
                if split == "anchor50"
                else trainable_index[sample_id]
            )
            normalized = build_model_input(source, preprocessing)
            normalized_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if row.get("normalized_text_sha256") != normalized_sha:
                raise ContractError(
                    "REFERENCE_PREDICTION_TEXT_MISMATCH",
                    "reference prediction text hash differs from canonical preprocessing",
                    sample_id=sample_id,
                )
            scalar = _require_mapping(
                row.get("scalar_heads"), artifact="reference scalar prediction"
            )
            if set(scalar) != set(SINGLE_LABEL_HEADS):
                raise ContractError(
                    "REFERENCE_PREDICTION_INVALID", "scalar prediction heads differ"
                )
            for head in SINGLE_LABEL_HEADS:
                item = _require_mapping(
                    scalar[head], artifact=f"reference prediction {head}"
                )
                classes = diagnostics["scalar_heads"][head]["class_order"]
                probabilities = item.get("probabilities")
                if item.get("classes") != classes or not isinstance(
                    probabilities, list
                ) or len(probabilities) != len(classes):
                    raise ContractError(
                        "REFERENCE_PREDICTION_INVALID",
                        "scalar classes/probability shape differs",
                        sample_id=sample_id,
                        head=head,
                    )
                values = [
                    _validate_probability(value, context=f"{sample_id}:{head}")
                    for value in probabilities
                ]
                if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2e-6):
                    raise ContractError(
                        "REFERENCE_PREDICTION_INVALID",
                        "scalar probabilities do not sum to one",
                        sample_id=sample_id,
                        head=head,
                    )
                predicted = classes[int(np.argmax(values))]
                if item.get("prediction") != predicted or item.get("truth") != label.get(
                    head
                ):
                    raise ContractError(
                        "REFERENCE_PREDICTION_TRUTH_MISMATCH",
                        "scalar truth/prediction differs from canonical labels or probabilities",
                        sample_id=sample_id,
                        head=head,
                    )
            reasoning = _require_mapping(
                row.get("reasoning_tags"), artifact="reference reasoning prediction"
            )
            labels = reasoning.get("labels")
            probabilities = reasoning.get("probabilities")
            row_thresholds = reasoning.get("thresholds")
            if (
                labels != list(schema.class_order["reasoning_tags"])
                or not isinstance(probabilities, Mapping)
                or list(probabilities) != labels
                or not isinstance(row_thresholds, Mapping)
                or list(row_thresholds) != labels
            ):
                raise ContractError(
                    "REFERENCE_PREDICTION_INVALID",
                    "reasoning label/probability/threshold order differs",
                    sample_id=sample_id,
                )
            if thresholds is None:
                thresholds = dict(row_thresholds)
            elif dict(row_thresholds) != dict(thresholds):
                raise ContractError(
                    "REFERENCE_PREDICTION_INVALID",
                    "reasoning thresholds vary by row",
                    sample_id=sample_id,
                )
            predicted_tags = [
                tag
                for tag in labels
                if _validate_probability(
                    probabilities[tag], context=f"{sample_id}:reasoning:{tag}"
                )
                >= _validate_probability(
                    row_thresholds[tag], context=f"{sample_id}:threshold:{tag}"
                )
            ]
            if not predicted_tags:
                predicted_tags = [max(labels, key=lambda tag: probabilities[tag])]
            if (
                reasoning.get("prediction") != predicted_tags
                or set(reasoning.get("truth", [])) != set(label.get("reasoning_tags", []))
            ):
                raise ContractError(
                    "REFERENCE_PREDICTION_TRUTH_MISMATCH",
                    "reasoning truth/prediction differs from canonical labels or thresholds",
                    sample_id=sample_id,
                )
    return {
        "rows": prediction_counts,
        "total_rows": sum(prediction_counts.values()),
        "reasoning_thresholds": dict(thresholds or {}),
    }


def audit_reference_package(
    config: ProjectConfig,
    *,
    data_package_manifest_id: str,
    schema: LabelSchema,
    preprocessing: PreprocessingContract,
    input_index: Mapping[str, Mapping[str, Any]],
    trainable_index: Mapping[str, Mapping[str, Any]],
    split_ids: Mapping[str, list[str]],
    anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = reference_root(config)
    if root is None or not root.exists():
        return {
            "available": False,
            "exact_reproduction_environment_match": False,
            "reproduction_blocker_codes": ["BLOCKED_MISSING_REFERENCE_PACKAGE"],
        }
    manifest = _verify_reference_manifest(
        config, root, data_package_manifest_id=data_package_manifest_id
    )
    policy = _validate_policy(_load_reference_json(root, "policy"))
    environment = _load_reference_json(root, "environment")
    if environment.get("environment_manifest_schema") != REFERENCE_ENVIRONMENT_SCHEMA:
        raise ContractError(
            "REFERENCE_ENVIRONMENT_INVALID", "reference environment schema differs"
        )
    capture_scope = environment.get("capture_scope")
    if not isinstance(capture_scope, str) or "Retrospective capture" not in capture_scope:
        raise ContractError(
            "REFERENCE_ENVIRONMENT_INVALID",
            "retrospective environment provenance limitation is missing",
        )
    if environment.get("artifact_unpickle_warnings") != []:
        raise ContractError(
            "REFERENCE_ENVIRONMENT_INVALID",
            "original model emitted compatibility warnings in the reference runtime",
        )
    model_sha256 = sha256_file(root / REFERENCE_FILES["model"])
    raw = _reference_config(config)
    assert raw is not None
    if model_sha256 != raw.get("expected_original_model_sha256"):
        raise ContractError(
            "REFERENCE_MODEL_HASH_MISMATCH", "original model hash differs"
        )
    diagnostics = _load_reference_json(root, "diagnostics")
    diagnostic_summary = _validate_diagnostics(
        diagnostics,
        schema=schema,
        preprocessing=preprocessing,
        model_sha256=model_sha256,
    )
    historical_metrics_sha256 = sha256_file(root / REFERENCE_FILES["historical_metrics"])
    if historical_metrics_sha256 != sha256_file(config.data_path("baseline_report")):
        raise ContractError(
            "REFERENCE_METRICS_BINDING_MISMATCH",
            "reference and data packages contain different historical metrics",
        )
    source_in_data_package = config.data_root / "provenance/train_semantic_baseline_v035_reference.py"
    if source_in_data_package.is_file() and sha256_file(
        root / REFERENCE_FILES["source"]
    ) != sha256_file(source_in_data_package):
        raise ContractError(
            "REFERENCE_SOURCE_BINDING_MISMATCH",
            "reference and data packages contain different training scripts",
        )
    metrics_comparison = _load_reference_json(root, "metrics_comparison")
    if (
        metrics_comparison.get("status") != "EXACT"
        or metrics_comparison.get("maximum_absolute_difference") != 0.0
        or not isinstance(metrics_comparison.get("comparisons"), list)
        or any(
            item.get("absolute_difference") != 0.0
            for item in metrics_comparison["comparisons"]
            if isinstance(item, Mapping)
        )
    ):
        raise ContractError(
            "REFERENCE_METRICS_RECOMPUTATION_MISMATCH",
            "original model does not exactly regenerate historical metrics",
        )
    predictions = _validate_reference_predictions(
        root,
        schema=schema,
        preprocessing=preprocessing,
        diagnostics=diagnostics,
        input_index=input_index,
        trainable_index=trainable_index,
        split_ids=split_ids,
        anchors=anchors,
    )
    runtime_comparison = compare_runtime_to_reference(_observed_runtime(), environment)
    return {
        "available": True,
        "path": str(root),
        "package_manifest_id": manifest["sha256"],
        "payload_file_count": manifest["payload_file_count"],
        "payload_total_bytes": manifest["payload_total_bytes"],
        "binding_data_package_content_address": manifest[
            "binding_data_package_content_address"
        ],
        "original_model_sha256": model_sha256,
        "historical_metrics_sha256": historical_metrics_sha256,
        "metrics_recomputation_status": "EXACT",
        "metrics_recomputation_maximum_absolute_difference": 0.0,
        "environment_capture_scope": capture_scope,
        "environment_capture_is_retrospective": True,
        "reference_environment": {
            "python": "3.12.13",
            "system": environment["operating_system"]["system"],
            "machine": environment["operating_system"]["machine"],
            "packages": dict(environment["packages"]),
            "cpu_model": environment["cpu"]["model_name"],
        },
        "local_environment": runtime_comparison,
        "exact_reproduction_environment_match": runtime_comparison[
            "matches_reference"
        ],
        "diagnostics": diagnostic_summary,
        "predictions": predictions,
        "policy": policy,
        "exact_reproduction_authorized_for_current_environment": runtime_comparison[
            "matches_reference"
        ],
        "allowed_current_status": (
            "EXACT_REPRODUCTION_CANDIDATE"
            if runtime_comparison["matches_reference"]
            else "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
        ),
        "production_approval": False,
        "reproduction_blocker_codes": (
            []
            if runtime_comparison["matches_reference"]
            else ["BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"]
        ),
    }


def _current_predictions(
    model: ClassicalMultiHeadModel,
    texts: Sequence[str],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]]]:
    probabilities = model.predict_probabilities(texts)
    labels: dict[str, list[Any]] = {}
    for head in SINGLE_LABEL_HEADS:
        order = model.class_order[head]
        labels[head] = [order[int(index)] for index in probabilities[head].argmax(axis=1)]
    reasoning_order = model.class_order["reasoning_tags"]
    threshold_array = np.asarray(
        [float(thresholds["reasoning_tags"][tag]) for tag in reasoning_order]
    )
    selected = (
        probabilities["reasoning_tags"] >= threshold_array.reshape(1, -1)
    ).astype(int)
    if thresholds.get("ensure_at_least_one_reasoning_tag"):
        empty = np.where(selected.sum(axis=1) == 0)[0]
        if len(empty):
            selected[empty, np.argmax(probabilities["reasoning_tags"][empty], axis=1)] = 1
    labels["reasoning_tags"] = [
        [tag for index, tag in enumerate(reasoning_order) if row[index]]
        for row in selected
    ]
    return probabilities, labels


def _metric_comparison(
    *,
    split: str,
    current_labels: Mapping[str, list[Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    reference_metrics: Mapping[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    split_reference = _require_mapping(reference_metrics.get(split), artifact=split)
    scalar_reference = _require_mapping(
        split_reference.get("scalar_fields"), artifact=f"{split}.scalar_fields"
    )
    for head in SINGLE_LABEL_HEADS:
        truth = [row["scalar_heads"][head]["truth"] for row in reference_rows]
        predicted = current_labels[head]
        observed = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(truth, predicted, average="weighted", zero_division=0)
            ),
        }
        expected = _require_mapping(scalar_reference.get(head), artifact=f"{split}.{head}")
        for metric, value in observed.items():
            reference_value = float(expected[metric])
            delta = abs(value - reference_value)
            comparisons.append(
                {
                    "split": split,
                    "head": head,
                    "metric": metric,
                    "reference": reference_value,
                    "observed": value,
                    "absolute_delta": delta,
                    "within_tolerance": delta <= tolerance,
                }
            )
    reasoning_order = list(reference_rows[0]["reasoning_tags"]["labels"])
    truth_matrix = np.asarray(
        [
            [int(tag in row["reasoning_tags"]["truth"]) for tag in reasoning_order]
            for row in reference_rows
        ],
        dtype=int,
    )
    predicted_matrix = np.asarray(
        [
            [int(tag in tags) for tag in reasoning_order]
            for tags in current_labels["reasoning_tags"]
        ],
        dtype=int,
    )
    observed_reasoning = {
        "micro_f1": float(
            f1_score(truth_matrix, predicted_matrix, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth_matrix, predicted_matrix, average="macro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(truth_matrix, predicted_matrix, average="samples", zero_division=0)
        ),
    }
    expected_reasoning = _require_mapping(
        split_reference.get("reasoning_tags"), artifact=f"{split}.reasoning_tags"
    )
    for metric, value in observed_reasoning.items():
        reference_value = float(expected_reasoning[metric])
        delta = abs(value - reference_value)
        comparisons.append(
            {
                "split": split,
                "head": "reasoning_tags",
                "metric": metric,
                "reference": reference_value,
                "observed": value,
                "absolute_delta": delta,
                "within_tolerance": delta <= tolerance,
            }
        )
    return comparisons


def compare_trained_model_to_reference(
    config: ProjectConfig,
    prepared: Any,
    model: ClassicalMultiHeadModel,
    thresholds: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
) -> dict[str, Any]:
    root = reference_root(config)
    if root is None or not reference_audit.get("available"):
        raise ContractError(
            "REFERENCE_PACKAGE_NOT_FOUND", "reference comparison requires the package"
        )
    policy = reference_audit["policy"]
    probability_tolerance = float(policy["probability_absolute_tolerance"])
    metric_tolerance = float(policy["metrics_absolute_tolerance"])
    reference_metrics = _load_reference_json(root, "recomputed_metrics")
    partitions = {
        "train": prepared.partition("train"),
        "dev": prepared.partition("dev"),
        "test": prepared.partition("test"),
    }
    anchor_texts = build_model_inputs(prepared.anchors, prepared.preprocessing)
    partitions["anchor50"] = type(
        "AnchorPartition",
        (),
        {
            "sample_ids": [str(row["sample_id"]) for row in prepared.anchors],
            "texts": anchor_texts,
        },
    )()
    per_split: dict[str, Any] = {}
    metric_comparisons: list[dict[str, Any]] = []
    global_probability_delta = 0.0
    global_label_matches = 0
    global_label_comparisons = 0
    exact_rows = 0
    total_rows = 0
    per_head: dict[str, dict[str, Any]] = {
        head: {
            "label_matches": 0,
            "label_comparisons": 0,
            "maximum_probability_absolute_delta": 0.0,
        }
        for head in (*SINGLE_LABEL_HEADS, "reasoning_tags")
    }
    for split in REFERENCE_SPLITS:
        reference_rows = read_jsonl(root / REFERENCE_FILES[f"prediction_{split}"])
        partition = partitions[split]
        if [row["sample_id"] for row in reference_rows] != list(partition.sample_ids):
            raise ContractError(
                "REFERENCE_PREDICTION_IDENTITY_MISMATCH",
                "prepared partition order differs from reference predictions",
                split=split,
            )
        probabilities, labels = _current_predictions(model, partition.texts, thresholds)
        split_probability_delta = 0.0
        split_label_matches = 0
        split_label_comparisons = 0
        split_exact_rows = 0
        for row_index, reference_row in enumerate(reference_rows):
            row_exact = True
            for head in SINGLE_LABEL_HEADS:
                reference_head = reference_row["scalar_heads"][head]
                reference_prediction = reference_head["prediction"]
                observed_prediction = labels[head][row_index]
                match = observed_prediction == reference_prediction
                split_label_matches += int(match)
                split_label_comparisons += 1
                per_head[head]["label_matches"] += int(match)
                per_head[head]["label_comparisons"] += 1
                row_exact &= match
                for class_index, class_name in enumerate(reference_head["classes"]):
                    observed_index = model.class_order[head].index(class_name)
                    delta = abs(
                        float(probabilities[head][row_index, observed_index])
                        - float(reference_head["probabilities"][class_index])
                    )
                    split_probability_delta = max(split_probability_delta, delta)
                    per_head[head]["maximum_probability_absolute_delta"] = max(
                        per_head[head]["maximum_probability_absolute_delta"], delta
                    )
                extra_classes = set(model.class_order[head]) - set(
                    reference_head["classes"]
                )
                for class_name in extra_classes:
                    observed_index = model.class_order[head].index(class_name)
                    split_probability_delta = max(
                        split_probability_delta,
                        abs(float(probabilities[head][row_index, observed_index])),
                    )
                    per_head[head]["maximum_probability_absolute_delta"] = max(
                        per_head[head]["maximum_probability_absolute_delta"],
                        abs(float(probabilities[head][row_index, observed_index])),
                    )
            reference_reasoning = reference_row["reasoning_tags"]
            observed_reasoning = labels["reasoning_tags"][row_index]
            reasoning_match = observed_reasoning == reference_reasoning["prediction"]
            split_label_matches += int(reasoning_match)
            split_label_comparisons += 1
            per_head["reasoning_tags"]["label_matches"] += int(reasoning_match)
            per_head["reasoning_tags"]["label_comparisons"] += 1
            row_exact &= reasoning_match
            for tag, reference_probability in reference_reasoning[
                "probabilities"
            ].items():
                observed_index = model.class_order["reasoning_tags"].index(tag)
                delta = abs(
                    float(probabilities["reasoning_tags"][row_index, observed_index])
                    - float(reference_probability)
                )
                split_probability_delta = max(split_probability_delta, delta)
                per_head["reasoning_tags"][
                    "maximum_probability_absolute_delta"
                ] = max(
                    per_head["reasoning_tags"][
                        "maximum_probability_absolute_delta"
                    ],
                    delta,
                )
            split_exact_rows += int(row_exact)
        comparisons = _metric_comparison(
            split=split,
            current_labels=labels,
            reference_rows=reference_rows,
            reference_metrics=reference_metrics,
            tolerance=metric_tolerance,
        )
        metric_comparisons.extend(comparisons)
        per_split[split] = {
            "rows": len(reference_rows),
            "rows_with_all_seven_labels_exact": split_exact_rows,
            "label_matches": split_label_matches,
            "label_comparisons": split_label_comparisons,
            "labels_exact": split_label_matches == split_label_comparisons,
            "maximum_probability_absolute_delta": split_probability_delta,
            "probabilities_within_tolerance": split_probability_delta
            <= probability_tolerance,
        }
        total_rows += len(reference_rows)
        exact_rows += split_exact_rows
        global_label_matches += split_label_matches
        global_label_comparisons += split_label_comparisons
        global_probability_delta = max(global_probability_delta, split_probability_delta)
    maximum_metric_delta = max(
        (item["absolute_delta"] for item in metric_comparisons), default=0.0
    )
    labels_exact = global_label_matches == global_label_comparisons
    probabilities_exact = global_probability_delta <= probability_tolerance
    metrics_exact = maximum_metric_delta <= metric_tolerance
    same_environment = bool(
        reference_audit.get("exact_reproduction_environment_match")
    )
    exact_authorized = same_environment and labels_exact and probabilities_exact and metrics_exact
    for values in per_head.values():
        values["labels_exact"] = (
            values["label_matches"] == values["label_comparisons"]
        )
        values["probabilities_within_tolerance"] = (
            values["maximum_probability_absolute_delta"] <= probability_tolerance
        )
    if exact_authorized:
        status = "BASELINE_V0_3_5_REPRODUCED_DIAGNOSTIC_ONLY"
        blocker_codes: list[str] = []
    elif not same_environment:
        status = "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
        blocker_codes = ["BLOCKED_REFERENCE_ENVIRONMENT_MISMATCH"]
    else:
        status = "BASELINE_V0_3_5_REPRODUCTION_MISMATCH"
        blocker_codes = ["BASELINE_REFERENCE_PREDICTION_MISMATCH"]
    return {
        "comparison_schema_version": "semantic-baseline-reference-comparison-v0.3.5",
        "status": status,
        "reference_package_manifest_id": reference_audit["package_manifest_id"],
        "reference_model_sha256": reference_audit["original_model_sha256"],
        "comparison_mode": (
            "SAME_REFERENCE_ENVIRONMENT"
            if same_environment
            else "CROSS_ENVIRONMENT_DIAGNOSTIC_ONLY"
        ),
        "same_reference_environment": same_environment,
        "rows": total_rows,
        "rows_with_all_seven_labels_exact": exact_rows,
        "label_matches": global_label_matches,
        "label_comparisons": global_label_comparisons,
        "prediction_labels_exact": labels_exact,
        "maximum_probability_absolute_delta": global_probability_delta,
        "probability_absolute_tolerance": probability_tolerance,
        "probabilities_within_tolerance": probabilities_exact,
        "maximum_metric_absolute_delta": maximum_metric_delta,
        "metrics_absolute_tolerance": metric_tolerance,
        "metrics_within_tolerance": metrics_exact,
        "metric_comparisons": metric_comparisons,
        "per_head": per_head,
        "per_split": per_split,
        "exact_reproduction_authorized": exact_authorized,
        "production_approval": False,
        "blocker_codes": blocker_codes,
    }
