"""Read-only Dev-448 disagreement analysis for the accepted M1 Encoder.

This module deliberately consumes only the frozen Dev weak-label file and the
matching canonical-input rows.  It never opens a Test/Gold/OOD label file,
never invokes a training entry point, and uses local-only model loading.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ProjectConfig
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, verify_content_addressed_id
from .preprocessing import PreprocessingContract, build_model_input
from .schema import SINGLE_LABEL_HEADS, V1_HEADS


ANALYSIS_SCHEMA_VERSION = "myresearcher.m2-dev-disagreement-analysis.v1"
EXPECTED_ENCODER_CONTENT_ADDRESS = (
    "b898ac50ac45baf56d094719213c4e3e23de10e2018cf825a69a372e748e8e58"
)
DEV_ROWS = 448
SAMPLE_ID_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')


@dataclass(frozen=True)
class VerifiedEncoderArtifact:
    root: Path
    cache_root: Path
    manifest: Mapping[str, Any]
    training_config: Mapping[str, Any]
    class_order: Mapping[str, list[str]]
    snapshot_sha256: str


@dataclass(frozen=True)
class VerifiedClassicalRun:
    root: Path
    run_manifest: Mapping[str, Any]
    model_manifest: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    preprocessing: PreprocessingContract


def _read_json(path: Path, *, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(code, "expected a readable JSON object", path=str(path)) from exc
    if not isinstance(value, Mapping):
        raise ContractError(code, "expected a JSON object", path=str(path))
    return value


def _require(value: bool, code: str, message: str, **details: Any) -> None:
    if not value:
        raise ContractError(code, message, **details)


def _as_hex(value: Any, *, code: str, field: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        code,
        f"{field} must be a lowercase SHA-256 hexadecimal digest",
    )
    return str(value)


def _safe_relative_path(value: Any, *, code: str) -> PurePosixPath:
    _require(isinstance(value, str) and value, code, "artifact path is required")
    relative = PurePosixPath(str(value))
    _require(
        not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(value),
        code,
        "artifact path escapes its root",
        path=value,
    )
    return relative


def _normalise(value: Any, *, field: str, sample_id: str) -> str:
    _require(isinstance(value, str), "M2_DEV_INPUT_INVALID", f"{field} must be a string", sample_id=sample_id)
    if field == "model_text":
        _require(bool(value.strip()), "M2_DEV_INPUT_INVALID", "model_text must be nonblank", sample_id=sample_id)
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _read_dev_labels(path: Path, class_order: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise ContractError("M2_DEV_LABELS_NOT_FOUND", "Dev weak-label file is unavailable", path=str(path)) from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError("M2_DEV_LABELS_INVALID", "Dev weak-label JSONL is invalid", line=line_number) from exc
            _require(isinstance(row, Mapping), "M2_DEV_LABELS_INVALID", "Dev weak-label row must be an object", line=line_number)
            sample_id = row.get("sample_id")
            _require(isinstance(sample_id, str) and sample_id, "M2_DEV_LABELS_INVALID", "Dev weak-label row lacks sample_id", line=line_number)
            _require(sample_id not in seen, "M2_DEV_DUPLICATE_SAMPLE_ID", "duplicate Dev sample_id", sample_id=sample_id)
            seen.add(sample_id)
            for head in SINGLE_LABEL_HEADS:
                value = row.get(head)
                _require(value in class_order[head], "M2_DEV_WEAK_LABEL_INVALID", "Dev weak label is outside the frozen class order", sample_id=sample_id, head=head)
            reasoning = row.get("reasoning_tags")
            _require(isinstance(reasoning, list), "M2_DEV_WEAK_LABEL_INVALID", "reasoning_tags must be a list", sample_id=sample_id)
            _require(
                all(isinstance(tag, str) and tag in class_order["reasoning_tags"] for tag in reasoning)
                and len(set(reasoning)) == len(reasoning),
                "M2_DEV_WEAK_LABEL_INVALID",
                "reasoning_tags must be unique frozen labels",
                sample_id=sample_id,
            )
            rows.append(dict(row))
    _require(len(rows) == DEV_ROWS, "M2_DEV_ROW_COUNT_MISMATCH", "Dev analysis requires exactly 448 weak-label rows", observed=len(rows), expected=DEV_ROWS)
    return rows


def _read_selected_canonical_inputs(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Decode only Dev rows from the mixed canonical-input JSONL file.

    The underlying canonical file contains rows from other roles.  The regular
    expression is used to select Dev identities before JSON decoding; this
    keeps Test, Gold, and OOD payloads out of the analysis process.
    """

    selected: dict[str, dict[str, Any]] = {}
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise ContractError("M2_DEV_INPUTS_NOT_FOUND", "canonical input file is unavailable", path=str(path)) from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            match = SAMPLE_ID_RE.search(line)
            if match is None or match.group(1) not in sample_ids:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError("M2_DEV_INPUTS_INVALID", "selected canonical JSONL is invalid", line=line_number) from exc
            _require(isinstance(row, Mapping), "M2_DEV_INPUTS_INVALID", "selected canonical row must be an object", line=line_number)
            sample_id = row.get("sample_id")
            _require(isinstance(sample_id, str) and sample_id in sample_ids, "M2_DEV_INPUTS_INVALID", "selected canonical row has an invalid sample_id", line=line_number)
            _require(sample_id not in selected, "M2_DEV_DUPLICATE_SAMPLE_ID", "duplicate canonical Dev sample_id", sample_id=sample_id)
            selected[sample_id] = dict(row)
    missing = sorted(sample_ids - set(selected))
    _require(not missing, "M2_DEV_CANONICAL_ROWS_MISSING", "canonical inputs are missing Dev rows", missing_count=len(missing), sample_ids=missing[:10])
    return selected


def load_dev_records(config_path: str | Path, class_order: Mapping[str, Sequence[str]]) -> tuple[ProjectConfig, list[dict[str, Any]], str]:
    """Load only Dev weak labels plus matching canonical input rows."""

    config = ProjectConfig.load(config_path)
    labels_path = config.data_path("split_labels_dev")
    label_rows = _read_dev_labels(labels_path, class_order)
    sample_ids = {str(row["sample_id"]) for row in label_rows}
    inputs = _read_selected_canonical_inputs(config.data_path("canonical_inputs"), sample_ids)
    records: list[dict[str, Any]] = []
    for label in label_rows:
        sample_id = str(label["sample_id"])
        input_row = inputs[sample_id]
        for field in ("stock_code", "stock_name"):
            if label.get(field) is not None:
                _require(input_row.get(field) == label.get(field), "M2_DEV_CANONICAL_METADATA_MISMATCH", "Dev weak-label metadata differs from canonical input", sample_id=sample_id, field=field)
        records.append(
            {
                "sample_id": sample_id,
                "stock_code": _normalise(input_row.get("stock_code") or "", field="stock_code", sample_id=sample_id),
                "stock_name": _normalise(input_row.get("stock_name") or "", field="stock_name", sample_id=sample_id),
                "model_text": _normalise(input_row.get("model_text"), field="model_text", sample_id=sample_id),
                "weak_label": {
                    **{head: str(label[head]) for head in SINGLE_LABEL_HEADS},
                    "reasoning_tags": list(label["reasoning_tags"]),
                },
            }
        )
    return config, records, sha256_file(labels_path)


def _validate_encoder_artifact_file_list(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "M2_ENCODER_ARTIFACT_INVALID", "accepted artifact requires a nonempty file list")
    seen: set[PurePosixPath] = set()
    for item in files:
        _require(isinstance(item, Mapping), "M2_ENCODER_ARTIFACT_INVALID", "artifact file entry must be an object")
        relative = _safe_relative_path(item.get("path"), code="M2_ENCODER_ARTIFACT_INVALID")
        _require(relative not in seen, "M2_ENCODER_ARTIFACT_INVALID", "artifact file list has duplicates", path=str(relative))
        seen.add(relative)
        expected_hash = _as_hex(item.get("sha256"), code="M2_ENCODER_ARTIFACT_INVALID", field=f"files[{relative}].sha256")
        artifact_file = root.joinpath(*relative.parts)
        _require(artifact_file.is_file(), "M2_ENCODER_ARTIFACT_FILE_MISSING", "accepted artifact file is missing", path=str(relative))
        _require(sha256_file(artifact_file) == expected_hash, "M2_ENCODER_ARTIFACT_HASH_MISMATCH", "accepted artifact file hash differs", path=str(relative))


def _expected_snapshot_sha256(contract_path: Path) -> str:
    contract = _read_json(contract_path, code="M2_ENCODER_CONTRACT_INVALID")
    selected = contract.get("selected_model")
    _require(isinstance(selected, Mapping), "M2_ENCODER_CONTRACT_INVALID", "selected_model is required")
    hashes = selected.get("artifact_hashes")
    _require(isinstance(hashes, Mapping), "M2_ENCODER_CONTRACT_INVALID", "selected_model.artifact_hashes is required")
    return _as_hex(hashes.get("pytorch_model_bin_sha256"), code="M2_ENCODER_CONTRACT_INVALID", field="pytorch_model_bin_sha256")


def verify_encoder_artifact(
    artifact_root: str | Path,
    *,
    expected_content_address: str = EXPECTED_ENCODER_CONTENT_ADDRESS,
    cache_root: str | Path | None = None,
    contract_path: str | Path = "manifests/encoder-experiment-contract-v1.json",
) -> VerifiedEncoderArtifact:
    """Verify the accepted local M1 artifact and its fixed local base snapshot."""

    root = Path(artifact_root).resolve()
    manifest = _read_json(root / "content-addressed-manifest.json", code="M2_ENCODER_MANIFEST_INVALID")
    _require(manifest.get("manifest_schema_version") == "myresearcher.encoder-m1-artifact-manifest.v2", "M2_ENCODER_MANIFEST_INVALID", "unexpected encoder artifact manifest schema")
    _require(manifest.get("diagnostic_only") is True, "M2_ENCODER_MANIFEST_INVALID", "Encoder artifact must remain diagnostic-only")
    observed_address = _as_hex(manifest.get("content_address"), code="M2_ENCODER_MANIFEST_INVALID", field="content_address")
    _require(observed_address == expected_content_address, "M2_ENCODER_ARTIFACT_NOT_ACCEPTED", "artifact does not match the accepted M1 content address", observed=observed_address, expected=expected_content_address)
    _require(
        content_addressed_id(manifest, omit_keys={"content_address"}) == observed_address,
        "CONTENT_ADDRESS_MISMATCH",
        "Encoder artifact content address differs",
    )
    _require(Path(str(manifest.get("artifact_root", ""))).resolve() == root, "M2_ENCODER_ARTIFACT_ROOT_MISMATCH", "artifact root differs from its signed manifest")
    _validate_encoder_artifact_file_list(root, manifest)
    training_config = _read_json(root / "training-config.json", code="M2_ENCODER_TRAINING_CONFIG_INVALID")
    class_order_value = _read_json(root / "class-order.json", code="M2_ENCODER_CLASS_ORDER_INVALID")
    classes = class_order_value.get("class_order")
    _require(isinstance(classes, Mapping) and set(classes) == set(V1_HEADS), "M2_ENCODER_CLASS_ORDER_INVALID", "encoder artifact must contain all seven head class orders")
    class_order: dict[str, list[str]] = {}
    for head in V1_HEADS:
        values = classes.get(head)
        _require(isinstance(values, list) and values and all(isinstance(value, str) for value in values), "M2_ENCODER_CLASS_ORDER_INVALID", "class order must be a nonempty label list", head=head)
        class_order[head] = list(values)
    _require(training_config.get("class_order") == class_order, "M2_ENCODER_CLASS_ORDER_INVALID", "training config and class-order artifact disagree")
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, Mapping), "M2_ENCODER_MANIFEST_INVALID", "Encoder provenance is required")
    _require(provenance.get("data_package_content_id") and provenance.get("reference_package_content_id"), "M2_ENCODER_MANIFEST_INVALID", "Encoder provenance lacks immutable data/reference identities")
    _require(training_config.get("model_id") == manifest.get("model_id"), "M2_ENCODER_TRAINING_CONFIG_INVALID", "model ID differs from artifact manifest")
    _require(training_config.get("revision") == manifest.get("resolved_revision"), "M2_ENCODER_TRAINING_CONFIG_INVALID", "revision differs from artifact manifest")
    manifest_cache = Path(str(manifest.get("cache_root", ""))).resolve()
    selected_cache = Path(cache_root).resolve() if cache_root is not None else manifest_cache
    _require(selected_cache == manifest_cache, "M2_ENCODER_CACHE_ROOT_MISMATCH", "analysis cache root differs from signed artifact cache root")
    snapshot = selected_cache / "official-snapshot" / str(manifest["resolved_revision"])
    expected_snapshot_hash = _expected_snapshot_sha256(Path(contract_path))
    _require(snapshot.is_dir() and (snapshot / "pytorch_model.bin").is_file(), "M2_ENCODER_SNAPSHOT_MISSING", "accepted local encoder snapshot is missing")
    observed_snapshot_hash = sha256_file(snapshot / "pytorch_model.bin")
    _require(observed_snapshot_hash == expected_snapshot_hash, "M2_ENCODER_SNAPSHOT_HASH_MISMATCH", "accepted local encoder snapshot hash differs")
    return VerifiedEncoderArtifact(
        root=root,
        cache_root=selected_cache,
        manifest=manifest,
        training_config=training_config,
        class_order=class_order,
        snapshot_sha256=observed_snapshot_hash,
    )


def _verify_run_candidate(
    run_root: Path,
    *,
    expected_data_id: str,
    expected_reference_id: str,
    expected_schema_version: str,
) -> VerifiedClassicalRun | None:
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        run_manifest = _read_json(manifest_path, code="M2_CLASSICAL_RUN_INVALID")
        expected_run_id = content_addressed_id(run_manifest, omit_keys={"run_manifest_id", "elapsed_seconds", "environment"})
        if run_manifest.get("run_manifest_id") != expected_run_id:
            return None
        if (
            run_manifest.get("status") != "COMPARABLE_DIAGNOSTIC_RUN_ONLY"
            or run_manifest.get("canonical_package_manifest_id") != expected_data_id
            or run_manifest.get("reference_package_manifest_id") != expected_reference_id
            or run_manifest.get("schema_version") != expected_schema_version
            or not isinstance(run_manifest.get("git"), Mapping)
            or run_manifest["git"].get("dirty") is not False
        ):
            return None
        model_manifest = _read_json(run_root / "model_manifest.json", code="M2_CLASSICAL_MODEL_MANIFEST_INVALID")
        verify_content_addressed_id(model_manifest, id_key="model_manifest_id")
        if run_manifest.get("model_manifest_id") != model_manifest.get("model_manifest_id"):
            return None
        required_hashes = {
            "model.joblib": model_manifest.get("model_sha256"),
            "thresholds.json": model_manifest.get("thresholds_sha256"),
            "preprocessing_contract.json": model_manifest.get("preprocessing_contract_sha256"),
        }
        for filename, expected_hash in required_hashes.items():
            if not isinstance(expected_hash, str) or not (run_root / filename).is_file() or sha256_file(run_root / filename) != expected_hash:
                return None
        thresholds = _read_json(run_root / "thresholds.json", code="M2_CLASSICAL_THRESHOLDS_INVALID")
        preprocessing = PreprocessingContract.load(run_root / "preprocessing_contract.json")
    except ContractError:
        return None
    return VerifiedClassicalRun(
        root=run_root,
        run_manifest=run_manifest,
        model_manifest=model_manifest,
        thresholds=thresholds,
        preprocessing=preprocessing,
    )


def select_unique_trusted_classical_run(
    run_catalog: str | Path,
    *,
    expected_data_id: str,
    expected_reference_id: str,
    expected_schema_version: str,
) -> VerifiedClassicalRun:
    """Select exactly one manifest-verified local comparable baseline run.

    Directory mtime, lexical recency, and a caller-provided "latest" shortcut
    are intentionally not used.  Zero or multiple valid candidates are a
    fail-closed blocker rather than an arbitrary model choice.
    """

    catalog = Path(run_catalog).resolve()
    _require(catalog.is_dir(), "M2_CLASSICAL_RUN_CATALOG_NOT_FOUND", "Classical run catalog is unavailable", path=str(catalog))
    candidates = [
        candidate
        for candidate in sorted(catalog.iterdir(), key=lambda item: item.name)
        if candidate.is_dir()
    ]
    trusted = [
        result
        for candidate in candidates
        if (result := _verify_run_candidate(candidate, expected_data_id=expected_data_id, expected_reference_id=expected_reference_id, expected_schema_version=expected_schema_version)) is not None
    ]
    if not trusted:
        raise ContractError("BLOCKED_CLASSICAL_TRUSTED_RUN_NOT_FOUND", "no local Classical run satisfied manifest/data/content-address requirements")
    if len(trusted) != 1:
        raise ContractError("BLOCKED_CLASSICAL_TRUSTED_RUN_AMBIGUOUS", "more than one local Classical run satisfied manifest/data/content-address requirements", candidate_run_ids=[item.run_manifest.get("run_id") for item in trusted])
    return trusted[0]


def _load_classical_model(run: VerifiedClassicalRun, class_order: Mapping[str, list[str]]) -> Any:
    import joblib

    model = joblib.load(run.root / "model.joblib")
    _require(getattr(model, "fitted", False) is True, "M2_CLASSICAL_MODEL_INVALID", "trusted Classical model is not fitted")
    _require(getattr(model, "class_order", None) == dict(class_order), "M2_CLASSICAL_CLASS_ORDER_MISMATCH", "trusted Classical model class order differs from accepted Encoder artifact")
    return model


def _ordered_probabilities(order: Sequence[str], probabilities: Sequence[float]) -> list[dict[str, Any]]:
    _require(len(order) == len(probabilities), "M2_PREDICTION_INVALID", "probability vector does not match class order")
    values = [float(value) for value in probabilities]
    _require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), "M2_PREDICTION_INVALID", "probability vector contains a non-probability")
    return [{"label": label, "probability": value} for label, value in zip(order, values, strict=True)]


def _argmax_label(order: Sequence[str], probabilities: Sequence[float]) -> tuple[str, float]:
    index = max(range(len(order)), key=lambda item: float(probabilities[item]))
    return str(order[index]), float(probabilities[index])


def _reasoning_prediction(
    order: Sequence[str],
    probabilities: Sequence[float],
    thresholds: Mapping[str, Any],
    *,
    ensure_at_least_one: bool,
) -> dict[str, Any]:
    ordered = _ordered_probabilities(order, probabilities)
    outcomes: dict[str, bool] = {}
    for item in ordered:
        threshold = thresholds.get(item["label"])
        _require(isinstance(threshold, (int, float)) and 0.0 <= float(threshold) <= 1.0, "M2_CLASSICAL_THRESHOLDS_INVALID", "Reasoning threshold is invalid", label=item["label"])
        outcomes[item["label"]] = item["probability"] >= float(threshold)
    labels = [label for label in order if outcomes[label]]
    if not labels and ensure_at_least_one:
        labels = [_argmax_label(order, probabilities)[0]]
        outcomes[labels[0]] = True
    decision_confidence = min(
        item["probability"] if outcomes[item["label"]] else 1.0 - item["probability"]
        for item in ordered
    )
    return {
        "predicted_labels": labels,
        "ordered_probabilities": ordered,
        "threshold_outcomes": outcomes,
        "confidence": float(decision_confidence),
    }


def _predict_classical(records: Sequence[Mapping[str, Any]], run: VerifiedClassicalRun, class_order: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    model = _load_classical_model(run, class_order)
    texts = [build_model_input(record, run.preprocessing) for record in records]
    probabilities = model.predict_probabilities(texts)
    reasoning_thresholds = run.thresholds.get("reasoning_tags")
    _require(isinstance(reasoning_thresholds, Mapping), "M2_CLASSICAL_THRESHOLDS_INVALID", "trusted run lacks Reasoning thresholds")
    ensure = run.thresholds.get("ensure_at_least_one_reasoning_tag")
    _require(isinstance(ensure, bool), "M2_CLASSICAL_THRESHOLDS_INVALID", "trusted run lacks the Reasoning fallback rule")
    predicted: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        result: dict[str, Any] = {}
        for head in SINGLE_LABEL_HEADS:
            order = class_order[head]
            vector = probabilities[head][row_index].tolist()
            label, confidence = _argmax_label(order, vector)
            result[head] = {
                "prediction": label,
                "ordered_probabilities": _ordered_probabilities(order, vector),
                "confidence": confidence,
            }
        result["reasoning_tags"] = _reasoning_prediction(
            class_order["reasoning_tags"],
            probabilities["reasoning_tags"][row_index].tolist(),
            reasoning_thresholds,
            ensure_at_least_one=ensure,
        )
        predicted.append(result)
    return predicted


def _encoder_input_ids(tokenizer: Any, record: Mapping[str, Any], config: Mapping[str, Any]) -> list[int]:
    def encode(value: str) -> list[int]:
        tokenized = tokenizer(value, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        values = tokenized.get("input_ids")
        _require(isinstance(values, list) and all(isinstance(item, int) for item in values), "M2_ENCODER_TOKENIZER_OUTPUT_INVALID", "tokenizer returned invalid input IDs", sample_id=record["sample_id"])
        return values

    code_ids = encode(str(record["stock_code"]))[: int(config["stock_code_token_cap"])]
    name_ids = encode(str(record["stock_name"]))[: int(config["stock_name_token_cap"])]
    text_ids = encode(str(record["model_text"]))
    remaining = int(config["max_length"]) - int(config["special_token_budget"]) - len(code_ids) - len(name_ids)
    _require(remaining >= 0, "M2_ENCODER_CONFIG_INVALID", "configured encoder segment caps exceed max length")
    if len(text_ids) > remaining:
        _require(config.get("truncation") == "HEAD_TAIL", "M2_ENCODER_CONFIG_INVALID", "accepted artifact must retain HEAD_TAIL truncation")
        leading = math.ceil(remaining / 2)
        trailing = remaining - leading
        text_ids = text_ids[:leading] + (text_ids[-trailing:] if trailing else [])
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    _require(isinstance(cls_id, int) and isinstance(sep_id, int), "M2_ENCODER_TOKENIZER_INVALID", "accepted tokenizer lacks CLS/SEP IDs")
    input_ids = [cls_id, *code_ids, sep_id, *name_ids, sep_id, *text_ids, sep_id]
    _require(len(input_ids) <= int(config["max_length"]), "M2_ENCODER_INPUT_LENGTH_INVALID", "manual encoder input exceeds frozen max length", sample_id=record["sample_id"])
    return input_ids


def _predict_encoder(records: Sequence[Mapping[str, Any]], artifact: VerifiedEncoderArtifact) -> list[dict[str, Any]]:
    """Run local CPU inference from the accepted artifact without training."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    config = artifact.training_config
    revision = str(artifact.manifest["resolved_revision"])
    snapshot = artifact.cache_root / "official-snapshot" / revision
    tokenizer = AutoTokenizer.from_pretrained(str(artifact.root / "tokenizer"), local_files_only=True, trust_remote_code=False, use_fast=True)
    base_model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    hidden_size = int(base_model.config.hidden_size)
    heads = torch.nn.ModuleDict({head: torch.nn.Linear(hidden_size, len(artifact.class_order[head])) for head in V1_HEADS})
    checkpoint = torch.load(artifact.root / "heads-checkpoint.pt", map_location="cpu", weights_only=True)
    _require(isinstance(checkpoint, Mapping), "M2_ENCODER_CHECKPOINT_INVALID", "accepted heads checkpoint must be an object")
    _require(checkpoint.get("model_id") == artifact.manifest.get("model_id") and checkpoint.get("revision") == revision, "M2_ENCODER_CHECKPOINT_INVALID", "accepted heads checkpoint identity differs from manifest")
    state = checkpoint.get("heads_state_dict")
    _require(isinstance(state, Mapping), "M2_ENCODER_CHECKPOINT_INVALID", "accepted heads checkpoint lacks head state")
    heads.load_state_dict(state)
    base_model.eval()
    heads.eval()
    pad_id = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
    batch_size = int(config["batch_size"])
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            batch_records = records[offset : offset + batch_size]
            rows = [_encoder_input_ids(tokenizer, record, config) for record in batch_records]
            maximum = max(len(row) for row in rows)
            input_ids = torch.tensor([row + [pad_id] * (maximum - len(row)) for row in rows], dtype=torch.long)
            attention_mask = torch.tensor([[1] * len(row) + [0] * (maximum - len(row)) for row in rows], dtype=torch.long)
            representation = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logits = {head: layer(representation) for head, layer in heads.items()}
            scalar_probabilities = {head: torch.softmax(logits[head], dim=1).cpu().tolist() for head in SINGLE_LABEL_HEADS}
            reasoning_probabilities = torch.sigmoid(logits["reasoning_tags"]).cpu().tolist()
            for row_index, _record in enumerate(batch_records):
                predictions: dict[str, Any] = {}
                for head in SINGLE_LABEL_HEADS:
                    order = artifact.class_order[head]
                    vector = scalar_probabilities[head][row_index]
                    label, confidence = _argmax_label(order, vector)
                    predictions[head] = {
                        "prediction": label,
                        "ordered_probabilities": _ordered_probabilities(order, vector),
                        "confidence": confidence,
                    }
                predictions["reasoning_tags"] = _reasoning_prediction(
                    artifact.class_order["reasoning_tags"],
                    reasoning_probabilities[row_index],
                    {tag: float(config["reasoning_probability_threshold"]) for tag in artifact.class_order["reasoning_tags"]},
                    ensure_at_least_one=False,
                )
                result.append(predictions)
    _require(len(result) == len(records), "M2_ENCODER_PREDICTION_COUNT_MISMATCH", "Encoder prediction count differs from Dev records")
    return result


def build_per_sample_analysis(
    records: Sequence[Mapping[str, Any]],
    classical_predictions: Sequence[Mapping[str, Any]],
    encoder_predictions: Sequence[Mapping[str, Any]],
    *,
    high_confidence_threshold: float,
) -> list[dict[str, Any]]:
    _require(0.0 < high_confidence_threshold <= 1.0, "M2_ANALYSIS_CONFIG_INVALID", "high-confidence threshold must be in (0, 1]")
    _require(len(records) == len(classical_predictions) == len(encoder_predictions) == DEV_ROWS, "M2_PREDICTION_COUNT_MISMATCH", "all analysis populations must be the frozen Dev 448")
    rows: list[dict[str, Any]] = []
    for record, classical, encoder in zip(records, classical_predictions, encoder_predictions, strict=True):
        heads: dict[str, Any] = {}
        disagreements: list[str] = []
        high_confidence_disagreements: list[str] = []
        weak_mismatch: list[str] = []
        for head in SINGLE_LABEL_HEADS:
            weak = record["weak_label"][head]
            c_value, e_value = classical[head], encoder[head]
            agreement = c_value["prediction"] == e_value["prediction"]
            c_match = c_value["prediction"] == weak
            e_match = e_value["prediction"] == weak
            if not agreement:
                disagreements.append(head)
                if min(float(c_value["confidence"]), float(e_value["confidence"])) >= high_confidence_threshold:
                    high_confidence_disagreements.append(head)
            if not c_match and not e_match:
                weak_mismatch.append(head)
            heads[head] = {
                "weak_label": weak,
                "classical": {**dict(c_value), "matches_weak_label": c_match},
                "encoder": {**dict(e_value), "matches_weak_label": e_match},
                "agreement": agreement,
            }
        weak_reasoning = list(record["weak_label"]["reasoning_tags"])
        c_reasoning, e_reasoning = classical["reasoning_tags"], encoder["reasoning_tags"]
        c_match = set(c_reasoning["predicted_labels"]) == set(weak_reasoning)
        e_match = set(e_reasoning["predicted_labels"]) == set(weak_reasoning)
        agreement = set(c_reasoning["predicted_labels"]) == set(e_reasoning["predicted_labels"])
        if not agreement:
            disagreements.append("reasoning_tags")
            if min(float(c_reasoning["confidence"]), float(e_reasoning["confidence"])) >= high_confidence_threshold:
                high_confidence_disagreements.append("reasoning_tags")
        if not c_match and not e_match:
            weak_mismatch.append("reasoning_tags")
        heads["reasoning_tags"] = {
            "weak_label": weak_reasoning,
            "classical": {**dict(c_reasoning), "matches_weak_label": c_match},
            "encoder": {**dict(e_reasoning), "matches_weak_label": e_match},
            "agreement": agreement,
        }
        review_score = len(disagreements) + 2 * len(high_confidence_disagreements) + len(weak_mismatch)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "heads": heads,
                "disagreement_heads": disagreements,
                "high_confidence_disagreement_heads": high_confidence_disagreements,
                "both_models_mismatch_weak_label_heads": weak_mismatch,
                "review_priority_score": review_score,
            }
        )
    return rows


def _aggregate_class(rows: Sequence[Mapping[str, Any]], head: str, label: str) -> dict[str, int]:
    if head == "reasoning_tags":
        def state(row: Mapping[str, Any], model: str) -> bool:
            return label in row["heads"][head][model]["predicted_labels"]

        def weak(row: Mapping[str, Any]) -> bool:
            return label in row["heads"][head]["weak_label"]
    else:
        def state(row: Mapping[str, Any], model: str) -> bool:
            return row["heads"][head][model]["prediction"] == label

        def weak(row: Mapping[str, Any]) -> bool:
            return row["heads"][head]["weak_label"] == label

    return {
        "weak_label_support": sum(weak(row) for row in rows),
        "classical_predicted": sum(state(row, "classical") for row in rows),
        "encoder_predicted": sum(state(row, "encoder") for row in rows),
        "disagreements": sum(state(row, "classical") != state(row, "encoder") for row in rows),
        "classical_only_matches_weak_label": sum(weak(row) and state(row, "classical") and not state(row, "encoder") for row in rows),
        "encoder_only_matches_weak_label": sum(weak(row) and state(row, "encoder") and not state(row, "classical") for row in rows),
        "both_models_mismatch_weak_label": sum(weak(row) and not state(row, "classical") and not state(row, "encoder") for row in rows),
    }


def aggregate_analysis(
    rows: Sequence[Mapping[str, Any]],
    class_order: Mapping[str, Sequence[str]],
    *,
    high_confidence_threshold: float,
    review_queue_size: int,
) -> dict[str, Any]:
    _require(len(rows) == DEV_ROWS, "M2_ANALYSIS_ROW_COUNT_MISMATCH", "aggregate analysis requires exactly Dev 448", observed=len(rows))
    _require(review_queue_size > 0, "M2_ANALYSIS_CONFIG_INVALID", "review queue size must be positive")
    heads: dict[str, Any] = {}
    for head in V1_HEADS:
        values = [row["heads"][head] for row in rows]
        disagreement_count = sum(not value["agreement"] for value in values)
        heads[head] = {
            "rows": len(rows),
            "disagreement_count": disagreement_count,
            "disagreement_rate": disagreement_count / len(rows),
            "classical_only_matches_weak_label": sum(value["classical"]["matches_weak_label"] and not value["encoder"]["matches_weak_label"] for value in values),
            "encoder_only_matches_weak_label": sum(value["encoder"]["matches_weak_label"] and not value["classical"]["matches_weak_label"] for value in values),
            "both_models_mismatch_weak_label": sum(not value["classical"]["matches_weak_label"] and not value["encoder"]["matches_weak_label"] for value in values),
            "high_confidence_disagreements": sum(head in row["high_confidence_disagreement_heads"] for row in rows),
            "classes": {label: _aggregate_class(rows, head, label) for label in class_order[head]},
        }
    queue = sorted(
        (
            {
                "sample_id": row["sample_id"],
                "review_priority_score": row["review_priority_score"],
                "disagreement_heads": row["disagreement_heads"],
                "high_confidence_disagreement_heads": row["high_confidence_disagreement_heads"],
                "both_models_mismatch_weak_label_heads": row["both_models_mismatch_weak_label_heads"],
            }
            for row in rows
            if row["disagreement_heads"]
        ),
        key=lambda row: (-int(row["review_priority_score"]), str(row["sample_id"])),
    )[:review_queue_size]
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "scope": "DEV_448_WEAK_LABEL_DIAGNOSTIC_ONLY_NOT_GOLD_NOT_TEST_NOT_OOD_NOT_PRODUCTION",
        "weak_label_interpretation": "All match/mismatch counts compare only to frozen Dev weak labels; they are not Gold, truth, or production-quality judgements.",
        "high_confidence_threshold": high_confidence_threshold,
        "heads": heads,
        "review_queue": queue,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _markdown_summary(summary: Mapping[str, Any], identity: Mapping[str, Any], analysis_id: str) -> str:
    lines = [
        "# M2 Dev-only Encoder / Classical Disagreement Analysis",
        "",
        f"Content address: `{analysis_id}`",
        "",
        "Scope: frozen Dev 448 weak labels only. All \"matches\" below mean a match to a weak label; they are neither Gold nor truth determinations.",
        "",
        "| Head | Disagreement rate | Classical-only weak-label matches | Encoder-only weak-label matches | Both mismatch weak labels | High-confidence disagreements |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for head, values in summary["heads"].items():
        lines.append(
            "| {head} | {rate:.2%} | {classical} | {encoder} | {both} | {high} |".format(
                head=head,
                rate=float(values["disagreement_rate"]),
                classical=int(values["classical_only_matches_weak_label"]),
                encoder=int(values["encoder_only_matches_weak_label"]),
                both=int(values["both_models_mismatch_weak_label"]),
                high=int(values["high_confidence_disagreements"]),
            )
        )
    lines.extend(
        [
            "",
            "## Input identities",
            "",
            f"- Encoder artifact: `{identity['encoder_content_address']}`",
            f"- Classical run manifest: `{identity['classical_run_manifest_id']}`",
            f"- Classical model SHA-256: `{identity['classical_model_sha256']}`",
            f"- Canonical data package: `{identity['data_package_content_id']}`",
            f"- Reference package: `{identity['reference_package_content_id']}`",
            f"- Dev weak-label SHA-256: `{identity['dev_weak_labels_sha256']}`",
            "",
            "## Highest-priority review IDs",
            "",
        ]
    )
    for row in summary["review_queue"][:10]:
        lines.append(
            "- `{sample_id}` — score {score}; disagreements: {heads}; high-confidence: {high}".format(
                sample_id=row["sample_id"],
                score=row["review_priority_score"],
                heads=", ".join(row["disagreement_heads"]),
                high=", ".join(row["high_confidence_disagreement_heads"]) or "none",
            )
        )
    return "\n".join(lines) + "\n"


def write_analysis_output(
    output_root: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[Path, str]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tmp-m2-dev-", dir=root))
    try:
        per_sample_path = temporary / "per-sample-analysis.jsonl"
        summary_path = temporary / "aggregate-report.json"
        _write_jsonl(per_sample_path, rows)
        _write_json(summary_path, summary)
        complete_identity = {
            **dict(identity),
            "per_sample_analysis_sha256": sha256_file(per_sample_path),
            "aggregate_report_sha256": sha256_file(summary_path),
            "dev_row_count": len(rows),
        }
        analysis_id = content_addressed_id(complete_identity)
        manifest = {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_content_address": analysis_id,
            "identity": complete_identity,
            "observation": {"created_at_utc": datetime.now(UTC).isoformat()},
        }
        _write_json(temporary / "content-addressed-manifest.json", manifest)
        (temporary / "summary.md").write_text(_markdown_summary(summary, complete_identity, analysis_id), encoding="utf-8")
        target = root / analysis_id
        if target.exists():
            existing = _read_json(target / "content-addressed-manifest.json", code="M2_ANALYSIS_OUTPUT_INVALID")
            _require(existing.get("analysis_content_address") == analysis_id and existing.get("identity") == complete_identity, "M2_ANALYSIS_OUTPUT_CONFLICT", "existing analysis output differs from the same content address")
            return target, analysis_id
        os.replace(temporary, target)
        return target, analysis_id
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise


def run_dev_disagreement_analysis(
    *,
    config_path: str | Path,
    encoder_artifact: str | Path,
    classical_run_catalog: str | Path,
    output_root: str | Path,
    encoder_cache_root: str | Path | None = None,
    confidence_threshold: float = 0.80,
    review_queue_size: int = 50,
    expected_encoder_content_address: str = EXPECTED_ENCODER_CONTENT_ADDRESS,
    contract_path: str | Path = "manifests/encoder-experiment-contract-v1.json",
) -> dict[str, Any]:
    """Produce a bounded, local-only Dev weak-label disagreement artifact."""

    artifact = verify_encoder_artifact(
        encoder_artifact,
        expected_content_address=expected_encoder_content_address,
        cache_root=encoder_cache_root,
        contract_path=contract_path,
    )
    provenance = artifact.manifest["provenance"]
    data_id = str(provenance["data_package_content_id"])
    reference_id = str(provenance["reference_package_content_id"])
    schema_version = str(provenance["schema_version"])
    config, records, dev_labels_sha = load_dev_records(config_path, artifact.class_order)
    trusted_run = select_unique_trusted_classical_run(
        classical_run_catalog,
        expected_data_id=data_id,
        expected_reference_id=reference_id,
        expected_schema_version=schema_version,
    )
    classical = _predict_classical(records, trusted_run, artifact.class_order)
    encoder = _predict_encoder(records, artifact)
    rows = build_per_sample_analysis(
        records,
        classical,
        encoder,
        high_confidence_threshold=confidence_threshold,
    )
    summary = aggregate_analysis(
        rows,
        artifact.class_order,
        high_confidence_threshold=confidence_threshold,
        review_queue_size=review_queue_size,
    )
    identity = {
        "scope": summary["scope"],
        "encoder_content_address": artifact.manifest["content_address"],
        "encoder_checkpoint_sha256": artifact.manifest["checkpoint_sha256"],
        "encoder_snapshot_pytorch_model_sha256": artifact.snapshot_sha256,
        "encoder_model_id": artifact.manifest["model_id"],
        "encoder_revision": artifact.manifest["resolved_revision"],
        "classical_run_id": trusted_run.run_manifest["run_id"],
        "classical_run_manifest_id": trusted_run.run_manifest["run_manifest_id"],
        "classical_model_manifest_id": trusted_run.model_manifest["model_manifest_id"],
        "classical_model_sha256": trusted_run.model_manifest["model_sha256"],
        "classical_status": trusted_run.run_manifest["status"],
        "data_package_content_id": data_id,
        "reference_package_content_id": reference_id,
        "schema_version": schema_version,
        "dev_weak_labels_sha256": dev_labels_sha,
        "analysis_module_sha256": sha256_file(Path(__file__)),
        "confidence_threshold": confidence_threshold,
        "review_queue_size": review_queue_size,
        "config_sha256": sha256_file(config.path),
    }
    output_dir, analysis_id = write_analysis_output(
        output_root,
        rows=rows,
        summary=summary,
        identity=identity,
    )
    return {
        "status": "M2_DEV_DISAGREEMENT_ANALYSIS_COMPLETED_WEAK_LABEL_DIAGNOSTIC_ONLY",
        "analysis_content_address": analysis_id,
        "output_dir": str(output_dir),
        "dev_rows": len(rows),
        "input_identity": identity,
        "summary": summary,
        "production_approval": False,
        "test_accessed": False,
        "gold_accessed": False,
        "ood_accessed": False,
        "training_invoked": False,
        "external_model_download_invoked": False,
        "external_llm_invoked": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Dev-only Encoder/Classical disagreement analysis")
    parser.add_argument("--config", required=True, help="config that points to the immutable local data package")
    parser.add_argument("--encoder-artifact", required=True)
    parser.add_argument("--encoder-cache-root")
    parser.add_argument("--classical-run-catalog", required=True)
    parser.add_argument("--output-root", default="runs/m2-dev-disagreement")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--review-queue-size", type=int, default=50)
    parser.add_argument("--expected-encoder-content-address", default=EXPECTED_ENCODER_CONTENT_ADDRESS)
    parser.add_argument("--contract", default="manifests/encoder-experiment-contract-v1.json")
    args = parser.parse_args(argv)
    try:
        result = run_dev_disagreement_analysis(
            config_path=args.config,
            encoder_artifact=args.encoder_artifact,
            encoder_cache_root=args.encoder_cache_root,
            classical_run_catalog=args.classical_run_catalog,
            output_root=args.output_root,
            confidence_threshold=args.confidence_threshold,
            review_queue_size=args.review_queue_size,
            expected_encoder_content_address=args.expected_encoder_content_address,
            contract_path=args.contract,
        )
        exit_code = 0
    except ContractError as exc:
        result = {
            "status": "BLOCKED_M2_DEV_DISAGREEMENT_ANALYSIS",
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
            "production_approval": False,
            "training_invoked": False,
        }
        exit_code = 3
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
