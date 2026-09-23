"""Unified CPU inference runtime for the frozen M2.2 head-selected candidate.

The module is deliberately inference-only.  It loads the immutable selected
artifact, the fixed local LERT snapshot, six LERT specialist states, and the
byte-for-byte Classical fallback.  It never fits a model and it has no Test,
Anchor, Gold, OOD, or production-data code path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import hstack

from . import encoder_m1 as m1
from . import encoder_m2_1_softlabel as m21
from . import encoder_m2_lert_final as lert
from .errors import ContractError
from .hashes import content_addressed_id, sha256_file, verify_content_addressed_id
from .preprocessing import PreprocessingContract, build_model_inputs
from .schema import LabelSchema
from .output_conditioning import add_conditioned_outputs


STAGE_ID = "M2_SELECTED_CANDIDATE_FROZEN_FOR_M3"
ARTIFACT_SCHEMA_VERSION = "myresearcher.m2-selected-candidate-artifact.v1"
MODEL_ID = "hfl/chinese-lert-small"
REVISION = "69e3e69ba258be5b301b26937e5b55a076c90460"
LICENSE = "Apache-2.0"
PARENT_CONTENT_ADDRESS = "ca2d1399ab252a14d1fb33f4e6ee2581f2253b0d5c63d84ef0ebb5fdad3caf92"
HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "context_dependency",
    "reasoning_tags",
)
LERT_HEADS = (
    "target_mode",
    "stance",
    "emotion_primary",
    "emotion_target",
    "action_tendency",
    "reasoning_tags",
)
SCALAR_HEADS = HEADS[:-1]
REASONING_HEAD = "reasoning_tags"
SARCASM_LABEL = "SARCASM_IRONY"
SARCASM_THRESHOLD = 0.45
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".encoder-artifacts" / "hf-cache-lert-small"
DEFAULT_PARENT_BUNDLE = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-2-action-balanced-three-seed-20260904" / "unified-candidate"
DEFAULT_SELECTED_ARTIFACT = Path(__file__).resolve().parents[3] / "model-artifacts" / "m2-selected-candidate-frozen-for-m3-20260904"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "manifests" / "encoder-m2-experiment-contract-v1.json"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "baseline_v0.3.5.yaml"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "semantic-schema-calibrated-v0.2.1.json"


@dataclass(frozen=True)
class InferenceRecord:
    """Input-only record; labels and field weights are intentionally absent."""

    sample_id: str
    stock_code: str
    stock_name: str
    model_text: str
    metadata: Mapping[str, Any] | None = None


def _require(ok: bool, code: str, message: str, **details: Any) -> None:
    if not ok:
        raise ContractError(code, message, **details)


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(code, "required JSON is missing", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContractError(code, "required JSON is invalid", path=str(path), detail=str(exc)) from exc
    _require(isinstance(value, dict), code, "required JSON must be an object", path=str(path))
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_manifest(root: Path, *, expected_content_address: str | None = None) -> dict[str, Any]:
    """Verify every listed file and the content address before loading state."""

    manifest_path = root / "content-addressed-manifest.json"
    manifest = _read_json(manifest_path, "M2_SELECTED_MANIFEST_MISSING")
    observed = verify_content_addressed_id(manifest, id_key="content_address")
    if expected_content_address is not None:
        _require(observed == expected_content_address, "M2_PARENT_CONTENT_ADDRESS_MISMATCH", "parent bundle content address changed", observed=observed, expected=expected_content_address)
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "M2_SELECTED_MANIFEST_INVALID", "manifest files are required")
    for item in files:
        _require(isinstance(item, Mapping), "M2_SELECTED_MANIFEST_INVALID", "manifest file entry must be an object")
        relative = item.get("path")
        _require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "M2_SELECTED_MANIFEST_INVALID", "manifest path is not relative", path=relative)
        path = root / relative
        _require(path.is_file(), "M2_SELECTED_ARTIFACT_FILE_MISSING", "manifest file is missing", path=relative)
        _require(int(item.get("bytes", -1)) == path.stat().st_size and item.get("sha256") == sha256_file(path), "M2_SELECTED_ARTIFACT_FILE_MISMATCH", "selected artifact file hash differs", path=relative)
    return manifest


def map_classical_scalar_probabilities(native: Any, classes: Sequence[Any], schema_order: Sequence[str]) -> np.ndarray:
    """Map estimator columns by class name, including absent schema classes as zero."""

    values = np.asarray(native, dtype=float)
    _require(values.ndim == 2 and values.shape[1] == len(classes), "M2_CLASSICAL_PROBABILITY_SHAPE_INVALID", "Classical probability columns do not match estimator.classes_")
    ordered = np.zeros((values.shape[0], len(schema_order)), dtype=float)
    index_by_label = {label: index for index, label in enumerate(schema_order)}
    for source_index, value in enumerate(classes):
        label = str(value)
        if label in index_by_label:
            ordered[:, index_by_label[label]] = values[:, source_index]
    _require(bool(np.isfinite(ordered).all()), "M2_CLASSICAL_NONFINITE_PROBABILITY", "Classical probabilities are non-finite")
    return ordered


def classical_positive_probability(native: Any, classes: Sequence[Any], *, positive_class: Any = 1) -> np.ndarray:
    """Return the positive column without assuming its position."""

    values = np.asarray(native, dtype=float)
    _require(values.ndim == 2 and values.shape[1] == len(classes), "M2_CLASSICAL_REASONING_SHAPE_INVALID", "reasoning probability columns do not match estimator.classes_")
    matches = [index for index, value in enumerate(classes) if str(value) == str(positive_class)]
    _require(len(matches) == 1, "M2_CLASSICAL_POSITIVE_CLASS_MISSING", "Classical reasoning estimator has no unique positive class")
    result = values[:, matches[0]]
    _require(bool(np.isfinite(result).all()), "M2_CLASSICAL_NONFINITE_PROBABILITY", "Classical reasoning probabilities are non-finite")
    return result


def _record_as_mapping(record: m1.M1Record) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "stock_code": record.stock_code,
        "stock_name": record.stock_name,
        "model_text": record.model_text,
    }


def _validate_inference_record(value: Mapping[str, Any]) -> InferenceRecord:
    required = ("sample_id", "stock_code", "stock_name", "model_text")
    _require(all(isinstance(value.get(field), str) and value[field] for field in required), "M2_INFERENCE_RECORD_INVALID", "unlabelled inference records require four non-empty string fields")
    forbidden = {"label", "labels", "weights", "target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "context_dependency", "reasoning_tags"}
    _require(not forbidden.intersection(value), "M2_INFERENCE_LABELS_PRESENT", "inference input must not contain labels or weights", fields=sorted(forbidden.intersection(value)))
    metadata = {key: value[key] for key in ("expected_scope", "challenge_category") if key in value}
    return InferenceRecord(sample_id=value["sample_id"], stock_code=value["stock_code"], stock_name=value["stock_name"], model_text=value["model_text"], metadata=metadata)


def load_unlabelled_jsonl(path: str | Path) -> list[InferenceRecord]:
    records: list[InferenceRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError("M2_INFERENCE_JSONL_INVALID", "input JSONL contains invalid JSON", line=line_number, path=str(path)) from exc
        _require(isinstance(value, Mapping), "M2_INFERENCE_RECORD_INVALID", "input JSONL rows must be objects", line=line_number)
        records.append(_validate_inference_record(value))
    _require(records and len({record.sample_id for record in records}) == len(records), "M2_INFERENCE_SAMPLE_ID_INVALID", "input JSONL requires unique non-empty sample IDs")
    return records


def build_inference_batch(torch: Any, tokenizer: Any, records: Sequence[Any], config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Construct only model inputs for records that carry no labels or weights."""

    _require(records, "M2_INFERENCE_EMPTY_BATCH", "cannot build an empty inference batch")
    rows = [m1.build_input_ids(tokenizer, record, config) for record in records]
    pad_id = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0
    max_len = max(len(row) for row in rows)
    input_ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _prediction_payload(
    records: Sequence[Any],
    schema: LabelSchema,
    scalar_probabilities: Mapping[str, np.ndarray],
    reasoning_probabilities: np.ndarray,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reasoning_order = list(schema.class_order[REASONING_HEAD])
    threshold_map = {label: float(thresholds.get(label, 0.5)) for label in reasoning_order}
    for row_index, record in enumerate(records):
        predictions: dict[str, Any] = {}
        for head in SCALAR_HEADS:
            order = list(schema.class_order[head])
            vector = np.asarray(scalar_probabilities[head][row_index], dtype=float)
            _require(bool(np.isfinite(vector).all()), "M2_RUNTIME_NONFINITE_OUTPUT", "scalar output is non-finite", head=head, sample_id=record.sample_id)
            _require(math.isclose(float(vector.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6), "M2_RUNTIME_SCALAR_PROBABILITY_NOT_NORMALIZED", "scalar probabilities do not sum to one", head=head, sample_id=record.sample_id, total=float(vector.sum()))
            predictions[head] = {
                "probabilities": {label: float(vector[index]) for index, label in enumerate(order)},
                "prediction": order[int(np.argmax(vector))],
            }
        reasoning_vector = np.asarray(reasoning_probabilities[row_index], dtype=float)
        _require(len(reasoning_vector) == len(reasoning_order) and bool(np.isfinite(reasoning_vector).all()), "M2_RUNTIME_NONFINITE_OUTPUT", "reasoning output is non-finite", sample_id=record.sample_id)
        predicted_tags = [label for index, label in enumerate(reasoning_order) if reasoning_vector[index] >= threshold_map[label]]
        predictions[REASONING_HEAD] = {
            "probabilities": {label: float(reasoning_vector[index]) for index, label in enumerate(reasoning_order)},
            "prediction": predicted_tags,
            "predicted_labels": predicted_tags,
            "thresholds": threshold_map,
        }
        output = {"sample_id": record.sample_id, "predictions": predictions}
        metadata = getattr(record, "metadata", None)
        if metadata:
            output.update({key: value for key, value in metadata.items() if key in {"expected_scope", "challenge_category"}})
        rows.append(add_conditioned_outputs(output))
    return rows


def _classical_predictions(records: Sequence[Any], schema: LabelSchema, prep: PreprocessingContract, legacy: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    for estimator in [*legacy["scalar_models"].values(), *legacy["reason_models"].values()]:
        if not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"
    texts = build_model_inputs([_record_as_mapping(record) for record in records], prep)
    matrix = hstack([legacy["char_vectorizer"].transform(texts), legacy["word_vectorizer"].transform(texts)], format="csr")
    scalar: dict[str, np.ndarray] = {}
    for head in SCALAR_HEADS:
        estimator = legacy["scalar_models"][head]
        scalar[head] = map_classical_scalar_probabilities(estimator.predict_proba(matrix), estimator.classes_, schema.class_order[head])
    reason_estimator = legacy["reason_models"][SARCASM_LABEL]
    sarcasm = classical_positive_probability(reason_estimator.predict_proba(matrix), reason_estimator.classes_)
    return scalar, sarcasm


class UnifiedCPUInferenceRuntime:
    """One-input, seven-head CPU runtime backed by the selected artifact."""

    def __init__(self, *, artifact_dir: Path, schema: LabelSchema, config: Mapping[str, Any], tokenizer: Any, model: Any, torch: Any, prep: PreprocessingContract, legacy: Mapping[str, Any], thresholds: Mapping[str, Any], snapshot_identity: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        self.artifact_dir = artifact_dir
        self.schema = schema
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.prep = prep
        self.legacy = legacy
        self.thresholds = thresholds
        self.snapshot_identity = snapshot_identity
        self.manifest = manifest

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, *, config_path: str | Path = CONFIG_PATH, cache_dir: str | Path = DEFAULT_CACHE, verify_manifest: bool = True) -> "UnifiedCPUInferenceRuntime":
        artifact = Path(artifact_dir).resolve()
        manifest: Mapping[str, Any]
        if verify_manifest:
            manifest = _verify_manifest(artifact)
            _require(manifest.get("status") == STAGE_ID and manifest.get("selected_candidate") is True and manifest.get("production_approval") is False, "M2_SELECTED_STATUS_INVALID", "artifact is not the frozen M2 selected candidate")
            _require(manifest.get("parent_artifact_content_address") == PARENT_CONTENT_ADDRESS, "M2_SELECTED_PARENT_INVALID", "selected artifact is not directly derived from M2.2 bundle")
        else:
            manifest = {"status": STAGE_ID, "selected_candidate": True, "production_approval": False, "parent_artifact_content_address": PARENT_CONTENT_ADDRESS}
        selection = _read_json(artifact / "selection-config.json", "M2_SELECTED_SELECTION_CONFIG_MISSING")
        _require(selection.get("status") == STAGE_ID and selection.get("selected_candidate") is True, "M2_SELECTED_SELECTION_CONFIG_INVALID", "selection config is not frozen selected state")
        bundle_path = artifact / "unified-runtime-bundle.pt"
        classical_path = artifact / "classical-fallback" / "original_model_v0.3.5.joblib"
        _require(bundle_path.is_file() and classical_path.is_file(), "M2_SELECTED_RUNTIME_FILES_MISSING", "selected runtime files are incomplete")
        frozen = lert._contract_requirements(Path(CONTRACT_PATH))
        schema = LabelSchema.load(Path(SCHEMA_PATH))
        project_config = m1.ProjectConfig.load(config_path)
        snapshot, snapshot_identity = m21._snapshot(Path(cache_dir).resolve())
        runtime = m21._load_runtime()
        np_module, torch, AutoModel, AutoTokenizer = runtime
        del np_module
        try:
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(bundle_path, map_location="cpu")
        _require(isinstance(bundle, Mapping), "M2_SELECTED_BUNDLE_INVALID", "selected bundle must be a mapping")
        identity = bundle.get("model_identity")
        _require(isinstance(identity, Mapping) and identity.get("model_id") == MODEL_ID and identity.get("revision") == REVISION and identity.get("license") == LICENSE, "M2_SELECTED_MODEL_IDENTITY_INVALID", "selected bundle model identity differs")
        _require(bundle.get("status") == STAGE_ID and bundle.get("selected_candidate") is True, "M2_SELECTED_BUNDLE_STATUS_INVALID", "selected runtime bundle is not frozen selected state")
        _require(set(bundle.get("lert_specialist_block_state_dict", {})) == set(LERT_HEADS) and set(bundle.get("lert_heads_state_dict", {})) == set(LERT_HEADS), "M2_SELECTED_LERT_STATE_INCOMPLETE", "selected bundle must contain six LERT specialist states")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
        runtime_config = lert._config(frozen["contract"], schema.class_order)
        model = lert._make_model(torch, AutoModel, snapshot, schema, float(runtime_config["head_dropout"]))
        for head in LERT_HEADS:
            model.specialist_blocks[head].load_state_dict(bundle["lert_specialist_block_state_dict"][head])
            model.heads[head].load_state_dict(bundle["lert_heads_state_dict"][head])
        model = model.to(torch.device("cpu"))
        model.eval()
        legacy = joblib.load(classical_path)
        prep = PreprocessingContract.load(project_config.data_path("preprocessing_contract"))
        thresholds = bundle.get("reasoning_thresholds", {})
        _require(float(thresholds.get(SARCASM_LABEL, -1.0)) == SARCASM_THRESHOLD, "M2_SELECTED_SARCASM_THRESHOLD_INVALID", "SARCASM_IRONY must use the Classical 0.45 threshold")
        return cls(artifact_dir=artifact, schema=schema, config=runtime_config, tokenizer=tokenizer, model=model, torch=torch, prep=prep, legacy=legacy, thresholds=thresholds, snapshot_identity=snapshot_identity, manifest=manifest)

    def predict_records(self, records: Sequence[Any]) -> list[dict[str, Any]]:
        if not records:
            return []
        lert_probabilities: dict[str, list[np.ndarray]] = {head: [] for head in SCALAR_HEADS}
        reasoning_batches: list[np.ndarray] = []
        self.model.eval()
        with self.torch.no_grad():
            for offset in range(0, len(records), 16):
                batch_records = records[offset:offset + 16]
                batch = build_inference_batch(self.torch, self.tokenizer, batch_records, self.config, self.torch.device("cpu"))
                outputs = self.model(batch["input_ids"], batch["attention_mask"])
                for head in SCALAR_HEADS:
                    values = self.torch.softmax(outputs[head], dim=-1).detach().cpu().numpy()
                    lert_probabilities[head].append(values)
                reasoning_batches.append(self.torch.sigmoid(outputs[REASONING_HEAD]).detach().cpu().numpy())
        scalar = {head: np.concatenate(chunks, axis=0) for head, chunks in lert_probabilities.items()}
        reasoning = np.concatenate(reasoning_batches, axis=0)
        classical_scalar, sarcasm = _classical_predictions(records, self.schema, self.prep, self.legacy)
        for head in ("context_dependency",):
            scalar[head] = classical_scalar[head]
        reasoning_order = list(self.schema.class_order[REASONING_HEAD])
        sarcasm_index = reasoning_order.index(SARCASM_LABEL)
        reasoning[:, sarcasm_index] = sarcasm
        return _prediction_payload(records, self.schema, scalar, reasoning, self.thresholds)


def _promote_bundle(parent_bundle: Path, selected_bundle: Path, torch: Any) -> None:
    try:
        bundle = torch.load(parent_bundle, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(parent_bundle, map_location="cpu")
    _require(isinstance(bundle, Mapping), "M2_PARENT_BUNDLE_INVALID", "M2.2 bundle must be a mapping")
    bundle = dict(bundle)
    bundle["stage"] = STAGE_ID
    bundle["status"] = STAGE_ID
    bundle["selected_candidate"] = True
    bundle["parent_m2_2_content_address"] = PARENT_CONTENT_ADDRESS
    torch.save(bundle, selected_bundle)


def freeze_selected_artifact(*, config_path: str | Path = CONFIG_PATH, cache_dir: str | Path = DEFAULT_CACHE, parent_bundle_dir: str | Path = DEFAULT_PARENT_BUNDLE, selected_artifact_dir: str | Path = DEFAULT_SELECTED_ARTIFACT) -> dict[str, Any]:
    """Create the immutable selected artifact and run the three-row Dev smoke."""

    parent = Path(parent_bundle_dir).resolve()
    selected = Path(selected_artifact_dir).resolve()
    _verify_manifest(parent, expected_content_address=PARENT_CONTENT_ADDRESS)
    _require(not selected.exists(), "M2_SELECTED_OUTPUT_ALREADY_EXISTS", "selected artifact directory already exists", path=str(selected))
    parent_bundle = parent / "unified-runtime-bundle.pt"
    parent_classical = parent / "classical-fallback" / "original_model_v0.3.5.joblib"
    _require(parent_bundle.is_file() and parent_classical.is_file(), "M2_PARENT_RUNTIME_FILES_MISSING", "M2.2 parent runtime files are incomplete")
    selected.mkdir(parents=True, exist_ok=False)
    (selected / "classical-fallback").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent_classical, selected / "classical-fallback" / parent_classical.name)
    np_module, torch, _, _ = m21._load_runtime()
    del np_module
    _promote_bundle(parent_bundle, selected / "unified-runtime-bundle.pt", torch)
    parent_manifest = _read_json(parent / "content-addressed-manifest.json", "M2_PARENT_MANIFEST_MISSING")
    selection = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": STAGE_ID,
        "selected_candidate": True,
        "production_approval": False,
        "parent_artifact": {"content_address": PARENT_CONTENT_ADDRESS, "relative_source": "m2-2-action-balanced-three-seed-20260904/unified-candidate"},
        "parent_manifest_sha256": sha256_file(parent / "content-addressed-manifest.json"),
        "model_identity": {"model_id": MODEL_ID, "revision": REVISION, "license": LICENSE, "cache_content_address": "64ff6776b1fccf14c1cfcd513908438c25a01871abae25e3f0d71982e2d3c4b1"},
        "head_sources": {"target_mode": "LERT seed-35 checkpoint", "stance": "LERT seed-35 checkpoint", "emotion_primary": "LERT seed-35 checkpoint", "emotion_target": "LERT seed-35 checkpoint", "action_tendency": "LERT balanced hard-label seed-35", "context_dependency": "Classical frozen estimator", "reasoning_tags": {"labels_14": "LERT seed-35 checkpoint", "SARCASM_IRONY": "Classical frozen estimator"}},
        "context_mapping": "estimator.classes_ mapped to Schema four-class order; absent classes are zero",
        "reasoning_sarcasm_irony_threshold": SARCASM_THRESHOLD,
        "forbidden_inputs": ["Test", "Anchor", "Gold", "OOD", "Human100", "reference_predictions", "LLM", "cloud", "production_inference"],
        "source_parent_manifest_content_address": parent_manifest.get("content_address"),
    }
    _write_json(selected / "selection-config.json", selection)
    runtime = UnifiedCPUInferenceRuntime.from_artifact(selected, config_path=config_path, cache_dir=cache_dir, verify_manifest=False)
    project_config = m1.ProjectConfig.load(config_path)
    _, _, dev = m1.load_m1_partitions(project_config)
    smoke_records = dev[:3]
    predictions = runtime.predict_records(smoke_records)
    expected_shapes = {"target_mode": 4, "stance": 5, "emotion_primary": 11, "emotion_target": 7, "action_tendency": 9, "context_dependency": 4, "reasoning_tags": 15}
    for row in predictions:
        for head, cardinality in expected_shapes.items():
            values = row["predictions"][head]["probabilities"]
            _require(len(values) == cardinality and bool(np.isfinite(list(values.values())).all()), "M2_SELECTED_SMOKE_OUTPUT_INVALID", "smoke output has wrong shape or non-finite value", head=head, sample_id=row["sample_id"])
    _write_json(selected / "smoke-predictions.json", predictions)
    smoke_summary = {
        "scope": "first_three_Dev_rows_only",
        "sample_ids": [record.sample_id for record in smoke_records],
        "output_shapes": expected_shapes,
        "scalar_probability_sums": {row["sample_id"]: {head: round(sum(row["predictions"][head]["probabilities"].values()), 12) for head in SCALAR_HEADS} for row in predictions},
        "all_outputs_finite": True,
        "device": "cpu",
        "selected_candidate": True,
        "production_approval": False,
    }
    _write_json(selected / "smoke-summary.json", smoke_summary)
    runtime_evidence = {
        "model_loaded": True,
        "training_invoked": False,
        "device": "cpu",
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "torch": {"version": str(torch.__version__), "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
        "transformers": {"version": _version_of("transformers")},
        "numpy": {"version": np.__version__},
        "tokenizers": {"version": _version_of("tokenizers")},
        "cache_snapshot": runtime.snapshot_identity,
        "smoke_summary_sha256": sha256_file(selected / "smoke-summary.json"),
    }
    _write_json(selected / "runtime-load-evidence.json", runtime_evidence)
    payload = {
        "manifest_schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": STAGE_ID,
        "selected_candidate": True,
        "production_approval": False,
        "parent_artifact_content_address": PARENT_CONTENT_ADDRESS,
        "model_identity": selection["model_identity"],
        "head_sources": selection["head_sources"],
        "context_mapping": selection["context_mapping"],
        "reasoning_sarcasm_irony_threshold": SARCASM_THRESHOLD,
        "cache_snapshot": runtime.snapshot_identity,
        "runtime_load_evidence_sha256": sha256_file(selected / "runtime-load-evidence.json"),
        "smoke_summary_sha256": sha256_file(selected / "smoke-summary.json"),
        "smoke_scope": "Dev_3_only",
        "forbidden_inputs": selection["forbidden_inputs"],
    }
    manifest = m1._write_content_manifest(selected, payload)
    return {"status": STAGE_ID, "selected_candidate": True, "artifact_dir": str(selected), "content_address": manifest["content_address"], "smoke_summary": smoke_summary, "runtime_evidence": runtime_evidence, "parent_content_address": PARENT_CONTENT_ADDRESS}


def _version_of(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


def _smoke_existing(*, artifact_dir: Path, config_path: Path, cache_dir: Path) -> dict[str, Any]:
    runtime = UnifiedCPUInferenceRuntime.from_artifact(artifact_dir, config_path=config_path, cache_dir=cache_dir)
    project_config = m1.ProjectConfig.load(config_path)
    _, _, dev = m1.load_m1_partitions(project_config)
    rows = runtime.predict_records(dev[:3])
    return {"artifact_dir": str(artifact_dir), "content_address": runtime.manifest["content_address"], "predictions": rows}


def _write_prediction_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--parent-bundle", type=Path, default=None)
    parser.add_argument("--selected-artifact", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--smoke", action="store_true", help="run the three-row Dev CPU smoke")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="input-only JSONL with sample_id, stock_code, stock_name, model_text")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="write predictions for --input-jsonl")
    args = parser.parse_args(argv)
    if args.parent_bundle is not None and not args.selected_artifact.exists():
        result = freeze_selected_artifact(config_path=args.config, cache_dir=args.cache_dir, parent_bundle_dir=args.parent_bundle, selected_artifact_dir=args.selected_artifact)
        if args.input_jsonl is not None:
            runtime = UnifiedCPUInferenceRuntime.from_artifact(args.selected_artifact.resolve(), config_path=args.config, cache_dir=args.cache_dir)
            rows = runtime.predict_records(load_unlabelled_jsonl(args.input_jsonl))
            if args.output_jsonl is not None:
                _write_prediction_jsonl(args.output_jsonl, rows)
            result = {"artifact_dir": result["artifact_dir"], "content_address": result["content_address"], "input_records": len(rows), "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None}
    elif args.selected_artifact.exists():
        if args.input_jsonl is not None:
            runtime = UnifiedCPUInferenceRuntime.from_artifact(args.selected_artifact.resolve(), config_path=args.config, cache_dir=args.cache_dir)
            rows = runtime.predict_records(load_unlabelled_jsonl(args.input_jsonl))
            if args.output_jsonl is not None:
                _write_prediction_jsonl(args.output_jsonl, rows)
            result = {"artifact_dir": str(args.selected_artifact.resolve()), "content_address": runtime.manifest["content_address"], "input_records": len(rows), "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None}
        else:
            _require(args.smoke, "M2_SELECTED_SMOKE_FLAG_REQUIRED", "an existing selected artifact requires --smoke or --input-jsonl for this CLI")
            result = _smoke_existing(artifact_dir=args.selected_artifact.resolve(), config_path=args.config, cache_dir=args.cache_dir)
    else:
        raise ContractError("M2_SELECTED_ARTIFACT_NOT_FOUND", "provide --parent-bundle to freeze the selected artifact")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke
    raise SystemExit(main())
