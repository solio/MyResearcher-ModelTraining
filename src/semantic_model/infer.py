from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
from jsonschema import Draft202012Validator

from .data import index_by_sample_id, read_json, read_jsonl
from .errors import ContractError
from .hashes import sha256_file, verify_content_addressed_id
from .models.classical import ClassicalMultiHeadModel
from .preprocessing import PreprocessingContract, build_model_inputs
from .schema import SINGLE_LABEL_HEADS, LabelSchema


def infer_records(
    model: ClassicalMultiHeadModel,
    records: Sequence[Mapping[str, Any]],
    *,
    schema: LabelSchema,
    preprocessing: PreprocessingContract,
    thresholds: Mapping[str, Any],
    model_version: str,
    output_schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    index_by_sample_id(records, role="inference-input")
    texts = build_model_inputs(records, preprocessing)
    probabilities = model.predict_probabilities(texts)
    outputs: list[dict[str, Any]] = []
    validator = Draft202012Validator(output_schema)
    for row_index, record in enumerate(records):
        predictions: dict[str, Any] = {}
        for head in SINGLE_LABEL_HEADS:
            order = schema.class_order[head]
            row = probabilities[head][row_index]
            predicted_index = int(row.argmax())
            confidence = float(row[predicted_index])
            threshold = float(thresholds["single_label"][head])
            predictions[head] = {
                "task_type": "single_label",
                "label": order[predicted_index],
                "confidence": confidence,
                "abstained": confidence < threshold,
                "threshold": threshold,
                "probabilities": {
                    label: float(row[index]) for index, label in enumerate(order)
                },
            }
        reasoning_order = schema.class_order["reasoning_tags"]
        reasoning_probabilities = probabilities["reasoning_tags"][row_index]
        reasoning_thresholds = {
            tag: float(thresholds["reasoning_tags"][tag])
            for tag in reasoning_order
        }
        selected = [
            tag
            for index, tag in enumerate(reasoning_order)
            if float(reasoning_probabilities[index]) >= reasoning_thresholds[tag]
        ]
        if not selected and thresholds.get("ensure_at_least_one_reasoning_tag"):
            selected = [reasoning_order[int(reasoning_probabilities.argmax())]]
        predictions["reasoning_tags"] = {
            "task_type": "multi_label",
            "labels": selected,
            "abstained": not selected,
            "thresholds": reasoning_thresholds,
            "probabilities": {
                tag: float(reasoning_probabilities[index])
                for index, tag in enumerate(reasoning_order)
            },
        }
        output = {
            "sample_id": str(record["sample_id"]),
            "schema_version": schema.schema_version,
            "model_version": model_version,
            "preprocessing_contract_version": preprocessing.contract_version,
            "predictions": predictions,
        }
        errors = sorted(validator.iter_errors(output), key=lambda error: list(error.path))
        if errors:
            raise ContractError(
                "INFERENCE_SCHEMA_VIOLATION",
                errors[0].message,
                sample_id=record.get("sample_id"),
                path=list(errors[0].path),
            )
        outputs.append(output)
    return outputs


def load_export(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir).resolve()
    export_manifest = read_json(path / "export_manifest.json")
    if not isinstance(export_manifest, Mapping):
        raise ContractError("EXPORT_MANIFEST_INVALID", "manifest must be an object")
    verify_content_addressed_id(export_manifest, id_key="export_manifest_id")
    artifacts = export_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("EXPORT_MANIFEST_INVALID", "artifacts must be an object")
    for filename, expected_hash in artifacts.items():
        artifact_path = path / str(filename)
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ContractError(
                "EXPORT_ARTIFACT_HASH_MISMATCH", "export artifact hash mismatch", file=filename
            )
    model_manifest = read_json(path / "model_manifest.json")
    verify_content_addressed_id(model_manifest, id_key="model_manifest_id")
    schema = LabelSchema.load(path / "schema.json")
    preprocessing = PreprocessingContract.load(path / "preprocessing_contract.json")
    thresholds = read_json(path / "thresholds.json")
    output_schema = read_json(path / "inference-output.schema.json")
    model = joblib.load(path / "model.joblib")
    if not isinstance(model, ClassicalMultiHeadModel):
        raise ContractError("MODEL_ARTIFACT_INVALID", "unexpected model type")
    model.assert_contract()
    if model.class_order != schema.class_order:
        raise ContractError(
            "MODEL_CLASS_ORDER_INVALID", "exported model and Schema class orders differ"
        )
    return {
        "manifest": export_manifest,
        "model_manifest": model_manifest,
        "model": model,
        "schema": schema,
        "preprocessing": preprocessing,
        "thresholds": thresholds,
        "output_schema": output_schema,
    }


def run_inference(
    model_dir: str | Path, input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise ContractError(
            "OUTPUT_EXISTS", "inference output is immutable and already exists", path=str(output)
        )
    bundle = load_export(model_dir)
    records = read_jsonl(input_path)
    outputs = infer_records(
        bundle["model"],
        records,
        schema=bundle["schema"],
        preprocessing=bundle["preprocessing"],
        thresholds=bundle["thresholds"],
        model_version=bundle["model_manifest"]["model_version"],
        output_schema=bundle["output_schema"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in outputs:
            handle.write(
                f"{json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            )
    return {
        "status": "INFERENCE_COMPLETE",
        "model_dir": str(Path(model_dir).resolve()),
        "input": str(Path(input_path).resolve()),
        "output": str(output.resolve()),
        "rows": len(outputs),
        "output_sha256": sha256_file(output),
        "execution_device": "CPU",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CPU inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_inference(args.model, args.input, args.output)
        exit_code = 0
    except ContractError as exc:
        result = {
            "status": "BLOCKED_INFERENCE_CONTRACT_ERROR",
            "blocker_codes": [exc.code],
            "error": exc.as_dict(),
        }
        exit_code = 3
    sys.stdout.write(
        f"{json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
