"""One-shot weak-label Test-467 evaluation for the frozen M3 candidate.

This entry point intentionally reads only the frozen Test split labels and the
matching canonical input rows.  It does not load Train, Dev, Anchor, Gold,
OOD, or any reference prediction package, and it never fits or calibrates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import encoder_m1 as m1
from .encoder_m2_selected_runtime import (
    DEFAULT_CACHE,
    DEFAULT_SELECTED_ARTIFACT,
    InferenceRecord,
    UnifiedCPUInferenceRuntime,
    _verify_manifest,
)
from .errors import ContractError
from .hashes import sha256_file
from .schema import LabelSchema, SINGLE_LABEL_HEADS, V1_HEADS


TEST_ROWS = 467
SELECTED_CONTENT_ADDRESS = "8f73bc437602ce25d6e8ba0c7fd787676a4b566077b1765f3510739ae85928cf"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "runs" / "m3g-test467-20260907"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "baseline_v0.3.5.yaml"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "semantic-schema-calibrated-v0.2.1.json"


@dataclass(frozen=True)
class TestRecord:
    sample_id: str
    stock_code: str
    stock_name: str
    model_text: str
    labels: Mapping[str, Any]


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_test_partition(config_path: str | Path = CONFIG_PATH) -> tuple[LabelSchema, list[TestRecord], dict[str, Any]]:
    """Load exactly Test-467 labels and matching inputs, without other roles."""

    config = m1.ProjectConfig.load(config_path)
    schema = LabelSchema.load(SCHEMA_PATH)
    rows = m1._read_jsonl(config.data_path("split_labels_test"))
    _require(len(rows) == TEST_ROWS, "M3G_TEST_SPLIT_SIZE_INVALID", "canonical Test split must contain 467 rows", observed=len(rows))
    ids = [row.get("sample_id") for row in rows]
    _require(all(isinstance(sample_id, str) and sample_id for sample_id in ids), "M3G_TEST_SAMPLE_ID_INVALID", "Test labels require non-empty sample IDs")
    _require(len(set(ids)) == TEST_ROWS, "M3G_TEST_SAMPLE_ID_INVALID", "Test labels require unique sample IDs")
    inputs = m1._read_selected_jsonl(config.data_path("canonical_inputs"), set(ids))
    records: list[TestRecord] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        source = inputs[sample_id]
        for field in ("stock_code", "stock_name"):
            _require(source.get(field) == row.get(field), "M3G_TEST_METADATA_MISMATCH", "Test label metadata differs from canonical input", sample_id=sample_id, field=field)
        text = source.get("model_text")
        _require(isinstance(text, str) and text.strip(), "M3G_TEST_MODEL_TEXT_INVALID", "Test model_text must be non-empty", sample_id=sample_id)
        labels: dict[str, Any] = {}
        for head in SINGLE_LABEL_HEADS:
            value = row.get(head)
            _require(isinstance(value, str) and value in schema.class_order[head], "M3G_TEST_LABEL_INVALID", "Test scalar label is outside frozen class order", sample_id=sample_id, head=head, value=value)
            labels[head] = value
        reasoning = row.get("reasoning_tags")
        _require(isinstance(reasoning, list) and len(set(reasoning)) == len(reasoning) and all(isinstance(tag, str) and tag in schema.class_order["reasoning_tags"] for tag in reasoning), "M3G_TEST_LABEL_INVALID", "Test reasoning labels are outside frozen class order", sample_id=sample_id)
        labels["reasoning_tags"] = list(reasoning)
        records.append(TestRecord(sample_id=sample_id, stock_code=str(source["stock_code"]), stock_name=str(source["stock_name"]), model_text=text, labels=labels))
    return schema, records, {"config_path": str(Path(config_path).resolve()), "split_labels_path": str(config.data_path("split_labels_test")), "canonical_inputs_path": str(config.data_path("canonical_inputs")), "rows": TEST_ROWS, "role": "Test_467_weak_labels_only"}


def _scalar_metrics(truth: Sequence[str], predicted: Sequence[str], order: Sequence[str]) -> dict[str, Any]:
    _require(len(truth) == len(predicted), "M3G_METRIC_LENGTH_INVALID", "scalar truth and predictions differ in length")
    matrix = {actual: {guess: 0 for guess in order} for actual in order}
    for actual, guess in zip(truth, predicted, strict=True):
        matrix[actual][guess] += 1
    per_class: dict[str, Any] = {}
    for label in order:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in order if actual != label)
        fn = sum(matrix[label][guess] for guess in order if guess != label)
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    accuracy = sum(actual == guess for actual, guess in zip(truth, predicted, strict=True)) / len(truth) if truth else 0.0
    return {"rows": len(truth), "accuracy": accuracy, "macro_f1": statistics.mean(item["f1"] for item in per_class.values()) if per_class else 0.0, "per_class": per_class, "confusion_matrix": matrix, "finite": True}


def _reasoning_metrics(truth: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]], order: Sequence[str]) -> dict[str, Any]:
    _require(len(truth) == len(predicted), "M3G_METRIC_LENGTH_INVALID", "reasoning truth and predictions differ in length")
    per_label: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for index, label in enumerate(order):
        tp = sum(actual[index] == 1 and guess[index] == 1 for actual, guess in zip(truth, predicted, strict=True))
        fp = sum(actual[index] == 0 and guess[index] == 1 for actual, guess in zip(truth, predicted, strict=True))
        fn = sum(actual[index] == 1 and guess[index] == 0 for actual, guess in zip(truth, predicted, strict=True))
        support = sum(actual[index] == 1 for actual in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        total_tp += tp; total_fp += fp; total_fn += fn
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    exact = sum(actual == guess for actual, guess in zip(truth, predicted, strict=True)) / len(truth) if truth else 0.0
    return {"rows": len(truth), "macro_f1": statistics.mean(item["f1"] for item in per_label.values()) if per_label else 0.0, "micro_f1": micro_f1, "exact_set_accuracy": exact, "per_label": per_label, "finite": True}


def evaluate_test(*, config_path: str | Path = CONFIG_PATH, artifact_dir: str | Path = DEFAULT_SELECTED_ARTIFACT, cache_dir: str | Path = DEFAULT_CACHE, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), "M3G_OUTPUT_ALREADY_EXISTS", "Test evaluation output must be a new directory", path=str(output))
    artifact = Path(artifact_dir).resolve()
    manifest = _verify_manifest(artifact, expected_content_address=SELECTED_CONTENT_ADDRESS)
    schema, records, data_identity = load_test_partition(config_path)
    runtime = UnifiedCPUInferenceRuntime.from_artifact(artifact, config_path=config_path, cache_dir=cache_dir, verify_manifest=True)
    predictions = runtime.predict_records([InferenceRecord(record.sample_id, record.stock_code, record.stock_name, record.model_text) for record in records])
    _require(len(predictions) == TEST_ROWS and [row["sample_id"] for row in predictions] == [record.sample_id for record in records], "M3G_PREDICTION_IDENTITY_INVALID", "predictions do not preserve Test ordering and identity")
    rows: list[dict[str, Any]] = []
    scalar_truth = {head: [] for head in SINGLE_LABEL_HEADS}
    scalar_pred = {head: [] for head in SINGLE_LABEL_HEADS}
    reasoning_truth: list[list[int]] = []
    reasoning_pred: list[list[int]] = []
    reasoning_order = list(schema.class_order["reasoning_tags"])
    for record, prediction in zip(records, predictions, strict=True):
        weak = {head: record.labels[head] for head in V1_HEADS}
        predicted = prediction["predictions"]
        for head in SINGLE_LABEL_HEADS:
            scalar_truth[head].append(str(weak[head])); scalar_pred[head].append(str(predicted[head]["prediction"]))
        truth_bits = [int(tag in weak["reasoning_tags"]) for tag in reasoning_order]
        pred_tags = list(predicted["reasoning_tags"]["prediction"])
        pred_bits = [int(tag in pred_tags) for tag in reasoning_order]
        reasoning_truth.append(truth_bits); reasoning_pred.append(pred_bits)
        rows.append({"sample_id": record.sample_id, "weak_labels": weak, "predictions": predicted})
    metrics = {head: _scalar_metrics(scalar_truth[head], scalar_pred[head], schema.class_order[head]) for head in SINGLE_LABEL_HEADS}
    metrics["reasoning_tags"] = _reasoning_metrics(reasoning_truth, reasoning_pred, reasoning_order)
    _require(all(math.isfinite(float(value)) for head in metrics.values() for value in ([head.get("accuracy", 0.0), head.get("macro_f1", 0.0), head.get("micro_f1", 0.0), head.get("exact_set_accuracy", 0.0)])), "M3G_NONFINITE_METRICS", "Test metrics are non-finite")
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "test-467-predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"status": "M3G_TEST_467_COMPLETED_DIAGNOSTIC", "selected_candidate_content_address": SELECTED_CONTENT_ADDRESS, "selected_candidate_manifest_sha256": sha256_file(artifact / "content-addressed-manifest.json"), "evaluation_scope": "frozen_candidate_Test_467_weak_label_generalization_only", "test_rows": TEST_ROWS, "device": "cpu", "threshold_source": "selected artifact reasoning thresholds; no Test threshold selection", "metrics": metrics, "data_identity": data_identity, "forbidden_inputs_not_read": ["Anchor", "Embargo", "Gold", "OOD", "Human100", "reference_predictions", "LLM", "cloud", "production_data"], "training_invoked": False, "candidate_modified": False, "predictions_file": predictions_path.name}
    _write_json(output / "test-467-summary.json", summary)
    result_manifest = m1._write_content_manifest(output, {"manifest_schema_version": "myresearcher.m3g-test467-evaluation-manifest.v1", "status": summary["status"], "selected_candidate_content_address": SELECTED_CONTENT_ADDRESS, "selected_candidate_manifest_sha256": summary["selected_candidate_manifest_sha256"], "evaluation_scope": summary["evaluation_scope"], "test_rows": TEST_ROWS, "device": "cpu", "summary_sha256": sha256_file(output / "test-467-summary.json"), "predictions_sha256": sha256_file(predictions_path), "forbidden_inputs_not_read": summary["forbidden_inputs_not_read"]})
    return {"status": summary["status"], "output_dir": str(output), "content_address": result_manifest["content_address"], "test_rows": TEST_ROWS, "metrics": metrics, "device": "cpu"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_SELECTED_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = evaluate_test(config_path=args.config, artifact_dir=args.artifact_dir, cache_dir=args.cache_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
