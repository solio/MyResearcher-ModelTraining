from __future__ import annotations

import json
from pathlib import Path

import yaml

from semantic_model.config import ProjectConfig
from semantic_model.evaluate import inspect_run
from semantic_model.export import export_run
from semantic_model.hashes import content_addressed_id
from semantic_model.infer import run_inference
from semantic_model.prepare import PreparedDataset
from semantic_model.preprocessing import PreprocessingContract, build_model_inputs
from semantic_model.schema import LabelSchema, V1_HEADS
from semantic_model.train import train_prepared


REPO_ROOT = Path(__file__).parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json"


def write_json(path: Path, value) -> None:
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def make_contract() -> PreprocessingContract:
    raw = {
        "contract_version": "synthetic-preprocessing-v1",
        "status": "TEST_ONLY",
        "include_board_context": True,
        "board_marker": "[BOARD]",
        "text_marker": "[TEXT]",
        "separator": "\n",
        "normalize_unicode": "NFC",
        "strip_outer_whitespace": True,
    }
    raw["preprocessing_contract_id"] = content_addressed_id(
        raw, omit_keys={"preprocessing_contract_id"}
    )
    return PreprocessingContract.from_mapping(raw)


def make_record(index: int) -> dict:
    positive = index % 2 == 0
    return {
        "sample_id": f"P{index:03d}",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "published_at": f"2026-07-{index:02d}T10:00:00+08:00",
        "board_context": "601012 隆基绿能",
        "model_text": (
            "订单增长，我会继续持有，看好未来。"
            if positive
            else "跌破支撑，我准备卖出，看空后市。"
        ),
    }


def make_label(record: dict, index: int) -> dict:
    positive = index % 2 == 0
    return {
        **record,
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "target_mode": "ON_TARGET" if positive else "MARKET_GENERAL",
        "stance": "BULL" if positive else "BEAR",
        "emotion_primary": "NONE_EXPLICIT" if positive else "ANXIETY",
        "emotion_target": "NOT_APPLICABLE" if positive else "PRICE",
        "action_tendency": "HOLD" if positive else "SELL",
        "reasoning_tags": ["FUNDAMENTAL"] if positive else ["TECHNICAL_PRICE"],
        "context_dependency": "SELF_CONTAINED" if positive else "PARTIAL_CONTEXT",
        "label_confidence": "HIGH",
        "evidence_spans": {},
    }


def make_prepared(tmp_path: Path) -> PreparedDataset:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for filename, value in {
        "baseline.json": {"reproduction_contract_complete": False},
        "split.json": {"split_manifest_id": "synthetic-split"},
        "quarantine.json": {"quarantine_manifest_id": "synthetic-quarantine"},
        "anchor-manifest.json": {"anchor_manifest_id": "synthetic-anchor"},
        "package.json": {"package_manifest_id": "synthetic-package"},
    }.items():
        write_json(data_root / filename, value)
    config_raw = {
        "config_schema_version": "myresearcher.semantic-baseline-config.v1",
        "project_root": str(REPO_ROOT),
        "schema_path": str(SCHEMA_PATH),
        "seed": 17,
        "data": {
            "root": str(data_root),
            "baseline_report": "baseline.json",
            "split_manifest": "split.json",
            "quarantine_manifest": "quarantine.json",
            "anchor_manifest": "anchor-manifest.json",
            "package_manifest": "package.json",
        },
        "weighting": {"config_path": "configs/weighting_v0.3.5.yaml"},
        "model": {
            "family": "tfidf-logistic-regression",
            "word_tfidf": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 200,
                "sublinear_tf": True,
            },
            "char_tfidf": {
                "analyzer": "char",
                "ngram_range": [2, 4],
                "min_df": 1,
                "max_features": 500,
                "sublinear_tf": True,
            },
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 300,
                "solver": "liblinear",
                "class_weight": None,
                "random_state": 17,
            },
        },
        "calibration": {"minimum_coverage": 0.5},
        "runtime": {"run_root": str(tmp_path / "runs"), "cpu_only": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    config = ProjectConfig.load(config_path)
    schema = LabelSchema.load(SCHEMA_PATH)
    preprocessing = make_contract()
    records = [make_record(index) for index in range(1, 9)]
    labels = [make_label(record, index) for index, record in enumerate(records, 1)]
    records_by_id = {record["sample_id"]: record for record in records}
    labels_by_id = {label["sample_id"]: label for label in labels}
    texts = build_model_inputs(records, preprocessing)
    texts_by_id = {
        record["sample_id"]: text for record, text in zip(records, texts, strict=True)
    }
    weights_by_id = {
        record["sample_id"]: {head: 1.0 for head in V1_HEADS} for record in records
    }
    anchors = [
        make_label(make_record(index), index) for index in (9, 10)
    ]
    return PreparedDataset(
        config=config,
        schema=schema,
        preprocessing=preprocessing,
        manifest={
            "prepare_manifest_id": "synthetic-prepare-v1",
            "input_artifacts": {"fixture": {"sha256": "synthetic"}},
        },
        records_by_id=records_by_id,
        labels_by_id=labels_by_id,
        texts_by_id=texts_by_id,
        weights_by_id=weights_by_id,
        split_ids={
            "train": ["P001", "P002", "P003", "P004"],
            "dev": ["P005", "P006"],
            "test": ["P007", "P008"],
            "embargo": [],
        },
        anchors=anchors,
        artifact_dir=None,
    )


def test_classical_harness_evaluate_export_and_cpu_infer(tmp_path):
    prepared = make_prepared(tmp_path)
    result = train_prepared(prepared, run_root=tmp_path / "runs")
    assert result["status"] == "BASELINE_HARNESS_TESTED"
    run_dir = Path(result["run_dir"])
    assert (run_dir / "model.joblib").is_file()
    inspection = inspect_run(run_dir)
    assert inspection["status"] == "BASELINE_HARNESS_TESTED"
    assert set(inspection["metrics"]) == {"dev", "test", "anchor"}
    assert len(inspection["metrics"]["test"]["reasoning_tags"]["class_order"]) == 15

    exported = export_run(run_dir)
    assert exported["status"] == "EXPORTED"
    model_dir = Path(exported["model_dir"])
    input_path = tmp_path / "infer-input.jsonl"
    input_records = [make_record(11), make_record(12)]
    input_path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n" for record in input_records
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "predictions.jsonl"
    inference = run_inference(model_dir, input_path, output_path)
    assert inference["execution_device"] == "CPU"
    assert inference["rows"] == 2
    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert all(output["schema_version"] == prepared.schema.schema_version for output in outputs)
    assert all(set(output["predictions"]) == set(V1_HEADS) for output in outputs)
    assert all(
        "abstained" in output["predictions"]["stance"] for output in outputs
    )


def test_same_logical_run_is_reused_without_overwrite(tmp_path):
    prepared = make_prepared(tmp_path)
    first = train_prepared(prepared, run_root=tmp_path / "runs")
    second = train_prepared(prepared, run_root=tmp_path / "runs")
    assert second["status"] == "EXISTING_IMMUTABLE_RUN"
    assert first["run_id"] == second["run_id"]
