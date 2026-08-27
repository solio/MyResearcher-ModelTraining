from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from semantic_model.audit_data import run_audit
from semantic_model.hashes import content_addressed_id, sha256_file
from semantic_model.prepare import run_prepare
from semantic_model.schema import V1_HEADS
from semantic_model.train import run_train


REPO_ROOT = Path(__file__).parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json"


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def address(value: dict, id_key: str) -> dict:
    value[id_key] = content_addressed_id(value, omit_keys={id_key})
    return value


def input_record(prefix: str, index: int) -> dict:
    positive = index % 2 == 0
    return {
        "sample_id": f"{prefix}{index:03d}",
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


def label_record(record: dict, index: int, *, invalid_evidence: bool = False) -> dict:
    positive = index % 2 == 0
    stance = "UNKNOWN" if invalid_evidence else ("BULL" if positive else "BEAR")
    evidence = {"stance": ["跌破支撑"]} if invalid_evidence else {}
    return {
        **record,
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "target_mode": "ON_TARGET" if positive else "MARKET_GENERAL",
        "stance": stance,
        "emotion_primary": "NONE_EXPLICIT" if positive else "ANXIETY",
        "emotion_target": "NOT_APPLICABLE" if positive else "PRICE",
        "action_tendency": "HOLD" if positive else "SELL",
        "reasoning_tags": ["FUNDAMENTAL"] if positive else ["TECHNICAL_PRICE"],
        "context_dependency": "SELF_CONTAINED" if positive else "PARTIAL_CONTEXT",
        "label_confidence": "HIGH",
        "evidence_spans": evidence,
    }


def build_complete_package(tmp_path: Path) -> Path:
    data_root = tmp_path / "canonical-data"
    data_root.mkdir()
    inputs = [input_record("P", index) for index in range(1, 11)]
    labels = [
        label_record(record, index, invalid_evidence=index == 10)
        for index, record in enumerate(inputs, 1)
    ]
    input_path = data_root / "inputs.jsonl"
    input_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in inputs),
        encoding="utf-8",
    )
    metadata_path = data_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inputs[0]))
        writer.writeheader()
        writer.writerows(inputs)
    labels_path = data_root / "labels.jsonl"
    labels_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in labels),
        encoding="utf-8",
    )

    quarantine = address(
        {
            "schema_version": "myresearcher.quarantine-manifest.v1",
            "records": [
                {
                    "sample_id": "P010",
                    "violations": ["EVIDENCE_DEPENDENCY_VIOLATION"],
                }
            ],
        },
        "quarantine_manifest_id",
    )
    quarantine_path = data_root / "quarantine.json"
    write_json(quarantine_path, quarantine)

    split_names = ["train"] * 4 + ["dev"] * 2 + ["test"] * 2 + ["embargo"]
    assignments = [
        {
            "sample_id": f"P{index:03d}",
            "split": split,
            "duplicate_cluster_id": f"D{index:03d}",
            "event_group": f"E{index:03d}",
            "published_at": inputs[index - 1]["published_at"],
        }
        for index, split in enumerate(split_names, 1)
    ]
    split_manifest = address(
        {
            "schema_version": "myresearcher.semantic-split-manifest.v1",
            "time_source": "canonical_input.published_at",
            "assignments": assignments,
        },
        "split_manifest_id",
    )
    split_path = data_root / "split.json"
    write_json(split_path, split_manifest)

    weights_path = data_root / "weights.jsonl"
    weights_path.write_text(
        "".join(
            f"{json.dumps({'sample_id': f'P{index:03d}', 'weights': {head: 1.0 for head in V1_HEADS}}, sort_keys=True)}\n"
            for index in range(1, 10)
        ),
        encoding="utf-8",
    )

    anchors = [
        label_record(input_record("A", index), index) for index in (1, 2)
    ]
    anchor_path = data_root / "anchors.jsonl"
    anchor_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in anchors),
        encoding="utf-8",
    )
    anchor_manifest = address(
        {
            "schema_version": "myresearcher.anchor-manifest.v1",
            "records": [
                {"sample_id": "A001", "provenance": "human_confirmed"},
                {"sample_id": "A002", "provenance": "expert_weak_gold"},
            ],
            "provenance_counts": {"human_confirmed": 1, "expert_weak_gold": 1},
        },
        "anchor_manifest_id",
    )
    anchor_manifest_path = data_root / "anchor-manifest.json"
    write_json(anchor_manifest_path, anchor_manifest)

    baseline_report = address(
        {
            "schema_version": "baseline-report-test-v1",
            "reproduction_contract_complete": False,
        },
        "baseline_report_id",
    )
    baseline_path = data_root / "baseline.json"
    write_json(baseline_path, baseline_report)
    preprocessing = address(
        {
            "contract_version": "synthetic-preprocessing-v1",
            "status": "TEST_ONLY",
            "include_board_context": True,
            "board_marker": "[BOARD]",
            "text_marker": "[TEXT]",
            "separator": "\n",
            "normalize_unicode": "NFC",
            "strip_outer_whitespace": True,
        },
        "preprocessing_contract_id",
    )
    preprocessing_path = data_root / "preprocessing.json"
    write_json(preprocessing_path, preprocessing)

    packaged_paths = {
        "canonical_inputs": input_path,
        "canonical_metadata": metadata_path,
        "frozen_teacher_labels": labels_path,
        "quarantine_manifest": quarantine_path,
        "split_manifest": split_path,
        "field_weights": weights_path,
        "anchor_labels": anchor_path,
        "anchor_manifest": anchor_manifest_path,
        "baseline_report": baseline_path,
        "preprocessing_contract": preprocessing_path,
    }
    package_manifest = address(
        {
            "schema_version": "myresearcher.canonical-package-manifest.v1",
            "artifacts": {
                logical_name: {"sha256": sha256_file(path)}
                for logical_name, path in packaged_paths.items()
            },
        },
        "package_manifest_id",
    )
    package_path = data_root / "package.json"
    write_json(package_path, package_manifest)

    config = {
        "config_schema_version": "myresearcher.semantic-baseline-config.v1",
        "config_version": "synthetic-complete-v1",
        "project_root": str(REPO_ROOT),
        "schema_path": str(SCHEMA_PATH),
        "seed": 17,
        "data": {
            "root": str(data_root),
            "canonical_inputs": input_path.name,
            "canonical_metadata": metadata_path.name,
            "frozen_teacher_labels": labels_path.name,
            "quarantine_manifest": quarantine_path.name,
            "split_manifest": split_path.name,
            "field_weights": weights_path.name,
            "anchor_labels": anchor_path.name,
            "anchor_manifest": anchor_manifest_path.name,
            "baseline_report": baseline_path.name,
            "preprocessing_contract": preprocessing_path.name,
            "package_manifest": package_path.name,
            "expected": {
                "inputs": 10,
                "frozen_labels": 10,
                "quarantine": 1,
                "trainable": 9,
                "split": {"train": 4, "dev": 2, "test": 2, "embargo": 1},
                "anchor": 2,
                "anchor_provenance": {
                    "human_confirmed": 1,
                    "expert_weak_gold": 1,
                },
            },
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
        "runtime": {
            "run_root": str(tmp_path / "runs"),
            "cpu_only": True,
            "immutable_runs": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path


def test_complete_synthetic_package_audits_prepares_and_trains(tmp_path):
    config_path = build_complete_package(tmp_path)
    audit, audit_exit = run_audit(config_path)
    assert audit_exit == 0
    assert audit["status"] == "READY_FOR_BASELINE_REPRODUCTION"
    assert audit["validation_summary"] == {
        "split_counts": {"train": 4, "dev": 2, "test": 2, "embargo": 1},
        "evidence_violation_count": 1,
        "trainable_count": 9,
        "anchor_count": 2,
    }
    prepared, prepare_exit = run_prepare(config_path)
    assert prepare_exit == 0
    assert prepared["status"] == "PREPARED"
    prepared_again, _ = run_prepare(config_path)
    assert prepared_again["prepare_manifest_id"] == prepared["prepare_manifest_id"]
    trained, train_exit = run_train(config_path)
    assert train_exit == 0
    assert trained["status"] == "BASELINE_HARNESS_TESTED"

