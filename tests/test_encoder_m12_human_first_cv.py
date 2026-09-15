from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_model import encoder_m1 as m1
from semantic_model import encoder_m12_human_first_cv as m12


def _record(index: int, label: str, reasoning: list[str] | None = None) -> m1.M1Record:
    labels = {
        "target_mode": label,
        "reasoning_tags": reasoning or ["NO_REASON_GIVEN"],
    }
    return m1.M1Record(str(index), "000001", "示例", f"文本{index}", labels, {head: 1.0 for head in m12.V1_HEADS})


def test_fold_indices_cover_every_sample_once() -> None:
    records = [_record(index, "ON_TARGET" if index % 3 else "CROSS_TARGET") for index in range(100)]
    folds = m12.fold_indices(records, "target_mode")
    assert len(folds) == 5
    assert sorted(index for _train, validation in folds for index in validation) == list(range(100))
    assert all(not (set(train) & set(validation)) for train, validation in folds)


def test_fold_indices_cover_human109_once() -> None:
    records = [_record(index, "ON_TARGET" if index % 3 else "CROSS_TARGET") for index in range(109)]
    folds = m12.fold_indices(records, "target_mode")
    assert sorted(index for _train, validation in folds for index in validation) == list(range(109))


def _recent_rows(sample_id: str, *, conflicting: bool = False) -> list[dict]:
    answers = {
        "target_mode": "ON_TARGET",
        "stance": "BULL",
        "emotion_primary": "NONE_EXPLICIT" if conflicting else "HOPE",
        "emotion_target": "OTHER" if conflicting else "PRICE",
        "action_tendency": "WATCH",
        "reasoning_tags": ["TECHNICAL_PRICE"],
        "context_dependency": "SELF_CONTAINED",
    }
    return [
        {
            "sample_id": sample_id,
            "head": head,
            "answer": answers[head],
            "is_final": True,
            "schema_version": "semantic-schema-calibrated-v0.2.1",
        }
        for head in m12.V1_HEADS
    ]


def test_recent_feedback_loader_preserves_answers_and_excludes_conflict(tmp_path) -> None:
    feed = tmp_path / "feed.jsonl"
    labels = tmp_path / "labels.jsonl"
    feed.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "stock_code": "000001", "stock_name": "示例", "model_text": f"文本 {sample_id}"}, ensure_ascii=False) + "\n"
            for sample_id in ("valid", "conflict")
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*_recent_rows("valid"), *_recent_rows("conflict", conflicting=True)]),
        encoding="utf-8",
    )
    records, audit = m12.load_recent_feedback_records(
        labels,
        feed,
        role="recent_train",
        exclude_dependency_conflicts=True,
    )
    assert [record.sample_id for record in records] == ["valid"]
    assert records[0].label["emotion_primary"] == "HOPE"
    assert records[0].label["emotion_target"] == "PRICE"
    assert audit["received_samples"] == 2
    assert audit["used_samples"] == 1
    assert audit["excluded_sample_ids"] == ["conflict"]


def test_recent_check_dependency_conflict_fails_instead_of_rewriting(tmp_path) -> None:
    feed = tmp_path / "feed.jsonl"
    labels = tmp_path / "labels.jsonl"
    feed.write_text(json.dumps({"sample_id": "conflict", "stock_code": "000001", "stock_name": "示例", "model_text": "文本"}, ensure_ascii=False) + "\n", encoding="utf-8")
    labels.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _recent_rows("conflict", conflicting=True)), encoding="utf-8")
    with pytest.raises(Exception) as error:
        m12.load_recent_feedback_records(labels, feed, expected_samples=1, role="recent_check")
    assert getattr(error.value, "code", None) == "M12_RECENT_DEPENDENCY_CONFLICT"


def test_base_dependency_conflicts_are_excluded_without_rewriting() -> None:
    valid = _record(1, "ON_TARGET")
    valid.label.update({"emotion_primary": "HOPE", "emotion_target": "PRICE"})
    conflict = _record(2, "ON_TARGET")
    conflict.label.update({"emotion_primary": "NONE_EXPLICIT", "emotion_target": "MARKET"})
    used, audit = m12.exclude_dependency_conflicts([valid, conflict], role="base_human_train")
    assert [record.sample_id for record in used] == ["1"]
    assert conflict.label["emotion_target"] == "MARKET"
    assert audit["received_samples"] == 2
    assert audit["used_samples"] == 1
    assert audit["excluded_sample_ids"] == ["2"]


def test_reasoning_threshold_calibration_is_model_specific() -> None:
    records = [
        _record(0, "ON_TARGET", ["NO_REASON_GIVEN"]),
        _record(1, "ON_TARGET", ["TECHNICAL_PRICE"]),
        _record(2, "ON_TARGET", ["TECHNICAL_PRICE"]),
        _record(3, "ON_TARGET", ["NO_REASON_GIVEN"]),
    ]
    order = ["NO_REASON_GIVEN", "TECHNICAL_PRICE"]
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]])
    thresholds = m12.calibrate_reasoning_thresholds(probabilities, records, order)
    assert set(thresholds) == set(order)
    assert all(0.1 <= value <= 0.9 for value in thresholds.values())
