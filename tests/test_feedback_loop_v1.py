from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import feedback_loop_v1 as fl


def _pilot_row(index: int, *, group: str | None = None, reason: str = "GENERAL_RANDOM", length: int = 20) -> dict:
    return {
        "observation_id": f"obs-{index:03d}",
        "content": "x" * max(1, length),
        "exact_content_group": group or f"group-{index:03d}",
        "sampling_reasons": [reason],
        "text_length": length,
        "collector_lineage": {"scopes": [{"scope_key": "stock:601012", "requested_bar_code": "601012"}]},
        "author_id": "must-not-be-projected",
        "author_name": "must-not-be-projected",
    }


def test_projection_uses_scope_and_omits_identity_and_labels():
    projected = fl.project_pilot_row(_pilot_row(1), {"601012": "隆基绿能"})
    assert projected["sample_id"] == "fl1-obs-001"
    assert projected["stock_code"] == "601012"
    assert projected["stock_name"] == "隆基绿能"
    assert projected["model_text"] == projected["text"]
    assert not fl.FORBIDDEN_INPUT_FIELDS.intersection(projected)
    assert "author_id" not in json.dumps(projected, ensure_ascii=False)


def test_ambiguous_scope_fails_closed():
    row = _pilot_row(1)
    row["collector_lineage"] = {"scopes": [{"scope_key": "stock:601012"}, {"scope_key": "stock:600519"}]}
    with pytest.raises(fl.FeedbackLoopError):
        fl.project_pilot_row(row, {"601012": "隆基绿能", "600519": "贵州茅台"})


def test_deterministic_selection_has_fourteen_distinct_groups_and_frozen_split():
    feed = [fl.project_pilot_row(_pilot_row(i, reason=("VERY_SHORT" if i % 3 == 0 else "LONGER_TEXT"), length=(5 if i % 3 == 0 else 35)), {"601012": "隆基绿能"}) for i in range(30)]
    snapshots = [{"sample_id": f"fl1-obs-{i:03d}", "aggregate_uncertainty": 0.50 + i / 1000} for i in range(30)]
    train, check = fl.select_samples(feed, snapshots)
    assert len(train) == 10 and len(check) == 4
    assert len({row["sample_id"] for row in train + check}) == 14
    assert len({row["exact_content_group"] for row in train + check}) == 14
    assert {row["exact_content_group"] for row in train}.isdisjoint({row["exact_content_group"] for row in check})
    assert all(row["frozen_role"] == "HUMAN_TRAIN" for row in train)
    assert all(row["frozen_role"] == "HUMAN_CHECK" for row in check)
    train2, check2 = fl.select_samples(feed, snapshots)
    assert [row["sample_id"] for row in train + check] == [row["sample_id"] for row in train2 + check2]


def test_task_payloads_have_no_prediction_or_labels():
    feed = [fl.project_pilot_row(_pilot_row(i), {"601012": "隆基绿能"}) for i in range(14)]
    snapshots = [{"sample_id": f"fl1-obs-{i:03d}", "aggregate_uncertainty": 0.4} for i in range(14)]
    train, _ = fl.select_samples(feed, snapshots)
    payload = [{"sample_id": row["sample_id"], "text": row["model_text"], "title": row["stock_name"], "metadata": {"stock_code": row["stock_code"]}} for row in train]
    assert len(payload) == 10
    assert all("predictions" not in row and "labels" not in row for row in payload)
    assert all(set(row).issubset({"sample_id", "text", "title", "metadata"}) for row in payload)


def test_pull_merges_latest_revisable_answers_and_preserves_provenance():
    rows = [
        {"batch": fl.TRAIN_BATCH, "sample_id": "fl1-a", "head": "stance", "answer": "BULL", "disposition": "IN_PROGRESS", "is_final": False, "updated_at": "2026-09-09T10:00:00Z", "schema_version": "v1"},
        {"batch": fl.TRAIN_BATCH, "sample_id": "fl1-a", "head": "stance", "answer": "BEAR", "disposition": "FINAL", "is_final": True, "updated_at": "2026-09-09T11:00:00Z", "schema_version": "v1"},
        {"batch": fl.TRAIN_BATCH, "sample_id": "fl1-a", "head": "stance", "answer": None, "disposition": "IN_PROGRESS", "is_final": False, "updated_at": "2026-09-09T12:00:00Z", "schema_version": "v1"},
        {"batch": fl.CHECK_BATCH, "sample_id": "fl1-b", "head": "reasoning_tags", "answer": ["FUNDAMENTAL"], "disposition": "FINAL", "is_final": True, "updated_at": "2026-09-09T11:00:00Z", "schema_version": "v1"},
    ]
    merged = fl.merge_pull_rows(rows)
    assert len(merged) == 2
    assert merged[0]["answer"] == "BEAR"
    assert merged[0]["batch"] == fl.TRAIN_BATCH
    assert merged[0]["schema_version"] == "v1"
    assert merged[0]["snapshot_role"] == "HUMAN_FEED_REVISION"


def test_real_pilot_has_300_rows_and_mapping_is_read_only():
    path = fl.PILOT_PATH
    if not path.is_file():
        pytest.skip("external DataClean pilot is not mounted")
    rows = fl.load_pilot(path)
    assert len(rows) == 300
    assert len({row["sample_id"] for row in rows}) == 300
    assert {row["stock_code"] for row in rows} == {"601012"}


def _prediction_for_uncertainty(*, reasoning_value: float = 0.5, scalar_peak: float = 0.7) -> dict:
    scalar_heads = {
        "target_mode": 4,
        "stance": 5,
        "emotion_primary": 11,
        "emotion_target": 7,
        "action_tendency": 9,
        "context_dependency": 4,
    }
    outputs = {}
    for head, count in scalar_heads.items():
        tail = (1.0 - scalar_peak) / (count - 1)
        outputs[head] = {"probabilities": {f"c{i}": scalar_peak if i == 0 else tail for i in range(count)}, "prediction": "c0"}
    labels = ["FUNDAMENTAL", "NO_REASON_GIVEN"]
    probs = {label: reasoning_value if label == "FUNDAMENTAL" else 0.1 for label in labels}
    outputs["reasoning_tags"] = {"probabilities": probs, "thresholds": {"FUNDAMENTAL": 0.5, "NO_REASON_GIVEN": 0.5}, "prediction": []}
    return {"sample_id": "s", "predictions": outputs}


def test_normalised_entropy_accounts_for_class_cardinality():
    assert fl._normalised_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)
    assert fl._normalised_entropy({str(i): 1 / 11 for i in range(11)}) == pytest.approx(1.0)
    assert fl._normalised_entropy({str(i): (0.7 if i == 0 else 0.3 / 10) for i in range(11)}) < 1.0


def test_reasoning_uncertainty_changes_seven_head_aggregate():
    near, near_details = fl._prediction_uncertainty(_prediction_for_uncertainty(reasoning_value=0.5))
    far, far_details = fl._prediction_uncertainty(_prediction_for_uncertainty(reasoning_value=0.99))
    assert near > far
    assert near_details["reasoning_boundary_uncertainty"] > far_details["reasoning_boundary_uncertainty"]
    assert near_details["aggregate_uncertainty"] > far_details["aggregate_uncertainty"]


def test_low_information_text_is_excluded_from_future_selection():
    feed = [fl.project_pilot_row(_pilot_row(i, length=20), {"601012": "隆基绿能"}) for i in range(20)]
    feed.extend([
        fl.project_pilot_row(_pilot_row(100, length=3, group="placeholder-ellipsis"), {"601012": "隆基绿能"}) | {"model_text": "……", "text": "……"},
        fl.project_pilot_row(_pilot_row(101, length=2, group="placeholder-youliao"), {"601012": "隆基绿能"}) | {"model_text": "又了", "text": "又了"},
    ])
    assert fl.is_low_information_input("……")
    assert fl.is_low_information_input("又了")
    snapshots = [{"sample_id": row["sample_id"], "aggregate_uncertainty": 0.99 if fl.is_low_information_input(row["model_text"]) else 0.1} for row in feed]
    train, check = fl.select_samples(feed, snapshots)
    selected_ids = {row["sample_id"] for row in train + check}
    assert "fl1-obs-100" not in selected_ids
    assert "fl1-obs-101" not in selected_ids
    assert len(selected_ids) == 14


def test_short_semantic_text_is_not_low_information():
    for text in ("已清仓，随便吧", "明天涨停板", "抄底", "看好", "垃圾"):
        assert not fl.is_low_information_input(text)
        assert fl.legacy_short_filter_misexcludes(text)
    assert fl.legacy_short_filter_misexcludes("抄底")


def test_v2_selection_excludes_all_v1_ids_and_keeps_frozen_counts():
    feed = [fl.project_pilot_row(_pilot_row(i, length=5 if i % 2 else 30), {"601012": "隆基绿能"}) for i in range(40)]
    snapshots = [{"sample_id": row["sample_id"], "aggregate_uncertainty": 0.5 + i / 1000} for i, row in enumerate(feed)]
    excluded = {row["sample_id"] for row in feed[:14]}
    train, check = fl.select_samples(feed, snapshots, exclude_sample_ids=excluded)
    selected = train + check
    assert len(train) == 10 and len(check) == 4
    assert not excluded.intersection(row["sample_id"] for row in selected)
    assert len({row["exact_content_group"] for row in selected}) == 14
    assert {row["exact_content_group"] for row in train}.isdisjoint(row["exact_content_group"] for row in check)
