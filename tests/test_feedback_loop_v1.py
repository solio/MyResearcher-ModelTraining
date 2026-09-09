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
    snapshots = [{"sample_id": f"fl1-obs-{i:03d}", "six_head_mean_max_probability": 0.50 + i / 1000} for i in range(30)]
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
    snapshots = [{"sample_id": f"fl1-obs-{i:03d}", "six_head_mean_max_probability": 0.4} for i in range(14)]
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
