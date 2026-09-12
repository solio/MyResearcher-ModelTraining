from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_model.encoder_m6c_auxiliary import load_auxiliary_task_views
from semantic_model.errors import ContractError
from semantic_model.schema import SemanticSchemaCandidate
from tools import feedback_loop_m7a as m7a


SCHEMA_PATH = Path(__file__).parents[1] / "schema" / "semantic-schema-candidate-v0.3.json"


@pytest.fixture
def schema() -> SemanticSchemaCandidate:
    return SemanticSchemaCandidate.load(SCHEMA_PATH)


def _task(sample_id: str, heads: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "text": "明天市场可能上涨，作者认为值得观察。",
        "title": "合成样本",
        "metadata": {"stock_code": "000001", "stock_name": "合成"},
        "heads": heads,
    }


def _assignment(sample_id: str, head: str, answer, *, final: bool = True, disposition=None) -> dict:
    return {
        "sample_id": sample_id,
        "head": head,
        "answer": answer,
        "is_final": final,
        "disposition": disposition,
        "schema_version": "semantic-schema-candidate-v0.3",
    }


def test_task_only_loader_does_not_require_or_fabricate_legacy_labels(schema, tmp_path):
    tasks = [
        _task("m7a-a", ["arousal_level", "market_scope"]),
        _task("m7a-b", ["expectation_phase"]),
    ]
    assignments = [
        _assignment("m7a-a", "arousal_level", "UNKNOWN"),
        _assignment("m7a-a", "market_scope", None, final=False),
        _assignment("m7a-b", "expectation_phase", {"not": "a label"}, final=True, disposition="SKIPPED"),
    ]
    task_path = tmp_path / "tasks.jsonl"
    assignment_path = tmp_path / "assignments.jsonl"
    task_path.write_text("\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8")
    assignment_path.write_text("\n".join(json.dumps(row) for row in assignments) + "\n", encoding="utf-8")
    records, views = load_auxiliary_task_views(task_path, assignment_path, schema=schema)
    assert [record.sample_id for record in records] == ["m7a-a", "m7a-b"]
    assert all(view.legacy_label == {} for view in views)
    assert views[0].auxiliary.loss_mask["arousal_level"] == 1
    assert views[0].auxiliary.values["arousal_level"] == "UNKNOWN"
    assert views[0].auxiliary.loss_mask["market_scope"] == 0
    assert views[1].auxiliary.loss_mask["expectation_phase"] == 0


def test_task_only_loader_rejects_undeclared_sample_and_head(schema):
    tasks = [_task("m7a-a", ["arousal_level"])]
    with pytest.raises(ContractError, match="M7A_TASK_SAMPLE_ID_NOT_FOUND"):
        load_auxiliary_task_views(tasks, [_assignment("missing", "arousal_level", "LOW")], schema=schema)
    with pytest.raises(ContractError, match="M7A_TASK_HEAD_NOT_DECLARED"):
        load_auxiliary_task_views(tasks, [_assignment("m7a-a", "market_scope", ["UNKNOWN"])], schema=schema)


def test_task_payload_rejects_predictions_and_selection_caps_two_heads():
    candidates = []
    factor_sets = [
        ("expectation_phase", "arousal_level"),
        ("market_scope", "discourse_focus"),
        ("desired_price_direction",),
        ("expectation_phase", "desired_price_direction"),
        ("arousal_level", "discourse_focus"),
        ("market_scope",),
    ]
    for index, factors in enumerate(factor_sets):
        candidates.append({
            "sample_id": f"m7a-synthetic-{index}",
            "text": "合成可判断文本上涨市场",
            "title": "合成",
            "stock_code": "000001",
            "stock_name": "合成",
            "exact_content_group": f"m7a-group-{index}",
            "observation_id": f"obs-{index}",
            "sampling_reasons": ["SYNTHETIC"],
            "text_length": 12,
            "length_bucket": "MEDIUM",
            "factors": factors,
            "aggregate_uncertainty": 0.5 + index / 100,
        })
    selected = m7a.select_tasks(candidates, target_assignments=10)
    assert sum(len(row["heads"]) for row in selected) == 10
    assert all(1 <= len(row["heads"]) <= 2 for row in selected)
    payload = m7a._task_payload(selected[0], "m7a-v03-vtest")
    assert not ({"answer", "prediction", "predictions", "probabilities"} & set(payload))
    assert not ({"answer", "prediction", "predictions", "probabilities"} & set(payload["metadata"]))


def test_input_filter_rejects_low_information_and_39_char_cuts(tmp_path):
    feed_path = tmp_path / "feed.jsonl"
    prediction_path = tmp_path / "pred.jsonl"
    predictions = {
        head: {"prediction": "MARKET" if head == "emotion_target" else "BULL", "probabilities": {"BULL": 1.0}}
        for head in ("target_mode", "stance", "emotion_primary", "emotion_target", "action_tendency", "context_dependency", "reasoning_tags")
    }
    rows = [
        {"sample_id": "m7a-empty", "text": "……", "text_length": 2, "exact_content_group": "g-empty"},
        {"sample_id": "m7a-cut", "text": "上涨" * 20, "text_length": 40, "exact_content_group": "g-cut"},
        {"sample_id": "m7a-ok", "text": "明天市场上涨", "text_length": 6, "exact_content_group": "g-ok", "stock_code": "000001", "stock_name": "合成", "sampling_reasons": []},
    ]
    feed_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    prediction_path.write_text("\n".join(json.dumps({"sample_id": row["sample_id"], "predictions": predictions, "reasoning_mean_uncertainty": 0.2}) for row in rows) + "\n", encoding="utf-8")
    candidates = m7a.load_feed_and_predictions(feed_path, prediction_path, require_300=False)
    assert [row["sample_id"] for row in candidates] == ["m7a-ok"]
