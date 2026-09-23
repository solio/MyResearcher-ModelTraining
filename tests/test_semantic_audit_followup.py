from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import semantic_audit_followup as audit


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def package(tmp_path, review=(), natural=()):
    (tmp_path / "plan.json").write_text(json.dumps({"batches": {"review": "r", "natural": "n"}, "sample_budget": 40, "max_heads_per_sample": 3}))
    write_jsonl(tmp_path / "review-tasks.jsonl", review)
    write_jsonl(tmp_path / "natural-tasks.jsonl", natural)
    return tmp_path


def label(sid, head, answer, *, batch="n", final=True, **extra):
    return {"sample_id": sid, "head": head, "answer": answer, "batch": batch, "is_final": final, **extra}


def test_sparse_unknown_disposition_and_missing_are_distinct(tmp_path):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["stance", "emotion_target", "context_dependency"]}, {"sample_id": "b", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "stance", "UNKNOWN"), label("a", "emotion_target", None, disposition="RESOLVED_WITHOUT_LABEL"), label("a", "context_dependency", None, disposition="NEEDS_ADJUDICATION")])
    report = audit.evaluate(p, annotations)
    assert report["status"] == "WAITING_FOR_HUMAN"
    assert report["assignments"] == 4 and report["tasks_processed"] == 3
    assert report["eligible_for_analysis"] == 1
    assert report["by_head"]["stance"]["explicit_unknown"] == 1
    assert report["by_head"]["stance"]["missing"] == ["b"]
    assert "action_tendency" not in report["by_head"]
    assert report["natural"]["emotion_target"]["support"]["UNKNOWN"] == 0


def test_cross_head_conflict_requires_joint_requested_heads_and_never_rewrites(tmp_path):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["emotion_primary", "emotion_target"]}, {"sample_id": "b", "heads": ["emotion_target"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "emotion_primary", "NONE_EXPLICIT"), label("a", "emotion_target", "MARKET"), label("b", "emotion_target", "MARKET")])
    original = annotations.read_bytes()
    report = audit.evaluate(p, annotations)
    assert report["emotion_dependency"]["jointly_requested"] == 1
    assert len(report["emotion_dependency"]["violations"]) == 1
    assert report["eligible_for_analysis"] == 1
    assert report["status"] == "HUMAN_REVIEW_COMPLETE_ANALYSIS_LIMITED"
    assert annotations.read_bytes() == original


def test_review_excluded_from_natural_metrics_and_probabilities(tmp_path):
    p = package(tmp_path, review=[{"sample_id": "r", "heads": ["stance"]}], natural=[{"sample_id": "n", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("r", "stance", "BEAR", batch="r"), label("n", "stance", "BULL")])
    write_jsonl(p / "baseline-labels.jsonl", [label("r", "stance", "BULL", batch="old")])
    probs = {"BULL": .6, "BEAR": .1, "NEUTRAL": .1, "MIXED": .1, "UNKNOWN": .1}
    write_jsonl(p / "predictions.jsonl", [{"sample_id": sid, "predictions": {"stance": {"prediction": "BULL", "probabilities": probs}}} for sid in ["r", "n"]])
    report = audit.evaluate(p, annotations)
    assert report["review"]["changed"] == 1
    stats = report["natural"]["stance"]
    assert stats["metrics"]["n"] == 1 and stats["metrics"]["accuracy"] == 1
    assert stats["metrics"]["macro_f1"] == .2
    assert stats["probability_statistics"]["brier"] == pytest.approx(.2)
    assert stats["probability_statistics"]["ece_5_equal_width_bins"] == pytest.approx(.4)
    assert report["promotion_decision"] == "NOT_ELIGIBLE_FOR_PROMOTION_OR_AUTOMATIC_RETRAINING"


def test_newer_draft_withdraws_old_final_and_ambiguous_revision_is_not_guessed(tmp_path):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["stance"]}, {"sample_id": "b", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "stance", "BULL", updated_at="2026-09-22T00:00:00Z"), label("a", "stance", "BEAR", updated_at="2026-09-22T01:00:00Z"), label("a", "stance", None, final=False, updated_at="2026-09-22T02:00:00Z"), label("b", "stance", "BULL"), label("b", "stance", "BEAR")])
    report = audit.evaluate(p, annotations)
    assert report["natural"]["stance"]["support"]["BEAR"] == 0
    assert report["status"] == "WAITING_FOR_HUMAN"
    assert report["tasks_processed"] == 0
    assert report["by_head"]["stance"]["states"]["IN_PROGRESS"] == 1
    assert report["by_head"]["stance"]["states"]["CONFLICTING_REVISIONS"] == 1


@pytest.mark.parametrize("disposition", ["跳过", "无法判断", "缺少上下文"])
def test_actual_app_terminal_disposition_without_final_is_processed_not_label(tmp_path, disposition):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "stance", None, final=False, disposition=disposition)])
    report = audit.evaluate(p, annotations)
    assert report["status"] == "HUMAN_REVIEW_COMPLETE_ANALYSIS_LIMITED"
    assert report["tasks_processed"] == 1
    assert report["eligible_for_analysis"] == 0
    assert report["by_head"]["stance"]["states"][disposition] == 1
    assert report["by_head"]["stance"]["explicit_unknown"] == 0


def test_reasoning_does_not_add_an_unfrozen_exclusivity_rule():
    order = {"reasoning_tags": ["FUNDAMENTAL", "NO_REASON_GIVEN", "UNKNOWN"]}
    assert audit._answer(["NO_REASON_GIVEN", "FUNDAMENTAL"], "reasoning_tags", order) == ["FUNDAMENTAL", "NO_REASON_GIVEN"]


def test_multilabel_empty_is_not_negative_and_existing_sets_are_scored(tmp_path):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["reasoning_tags"]}, {"sample_id": "b", "heads": ["reasoning_tags"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "reasoning_tags", ["FUNDAMENTAL", "NEWS_EVENT"]), label("b", "reasoning_tags", [])])
    write_jsonl(p / "predictions.jsonl", [{"sample_id": "a", "predictions": {"reasoning_tags": {"labels": ["FUNDAMENTAL"]}}}])
    report = audit.evaluate(p, annotations)
    assert report["by_head"]["reasoning_tags"]["states"]["INVALID_LABEL"] == 1
    assert report["natural"]["reasoning_tags"]["metrics"]["micro_f1"] == pytest.approx(2 / 3)
    assert report["natural"]["reasoning_tags"]["metrics"]["exact_set_accuracy"] == 0


def test_input_batch_and_budget_errors_fail_without_writes(tmp_path):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [label("a", "stance", "BULL", batch="other")])
    with pytest.raises(audit.FollowupError, match="batch"):
        audit.evaluate(p, annotations)
    write_jsonl(p / "natural-tasks.jsonl", [{"sample_id": str(i), "heads": ["stance"]} for i in range(41)])
    with pytest.raises(audit.FollowupError, match="budget"):
        audit.evaluate(p, annotations)


def test_invalid_probabilities_and_empty_model_set_are_not_repaired():
    with pytest.raises(audit.FollowupError, match="sum"):
        audit._probabilities({"probabilities": {"A": .1, "B": .1}}, ["A", "B"], multilabel=False)
    assert audit._prediction({"prediction": []}, "reasoning_tags", {"reasoning_tags": ["UNKNOWN"]}) == []


def test_cli_default_no_write_and_output_never_overwrites(tmp_path, capsys):
    p = package(tmp_path, natural=[{"sample_id": "a", "heads": ["stance"]}])
    annotations = p / "human-labels.jsonl"
    write_jsonl(annotations, [])
    before = {x.name: x.read_bytes() for x in p.iterdir()}
    assert audit.main(["--package", str(p), "--annotations", str(annotations)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "WAITING_FOR_HUMAN"
    assert {x.name: x.read_bytes() for x in p.iterdir()} == before
    assert audit.main(["--package", str(p), "--annotations", str(annotations), "--output", str(annotations)]) == 2
    assert annotations.read_bytes() == before[annotations.name]
    output = p / "report.json"
    assert audit.main(["--package", str(p), "--annotations", str(annotations), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert audit.main(["--package", str(p), "--annotations", str(annotations), "--output", str(output)]) == 2
    assert output.read_bytes() == saved
