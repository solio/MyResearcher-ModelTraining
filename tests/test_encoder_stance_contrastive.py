from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from semantic_model import encoder_stance_contrastive as stance
from semantic_model.schema import LabelSchema


ROOT = Path(__file__).resolve().parents[1]


def test_paired_auxiliary_examples_are_distinct_and_not_human_labels() -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    from semantic_model.encoder_m1 import M1Record

    human = [M1Record(f"human-{i}", f"60{i:04d}", f"股票{i}", f"已有人工样本{i}", {"stance": "NEUTRAL"}, {"stance": 1.0}) for i in range(8)]
    rows = stance._auxiliary_records(schema, human)
    assert len(rows) == 2 * len(stance.PAIRED_AUXILIARY) == 12
    assert len({row.sample_id for row in rows}) == len(rows)
    assert {row.label["stance"] for row in rows} == {"BULL", "BEAR", "NEUTRAL"}
    assert all(row.weights["stance"] == stance.AUXILIARY_WEIGHT < 1.0 for row in rows)
    assert all(row.sample_id.startswith("stance-pair-") for row in rows)
    assert {stance._norm(row.model_text) for row in rows}.isdisjoint(stance._norm(row.model_text) for row in human)


def test_normalization_and_html_text_extraction_are_deterministic() -> None:
    assert stance._norm("卖飞了！") == stance._norm("卖飞了")
    assert stance._plain_text("<div>之前卖早了&nbsp;，后来涨了</div>") == "之前卖早了 ，后来涨了"


def test_directional_error_counts_separate_future_reversal_and_past_regret() -> None:
    rows = [
        {"review_label": "BULL", "probe_type": "future_bull"},
        {"review_label": "BEAR", "probe_type": "future_bear"},
        {"review_label": "NEUTRAL", "probe_type": "past_regret_only"},
        {"review_label": "BULL", "probe_type": "future_bull"},
    ]
    result = stance._directional_counts(rows, ["BEAR", "BULL", "BEAR", "NEUTRAL"])
    assert result["BULL_to_BEAR"] == 1
    assert result["BEAR_to_BULL"] == 1
    assert result["past_regret_to_directional"] == 1
    assert result["target_error_total"] == 3
    assert result["future_direction_accuracy"] == 0.0


def test_candidate_merge_changes_stance_only_and_preserves_conditioned_payload() -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    from semantic_model import encoder_m2_selected_runtime as selected

    row = {
        "sample_id": "sample-1",
        "predictions": {
            head: {"prediction": label, "probabilities": {value: float(value == label) for value in schema.class_order[head]}}
            for head, label in {
                "target_mode": "ON_TARGET",
                "stance": "BEAR",
                "emotion_primary": "NONE_EXPLICIT",
                "emotion_target": "NOT_APPLICABLE",
                "action_tendency": "HOLD",
                "reasoning_tags": ["NO_REASON_GIVEN"],
                "context_dependency": "SELF_CONTAINED",
            }.items()
        },
        "conditioned_outputs": {"emotion_target": {"effective_prediction": "NOT_APPLICABLE"}},
    }
    probabilities = np.zeros((1, len(schema.class_order["stance"])), dtype=float)
    probabilities[0, schema.class_order["stance"].index("BULL")] = 1.0
    before = deepcopy(row)
    merged = stance._merge_stance_rows([row], probabilities, schema, "CANDIDATE")
    assert set(merged[0]["predictions"]) == set(selected.HEADS)
    assert merged[0]["predictions"]["stance"]["prediction"] == "BULL"
    for head in set(selected.HEADS) - {"stance"}:
        assert merged[0]["predictions"][head] == before["predictions"][head]
    assert merged[0]["conditioned_outputs"] == before["conditioned_outputs"]
    assert row == before


def test_promotion_rule_is_predeclared_and_never_uses_market_returns() -> None:
    assert stance.PROMOTION_RULE["requires_reviewed_future_bull"] >= 5
    assert stance.PROMOTION_RULE["requires_reviewed_future_bear"] >= 5
    assert stance.PROMOTION_RULE["requires_reviewed_past_regret_only"] >= 3
    assert stance.PROMOTION_RULE["market_outcomes_allowed"] is False


def test_postfit_recovery_reuses_saved_blind_predictions_without_inference(tmp_path: Path) -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    from semantic_model.encoder_m1 import M1Record

    order = list(schema.class_order["stance"])
    vector = {label: (1.0 if label == "NEUTRAL" else 0.0) for label in order}
    review = [{"sample_id": "blind-1", "text": "刚才卖早了", "review_label": "NEUTRAL", "evidence": "只回顾已完成交易", "text_sha256": "same", "probe_type": "past_regret_only"}]
    records = [M1Record("blind-1", "600000", "示例", "刚才卖早了", {"stance": "NEUTRAL"}, {"stance": 1.0})]
    saved = {
        "sample_id": "blind-1",
        "text": "刚才卖早了",
        "review_label": "NEUTRAL",
        "evidence": "只回顾已完成交易",
        "a5_prediction": "NEUTRAL",
        "candidate_prediction": "NEUTRAL",
        "a5_probabilities": vector,
        "candidate_probabilities": vector,
    }
    (tmp_path / "blind-evaluation-predictions.jsonl").write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    evaluation = stance._load_saved_blind_evaluation(tmp_path, review, records, schema)
    assert evaluation is not None
    assert evaluation[0] is None  # a crash after saving blind predictions can resume without rerunning inference
    assert evaluation[1][0]["candidate_prediction"] == "NEUTRAL"


def test_postfit_recovery_rejects_prediction_rows_rebound_to_other_text(tmp_path: Path) -> None:
    schema = LabelSchema.load(ROOT / "schema" / "semantic-schema-calibrated-v0.2.1.json")
    from semantic_model.encoder_m1 import M1Record

    order = list(schema.class_order["stance"])
    vector = {label: (1.0 if label == "NEUTRAL" else 0.0) for label in order}
    review = [{"sample_id": "blind-1", "text": "真实原文", "review_label": "NEUTRAL", "evidence": "文本证据", "text_sha256": "same", "probe_type": "past_regret_only"}]
    records = [M1Record("blind-1", "600000", "示例", "真实原文", {"stance": "NEUTRAL"}, {"stance": 1.0})]
    saved = {"sample_id": "blind-1", "text": "不同文本", "review_label": "NEUTRAL", "evidence": "文本证据", "a5_prediction": "NEUTRAL", "candidate_prediction": "NEUTRAL", "a5_probabilities": vector, "candidate_probabilities": vector}
    (tmp_path / "blind-evaluation-predictions.jsonl").write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "evaluation.json").write_text(json.dumps({"blind_review_was_frozen_before_training_and_predictions": True, "market_outcomes_read": False}), encoding="utf-8")
    try:
        stance._load_saved_blind_evaluation(tmp_path, review, records, schema)
    except Exception as exc:
        assert getattr(exc, "code", None) == "STANCE_BLIND_EVALUATION_REVIEW_BINDING"
    else:
        raise AssertionError("recovery accepted a prediction bound to different text")
