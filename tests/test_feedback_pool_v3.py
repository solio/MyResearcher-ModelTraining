from __future__ import annotations

import json

import pytest

from tools import feedback_pool_v3 as pool


def _prediction(sample_id: str, index: int) -> dict:
    scalar_classes = {
        "target_mode": ("ON_TARGET", "CROSS_TARGET", "MARKET_GENERAL"),
        "stance": ("UNKNOWN", "BEAR", "NEUTRAL", "BULL"),
        "emotion_primary": ("FRUSTRATION", "NONE_EXPLICIT", "ANGER", "HOPE"),
        "emotion_target": ("PRICE", "POSITION", "NOT_APPLICABLE", "COMPANY"),
        "action_tendency": ("NO_ACTION_SIGNAL", "BUY", "SELL", "HOLD"),
        "context_dependency": ("SELF_CONTAINED", "EXTERNAL_CONTEXT_REQUIRED"),
    }
    outputs = {}
    for head, classes in scalar_classes.items():
        selected = classes[index % len(classes)]
        tail = (1.0 - 0.65) / (len(classes) - 1)
        outputs[head] = {
            "prediction": selected,
            "probabilities": {value: (0.65 if value == selected else tail) for value in classes},
        }
    labels = ["NO_REASON_GIVEN"] if index % 3 == 0 else ["TECHNICAL_PRICE", "FUNDAMENTAL"]
    reasoning_probs = {label: (0.52 if label in labels else 0.12) for label in ("NO_REASON_GIVEN", "TECHNICAL_PRICE", "FUNDAMENTAL")}
    outputs["reasoning_tags"] = {
        "prediction": labels,
        "predicted_labels": labels,
        "probabilities": reasoning_probs,
        "thresholds": {label: 0.5 for label in reasoning_probs},
    }
    return {"sample_id": sample_id, "predictions": outputs}


def _feed(index: int) -> dict:
    if index < 35:
        length = 4
    elif index < 75:
        length = 16
    else:
        length = 28
    return {
        "sample_id": f"fl1-synthetic-{index:03d}",
        "observation_id": f"obs-{index:03d}",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "model_text": f"这是可判断的反馈样本 {index}",
        "text": f"这是可判断的反馈样本 {index}",
        "text_length": length,
        "exact_content_group": f"group-{index:03d}",
        "sampling_reasons": ["VERY_SHORT" if length < 10 else "MEDIUM_LENGTH" if length <= 25 else "LONGER_TEXT"],
    }


def test_select_pool_is_deterministic_disjoint_and_balanced(monkeypatch):
    monkeypatch.setattr(pool, "EXPECTED_SOURCE_ROWS", 100)
    feed = [_feed(i) for i in range(100)]
    predictions = [_prediction(row["sample_id"], i) for i, row in enumerate(feed)]
    train, check, summary = pool.select_pool(feed, predictions, exclude_sample_ids={feed[0]["sample_id"], feed[1]["sample_id"]})
    assert len(train) == 80 and len(check) == 14
    assert len({row["sample_id"] for row in train + check}) == 94
    assert {row["exact_content_group"] for row in train}.isdisjoint({row["exact_content_group"] for row in check})
    assert {row["length_bucket"] for row in train} == {"SHORT", "MEDIUM", "LONG"}
    assert {row["length_bucket"] for row in check} == {"SHORT", "MEDIUM", "LONG"}
    assert summary["human_labels_used_for_selection"] is False
    train2, check2, _ = pool.select_pool(feed, predictions, exclude_sample_ids={feed[0]["sample_id"], feed[1]["sample_id"]})
    assert [r["sample_id"] for r in train + check] == [r["sample_id"] for r in train2 + check2]


def test_known_low_information_and_prior_ids_are_not_selected(monkeypatch):
    monkeypatch.setattr(pool, "EXPECTED_SOURCE_ROWS", 100)
    feed = [_feed(i) for i in range(100)]
    feed[2]["model_text"] = feed[2]["text"] = "……"
    feed[3]["model_text"] = feed[3]["text"] = "又了"
    predictions = [_prediction(row["sample_id"], i) for i, row in enumerate(feed)]
    train, check, summary = pool.select_pool(feed, predictions, exclude_sample_ids={feed[0]["sample_id"]})
    selected = {row["sample_id"] for row in train + check}
    assert feed[0]["sample_id"] not in selected
    assert feed[2]["sample_id"] not in selected and feed[3]["sample_id"] not in selected
    assert summary["filter_counts"]["low_information"] == 2


def test_reasoning_and_scalar_uncertainty_are_in_selection_features(monkeypatch):
    monkeypatch.setattr(pool, "EXPECTED_SOURCE_ROWS", 1)
    row = _feed(1)
    row["text_length"] = 16
    row.update(pool._refresh_snapshot([row], [_prediction(row["sample_id"], 1)])[0])
    tokens = pool._feature_tokens(row)
    assert "stance:BEAR" in tokens
    assert "reasoning:TECHNICAL_PRICE" in tokens
    assert "length:MEDIUM" in tokens


def test_labeling_payload_contains_no_model_predictions_or_labels(monkeypatch):
    monkeypatch.setattr(pool, "EXPECTED_SOURCE_ROWS", 1)
    row = _feed(1)
    row.update(pool._refresh_snapshot([row], [_prediction(row["sample_id"], 1)])[0])
    row.update({"frozen_role": "HUMAN_TRAIN", "selection_stratum": "SEVEN_HEAD_COVERAGE", "length_bucket": "MEDIUM"})
    payload = pool.build_labeling_payload([row])[0]
    assert "predictions" not in payload
    assert "probabilities" not in json.dumps(payload, ensure_ascii=False)
    assert set(payload) == {"sample_id", "text", "title", "metadata"}
    assert not any(key in payload["metadata"] for key in ("labels", "weights", "predicted_classes", "uncertainty_score"))


def test_known_short_semantic_text_remains_eligible():
    for text in ("已清仓，随便吧", "明天涨停板", "抄底", "看好", "垃圾"):
        assert not pool.is_known_incomplete({
            "sample_id": "s",
            "model_text": text,
            "text": text,
            "stock_code": "601012",
            "stock_name": "隆基绿能",
            "exact_content_group": "g",
            "text_length": len(text),
        })


def test_known_incomplete_url_or_trailing_separator_is_excluded():
    base = {
        "sample_id": "s",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "exact_content_group": "g",
        "text_length": 20,
    }
    assert pool.is_known_incomplete({**base, "model_text": "https://example.test/post"})
    assert pool.is_known_incomplete({**base, "model_text": "这公司真，"})


def test_pool_sizes_are_frozen():
    with pytest.raises(pool.FeedbackLoopError):
        pool.select_pool([], [], train_count=10, check_count=4)
