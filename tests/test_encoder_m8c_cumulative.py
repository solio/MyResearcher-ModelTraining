from __future__ import annotations

import torch

from semantic_model.auxiliary_heads import AuxiliaryTargetBatch
from semantic_model.encoder_m8c_cumulative import (
    merge_task_rows,
    restrict_batch_to_pairs,
    summarize_fresh_rows,
)


def test_duplicate_tasks_union_heads_without_duplicate_samples():
    first = [{"sample_id": "s1", "text": "大盘上涨", "title": "", "heads": ["market_scope"], "metadata": {"exact_content_group": "g1"}}]
    second = [{"sample_id": "s1", "text": "大盘上涨", "title": "", "heads": ["desired_price_direction"], "metadata": {"exact_content_group": "g1"}}]
    merged = merge_task_rows((first, second))
    assert len(merged) == 1
    assert merged[0]["heads"] == ["market_scope", "desired_price_direction"]


def test_restrict_batch_zeroes_unrequested_pairs_without_changing_targets():
    batch = AuxiliaryTargetBatch(
        targets={"expectation_phase": torch.tensor([1, 0]), "arousal_level": torch.tensor([2, 1])},
        loss_masks={"expectation_phase": torch.tensor([1.0, 1.0]), "arousal_level": torch.tensor([1.0, 1.0])},
        sample_ids=("s1", "s2"),
    )
    restricted = restrict_batch_to_pairs(batch, ["s1", "s2"], {("s1", "expectation_phase")})
    assert restricted.targets["expectation_phase"].tolist() == [1, 0]
    assert restricted.loss_masks["expectation_phase"].tolist() == [1.0, 0.0]
    assert restricted.loss_masks["arousal_level"].tolist() == [0.0, 0.0]


def test_fresh_summary_keeps_empty_multilabel_as_empty_and_accepts_long_text():
    rows = [
        {
            "sample_id": "f1",
            "model_text": "这是一段超过历史三十九字过滤边界但必须进入 fresh pool 的完整文本。" * 2,
            "applicable_factors": ["market_scope", "discourse_focus"],
            "auxiliary": {
                "market_scope": {"prediction": [], "confidence": 0.51, "normalized_entropy": 0.9},
                "discourse_focus": {"prediction": ["AUTHOR_VIEWPOINT"], "confidence": 0.7, "normalized_entropy": 0.4},
            },
        },
        {
            "sample_id": "f2",
            "model_text": "期待大盘上涨",
            "applicable_factors": ["expectation_phase"],
            "auxiliary": {
                "expectation_phase": {"prediction": ["ANTICIPATED"], "confidence": 0.8, "normalized_entropy": 0.2},
            },
        },
    ]
    summary = summarize_fresh_rows(rows)
    assert summary["sample_count"] == 2
    market = summary["factor_summary"]["market_scope"]
    assert market["applicable_sample_count"] == 1
    assert market["empty_prediction_count"] == 1
    assert market["empty_prediction_rate"] == 1.0
    discourse = summary["factor_summary"]["discourse_focus"]
    assert discourse["empty_prediction_count"] == 0
    assert discourse["prediction_distribution"] == {"AUTHOR_VIEWPOINT": 1}
