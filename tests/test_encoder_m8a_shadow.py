from __future__ import annotations

import torch

from semantic_model.encoder_m8a_shadow import (
    AUXILIARY_THRESHOLD,
    MULTI_FACTORS,
    prepare_shadow_rows,
    summarize_shadow,
)


def test_shadow_preparation_uses_text_cues_and_pair_coverage_only():
    feed = [
        {"sample_id": "s1", "model_text": "期待大盘上涨", "exact_content_group": "g1", "text_length": 6},
        {"sample_id": "s2", "model_text": "市场", "exact_content_group": "g2", "text_length": 2},
        {"sample_id": "s3", "model_text": "买入后持有", "exact_content_group": "g3", "text_length": 5},
        {"sample_id": "s4", "model_text": "...", "exact_content_group": "g4", "text_length": 3},
        {"sample_id": "s5", "model_text": "这是一条未完的文本" + "x" * 35, "exact_content_group": "g5", "text_length": 39},
        {"sample_id": "s6", "model_text": "期待大盘上涨", "exact_content_group": "duplicate", "text_length": 6},
        {"sample_id": "s7", "model_text": "期待大盘上涨", "exact_content_group": "duplicate", "text_length": 6},
    ]
    records, metadata = prepare_shadow_rows(feed, {("s1", "market_scope")})
    assert [record.sample_id for record in records] == ["s1", "s2", "s3"]
    assert "market_scope" in metadata[0]["applicable_factors"]
    assert "market_scope" not in metadata[0]["uncovered_factors"]
    assert "desired_price_direction" in metadata[0]["uncovered_factors"]
    assert "market_scope" not in metadata[1]["applicable_factors"]
    assert "desired_price_direction" not in metadata[2]["applicable_factors"]


def test_shadow_summary_reports_multilabel_threshold_entropy_and_collapse():
    rows = []
    for index in range(4):
        rows.append(
            {
                "sample_id": f"s{index}",
                "auxiliary": {
                    "expectation_phase": {
                        "prediction": ["ANTICIPATED"], "confidence": 0.95, "normalized_entropy": 0.1,
                    },
                    "market_scope": {
                        "prediction": ["BROAD_MARKET"], "confidence": 0.95, "normalized_entropy": 0.9, "high_uncertainty": True,
                    },
                },
            }
        )
    summary = summarize_shadow(rows)
    market = summary["factor_summary"]["market_scope"]
    assert market["applicable_uncovered_sample_count"] == 4
    assert market["single_category_collapse"] is True
    assert market["high_uncertainty_count"] == 4
    assert summary["high_uncertainty_union_count"] == 4
    assert summary["criteria"]["multi_label_threshold"] == AUXILIARY_THRESHOLD


def test_shadow_summary_accepts_empty_factor_without_fabricating_labels():
    summary = summarize_shadow([{"sample_id": "s", "auxiliary": {}}])
    assert summary["factor_summary"]["discourse_focus"]["applicable_uncovered_sample_count"] == 0
    assert summary["factor_summary"]["discourse_focus"]["prediction_distribution"] == {}
    assert MULTI_FACTORS == {"market_scope", "discourse_focus"}
