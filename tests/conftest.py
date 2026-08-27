from __future__ import annotations

from copy import deepcopy

import pytest


@pytest.fixture
def canonical_input() -> dict:
    return {
        "sample_id": "P001",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "published_at": "2026-07-13T16:29:31+08:00",
        "board_context": "601012 隆基绿能",
        "model_text": "我会继续持有，因为订单增长。",
        "duplicate_cluster_id": "D001",
        "event_group": "E001",
    }


@pytest.fixture
def valid_label() -> dict:
    return {
        "sample_id": "P001",
        "schema_version": "semantic-schema-calibrated-v0.2.1",
        "stock_code": "601012",
        "stock_name": "隆基绿能",
        "published_at": "2026-07-13T16:29:31+08:00",
        "target_mode": "ON_TARGET",
        "stance": "BULL",
        "emotion_primary": "NONE_EXPLICIT",
        "emotion_target": "NOT_APPLICABLE",
        "action_tendency": "HOLD",
        "reasoning_tags": ["FUNDAMENTAL"],
        "context_dependency": "SELF_CONTAINED",
        "label_confidence": "HIGH",
        "evidence_spans": {
            "target_mode": ["继续持有"],
            "stance": ["订单增长"],
            "emotion_primary": [],
            "emotion_target": [],
            "action_tendency": ["继续持有"],
            "reasoning_tags": {"FUNDAMENTAL": ["订单增长"]},
            "context_dependency": [],
        },
    }


@pytest.fixture
def clone():
    return deepcopy

