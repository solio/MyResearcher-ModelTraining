from __future__ import annotations

import pytest

from tools import feedback_adapter_cumulative as cumulative
from tools import feedback_adapter_v1 as adapter


def _rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [{"sample_id": f"{prefix}-{index}", "model_text": "semantic input"} for index in range(count)]


def _answers(sample_ids: list[str]) -> dict[tuple[str, str], object]:
    return {
        (sample_id, head): ([adapter.CLASS_ORDER[head][0]] if head == "reasoning_tags" else adapter.CLASS_ORDER[head][0])
        for sample_id in sample_ids
        for head in adapter.HEADS
    }


def test_validate_layout_accepts_twenty_train_and_eight_check_ids():
    v1_train, v1_check = _rows("v1-train", 10), _rows("v1-check", 4)
    v2_train, v2_check = _rows("v2-train", 10), _rows("v2-check", 4)
    all_rows = v1_train + v1_check + v2_train + v2_check
    result = cumulative._validate_layout(v1_train, v1_check, v2_train, v2_check, _answers([row["sample_id"] for row in all_rows]), all_rows)
    assert result["train_rows"] == 20
    assert result["check_rows"] == 8
    assert len(result["train_sample_ids"]) == 20
    assert len(result["check_sample_ids"]) == 8


def test_validate_layout_rejects_train_check_overlap():
    v1_train, v1_check = _rows("v1-train", 10), _rows("v1-check", 4)
    v2_train, v2_check = _rows("v2-train", 10), _rows("v2-check", 4)
    v2_check[0]["sample_id"] = v1_train[0]["sample_id"]
    all_rows = v1_train + v1_check + v2_train + v2_check
    with pytest.raises(cumulative.CumulativeAdapterError):
        cumulative._validate_layout(v1_train, v1_check, v2_train, v2_check, _answers([row["sample_id"] for row in all_rows]), all_rows)
