from __future__ import annotations

from types import SimpleNamespace

import pytest

from semantic_model import evaluate_m3g_test as m3g


def test_scalar_metrics_reports_frozen_order_and_confusion_matrix():
    result = m3g._scalar_metrics(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "A"],
        ["A", "B"],
    )
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == {"A": {"A": 1, "B": 1}, "B": {"A": 1, "B": 1}}
    assert result["per_class"]["A"]["support"] == 2
    assert result["per_class"]["A"]["f1"] == pytest.approx(0.5)


def test_reasoning_metrics_reports_macro_micro_exact_and_each_label():
    result = m3g._reasoning_metrics([[1, 0], [0, 1]], [[1, 1], [0, 1]], ["FUNDAMENTAL", "NO_REASON_GIVEN"])
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert result["micro_f1"] == pytest.approx(0.8)
    assert result["exact_set_accuracy"] == pytest.approx(0.5)
    assert result["per_label"]["FUNDAMENTAL"]["support"] == 1
    assert result["per_label"]["NO_REASON_GIVEN"]["f1"] == pytest.approx(2 / 3)


def test_load_test_partition_reads_only_test_split_and_selected_inputs(monkeypatch):
    paths: list[str] = []
    labels = [
        {
            "sample_id": f"t{i}",
            "stock_code": "000001",
            "stock_name": "测试",
            "target_mode": "ON_TARGET",
            "stance": "BULL",
            "emotion_primary": "HOPE",
            "emotion_target": "PRICE",
            "action_tendency": "BUY",
            "context_dependency": "SELF_CONTAINED",
            "reasoning_tags": ["FUNDAMENTAL"],
        }
        for i in range(m3g.TEST_ROWS)
    ]

    class FakeConfig:
        def data_path(self, key: str):
            paths.append(key)
            return key

    monkeypatch.setattr(m3g.m1.ProjectConfig, "load", lambda _path: FakeConfig())
    monkeypatch.setattr(m3g.LabelSchema, "load", lambda _path: SimpleNamespace(class_order={head: list(order) for head, order in {
        "target_mode": ["ON_TARGET"], "stance": ["BULL"], "emotion_primary": ["HOPE"], "emotion_target": ["PRICE"], "action_tendency": ["BUY"], "context_dependency": ["SELF_CONTAINED"], "reasoning_tags": ["FUNDAMENTAL"]
    }.items()}))
    monkeypatch.setattr(m3g.m1, "_read_jsonl", lambda path: labels if path == "split_labels_test" else pytest.fail(f"unexpected read: {path}"))
    monkeypatch.setattr(m3g.m1, "_read_selected_jsonl", lambda path, ids: {sample_id: {"sample_id": sample_id, "stock_code": "000001", "stock_name": "测试", "model_text": "文本"} for sample_id in ids})

    schema, records, identity = m3g.load_test_partition("config.yaml")
    assert len(records) == m3g.TEST_ROWS
    assert identity["role"] == "Test_467_weak_labels_only"
    assert paths.count("split_labels_test") == 2
    assert paths.count("canonical_inputs") == 2
    assert not any("train" in path or "dev" in path for path in paths)
