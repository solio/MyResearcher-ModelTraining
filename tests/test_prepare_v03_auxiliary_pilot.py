from __future__ import annotations

from tools.prepare_v03_auxiliary_pilot import (
    AUXILIARY_FACTORS,
    PilotPreparationError,
    _is_low_information,
    build_pilot_package,
)


def _fixtures():
    texts = [
        "明天涨停，我很期待",
        "今天跌了，心里很焦虑",
        "下周看大盘和行业板块，大家怎么看，这个季度政策和资金会怎么走",
        "已经清仓，后悔又难受",
        "机构资金流入，市场可能反弹",
        "我要买入，感觉机会来了",
        "上周卖出，今天又接回",
        "光伏行业价格回升，值得关注",
        "大家都说要抄底，我不太相信",
        "主力拉升，散户跟上",
        "未来会不会继续上涨",
        "跌停以后还能反弹吗",
        "大盘震荡，行业轮动明显",
        "我持仓很久，终于等到回本",
        "消息发布后市场情绪兴奋",
        "卖出之后觉得很后悔",
        "明年政策支持光伏板块",
        "你们觉得现在适合买入吗",
        "基金资金撤出，板块下跌",
        "今天看好公司，准备加仓",
    ]
    human = {f"s{i:03d}": {} for i in range(len(texts))}
    sources = {}
    for i, text in enumerate(texts):
        source = "v1" if i < 6 else "v2" if i < 12 else "v3"
        sources[f"s{i:03d}"] = {
            "sample_id": f"s{i:03d}",
            "text": text,
            "observation_id": f"obs-{i}",
            "stock_code": "600000",
            "stock_name": "测试股票",
            "exact_content_group": f"group-{i}",
            "sampling_reasons": ["SYNTHETIC"],
            "text_length": len(text),
            "source_batch": source,
            "selection_score": 1.0 - i / 100,
        }
    return human, sources


def test_pilot_is_deterministic_sparse_and_answer_free():
    human, sources = _fixtures()
    first, summary = build_pilot_package(human, sources, batch_id="synthetic", target_samples=12)
    second, _ = build_pilot_package(human, sources, batch_id="synthetic", target_samples=12)

    assert first == second
    assert len(first) == 12
    assert summary["sample_count"] == 12
    assert summary["assignment_count"] <= 50
    assert summary["distinct_exact_content_groups"] == 12
    assert set(summary["factor_task_counts"]) == set(AUXILIARY_FACTORS)
    assert all(summary["factor_task_counts"][factor] > 0 for factor in AUXILIARY_FACTORS)
    assert set(summary["length_distribution"]) == {"SHORT", "MEDIUM", "LONG"}
    assert set(summary["source_distribution"]) == {"v1", "v2", "v3"}

    forbidden = {
        "answer",
        "label",
        "predictions",
        "probabilities",
        "target_mode",
        "stance",
        "emotion_primary",
        "emotion_target",
        "action_tendency",
        "context_dependency",
        "reasoning_tags",
    }
    for row in first:
        assert set(row) == {"sample_id", "text", "title", "metadata", "heads"}
        assert set(row["heads"]) <= set(AUXILIARY_FACTORS)
        assert row["heads"]
        assert not forbidden.intersection(row)
        assert not forbidden.intersection(row["metadata"])
        assert row["text"].strip()


def test_low_information_filter_does_not_rely_on_short_length():
    assert _is_low_information("   ")
    assert _is_low_information("……")
    assert _is_low_information("又了")
    assert _is_low_information("暂无")
    assert _is_low_information("!!!")
    for text in ("抄底", "看好", "垃圾", "明天涨停板", "已清仓，随便吧"):
        assert not _is_low_information(text)


def test_human_and_source_identity_must_match():
    human, sources = _fixtures()
    sources.pop("s000")
    try:
        build_pilot_package(human, sources, target_samples=12)
    except PilotPreparationError as exc:
        assert "identity mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("identity mismatch must fail closed")
