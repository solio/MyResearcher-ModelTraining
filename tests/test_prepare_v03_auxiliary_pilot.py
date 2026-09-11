from __future__ import annotations

from tools.prepare_v03_auxiliary_pilot import (
    AUXILIARY_FACTORS,
    PilotPreparationError,
    TRUNCATED_SAMPLE_IDS,
    _applicable_factors,
    _is_low_information,
    _looks_truncated,
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
    # Only the legacy emotion field is supplied to the applicability selector;
    # no auxiliary answer is present in this synthetic fixture.
    human = {
        f"s{i:03d}": {
            "answers": {
                "emotion_primary": "FEAR",
                "emotion_target": "MARKET" if i in {2, 4, 7, 8, 9, 12, 14, 16, 18, 19} else "COMPANY",
            }
        }
        for i in range(len(texts))
    }
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
    assert summary["market_scope_parent_validation"] is True
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


def test_market_scope_parent_and_price_action_gates():
    factors = _applicable_factors(
        "我买入后已经清仓，卖出又接回",
        {"emotion_primary": "NONE_EXPLICIT", "emotion_target": "COMPANY"},
    )
    assert "market_scope" not in factors
    assert "desired_price_direction" not in factors

    market_factors = _applicable_factors(
        "大盘和行业板块可能反弹",
        {"emotion_primary": "CALM", "emotion_target": "MARKET"},
    )
    assert "market_scope" in market_factors
    assert "desired_price_direction" in market_factors


def test_known_and_boundary_truncation_are_excluded():
    known = next(iter(TRUNCATED_SAMPLE_IDS))
    assert _looks_truncated(known, "完整与否未知", 12)
    assert _looks_truncated("other", "没有句末标点的长前缀", 39)
    assert not _looks_truncated("complete", "这是完整的一句话。", 12)

    human, sources = _fixtures()
    moved_human = human.pop("s000")
    moved_source = sources.pop("s000")
    moved_source["sample_id"] = known
    moved_source["text_length"] = 40
    human[known] = moved_human
    sources[known] = moved_source
    tasks, _ = build_pilot_package(human, sources, target_samples=12)
    assert all(row["sample_id"] != known for row in tasks)
