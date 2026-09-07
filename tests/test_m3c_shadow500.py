from __future__ import annotations

import hashlib
import json

import pytest

from tools import m3c_shadow500 as shadow


def _input(sample_id: str, stock: str = "000001", **extra: str) -> dict[str, str]:
    row = {
        "sample_id": sample_id,
        "stock_code": stock,
        "stock_name": "Synthetic",
        "model_text": "A sufficiently long synthetic post for inference.",
        "text_quality_tier": "B_TITLE_RICH",
        "model_eligibility": "ELIGIBLE",
        "quality_flags": "",
    }
    row.update(extra)
    return row


def test_selection_is_hash_ordered_and_uses_stock_quotas():
    rows = [_input(f"s-{stock}-{index}", stock) for stock in ("000001", "000002") for index in range(40)]
    # The production selector requires sixteen stocks; provide explicit small
    # quotas here to test deterministic ordering without a large fixture.
    quotas = {f"{index:06d}": 1 for index in range(1, 17)}
    rows = [_input(f"s-{stock}", stock) for stock in quotas]
    selected = shadow.select_shadow_rows(rows, quotas=quotas)
    assert [row["stock_code"] for row in selected] == sorted(quotas)
    assert [row["sample_id"] for row in selected] == [f"s-{stock}" for stock in sorted(quotas)]


def test_selection_hash_key_is_stable():
    sample_id = "P-fixed"
    assert shadow._sha256_selection(sample_id) == hashlib.sha256(
        f"m3c-shadow500-v1|{sample_id}".encode("utf-8")
    ).hexdigest()


def test_candidate_buckets_are_ordered_deduplicated_and_safe():
    rows = [
        {
            **_input("low", quality_flags=""),
            "six_head_mean_max_probability": 0.10,
            "tokenizer_unk_ratio": 0.01,
            "abstain": True,
        },
        {
            **_input("risk", quality_flags="SHORT_CONTEXT"),
            "six_head_mean_max_probability": 0.90,
            "tokenizer_unk_ratio": 0.50,
            "abstain": False,
        },
        {
            **_input("short", text_quality_tier="C_SHORT_TEXT"),
            "six_head_mean_max_probability": 0.80,
            "tokenizer_unk_ratio": 0.02,
            "abstain": False,
        },
    ]
    result = shadow.choose_review_candidates(rows)
    assert result[0]["review_bucket"] == "LOW_CONFIDENCE"
    assert result[1]["review_bucket"] == "HIGHEST_UNK_RATIO"
    assert {row["sample_id"] for row in result} == {"low", "risk", "short"}
    assert [row["review_bucket"] for row in result] == sorted(
        [row["review_bucket"] for row in result],
        key=[
            "LOW_CONFIDENCE",
            "HIGHEST_UNK_RATIO",
            "RISK_FLAGGED_HIGH_CONFIDENCE",
            "SHORT_OR_MISSING_CONTEXT_RELEASED",
        ].index,
    )
    assert len({row["sample_id"] for row in result}) == len(result)
    forbidden = {"author_id", "author_name", "url", "label", "labels", "split", "role"}
    assert not forbidden.intersection(result[0])


def test_risk_and_short_rules_are_explicit():
    assert shadow._risk_flagged(_input("a", quality_flags="LOW_INFORMATION"))
    assert not shadow._risk_flagged(_input("b"))
    assert shadow._short_or_missing_context(_input("c", text_quality_tier="C_SHORT_TEXT"))
    assert shadow._short_or_missing_context(_input("d", model_text="tiny"))


def test_fixed_abstention_threshold_is_strict():
    assert shadow.THRESHOLD == pytest.approx(0.5243767648935318)
    assert (shadow.THRESHOLD - 1e-12) < shadow.THRESHOLD


def test_input_projection_keeps_only_inputs_and_risk_metadata():
    source = _input("sample")
    source.update({"author_id": "secret", "author_name": "secret", "url": "secret", "target_mode": "label"})
    projected = {field: source[field] for field in (*shadow.SAFE_INPUT_FIELDS, *shadow.RISK_FIELDS)}
    assert set(projected) == set((*shadow.SAFE_INPUT_FIELDS, *shadow.RISK_FIELDS))
    assert "author_id" not in json.dumps(projected)
    assert "target_mode" not in projected
