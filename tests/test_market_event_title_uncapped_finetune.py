from __future__ import annotations

import copy

import pytest

from tools import market_event_title_uncapped_finetune as finetune


def _cases():
    return [{
        "event_id": "000001:2026-09-09",
        "stock_code": "000001",
        "shock_date": "2026-09-09",
        "direction": "UP",
        "shock_close_return": 0.08,
        "candidates": [
            {
                "stock_code": "000001",
                "entry_date": "2026-09-07",
                "outcome_date": "2026-09-21",
                "return": 0.04,
                "market": [1.0, 2.0],
                "market_scaled": [0.1, 0.2],
                "post_ids": ["old-monday"],
                "ages": [0.4],
            },
            {
                "stock_code": "000001",
                "entry_date": "2026-09-08",
                "outcome_date": "2026-09-22",
                "return": -0.03,
                "market": [3.0, 4.0],
                "market_scaled": [0.3, 0.4],
                "post_ids": ["old-tuesday"],
                "ages": [0.7],
            },
        ],
    }]


def _bags():
    return {
        ("000001", "2026-09-07"): {
            "sample16": ["sample-monday"],
            "all_titles": ["all-monday-1", "all-monday-2"],
        },
        ("000001", "2026-09-08"): {
            "sample16": ["sample-tuesday"],
            "all_titles": ["all-tuesday"],
        },
    }


@pytest.mark.parametrize("arm", ["sample16", "all_titles"])
def test_rebag_cases_only_replaces_post_ids_and_ages(arm):
    cases = _cases()
    bags = _bags()
    original = copy.deepcopy(cases)

    rewritten = finetune.rebag_cases(cases, bags, arm)

    expected = copy.deepcopy(original)
    for row in expected[0]["candidates"]:
        ids = bags[(row["stock_code"], row["entry_date"])][arm]
        row["post_ids"] = ids
        row["ages"] = [0.0] * len(ids)
    assert rewritten == expected
    assert cases == original
    assert rewritten is not cases
    assert rewritten[0] is not cases[0]
    for new_row, old_row in zip(rewritten[0]["candidates"], cases[0]["candidates"]):
        assert new_row is not old_row
        unit = (new_row["stock_code"], new_row["entry_date"])
        assert new_row["post_ids"] is not bags[unit][arm]


def test_rebag_cases_rejects_empty_title_bag_without_mutating_source():
    cases = _cases()
    original = copy.deepcopy(cases)
    bags = _bags()
    bags[("000001", "2026-09-08")]["all_titles"] = []

    with pytest.raises(ValueError, match="empty title bag"):
        finetune.rebag_cases(cases, bags, "all_titles")

    assert cases == original
