from __future__ import annotations

import pytest

from tools import market_event_top3_encoder as study


def test_event_catalog_keeps_only_complete_pre_shock_candidates_and_thins_overlap():
    calendar = [f"2026-07-{day:02d}" for day in range(1, 13)]
    close = [10, 10, 10, 10, 10, 12, 12, 9, 9, 9, 9, 9]
    qfq = {("600000", day): {"close": value} for day, value in zip(calendar, close)}
    rows = [{"stock_code": "600000", "entry_date": day,
             "outcome_date": "2026-07-20", "return": index / 100}
            for index, day in enumerate(calendar[:9])]
    all_cases = study.event_cases(rows, calendar, qfq, threshold=0.1,
                                  first_shock=calendar[0], last_shock=calendar[-1],
                                  prior_days=5, minimum_gap=0)
    assert [(case["shock_date"], case["direction"]) for case in all_cases] == [
        ("2026-07-06", "UP"), ("2026-07-08", "DOWN")]
    assert [row["entry_date"] for row in all_cases[0]["candidates"]] == calendar[:5]
    thinned = study.event_cases(rows, calendar, qfq, threshold=0.1,
                                first_shock=calendar[0], last_shock=calendar[-1],
                                prior_days=5, minimum_gap=10)
    assert [case["shock_date"] for case in thinned] == ["2026-07-06"]


def test_top3_selection_uses_returns_for_oracle_but_only_scores_for_model():
    candidates = [{"entry_date": f"2026-07-{day:02d}",
                   "outcome_date": "2026-07-20", "return": day / 100}
                  for day in range(1, 6)]
    case = {"event_id": "600000:2026-07-06", "stock_code": "600000",
            "shock_date": "2026-07-06", "direction": "UP",
            "shock_close_return": 0.12, "candidates": candidates}
    record = study.case_record(case, [5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
    assert record["selected_indices"]["oracle"] == [4, 3, 2]
    assert record["selected_indices"]["text"] == [0, 1, 2]
    assert record["selected_indices"]["market_count"] == [4, 3, 2]
    assert record["selected_mean_return"]["text"] == pytest.approx(0.02)
    assert record["selected_mean_return"]["oracle"] == pytest.approx(0.04)
    assert record["oracle_regret"]["text"] == pytest.approx(0.02)
    aggregate = study.aggregate([record])
    assert aggregate["text_vs_market_events"] == {"better": 0, "equal": 0, "worse": 1}


def test_boundary_rank_loss_ignores_orders_within_top3_and_bottom2():
    torch = pytest.importorskip("torch")
    returns = [0.5, 0.4, 0.3, 0.2, 0.1]
    scores = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], requires_grad=True)
    changed_internal_order = torch.tensor([0.3, 0.4, 0.5, 0.1, 0.2])
    first = study.boundary_rank_loss(torch, scores, returns, temperature=0.1)
    second = study.boundary_rank_loss(torch, changed_internal_order, returns, temperature=0.1)
    assert float(first.detach()) == pytest.approx(float(second.detach()))
    first.backward()
    assert scores.grad is not None
    assert bool(torch.any(scores.grad != 0))
