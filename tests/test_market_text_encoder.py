from __future__ import annotations

import pytest

from tools import market_text_encoder as study


def test_fit_uses_only_outcomes_matured_before_diagnostic_entry():
    rows = [
        {"entry_date": "2026-07-20", "outcome_date": "2026-07-31"},
        {"entry_date": "2026-07-21", "outcome_date": "2026-08-03"},
        {"entry_date": "2026-08-03", "outcome_date": "2026-08-14"},
    ]
    fit, diagnostic = study.split(rows, "2026-07-31", "2026-08-03")
    assert fit == [rows[0]]
    assert diagnostic == [rows[2]]
    with pytest.raises(ValueError, match="precede"):
        study.split(rows, "2026-08-03", "2026-08-03")


def test_up_event_groups_never_use_down_shock_or_future_diagnostic_label():
    calendar = [f"2026-07-{day:02d}" for day in range(1, 13)]
    qfq = {("600000", day): {"close": 10.0} for day in calendar}
    qfq[("600000", "2026-07-06")]["close"] = 12.0
    qfq[("600000", "2026-07-07")]["close"] = 10.0
    fit = [{"stock_code": "600000", "entry_date": day, "outcome_date": "2026-07-12"}
           for day in calendar[:5]]
    threshold, groups = study.training_up_events(fit, calendar, qfq, fit_end="2026-07-12",
                                                   prior_days=5, top_gap_sessions=1)
    assert threshold > 0
    assert groups == [[0, 1, 2, 3, 4]]


def test_market_encoder_has_no_seven_head_prediction_dependency():
    # The raw-title training module may reuse price-panel helpers but must not
    # read m12/A0/A5 post-level probability files in its own data path.
    import inspect

    source = inspect.getsource(study.build_text_rows)
    assert "predictions" not in source
    assert "model_text" not in source
    assert 'row.get("title")' in source
