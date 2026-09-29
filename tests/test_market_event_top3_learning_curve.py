from __future__ import annotations

import pytest

from tools import market_event_top3_learning_curve as study


def test_internal_partition_has_matured_fit_embargo_and_no_august_rows():
    rows = [
        {"entry_date": "2026-06-10", "outcome_date": "2026-06-23"},
        {"entry_date": "2026-06-24", "outcome_date": "2026-07-07"},
        {"entry_date": "2026-06-26", "outcome_date": "2026-07-09"},
        {"entry_date": "2026-07-20", "outcome_date": "2026-08-03"},
        {"entry_date": "2026-08-03", "outcome_date": "2026-08-14"},
    ]
    fit, validation = study.internal_partition(rows)
    assert fit == [rows[0]]
    assert validation == [rows[2]]
    assert fit[0]["outcome_date"] < validation[0]["entry_date"]


def test_best_epoch_tie_prefers_earliest_snapshot():
    curve = [
        {"epoch": 0, "mean_validation_regret": 0.03},
        {"epoch": 8, "mean_validation_regret": 0.01},
        {"epoch": 3, "mean_validation_regret": 0.01},
    ]
    assert study.select_best_epoch(curve) == 3
    with pytest.raises(ValueError, match="empty"):
        study.select_best_epoch([])
