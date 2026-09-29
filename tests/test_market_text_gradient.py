from __future__ import annotations

import numpy as np
import pytest

from tools import market_text_gradient as study


def _prediction():
    return {"sample_id": "p1", "predictions": {
        "stance": {"probabilities": {"BEAR": 0.6, "BULL": 0.1, "MIXED": 0.1, "NEUTRAL": 0.1, "UNKNOWN": 0.1}},
        "emotion_primary": {"probabilities": {label: 0.01 for label in (*study.NEGATIVE, *study.POSITIVE)}},
        "action_tendency": {"probabilities": {"BUY": 0.1, "ADD": 0.2, "SELL": 0.05, "REDUCE": 0.05}},
        "target_mode": {"probabilities": {"ON_TARGET": 0.8}},
    }}


def test_post_polarity_is_frozen_bear_minus_bull_not_market_outcome():
    rep, polarity = study.post_representation({"sample_id": "p1", "model_text": "看空"}, _prediction(), 4, 5)
    assert len(rep) == len(study.POST_KEYS)
    assert polarity == pytest.approx(0.5)
    assert rep[-1] == 1.0
    assert rep[7] == pytest.approx(0.3)


def test_fit_requires_matured_labels_before_diagnostic_entry():
    rows = [
        {"entry_date": "2026-07-20", "outcome_date": "2026-07-31"},
        {"entry_date": "2026-08-03", "outcome_date": "2026-08-14"},
    ]
    fit, diagnostic = study.split_rows(rows, "2026-07-31", "2026-08-03")
    assert len(fit) == len(diagnostic) == 1
    with pytest.raises(ValueError, match="precede"):
        study.split_rows(rows, "2026-08-03", "2026-08-03")
    with pytest.raises(ValueError, match="empty fit"):
        study.split_rows([rows[0] | {"outcome_date": "2026-08-04"}, rows[1]],
                         "2026-07-31", "2026-08-03")


def test_framework_free_checkpoint_replay_uses_only_saved_parameters():
    row = {"market": [2.0], "post_x": [[0.0] * len(study.POST_KEYS), [0.0] * len(study.POST_KEYS)],
           "polarity": [0.5, -0.5], "return": 0.0}
    state = {"beta": [1.0], "intercept": 0.0, "gamma": 1.0, "attention": None}
    checkpoint = {"market_mean": [1.0], "market_std": [1.0], "return_mean": 0.1, "return_std": 0.2,
                  "equal_semantic_mean": 0.0, "equal_semantic_std": 1.0,
                  "models": {"market_count": state | {"gamma": None},
                             "equal_semantic": state,
                             "learned_post_weight": state | {"attention": [0.0] * len(study.POST_KEYS)}}}
    replay = study.predict_from_checkpoint([row], checkpoint)
    assert replay["market_count"][0] == pytest.approx(0.3)
    assert replay["equal_semantic"][0] == pytest.approx(0.3)
    assert replay["learned_post_weight"][0] == pytest.approx(0.3)
    assert np.isfinite([values[0] for values in replay.values()]).all()


def test_event_oracle_is_not_used_as_fit_label_and_requires_prior_candidates():
    calendar = [f"2026-08-{day:02d}" for day in range(3, 10)]
    qfq = {("600000", day): {"close": 10.0 if i < 5 else 11.0}
           for i, day in enumerate(calendar)}
    predictions = []
    for i, day in enumerate(calendar[:5]):
        predictions.append({"stock_code": "600000", "entry_date": day,
                            "actual_horizon_return": 0.01 * i,
                            **{f"predicted_return__{mode}": -float(i) for mode in study.MODES}})
    result = study.event_diagnostic(predictions, qfq, calendar, "2026-08-04")
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["shock_date"] == "2026-08-08"
    assert event["oracle_top3_mean_return"] == pytest.approx(0.03)
    assert event["selected_top3_mean_return__market_count"] == pytest.approx(0.01)
    assert event["recent3_mean_return"] == pytest.approx(0.03)
