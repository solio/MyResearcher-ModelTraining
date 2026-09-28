from tools import market_cooling_shadow as shadow


def test_cooling_uses_only_observed_history_and_enters_next_session():
    calendar = [f"2026-01-{i + 1:02d}" for i in range(30)]
    states, raw = [], {}
    for i, day in enumerate(calendar):
        stance = -.8 if 15 <= i < 20 else 0.0 if 20 <= i < 25 else .2
        posts = 10 if 15 <= i < 20 else 3 if 20 <= i < 25 else 6
        volume = 100 if 15 <= i < 20 else 50 if 20 <= i < 25 else 70
        states.append({"stock_code": "000001", "stock_name": "Example", "entry_date": day,
                       "observed_posts": posts, "features": {"stance_index": stance}})
        raw[("000001", day)] = {"volume": volume}
    cases = shadow.find_candidates(states, calendar, raw)
    first = next(row for row in cases if row["signal_date"] == calendar[24])
    assert first["entry_date"] == calendar[25]
    assert first["cooled"] is True
    assert first["posts_previous_5"] == 50
    assert first["posts_recent_5"] == 15
    assert first["volume_previous_5"] == 500
    assert first["volume_recent_5"] == 250
    # A future day's volume cannot classify the already-known signal.
    raw[("000001", calendar[25])]["volume"] = 999999
    assert next(row for row in shadow.find_candidates(states, calendar, raw) if row["signal_date"] == calendar[24]) == first
    # The three-part cooling definition requires volume to decline too.
    for day in calendar[20:25]:
        raw[("000001", day)]["volume"] = 200
    changed = next(row for row in shadow.find_candidates(states, calendar, raw) if row["signal_date"] == calendar[24])
    assert changed["cooled"] is False


def test_unmatured_horizon_is_missing_not_zero():
    calendar = [f"2026-01-{i + 1:02d}" for i in range(30)]
    qfq = {("000001", day): {"open": 10, "close": 10 + i / 10,
                               "high": 11 + i / 10, "low": 9.5, "fetched_at": "same"}
           for i, day in enumerate(calendar)}
    raw = {key: dict(value) for key, value in qfq.items()}
    case = {"stock_code": "000001", "stock_name": "Example", "signal_date": calendar[24],
            "entry_date": calendar[25], "cooled": True}
    stocks = shadow.summarize([case], calendar, qfq, raw)
    assert stocks[0]["cooled"]["signal_dates"] == 1
    assert stocks[0]["cooled"]["complete"] == 0
    assert stocks[0]["cooled"]["complete_episodes"] == 0
    assert stocks[0]["cooled"]["mean_return"] is None
    assert case["outcomes"]["10"]["status"] == "PENDING_HORIZON"
    assert case["outcomes"]["20"]["status"] == "PENDING_HORIZON"
