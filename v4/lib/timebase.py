"""Strict, executable timebase for the v4 report.

Rules
-----
* A post is known at the close of the first trading day whose 15:00 cutoff is
  at or after its publication time.  Example: a post at 2026-10-16 14:59
  belongs to 2026-10-16; a post at 15:01 belongs to 2026-10-19.
* The signal is formed at that trading day's close.  Entry is the *next*
  trading day's open.  A 5-day holding period exits at the close of the fifth
  trading session after entry (signal-day index + 6).
* Nothing in this module reads a future post or a future bar.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime
from typing import Iterable, Mapping, Sequence


CUTOFF = "15:00:00"


def parse_timestamp(value: str) -> tuple[str, str]:
    """Return (calendar date, HH:MM:SS) from an ISO-like timestamp."""
    date = value[:10]
    time = value[11:19] if len(value) >= 19 else "00:00:00"
    return date, time


def next_trading_day(calendar: Sequence[str], day: str) -> str | None:
    """First trading day strictly after ``day``."""
    pos = bisect_right(calendar, day)
    return calendar[pos] if pos < len(calendar) else None


def trading_days_after(calendar: Sequence[str], day: str, n: int) -> str | None:
    pos = bisect_right(calendar, day) + n - 1
    return calendar[pos] if 0 <= pos < len(calendar) else None


def post_info_day(published_at: str, calendar: Sequence[str], cutoff: str = CUTOFF) -> str | None:
    """The trading day on which a post first becomes usable.

    Posts on a trading day at/before the cutoff are known at that day's close.
    Everything else is known at the next trading day's close.
    """
    if not published_at:
        return None
    day, time = parse_timestamp(published_at)
    pos = bisect_left(calendar, day)
    is_trading = pos < len(calendar) and calendar[pos] == day
    if is_trading and time <= cutoff:
        return day
    if is_trading:
        return calendar[pos + 1] if pos + 1 < len(calendar) else None
    return calendar[pos] if pos < len(calendar) else None


def index_of(calendar: Sequence[str], day: str) -> int | None:
    pos = bisect_left(calendar, day)
    if pos < len(calendar) and calendar[pos] == day:
        return pos
    return None


def prior_drawdown(closes: Mapping[str, float], calendar: Sequence[str], day: str, window: int = 20) -> float | None:
    """close(day) / max(close over the last window+1 sessions) - 1."""
    pos = index_of(calendar, day)
    if pos is None:
        return None
    start = max(0, pos - window)
    values = [closes.get(calendar[i]) for i in range(start, pos + 1)]
    values = [v for v in values if v is not None and v > 0]
    if not values or closes.get(day) is None:
        return None
    high = max(values)
    return closes[day] / high - 1 if high else None


def trailing_return(closes: Mapping[str, float], calendar: Sequence[str], day: str, lookback: int = 5) -> float | None:
    pos = index_of(calendar, day)
    if pos is None or pos - lookback < 0:
        return None
    start = calendar[pos - lookback]
    if closes.get(day) is None or not closes.get(start):
        return None
    return closes[day] / closes[start] - 1


def forward_return(
    bars: Mapping[str, Mapping[str, float]],
    calendar: Sequence[str],
    signal_day: str,
    horizon: int,
) -> tuple[str | None, str | None, float | None, str]:
    """Entry at next open, exit at close after ``horizon`` sessions.

    Returns (entry_day, exit_day, return, status).
    status ∈ COMPLETE / PENDING_ENTRY / PENDING_EXIT / MISSING_PRICE.
    """
    pos = index_of(calendar, signal_day)
    if pos is None or pos + 1 >= len(calendar):
        return None, None, None, "PENDING_ENTRY"
    entry_day = calendar[pos + 1]
    exit_pos = pos + 1 + horizon
    if exit_pos >= len(calendar):
        return entry_day, None, None, "PENDING_EXIT"
    exit_day = calendar[exit_pos]
    entry = bars.get(entry_day, {}).get("open")
    exit_price = bars.get(exit_day, {}).get("close")
    if entry is None or not exit_price:
        return entry_day, exit_day, None, "MISSING_PRICE"
    return entry_day, exit_day, exit_price / entry - 1, "COMPLETE"


def same_close_return(
    bars: Mapping[str, Mapping[str, float]],
    calendar: Sequence[str],
    signal_day: str,
    horizon: int,
) -> float | None:
    """K3-style close-to-close return, kept only for the reference comparison."""
    pos = index_of(calendar, signal_day)
    if pos is None or pos + horizon >= len(calendar):
        return None
    exit_day = calendar[pos + horizon]
    start = bars.get(signal_day, {}).get("close")
    end = bars.get(exit_day, {}).get("close")
    if not start or not end:
        return None
    return end / start - 1


def regime_series(
    benchmark: Mapping[str, Mapping[str, float]],
    calendar: Sequence[str],
    window: int = 200,
) -> dict[str, dict]:
    """Simple causal market regime: close vs 200d MA and 20d return."""
    out: dict[str, dict] = {}
    values = [benchmark.get(d, {}).get("close") for d in calendar]
    for i, day in enumerate(calendar):
        close = values[i]
        if close is None:
            continue
        hist = [v for v in values[max(0, i - window + 1) : i + 1] if v is not None]
        ma200 = sum(hist) / len(hist) if len(hist) >= window else None
        ret20 = values[i - 20] and (close / values[i - 20] - 1) if i >= 20 else None
        hi60 = [v for v in values[max(0, i - 59) : i + 1] if v is not None]
        dd60 = close / max(hi60) - 1 if hi60 else None
        if ma200 is None or ret20 is None:
            label = "unknown"
        elif close >= ma200 and ret20 >= 0:
            label = "risk_on"
        elif close < ma200 and ret20 < 0:
            label = "risk_off"
        else:
            label = "mixed"
        out[day] = {
            "d": day,
            "close": round(close, 4),
            "ma200": round(ma200, 4) if ma200 is not None else None,
            "ret20": ret20,
            "dd60": dd60,
            "label": label,
        }
    return out
