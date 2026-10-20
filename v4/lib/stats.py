"""Small, dependency-free statistics helpers used by the v4 report."""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def tstat(values: Sequence[float]) -> float | None:
    if len(values) < 3:
        return None
    m = sum(values) / len(values)
    sd = stdev(values)
    if not sd:
        return None
    return m / (sd / math.sqrt(len(values)))


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def percentile(values: Sequence[float], value: float) -> float | None:
    if not values:
        return None
    return sum(1 for x in values if x <= value) / len(values)


def cluster_by_gap(items: Sequence[dict], gap: int) -> list[list[dict]]:
    """Group dated items whose calendar positions are <= gap apart."""
    clusters: list[list[dict]] = []
    for item in items:
        if clusters and item["_index"] - clusters[-1][-1]["_index"] <= gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def bootstrap_mean_by_stock(
    items_by_stock: dict[str, list[float]],
    reps: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    """Bootstrap the mean by resampling stocks (not individual overlapping days)."""
    stocks = sorted(items_by_stock)
    if len(stocks) < 2:
        return None, None, None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(reps):
        sample = [rng.choice(stocks) for _ in stocks]
        pooled = [v for s in sample for v in items_by_stock[s]]
        if pooled:
            means.append(sum(pooled) / len(pooled))
    if not means:
        return None, None, None
    return mean(means), quantile(means, 0.025), quantile(means, 0.975)


def bootstrap_cluster_mean(
    clusters: Sequence[Sequence[float]],
    reps: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    """Resample whole event clusters; each cluster contributes one mean."""
    cluster_means = [sum(c) / len(c) for c in clusters if c]
    if len(cluster_means) < 2:
        return None, None, None
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(reps):
        sample = [rng.choice(cluster_means) for _ in cluster_means]
        out.append(sum(sample) / len(sample))
    return mean(out), quantile(out, 0.025), quantile(out, 0.975)


def cross_stock_mean_and_t(values_by_stock: dict[str, float]) -> tuple[float | None, float | None, int]:
    vals = [v for v in values_by_stock.values() if v is not None]
    return (mean(vals), tstat(vals), len(vals))
