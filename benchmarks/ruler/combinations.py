"""Intervals the Ruler validated but does not ship.

Clopper–Pearson, and the union-bound combinations for status strata and
two-tier labelling. Phase 1 (``simulate.py``) showed both combinations valid
but wider than a plain sample, so they left v1; they stay here so the
benchmark can keep showing why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ondine.ruler.intervals import _tails

if TYPE_CHECKING:
    from collections.abc import Sequence


def _binom_pmf(n: int, rate: np.ndarray) -> np.ndarray:
    """P(X = x | rate) for x = 0..n, one row per entry of ``rate`` (0 < rate < 1)."""
    rate = np.asarray(rate, dtype=float)[:, None]
    step = np.arange(n, dtype=float)[None, :]
    ratio = np.log(n - step) - np.log(step + 1) + np.log(rate) - np.log1p(-rate)
    logp = np.concatenate(
        [np.zeros((rate.shape[0], 1)), np.cumsum(ratio, axis=1)], axis=1
    )
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def binom_bounds(
    x: Sequence[int] | np.ndarray, n: int, alpha: float = 0.05, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Clopper–Pearson bounds on a rate, for one or many ``x`` out of ``n``.

    Used where the population size is unknown. Returns ``(low, high)``
    rates; ``low`` is 0 when x == 0 and ``high`` is 1 when x == n.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.int64))
    if n <= 0 or np.any((x < 0) | (x > n)):
        raise ValueError("need n > 0 and 0 <= x <= n")
    low = np.zeros(x.size)
    high = np.ones(x.size)
    for upper, out, edge in ((False, low, 0), (True, high, n)):
        todo = x != edge
        if not todo.any():
            continue
        xs = x[todo]
        a = np.zeros(xs.size)
        b = np.ones(xs.size)
        while np.max(b - a) > tol:
            mid = (a + b) / 2
            at_least, at_most = _tails(_binom_pmf(n, mid), xs)
            # Lower: the tail P(X >= x) grows with the rate; find where it
            # crosses alpha/2. Upper: P(X <= x) shrinks; find its crossing.
            above = (at_most > alpha / 2) if upper else (at_least > alpha / 2)
            if upper:
                a, b = np.where(above, mid, a), np.where(above, b, mid)
            else:
                a, b = np.where(above, a, mid), np.where(above, mid, b)
        # Keep the outer end of the bracket, so rounding only widens.
        out[todo] = b if upper else a
    return low, high


def stratum_alphas(
    weights: Sequence[float], census: Sequence[bool], alpha: float, split: str
) -> np.ndarray:
    """Share ``alpha`` among the sampled strata; census strata get none.

    ``split`` is "equal" or "weight" (in proportion to stratum weight). Any
    split keeps the combined bounds valid, by the union bound.
    """
    weights = np.asarray(weights, dtype=float)
    sampled = ~np.asarray(census, dtype=bool)
    shares = np.zeros(weights.size)
    if not sampled.any():
        return shares
    if split == "equal":
        shares[sampled] = 1.0
    elif split == "weight":
        shares[sampled] = weights[sampled]
    else:
        raise ValueError(f"unknown split {split!r}")
    return alpha * shares / shares.sum()


def allocate(sizes: Sequence[int], n: int, minimum: int = 20) -> np.ndarray:
    """Split a sample of ``n`` rows among strata of the given sizes.

    Proportional to size; every non-empty stratum gets at least
    ``min(minimum, size)``; the largest stratum gives up the difference.
    A stratum no larger than its share is taken whole (a census).
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if n >= sizes.sum():
        return sizes.copy()
    share = np.floor(n * sizes / sizes.sum()).astype(np.int64)
    share = np.maximum(share, np.minimum(minimum, sizes))
    share = np.minimum(share, sizes)
    largest = int(np.argmax(sizes))
    share[largest] += n - share.sum()
    if share[largest] < 1 or share[largest] > sizes[largest]:
        raise ValueError("sample too small for the stratum minimums")
    return share


def two_tier_bounds(
    disagree: tuple[np.ndarray, np.ndarray],
    agree_acc: tuple[np.ndarray, np.ndarray],
    disagree_acc: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy bounds from the worst corners of three intervals.

    Accuracy = (1 - share) * agree_acc + share * disagree_acc, where
    ``share`` is the population's disagreement share. Each argument is a
    ``(low, high)`` pair of arrays, each interval at level 1 - alpha/3.
    """
    (s_lo, s_hi), (a_lo, a_hi), (d_lo, d_hi) = disagree, agree_acc, disagree_acc
    low = np.minimum((1 - s_lo) * a_lo + s_lo * d_lo, (1 - s_hi) * a_lo + s_hi * d_lo)
    high = np.maximum((1 - s_lo) * a_hi + s_lo * d_hi, (1 - s_hi) * a_hi + s_hi * d_hi)
    return low, high
