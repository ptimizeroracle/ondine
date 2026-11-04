"""Exact confidence intervals for certified columns ("the Ruler").

Phase 1 validates these functions (``exact_coverage.py``, ``simulate.py``);
phase 2 moves them into Ondine unchanged. They use numpy only, because
Ondine does not depend on scipy.

All intervals are equal-tailed and exact by test inversion, so each covers
the truth at least ``1 - alpha`` of the time:

* :func:`hypergeom_bounds` — a sample of ``n`` rows drawn without
  replacement from ``N`` rows, ``x`` of them right. Returns bounds on the
  count ``K`` of right rows among the ``N``.
* :func:`binom_bounds` — Clopper–Pearson, for when the population size is
  unknown. Returns bounds on the rate.

The combinations (strata, unresolved rows, two-tier) are union bounds over
these; see ``.claude/designs/certified-columns.md`` section 8.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Rows of the pmf matrix built at once; bounds n+1 by n+1 float arrays.
_CHUNK = 256


def _hypergeom_pmf(N: int, K: np.ndarray, n: int) -> np.ndarray:
    """P(X = x | K) for x = 0..n, one row per entry of ``K``.

    Built from the ratio pmf(x+1)/pmf(x) in log space and normalised, so no
    factorial is ever formed; exact to float precision for N up to 10^9.
    """
    K = np.asarray(K, dtype=np.int64)[:, None]
    x = np.arange(n + 1, dtype=np.int64)[None, :]
    first = np.maximum(0, n - (N - K))
    last = np.minimum(n, K)
    step = x[:, :-1]
    inside = (step >= first) & (step < last)
    with np.errstate(divide="ignore"):
        ratio = (
            np.log(np.where(inside, K - step, 1))
            + np.log(np.where(inside, n - step, 1))
            - np.log(step + 1)
            - np.log(np.where(inside, N - K - n + step + 1, 1))
        )
    ratio = np.where(inside, ratio, 0.0)
    logp = np.concatenate([np.zeros((K.shape[0], 1)), np.cumsum(ratio, axis=1)], axis=1)
    logp = np.where((x >= first) & (x <= last), logp, -np.inf)
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


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


def _tails(pmf: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(X >= x) and P(X <= x) for each row's own ``x``."""
    rows = np.arange(pmf.shape[0])
    at_least = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1][rows, x]
    at_most = np.cumsum(pmf, axis=1)[rows, x]
    return at_least, at_most


def hypergeom_bounds(
    x: Sequence[int] | np.ndarray, n: int, N: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounds on the count of right rows among ``N``.

    Args:
        x: Right rows in the sample, one or many (0 <= x <= n).
        n: Sample size, drawn without replacement.
        N: Population size (n <= N).
        alpha: 1 minus the confidence level, split equally between the tails.

    Returns:
        ``(low, high)`` integer counts per ``x``: the smallest K with
        P(X >= x | K) > alpha/2 and the largest K with P(X <= x | K) > alpha/2.
        A census (n == N) returns x for both.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.int64))
    if not 0 < n <= N:
        raise ValueError(f"need 0 < n <= N, got n={n}, N={N}")
    if np.any((x < 0) | (x > n)):
        raise ValueError("x must be between 0 and n")
    if n == N:
        return x.copy(), x.copy()
    low = np.empty_like(x)
    high = np.empty_like(x)
    for start in range(0, x.size, _CHUNK):
        part = x[start : start + _CHUNK]
        low[start : start + _CHUNK] = _search(part, n, N, alpha, upper=False)
        high[start : start + _CHUNK] = _search(part, n, N, alpha, upper=True)
    return low, high


def _search(x: np.ndarray, n: int, N: int, alpha: float, *, upper: bool) -> np.ndarray:
    # K ranges over [x, N - (n - x)], the counts for which x is possible.
    # Lower bound: smallest K whose upper tail exceeds alpha/2 (tail grows with K).
    # Upper bound: largest K whose lower tail exceeds alpha/2 (tail shrinks with K).
    a = x.copy()
    b = N - (n - x)
    while np.any(a < b):
        mid = (a + b + 1) // 2 if upper else (a + b) // 2
        at_least, at_most = _tails(_hypergeom_pmf(N, mid, n), x)
        if upper:
            ok = at_most > alpha / 2
            a, b = np.where(ok, mid, a), np.where(ok, b, mid - 1)
        else:
            ok = at_least > alpha / 2
            a, b = np.where(ok, a, mid + 1), np.where(ok, mid, b)
    return a


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
