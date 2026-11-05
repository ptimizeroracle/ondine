"""Exact confidence intervals for certified columns ("the Ruler").

Validated by ``benchmarks/ruler`` (exact coverage over n 2-2,000 and N to
10^6, and simulation on populations built from real rows) before they
shipped. They use numpy only, because Ondine does not depend on scipy.

The interval is equal-tailed and exact by test inversion, so it covers the
truth at least ``1 - alpha`` of the time.

:func:`hypergeom_bounds` takes a sample of ``n`` rows drawn without
replacement from ``N`` rows, ``x`` of them right, and returns bounds on the
count ``K`` of right rows among the ``N``. Unresolved rows widen them by the
union bound (``ondine.ruler.ruler``).
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
    normalised: np.ndarray = p / p.sum(axis=1, keepdims=True)
    return normalised


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
