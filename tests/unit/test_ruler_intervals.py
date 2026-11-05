"""Exact hypergeometric bounds, on cases small enough to check by hand.

The full proof of coverage is ``benchmarks/ruler/exact_coverage.py`` (n up to
2,000, N up to 10^6, scipy as the reference). These tests pin the behaviour
CI can check without the benchmark: the definition at small sizes, the
edges, and the refusals.
"""

from __future__ import annotations

from math import comb

import pytest

from ondine.ruler.intervals import hypergeom_bounds


def _pmf(x: int, K: int, n: int, N: int) -> float:  # noqa: N803
    return comb(K, x) * comb(N - K, n - x) / comb(N, n)


def _by_definition(x: int, n: int, N: int, alpha: float = 0.05) -> tuple[int, int]:  # noqa: N803
    """Smallest K with P(X >= x) > alpha/2; largest K with P(X <= x) > alpha/2."""
    possible = range(x, N - (n - x) + 1)
    low = min(
        K
        for K in possible
        if sum(_pmf(y, K, n, N) for y in range(x, n + 1)) > alpha / 2
    )
    high = max(
        K for K in possible if sum(_pmf(y, K, n, N) for y in range(x + 1)) > alpha / 2
    )
    return low, high


@pytest.mark.parametrize(("n", "N"), [(4, 5), (10, 30), (20, 25), (12, 60)])
def test_every_bound_matches_the_definition_computed_with_exact_fractions(n, N):  # noqa: N803
    low, high = hypergeom_bounds(list(range(n + 1)), n, N)

    assert [(int(lo), int(hi)) for lo, hi in zip(low, high, strict=True)] == [
        _by_definition(x, n, N) for x in range(n + 1)
    ]


def test_a_census_is_exact():
    low, high = hypergeom_bounds([0, 7, 9], 9, 9)

    assert list(low) == list(high) == [0, 7, 9]


def test_all_right_or_all_wrong_reaches_the_edge_of_the_population():
    low, high = hypergeom_bounds([0, 50], 50, 1000)

    assert (low[0], high[1]) == (0, 1000)
    assert 0 < high[0] < 1000
    assert 0 < low[1] < 1000


@pytest.mark.parametrize(
    ("x", "n", "N"), [([1], 0, 10), ([1], 11, 10), ([-1], 5, 10), ([6], 5, 10)]
)
def test_impossible_inputs_are_refused(x, n, N):  # noqa: N803
    with pytest.raises(ValueError):
        hypergeom_bounds(x, n, N)
