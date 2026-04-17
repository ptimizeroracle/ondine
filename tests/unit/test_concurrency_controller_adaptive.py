"""Tests for adaptive wiring on ConcurrencyController.

Covers: opt-in adaptive mode, 429 propagation to both limiter and
rate-limit token bucket, and non-regression on the default
(fixed-semaphore) path.
"""

from __future__ import annotations

import asyncio

import pytest

from ondine.orchestration.concurrency_controller import ConcurrencyController
from ondine.utils.rate_limiter import RateLimiter


def test_default_mode_is_fixed_semaphore_backward_compatible() -> None:
    """Regression: existing callers passing only max_concurrent must
    get identical behaviour to today (fixed-N asyncio.Semaphore)."""
    ctrl = ConcurrencyController(max_concurrent=5)
    assert ctrl.max_concurrent == 5
    assert ctrl.adaptive is False


def test_adaptive_mode_surfaces_limit_via_max_concurrent() -> None:
    """Regression: when adaptive=True, max_concurrent reports the
    *current* live limit — useful for snapshots and tests."""
    ctrl = ConcurrencyController(max_concurrent=5, adaptive=True)
    assert ctrl.adaptive is True
    # Initial live limit equals the configured ceiling.
    assert ctrl.max_concurrent == 5


@pytest.mark.asyncio
async def test_on_rate_limit_shrinks_adaptive_limit() -> None:
    """Regression: a 429 signal must propagate into the adaptive
    limiter so subsequent acquires see reduced concurrency."""
    ctrl = ConcurrencyController(max_concurrent=10, adaptive=True)
    starting = ctrl.max_concurrent
    ctrl.on_rate_limit(retry_after_s=1.0)
    assert ctrl.max_concurrent < starting


@pytest.mark.asyncio
async def test_on_rate_limit_penalises_rate_limiter_when_delay_given() -> None:
    """Regression: the server-issued delay must also drain the token
    bucket, otherwise a concurrent caller with local credit fires
    immediately and re-triggers the 429."""
    rl = RateLimiter(requests_per_minute=60_000, burst_size=100)
    ctrl = ConcurrencyController(max_concurrent=10, rate_limiter=rl, adaptive=True)
    ctrl.on_rate_limit(retry_after_s=0.3)
    assert rl.available_tokens == 0.0


@pytest.mark.asyncio
async def test_on_rate_limit_with_none_is_safe() -> None:
    """Regression: callers commonly have `retry_after_s is None`
    (no header); must still record the 429 for adaptation without
    touching the token bucket."""
    rl = RateLimiter(requests_per_minute=60_000, burst_size=100)
    ctrl = ConcurrencyController(max_concurrent=10, rate_limiter=rl, adaptive=True)
    before = rl.available_tokens
    ctrl.on_rate_limit(retry_after_s=None)
    # Bucket untouched when no server hint is available.
    assert rl.available_tokens == pytest.approx(before, rel=0.01)


@pytest.mark.asyncio
async def test_throttle_enforces_adaptive_limit() -> None:
    """Regression: throttle() must actually gate on the adaptive
    cap, not on the configured ceiling."""
    ctrl = ConcurrencyController(max_concurrent=8, adaptive=True)
    # Shrink aggressively.
    for _ in range(20):
        ctrl.on_rate_limit(retry_after_s=0.0)
    shrunk = ctrl.max_concurrent
    assert shrunk < 8

    counter = 0
    peak = 0

    async def worker() -> None:
        nonlocal counter, peak
        async with ctrl.throttle():
            counter += 1
            peak = max(peak, counter)
            await asyncio.sleep(0.01)
            counter -= 1

    await asyncio.gather(*(worker() for _ in range(20)))
    assert peak <= shrunk
