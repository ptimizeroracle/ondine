"""Tests for AdaptiveLimiter.

The limiter wraps a dynamic in-flight cap around coroutines. Tests
use an injected clock/RTT source so no real sleeping or wallclock
assumptions leak into the suite.

Every test articulates a regression: a real behaviour change in the
algorithm that a bug would silently break.
"""

from __future__ import annotations

import asyncio

import pytest

from ondine.utils.adaptive_limiter import AdaptiveLimiter

# ── basic mechanics ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_release_respects_initial_limit() -> None:
    """Regression: initial_limit must be the hard admission cap at
    time zero. Off-by-one here makes the whole adaptation meaningless."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=10, initial_limit=3)
    acquired: list[int] = []

    async def worker(i: int) -> None:
        async with lim.slot(rtt_source=lambda: 0.05):
            acquired.append(i)
            await asyncio.sleep(0.01)

    # Four coroutines, cap of 3 — fourth must wait.
    await asyncio.gather(*(worker(i) for i in range(4)))
    assert lim.peak_in_flight <= 3
    assert set(acquired) == {0, 1, 2, 3}


@pytest.mark.asyncio
async def test_slot_releases_even_on_exception() -> None:
    """Regression: without finally-release, one raising worker
    permanently pins a slot and the limiter leaks capacity."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=10, initial_limit=2)

    async def boom() -> None:
        async with lim.slot(rtt_source=lambda: 0.05):
            raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await boom()

    # After three raising workers the limiter must still grant slots.
    async with lim.slot(rtt_source=lambda: 0.05):
        assert lim.in_flight == 1


# ── Gradient2 adaptation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_limit_shrinks_on_explicit_rate_limit() -> None:
    """Regression: on_rate_limit() must shrink the current limit —
    otherwise a 429 produces no corrective action and we keep
    hammering the provider."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=20, initial_limit=10)
    lim.on_rate_limit(retry_after_s=2.0)
    # Gradient2 shrink factor is 0.9 on explicit 429.
    assert lim.current_limit < 10
    assert lim.current_limit >= 1


@pytest.mark.asyncio
async def test_limit_never_drops_below_min() -> None:
    """Regression: repeated 429s must not collapse the limit to zero
    (would deadlock the pipeline). Uses an injected clock that
    advances past the cooldown window on every call so each 429
    contributes an effective shrink — the point of this test is the
    floor, not the debounce."""
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 1.0  # well past _RATE_LIMIT_COOLDOWN_S
        return ticks[0]

    lim = AdaptiveLimiter(min_limit=2, max_limit=20, initial_limit=10, monotonic=clock)
    for _ in range(50):
        lim.on_rate_limit(retry_after_s=1.0)
    assert lim.current_limit == 2


@pytest.mark.asyncio
async def test_close_bursts_of_429s_debounce_to_one_shrink() -> None:
    """Regression (CodeRabbit #141): N workers retrying M times each
    must not collapse the limit by 0.9^(N*M) for one logical upstream
    overload. A burst of 429s arriving within the cooldown window
    compounds to one shrink."""
    # Fixed clock — all calls appear simultaneous.
    lim = AdaptiveLimiter(
        min_limit=1, max_limit=20, initial_limit=10, monotonic=lambda: 1.0
    )
    for _ in range(20):
        lim.on_rate_limit(retry_after_s=0.0)
    # One effective shrink: 10 * 0.9 = 9.0
    assert lim.current_limit == 9
    # But the counter still reflects every hit for observability.
    assert lim.snapshot()["rate_limit_hits"] == 20


@pytest.mark.asyncio
async def test_limit_never_exceeds_max() -> None:
    """Regression: sustained low-latency success must not grow the
    limit unbounded — max_limit is the user's configured ceiling."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=5, initial_limit=3)
    # Feed low RTT observations and force growth ticks.
    for _ in range(100):
        lim.observe(rtt_s=0.01, in_flight_at_sample=lim.current_limit)
    assert lim.current_limit <= 5


@pytest.mark.asyncio
async def test_limit_grows_when_queue_full_and_rtt_stable() -> None:
    """Regression: Gradient2's core promise — when the limiter is
    saturated AND RTT is near the no-load baseline, capacity should
    increase. Without this the limit only ever shrinks."""
    lim = AdaptiveLimiter(
        min_limit=1,
        max_limit=20,
        initial_limit=4,
    )
    # Establish a baseline RTT with some observations.
    for _ in range(5):
        lim.observe(rtt_s=0.05, in_flight_at_sample=4)
    start_limit = lim.current_limit
    # Many more observations at near-baseline RTT, with the queue
    # saturated (in_flight == current_limit).
    for _ in range(30):
        lim.observe(rtt_s=0.05, in_flight_at_sample=lim.current_limit)
    assert lim.current_limit > start_limit


@pytest.mark.asyncio
async def test_limit_shrinks_on_rtt_inflation() -> None:
    """Regression: latency rising above the no-load baseline is the
    earliest signal of upstream overload. Gradient2 must shrink
    before an explicit 429 arrives."""
    lim = AdaptiveLimiter(
        min_limit=1,
        max_limit=20,
        initial_limit=10,
    )
    # Establish a low-latency baseline.
    for _ in range(5):
        lim.observe(rtt_s=0.05, in_flight_at_sample=10)
    baseline = lim.current_limit
    # Now inflated RTT (5x baseline) — limiter should react.
    for _ in range(10):
        lim.observe(rtt_s=0.25, in_flight_at_sample=10)
    assert lim.current_limit < baseline


# ── fairness / shrink-while-queued ────────────────────────────────────


@pytest.mark.asyncio
async def test_shrink_does_not_cancel_in_flight_tasks() -> None:
    """Regression: if shrink below in_flight cancelled running tasks,
    a brief 429 would kill perfectly-valid work."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=10, initial_limit=5)

    async def slow_worker() -> str:
        async with lim.slot(rtt_source=lambda: 0.05):
            await asyncio.sleep(0.05)
            return "ok"

    t1 = asyncio.create_task(slow_worker())
    t2 = asyncio.create_task(slow_worker())
    await asyncio.sleep(0.01)  # let them enter the slot
    # Shrink aggressively while both are in-flight.
    for _ in range(20):
        lim.on_rate_limit(retry_after_s=1.0)
    results = await asyncio.gather(t1, t2)
    assert results == ["ok", "ok"]


# ── snapshot / observability ───────────────────────────────────────────


def test_snapshot_exposes_load_bearing_fields() -> None:
    """Regression: observability contract. Removing any of these
    fields blinds production debugging of 429 storms."""
    lim = AdaptiveLimiter(min_limit=1, max_limit=20, initial_limit=10)
    snap = lim.snapshot()
    for field in (
        "current_limit",
        "in_flight",
        "queue_depth",
        "rtt_noload_ms",
        "rtt_smoothed_ms",
        "rate_limit_hits",
    ):
        assert field in snap, f"missing observability field: {field}"
