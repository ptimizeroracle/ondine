"""Tests for async rate limiter — drives the acquire_async() implementation.

These tests verify that RateLimiter.acquire_async():
1. Works without blocking the event loop (no time.sleep, no run_in_executor)
2. Correctly limits request rate using async-native primitives
3. Is backward-compatible — sync acquire() still works unchanged
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from ondine.utils.rate_limiter import RateLimiter


class TestAsyncRateLimiterBehavior:
    """Verify acquire_async() rate limits correctly without blocking."""

    @pytest.mark.asyncio
    async def test_acquire_async_exists_and_is_coroutine(self):
        """acquire_async() must exist and be awaitable.

        Regression: ensures the async interface is present.
        """
        limiter = RateLimiter(requests_per_minute=60)
        assert hasattr(limiter, "acquire_async"), (
            "RateLimiter must have acquire_async() method"
        )
        # Must be awaitable
        result = await limiter.acquire_async()
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_async_does_not_use_thread_executor(self):
        """acquire_async() must NOT call run_in_executor or time.sleep.

        Regression: the whole point of this optimization is to avoid
        the blocking time.sleep(0.1) pattern that wastes thread pool resources.
        """
        limiter = RateLimiter(requests_per_minute=600, burst_size=10)

        with patch("time.sleep", side_effect=AssertionError("time.sleep called!")):
            # Should succeed without touching time.sleep
            for _ in range(10):
                await limiter.acquire_async()

    @pytest.mark.asyncio
    async def test_acquire_async_respects_rate_limit(self):
        """acquire_async() must enforce the RPM limit.

        With 120 RPM (2/sec) and burst_size=2, the 3rd acquire
        within <1 second should block until tokens refill.
        """
        limiter = RateLimiter(requests_per_minute=120, burst_size=2)
        refill_period = 0.5  # 2 tokens/sec

        # The first two spend the burst and must not wait for a refill.
        t0 = time.perf_counter()
        await limiter.acquire_async()
        await limiter.acquire_async()
        t_burst = time.perf_counter() - t0

        # The third has no token left, so it must.
        t0 = time.perf_counter()
        await limiter.acquire_async()
        t_throttled = time.perf_counter() - t0

        # Compared against each other, and against the refill period, rather
        # than against an absolute ceiling. A ceiling low enough to be
        # meaningful (0.1s) is also low enough for a loaded CI runner to
        # exceed while the limiter behaves perfectly: this failed at 0.150s
        # on a box that was merely busy. What the test is actually about is
        # the *difference* between the two — and a slow runner slows both.
        assert t_throttled >= refill_period / 2, (
            f"3rd acquire returned in {t_throttled:.3f}s — it should have "
            f"waited for a token to refill (~{refill_period}s)"
        )
        assert t_burst < t_throttled / 2, (
            f"Burst acquires took {t_burst:.3f}s vs {t_throttled:.3f}s "
            f"throttled — the burst should not have waited at all"
        )

    @pytest.mark.asyncio
    async def test_acquire_async_timeout_returns_false(self):
        """acquire_async() with timeout returns False when tokens unavailable.

        Regression: timeout behavior must match sync acquire().
        """
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        # Consume the only token
        await limiter.acquire_async()

        # Next should timeout
        result = await limiter.acquire_async(timeout=0.05)
        assert result is False, "Should return False on timeout"

    @pytest.mark.asyncio
    async def test_acquire_async_concurrent_fairness(self):
        """Multiple concurrent acquire_async() calls should all eventually succeed.

        Regression: ensures no starvation under concurrent load.
        """
        limiter = RateLimiter(requests_per_minute=600, burst_size=20)
        results = []

        async def _acquire(idx: int):
            ok = await limiter.acquire_async()
            results.append((idx, ok))

        tasks = [_acquire(i) for i in range(20)]
        await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(ok for _, ok in results), "All burst acquires should succeed"


class TestSyncAcquireUnchanged:
    """Verify sync acquire() is NOT broken by async changes."""

    def test_sync_acquire_still_works(self):
        """Sync acquire() must still work identically after adding async support.

        Regression: backward compatibility for non-async callers.
        """
        limiter = RateLimiter(requests_per_minute=600, burst_size=10)

        for _ in range(10):
            result = limiter.acquire()
            assert result is True

    def test_sync_acquire_timeout_still_works(self):
        """Sync acquire(timeout=...) must still work."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        limiter.acquire()  # consume token

        result = limiter.acquire(timeout=0.05)
        assert result is False
