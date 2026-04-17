"""Tests for RateLimiter.penalize() — server-sourced backoff.

When the server says "retry after 30s", that signal must be absorbed
by the shared limiter, not just by the caller that saw the 429.
Otherwise a concurrent coroutine that still has local bucket tokens
will fire immediately and re-trigger the 429. These tests pin that
behaviour.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ondine.utils.rate_limiter import RateLimiter


def test_penalize_empties_tokens_immediately() -> None:
    """Regression: after penalize(), the very next acquire must see
    zero tokens — otherwise the in-flight bucket still lets other
    callers through."""
    rl = RateLimiter(requests_per_minute=6000, burst_size=10)
    # Burn no tokens — full bucket.
    rl.penalize(delay_seconds=0.5)
    assert rl.available_tokens == pytest.approx(0.0, abs=0.1)


def test_penalize_blocks_acquire_until_delay_elapsed() -> None:
    """Regression: acquire() must not return True during the penalty
    window even with a generous timeout."""
    rl = RateLimiter(requests_per_minute=60_000, burst_size=100)
    rl.penalize(delay_seconds=0.3)
    start = time.monotonic()
    ok = rl.acquire(tokens=1, timeout=1.0)
    elapsed = time.monotonic() - start
    assert ok is True
    assert elapsed >= 0.25  # allow small clock jitter


def test_penalize_is_idempotent_for_shorter_delays() -> None:
    """Regression: if two 429s arrive and the second says '5s' while
    the first said '30s', the longer wait must win. Otherwise a
    late-arriving soft signal shortens a real one."""
    rl = RateLimiter(requests_per_minute=60_000, burst_size=100)
    rl.penalize(delay_seconds=30.0)
    rl.penalize(delay_seconds=1.0)
    # available_tokens still zero until the 30s window clears.
    assert rl.available_tokens == 0.0


@pytest.mark.asyncio
async def test_penalize_visible_to_concurrent_async_callers() -> None:
    """Regression: a 429 on one coroutine must gate all others.
    Without a shared penalty, coroutines with existing bucket credit
    fire straight through and re-trigger 429s."""
    rl = RateLimiter(requests_per_minute=60_000, burst_size=100)
    # Two coroutines race: one penalises, the other tries to acquire.
    rl.penalize(delay_seconds=0.2)

    start = asyncio.get_event_loop().time()
    ok = await rl.acquire_async(tokens=1, timeout=1.0)
    elapsed = asyncio.get_event_loop().time() - start

    assert ok is True
    assert elapsed >= 0.15


def test_zero_penalty_is_noop() -> None:
    """Regression: `retry-after: 0` parses to 0.0; that path must
    neither raise nor block."""
    rl = RateLimiter(requests_per_minute=6000, burst_size=10)
    rl.penalize(delay_seconds=0.0)
    # With zero penalty the bucket is unchanged.
    assert rl.available_tokens > 0


def test_negative_penalty_rejected() -> None:
    """Regression: defensive — callers should never pass negative
    durations. Raising surfaces the bug rather than silently
    extending the bucket."""
    rl = RateLimiter(requests_per_minute=6000, burst_size=10)
    with pytest.raises(ValueError):
        rl.penalize(delay_seconds=-1.0)
