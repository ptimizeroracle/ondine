"""Tests: penalize() propagates across workers.

The whole reason to add Redis: when worker A sees a 429 with
Retry-After=30, worker B must also be held back for 30s. Local
penalize() cannot do this — only a shared Redis key can.
"""

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from ondine.utils.redis_rate_limiter import RedisRateLimiter


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def clock():
    now = [1_000_000.0]

    def _clock() -> float:
        return now[0]

    _clock.advance = lambda dt: now.__setitem__(0, now[0] + dt)  # type: ignore[attr-defined]
    return _clock


def test_penalize_blocks_same_worker(redis_client, clock) -> None:
    """Regression: the worker that issued the penalty must itself
    respect it — baseline property before cross-worker."""
    rl = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        sync_redis_client=redis_client,
        scope="pen-1",
        monotonic=clock,
    )
    rl.penalize(delay_seconds=0.3)
    # Even with a large burst, acquire during the penalty window
    # must fail within a short sync-timeout budget.
    assert rl.acquire(timeout=0.1) is False


def test_penalize_is_visible_to_another_worker(redis_client, clock) -> None:
    """Regression (the A3 motivation): worker B must see a penalty
    worker A issued. This is what the in-process RateLimiter cannot
    provide and is the core value of A3."""
    worker_a = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        sync_redis_client=redis_client,
        scope="pen-shared",
        monotonic=clock,
    )
    worker_b = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        sync_redis_client=redis_client,
        scope="pen-shared",
        monotonic=clock,
    )
    worker_a.penalize(delay_seconds=0.4)
    # Worker B has never seen a 429 itself — but must still be gated.
    assert worker_b.acquire(timeout=0.1) is False


def test_longer_penalty_wins_over_shorter_late_signal(redis_client, clock) -> None:
    """Regression: a second penalty must not shorten an in-flight
    longer one. Mirrors the local RateLimiter's contract."""
    rl = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        sync_redis_client=redis_client,
        scope="pen-merge",
        monotonic=clock,
    )
    rl.penalize(delay_seconds=30.0)
    rl.penalize(delay_seconds=0.05)
    # The 30s window still holds — acquire must not slip through.
    assert rl.acquire(timeout=0.1) is False


def test_zero_penalty_is_noop(redis_client, clock) -> None:
    """Regression: Retry-After=0 parses to 0.0; the path must not
    raise or lock the bucket."""
    rl = RedisRateLimiter(
        requests_per_minute=60,
        sync_redis_client=redis_client,
        scope="pen-zero",
        monotonic=clock,
    )
    rl.penalize(delay_seconds=0.0)
    assert rl.acquire(timeout=0.2) is True


def test_negative_penalty_rejected(redis_client, clock) -> None:
    """Regression: callers must never pass negatives; surfacing
    the bug beats silently extending the bucket."""
    rl = RedisRateLimiter(
        requests_per_minute=60,
        sync_redis_client=redis_client,
        scope="pen-neg",
        monotonic=clock,
    )
    with pytest.raises(ValueError):
        rl.penalize(delay_seconds=-1.0)
