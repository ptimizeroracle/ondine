"""Tests for RedisRateLimiter — distributed token bucket.

Uses fakeredis with Lua support (via lupa) for deterministic,
in-process unit tests. Real Redis is exercised by a separate
integration test, not here.

Every test articulates a regression: a concrete misbehaviour that
would silently leak over-rate into production traffic, or that
would break the "same interface as local RateLimiter" contract.
"""

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from ondine.utils.redis_rate_limiter import RedisRateLimiter


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def aredis_client():
    return fakeredis.FakeAsyncRedis()


@pytest.fixture
def clock():
    """Injectable monotonic-ish clock. Returns caller-controlled time
    in seconds. Tests advance by mutating the list."""
    now = [1_000_000.0]

    def _clock() -> float:
        return now[0]

    _clock.advance = lambda dt: now.__setitem__(0, now[0] + dt)  # type: ignore[attr-defined]
    return _clock


# ── single-worker sanity ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_async_succeeds_when_bucket_full(aredis_client, clock) -> None:
    """Regression: first call into a fresh bucket must succeed.
    Without this, every fresh scope deadlocks new workers."""
    rl = RedisRateLimiter(
        requests_per_minute=60,
        redis_client=aredis_client,
        scope="test-scope-1",
        monotonic=clock,
    )
    ok = await rl.acquire_async(timeout=0.5)
    assert ok is True


@pytest.mark.asyncio
async def test_acquire_async_returns_false_on_timeout_when_empty(
    aredis_client, clock
) -> None:
    """Regression: when the bucket is empty and refill won't catch
    up in the timeout window, acquire must return False rather than
    hang or raise."""
    rl = RedisRateLimiter(
        requests_per_minute=1,  # refill: 1 token per 60s
        burst_size=1,
        redis_client=aredis_client,
        scope="test-scope-2",
        monotonic=clock,
    )
    assert await rl.acquire_async(timeout=0.5) is True
    # Second attempt: bucket empty, timeout short, clock frozen.
    assert await rl.acquire_async(timeout=0.2) is False


@pytest.mark.asyncio
async def test_refill_respects_requests_per_minute(aredis_client, clock) -> None:
    """Regression: refill rate is rpm/60 tokens per second. Drift
    here silently over-issues traffic to the provider."""
    rl = RedisRateLimiter(
        requests_per_minute=60,  # 1 token/s
        burst_size=2,
        redis_client=aredis_client,
        scope="test-scope-3",
        monotonic=clock,
    )
    # Drain.
    assert await rl.acquire_async() is True
    assert await rl.acquire_async() is True
    # Frozen clock: third acquire must fail fast.
    assert await rl.acquire_async(timeout=0.1) is False
    # Advance 1.5s: one full token should be available.
    clock.advance(1.5)
    assert await rl.acquire_async(timeout=0.1) is True


# ── the whole point: cross-worker sharing ─────────────────────────────


@pytest.mark.asyncio
async def test_two_workers_share_one_bucket(aredis_client, clock) -> None:
    """Regression (the entire motivation): two distinct limiter
    instances pointed at the same Redis + scope must contend on one
    bucket, not each operate their own local one. This is the bug
    the in-process RateLimiter cannot fix."""
    worker_a = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=2,
        redis_client=aredis_client,
        scope="shared",
        monotonic=clock,
    )
    worker_b = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=2,
        redis_client=aredis_client,
        scope="shared",
        monotonic=clock,
    )
    # Two tokens total across both workers.
    assert await worker_a.acquire_async() is True
    assert await worker_b.acquire_async() is True
    # Third anywhere must fail — the bucket is empty globally.
    assert await worker_a.acquire_async(timeout=0.1) is False
    assert await worker_b.acquire_async(timeout=0.1) is False


@pytest.mark.asyncio
async def test_scopes_are_isolated(aredis_client, clock) -> None:
    """Regression: different scopes (provider/model/tier) must not
    drain each other. Cross-scope leakage would cause the cheap
    model to throttle the expensive one."""
    anthropic = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=1,
        redis_client=aredis_client,
        scope="anthropic:sonnet",
        monotonic=clock,
    )
    openai = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=1,
        redis_client=aredis_client,
        scope="openai:gpt-4",
        monotonic=clock,
    )
    assert await anthropic.acquire_async() is True
    # anthropic is empty but openai must still have its own token.
    assert await openai.acquire_async() is True


# ── capacity guard ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_requesting_more_than_capacity_raises(aredis_client, clock) -> None:
    """Regression: asking for more tokens than the bucket holds is
    a caller bug that would deadlock forever if we silently retried.
    Mirrors the local RateLimiter contract."""
    rl = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=5,
        redis_client=aredis_client,
        scope="cap",
        monotonic=clock,
    )
    with pytest.raises(ValueError):
        await rl.acquire_async(tokens=10)
