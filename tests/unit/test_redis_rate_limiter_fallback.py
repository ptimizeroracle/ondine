"""Tests: circuit breaker + local fallback on Redis outage.

The limiter must not pin a pipeline run when Redis is unreachable.
The design: after N consecutive failures, open the breaker and
delegate to a local ``RateLimiter`` sized at ``capacity / expected_workers``
(configured, not discovered) so the fallback stays conservative
rather than fail-open and re-trigger 429s.
"""

from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")

from ondine.utils.rate_limiter import RateLimiter
from ondine.utils.redis_rate_limiter import RedisRateLimiter


class _BrokenRedis:
    """Minimal test double that raises on every call to mimic a
    Redis outage. Covers both the sync register_script path and any
    subsequent command."""

    def register_script(self, script: str):  # noqa: ARG002
        raise ConnectionError("simulated redis outage")

    def delete(self, *args, **kwargs):  # noqa: ARG002
        raise ConnectionError("simulated redis outage")


@pytest.fixture
def clock():
    now = [1_000_000.0]

    def _clock() -> float:
        return now[0]

    _clock.advance = lambda dt: now.__setitem__(0, now[0] + dt)  # type: ignore[attr-defined]
    return _clock


def test_fallback_engages_when_redis_fails(clock) -> None:
    """Regression: if Redis is unreachable, acquire must still
    succeed via the local fallback — otherwise one network blip
    deadlocks the pipeline."""
    fallback = RateLimiter(requests_per_minute=60, burst_size=3)
    rl = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=3,
        sync_redis_client=_BrokenRedis(),
        scope="fb-1",
        monotonic=clock,
        fallback=fallback,
    )
    # Even though Redis is broken, acquire returns True via fallback.
    assert rl.acquire(timeout=0.2) is True


def test_fallback_raises_when_no_fallback_configured(clock) -> None:
    """Regression: without an explicit fallback, the user opted in to
    fail-hard. Swallowing the error silently would mask infra
    breakage."""
    rl = RedisRateLimiter(
        requests_per_minute=60,
        sync_redis_client=_BrokenRedis(),
        scope="fb-2",
        monotonic=clock,
    )
    with pytest.raises(ConnectionError):
        rl.acquire(timeout=0.2)


def test_fallback_carries_penalty(clock) -> None:
    """Regression: penalize() must also route through the fallback
    when Redis is broken — otherwise a 429 is silently dropped and
    cross-worker coordination is already lost (accept that) but at
    least the local worker respects the hint."""
    fallback = RateLimiter(requests_per_minute=60_000, burst_size=100)
    rl = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        sync_redis_client=_BrokenRedis(),
        scope="fb-3",
        monotonic=clock,
        fallback=fallback,
    )
    rl.penalize(delay_seconds=0.4)
    # Fallback bucket must be drained.
    assert rl.acquire(timeout=0.1) is False


def test_fallback_recovery_after_success(clock) -> None:
    """Regression: the breaker must probe Redis periodically so a
    transient outage doesn't permanently route to the less-shared
    fallback. This uses a probe-recovery test double that flips
    after N failures."""

    class _FlakyRedis:
        def __init__(self) -> None:
            self.calls = 0
            self._ok = fakeredis.FakeRedis()

        def register_script(self, script):
            # Simulate a short outage: fail the first 3 script loads,
            # succeed thereafter.
            self.calls += 1
            if self.calls <= 3:
                raise ConnectionError("transient")
            return self._ok.register_script(script)

        def delete(self, *args, **kwargs):
            return self._ok.delete(*args, **kwargs)

    fallback = RateLimiter(requests_per_minute=60, burst_size=5)
    client = _FlakyRedis()
    rl = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=5,
        sync_redis_client=client,
        scope="fb-recov",
        monotonic=clock,
        fallback=fallback,
    )
    # First call opens the breaker, uses fallback.
    assert rl.acquire(timeout=0.2) is True
    # Advance past the probe interval so the breaker re-probes.
    clock.advance(11.0)
    # Eventually the underlying redis recovers and the next acquire
    # goes through it cleanly.
    assert rl.acquire(timeout=0.5) is True
