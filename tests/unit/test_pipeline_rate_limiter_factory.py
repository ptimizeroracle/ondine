"""Tests for the rate-limiter factory wiring in the Pipeline module.

Pins the three config paths:
1. rpm unset -> None (no throttle)
2. rpm set, redis_url unset -> in-process RateLimiter (byte-identical
   to pre-A3 behaviour)
3. rpm set, redis_url set -> RedisRateLimiter with local fallback
"""

from __future__ import annotations

import pytest

from ondine.api.pipeline import _build_rate_limiter
from ondine.core.specifications import ProcessingSpec
from ondine.utils.rate_limiter import RateLimiter


def test_no_rpm_returns_none() -> None:
    """Regression: when rate_limit_rpm is unset, the factory must
    return None so the stage sees 'no throttle' — not a zero-rpm
    limiter that deadlocks the pipeline."""
    spec = ProcessingSpec()
    assert _build_rate_limiter(spec) is None


def test_rpm_only_returns_local_limiter() -> None:
    """Regression: default path must still produce the in-process
    RateLimiter — exactly what users had before A3. No Redis
    dependency when none was requested."""
    spec = ProcessingSpec(rate_limit_rpm=120, concurrency=5)
    rl = _build_rate_limiter(spec)
    assert isinstance(rl, RateLimiter)
    # burst sized at min(20, concurrency) per pre-existing rule.
    assert rl.capacity == 5


def test_redis_url_returns_redis_limiter_with_local_fallback() -> None:
    """Regression: when a Redis URL is configured, the factory must
    return a RedisRateLimiter whose fallback is a local RateLimiter
    at the same rpm. Missing fallback would deadlock on outage."""
    pytest.importorskip("fakeredis")
    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    spec = ProcessingSpec(
        rate_limit_rpm=120,
        concurrency=5,
        rate_limit_redis_url="redis://localhost:6379/0",
        rate_limit_scope="anthropic:sonnet",
    )
    rl = _build_rate_limiter(spec)
    assert isinstance(rl, RedisRateLimiter)
    assert rl._scope == "anthropic:sonnet"
    assert rl._fallback is not None
    assert isinstance(rl._fallback, RateLimiter)
    assert rl._fallback.capacity == 5
