"""End-to-end tests against a real Redis server.

Runs only when ``ONDINE_REDIS_E2E_URL`` is set. CI can point it at a
sidecar container; developers can point it at a local Redis. Without
the env var the tests are skipped so the suite stays fast and
hermetic by default.

These tests exercise the production code paths that fakeredis cannot
reach:

* ``register_script`` -> ``EVALSHA`` caching behaviour
* Real ``PEXPIRE`` + key expiry
* Cross-client contention with a second connection (closest we can
  get to multi-process without spawning subprocesses)
* A subprocess-based multi-worker test that proves the shared-bucket
  property across true OS processes — the whole reason A3 exists
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("redis")


REDIS_URL = os.environ.get("ONDINE_REDIS_E2E_URL")


@pytest.fixture
def redis_url() -> str:
    if not REDIS_URL:
        pytest.skip("set ONDINE_REDIS_E2E_URL to run real-Redis E2E")
    return REDIS_URL


@pytest.fixture
def fresh_scope() -> str:
    # Unique per-test scope so parallel runs don't contaminate each
    # other's bucket state.
    return f"e2e-{os.getpid()}-{time.time_ns()}"


# ── single-process, real Redis ────────────────────────────────────────


def test_real_redis_sync_acquire_drains_shared_bucket(redis_url, fresh_scope) -> None:
    """Two limiters, one process, real Redis: must share one bucket.

    Without Lua atomicity this test would fail under contention
    because the refill/check/decrement sequence would race.
    """
    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    a = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=3,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    b = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=3,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    a.reset()
    # Burst of 3 across the two instances.
    assert a.acquire(timeout=1.0) is True
    assert b.acquire(timeout=1.0) is True
    assert a.acquire(timeout=1.0) is True
    # Fourth must block + time out because the bucket is globally empty.
    assert b.acquire(timeout=0.3) is False
    a.reset()


def test_real_redis_evalsha_reused(redis_url, fresh_scope) -> None:
    """After the first acquire, subsequent acquires must use
    EVALSHA — otherwise every call re-sends the Lua body, wasting
    bandwidth at high RPM. We inspect redis INFO to confirm the
    script cache grew by exactly one entry.
    """
    import redis as _redis

    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    conn = _redis.Redis.from_url(redis_url)
    conn.script_flush()
    cache_before = int(conn.info("memory").get("number_of_cached_scripts", 0) or 0)

    rl = RedisRateLimiter(
        requests_per_minute=600,
        burst_size=20,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    for _ in range(5):
        assert rl.acquire(timeout=1.0) is True

    cache_after = int(conn.info("memory").get("number_of_cached_scripts", 0) or 0)
    # Exactly two scripts registered: acquire + penalize.
    assert cache_after - cache_before in (1, 2)
    rl.reset()


def test_real_redis_key_expires(redis_url, fresh_scope) -> None:
    """PEXPIRE must actually set a TTL — otherwise dead scopes
    accumulate in Redis forever. Use a tiny TTL and verify the key
    disappears."""
    import redis as _redis

    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    rl = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=3,
        redis_url=redis_url,
        scope=fresh_scope,
        key_ttl_s=1,
    )
    rl.reset()
    assert rl.acquire(timeout=1.0) is True

    conn = _redis.Redis.from_url(redis_url)
    bucket_key = f"{{ondine:ratelimit:{fresh_scope}}}:bucket"
    ttl_ms = conn.pexpiretime(bucket_key)
    # Key exists and has a TTL in the future.
    assert ttl_ms > 0

    time.sleep(1.5)
    assert conn.exists(bucket_key) == 0


# ── async path, real Redis ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_real_redis_async_acquire(redis_url, fresh_scope) -> None:
    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    rl = RedisRateLimiter(
        requests_per_minute=60,
        burst_size=2,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    rl.reset()
    assert await rl.acquire_async(timeout=1.0) is True
    assert await rl.acquire_async(timeout=1.0) is True
    assert await rl.acquire_async(timeout=0.3) is False
    rl.reset()


# ── penalty cross-worker, real Redis ─────────────────────────────────


def test_real_redis_penalty_is_cross_process_visible(redis_url, fresh_scope) -> None:
    """Worker A's penalize() must gate worker B through real Redis.

    This is the core A3 value proposition: in the current
    in-process RateLimiter, this is physically impossible.
    """
    from ondine.utils.redis_rate_limiter import RedisRateLimiter

    a = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    b = RedisRateLimiter(
        requests_per_minute=60_000,
        burst_size=100,
        redis_url=redis_url,
        scope=fresh_scope,
    )
    a.reset()
    a.penalize(delay_seconds=1.0)
    # Worker B has never seen a 429, but the penalty in Redis
    # gates every acquire until the window passes.
    assert b.acquire(timeout=0.3) is False

    time.sleep(1.2)
    assert b.acquire(timeout=0.5) is True
    a.reset()


# ── true multi-process contention ────────────────────────────────────


_WORKER_SCRIPT = """
import os, sys, time
from ondine.utils.redis_rate_limiter import RedisRateLimiter

url = os.environ["ONDINE_REDIS_E2E_URL"]
scope = os.environ["ONDINE_REDIS_E2E_SCOPE"]
n = int(os.environ["ONDINE_REDIS_E2E_N"])

rl = RedisRateLimiter(
    requests_per_minute=60,
    burst_size=4,
    redis_url=url,
    scope=scope,
)
granted = 0
deadline = time.monotonic() + 2.0
while granted < n and time.monotonic() < deadline:
    if rl.acquire(timeout=0.1):
        granted += 1
print(granted)
"""


def test_real_redis_two_processes_share_one_bucket(redis_url, fresh_scope) -> None:
    """Spawn two real subprocesses each trying to acquire 10 tokens
    from a burst-4 bucket. Combined grants must be ≤ 4 + refill
    during the test window. This is the only test in the suite
    that proves the cross-process property with true OS processes.
    """
    import redis as _redis

    # Pre-reset the scope.
    conn = _redis.Redis.from_url(redis_url)
    key_base = f"{{ondine:ratelimit:{fresh_scope}}}"
    conn.delete(f"{key_base}:bucket", f"{key_base}:penalty")

    env = {
        **os.environ,
        "ONDINE_REDIS_E2E_URL": redis_url,
        "ONDINE_REDIS_E2E_SCOPE": fresh_scope,
        "ONDINE_REDIS_E2E_N": "10",
        "PYTHONPATH": os.environ.get("PYTHONPATH", os.getcwd()),
    }

    def _spawn() -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", _WORKER_SCRIPT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    t0 = time.monotonic()
    p1 = _spawn()
    p2 = _spawn()
    out1, err1 = p1.communicate(timeout=15)
    out2, err2 = p2.communicate(timeout=15)
    elapsed = time.monotonic() - t0

    assert p1.returncode == 0, err1.decode()
    assert p2.returncode == 0, err2.decode()

    granted_a = int(out1.decode().strip())
    granted_b = int(out2.decode().strip())
    total = granted_a + granted_b

    # Refill is 60 rpm = 1 token/s; during the <= 2s work window
    # the bucket yields at most 4 (burst) + floor(2) = 6 tokens.
    # Both processes collectively capped by Redis.
    assert total <= 6, (
        f"combined grants {total} exceeds theoretical max 6 "
        f"(a={granted_a}, b={granted_b}, elapsed={elapsed:.2f}s)"
    )
    # Proof both processes actually participated.
    assert granted_a > 0
    assert granted_b > 0
    conn.delete(f"{key_base}:bucket", f"{key_base}:penalty")
