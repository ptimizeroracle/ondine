"""Distributed token-bucket rate limiter backed by Redis.

Implements the same public surface as ``ondine.utils.rate_limiter.RateLimiter``
so it can be swapped in without touching the ConcurrencyController or
LLMInvocationStage. The entire token-bucket arithmetic (refill,
capacity cap, conditional decrement) is performed inside a single
atomic Lua script that runs server-side, so concurrent workers
across processes or machines share one true bucket — something a
Python ``threading.Lock`` fundamentally cannot provide.

Key design choices (see docs/architecture/decisions/ADR-008 pending):

* **Caller-supplied timestamp.** The script receives ``now`` as an
  argument rather than calling Redis ``TIME``. This keeps the script
  deterministic (replicable, cacheable) and behaves correctly on
  read replicas. Workers must have reasonably synchronised clocks
  (NTP). The script clamps backwards clock moves so small skew
  cannot corrupt state.
* **One hash per scope.** ``{namespace}:{scope}`` holds ``tokens``
  and ``ts`` fields. Separate ``:penalty`` key (added in Phase 3)
  holds the server-issued Retry-After deadline. Both are keyed
  under the same hash tag so Redis Cluster keeps them colocated.
* **EVALSHA via ``register_script``.** First call ``EVAL``s and
  caches the SHA; subsequent calls use the cheap ``EVALSHA``. On
  ``NOSCRIPT`` (e.g., after a failover to a replica with empty
  script cache) redis-py automatically falls back to ``EVAL``.

Later phases layer ``penalize()`` cross-worker propagation, a
circuit breaker, and a local ``RateLimiter`` fallback for Redis
outages.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis

    from ondine.utils.rate_limiter import RateLimiter


# Breaker tuning. ``_BREAKER_THRESHOLD`` failures within the lifetime
# of the limiter will open the breaker; after ``_BREAKER_PROBE_S``
# the next acquire will retry Redis. Keep conservative because the
# only failure mode here is routing through the lower-ceiling
# fallback — nothing gets dropped.
_BREAKER_THRESHOLD = 3
_BREAKER_PROBE_S = 10.0


# Lua script:
# KEYS[1] = bucket hash key
# KEYS[2] = penalty_until key (holds a float wallclock deadline)
# ARGV[1] = refill_rate (tokens per second, float)
# ARGV[2] = capacity (float)
# ARGV[3] = tokens_requested (float)
# ARGV[4] = now (seconds since epoch or monotonic, float)
# ARGV[5] = ttl_ms (integer, for PEXPIRE)
#
# Returns: {granted (0/1), current_tokens (float)}
#
# Penalty gate runs first: if a server-issued Retry-After is still
# in effect, all workers are held back regardless of local bucket
# depth. This is the whole reason the penalty key exists in Redis
# instead of per-worker memory.
_LUA_ACQUIRE = """
local now = tonumber(ARGV[4])

local penalty_until = tonumber(redis.call('GET', KEYS[2]))
if penalty_until ~= nil and now < penalty_until then
    return {0, tostring(0)}
end

local bucket = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local capacity = tonumber(ARGV[2])
local refill_rate = tonumber(ARGV[1])
local tokens_requested = tonumber(ARGV[3])

local stored_tokens = tonumber(bucket[1])
local stored_ts = tonumber(bucket[2])

-- Fresh bucket, or caller clock went backwards: start at full.
if stored_tokens == nil or stored_ts == nil or now < stored_ts then
    stored_tokens = capacity
    stored_ts = now
end

-- Refill based on elapsed time, capped at capacity.
local elapsed = now - stored_ts
local refilled = math.min(capacity, stored_tokens + elapsed * refill_rate)

if tokens_requested > 0 and refilled < tokens_requested then
    redis.call('HMSET', KEYS[1], 'tokens', refilled, 'ts', now)
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[5]))
    return {0, tostring(refilled)}
end

local remaining = refilled - tokens_requested
redis.call('HMSET', KEYS[1], 'tokens', remaining, 'ts', now)
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[5]))
return {1, tostring(remaining)}
"""


# Lua script:
# KEYS[1] = penalty_until key
# ARGV[1] = candidate_deadline (float, seconds)
# ARGV[2] = ttl_ms (integer)
#
# Set the key to the max of the current value and the candidate so
# that a late short signal can never shorten an earlier long one.
_LUA_PENALIZE = """
local current = tonumber(redis.call('GET', KEYS[1]))
local candidate = tonumber(ARGV[1])
if current == nil or candidate > current then
    redis.call('SET', KEYS[1], tostring(candidate), 'PX', tonumber(ARGV[2]))
end
return 1
"""


class RedisRateLimiter:
    """Token-bucket rate limiter with Redis as the shared source of
    truth.

    Duck-typed to match :class:`ondine.utils.rate_limiter.RateLimiter` —
    the same ``acquire``, ``acquire_async``, ``penalize``,
    ``available_tokens``, and ``reset`` methods.

    Args:
        requests_per_minute: RPM cap across all workers sharing this
            scope. Refill rate is ``rpm/60`` tokens/second.
        redis_client: Optional pre-built ``redis.asyncio.Redis``
            instance. When ``None``, callers must pass ``redis_url``.
        redis_url: Connection URL (``redis://host:port/db``). Used
            only when ``redis_client`` is ``None``.
        key_namespace: Top-level key prefix. Defaults to
            ``"ondine:ratelimit"``.
        scope: Sub-key identifying which bucket this limiter uses.
            Typically ``"{provider}:{model}"``. Workers that must
            share a budget use the same scope; different scopes are
            isolated.
        burst_size: Maximum burst / bucket capacity. Defaults to
            ``requests_per_minute``.
        monotonic: Injectable clock — returns seconds as a float.
            Tests pass a controlled fake; production passes nothing
            and the limiter uses :func:`time.time` (a wallclock is
            used here, not :func:`time.monotonic`, because the
            timestamp is shared across hosts and must be comparable
            regardless of each host's process uptime).
        key_ttl_s: PEXPIRE horizon for the bucket key. A long-idle
            bucket is garbage-collected so dead scopes don't pile
            up. Refreshed on every acquire.
    """

    def __init__(
        self,
        requests_per_minute: int,
        redis_client: AsyncRedis | None = None,
        sync_redis_client: Redis | None = None,
        redis_url: str | None = None,
        key_namespace: str = "ondine:ratelimit",
        scope: str = "default",
        burst_size: int | None = None,
        monotonic: Callable[[], float] | None = None,
        key_ttl_s: int = 600,
        fallback: RateLimiter | None = None,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError(
                f"requests_per_minute must be positive, got {requests_per_minute}"
            )
        if redis_client is None and sync_redis_client is None and redis_url is None:
            raise ValueError("pass redis_client, sync_redis_client, or redis_url")

        self.rpm = requests_per_minute
        self.capacity = float(burst_size) if burst_size else float(requests_per_minute)
        self.refill_rate = requests_per_minute / 60.0
        self._namespace = key_namespace
        self._scope = scope
        # Hash tag ``{namespace:scope}`` keeps bucket + penalty
        # co-located in Redis Cluster (same slot), so the single
        # multi-key EVAL is legal.
        tag = f"{{{key_namespace}:{scope}}}"
        self._key = f"{tag}:bucket"
        self._penalty_key = f"{tag}:penalty"
        self._ttl_ms = int(key_ttl_s * 1000)
        self._monotonic = monotonic or time.time

        self._async_client = redis_client
        self._sync_client = sync_redis_client
        self._async_url = redis_url
        self._async_script: Any | None = None
        self._sync_script: Any | None = None
        self._async_penalize: Any | None = None
        self._sync_penalize: Any | None = None

        # Circuit breaker state. ``_breaker_opened_at`` is ``None``
        # when closed; otherwise the monotonic timestamp when it
        # flipped open. ``_fallback`` is the local RateLimiter used
        # when the breaker is open. When no fallback is configured,
        # Redis errors propagate — intentional: the caller opted in
        # to fail-hard.
        self._fallback = fallback
        self._breaker_failures = 0
        self._breaker_opened_at: float | None = None

    # ── async path ────────────────────────────────────────────────

    async def acquire_async(
        self, tokens: int = 1, timeout: float | None = None
    ) -> bool:
        """Acquire ``tokens`` from the shared bucket.

        Blocks via ``asyncio.sleep`` until either the tokens are
        granted by the server-side script or ``timeout`` elapses.
        Polling interval is 50 ms, matching the local RateLimiter.

        Raises:
            ValueError: If ``tokens`` exceeds the bucket capacity.
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds capacity {self.capacity}"
            )

        # Deadline uses the event-loop clock, not the injectable
        # ``monotonic`` callable. ``monotonic`` is the *bucket
        # timestamp*: it feeds the Lua script so tests can control
        # refill determinism. The deadline must tick with real
        # elapsed wall-time, otherwise a frozen test clock pins a
        # short timeout into an infinite loop.
        loop = asyncio.get_event_loop()
        deadline = None if timeout is None else loop.time() + timeout
        client = await self._get_async_client()
        script = await self._get_async_script(client)

        while True:
            granted, _remaining = await self._eval_async(script, tokens)
            if granted:
                return True
            if deadline is not None and loop.time() >= deadline:
                return False
            await asyncio.sleep(0.05)

    async def _eval_async(self, script: Any, tokens: int) -> tuple[int, float]:
        now = self._monotonic()
        result = await script(
            keys=[self._key, self._penalty_key],
            args=[
                self.refill_rate,
                self.capacity,
                float(tokens),
                now,
                self._ttl_ms,
            ],
        )
        granted_raw, remaining_raw = result
        granted = int(granted_raw)
        remaining = float(
            remaining_raw.decode()
            if isinstance(remaining_raw, bytes)
            else remaining_raw
        )
        return granted, remaining

    async def _get_async_client(self) -> AsyncRedis:
        if self._async_client is not None:
            return self._async_client
        # Lazy — only import redis.asyncio if we actually need it.
        from redis.asyncio import Redis as _AsyncRedis

        self._async_client = _AsyncRedis.from_url(self._async_url)  # type: ignore[arg-type]
        return self._async_client

    async def _get_async_script(self, client: AsyncRedis) -> Any:
        if self._async_script is None:
            self._async_script = client.register_script(_LUA_ACQUIRE)
        return self._async_script

    # ── sync path ─────────────────────────────────────────────────

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Blocking sync acquire. See :meth:`acquire_async`."""
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds capacity {self.capacity}"
            )

        if self._should_use_fallback():
            return self._fallback.acquire(tokens=tokens, timeout=timeout)  # type: ignore[union-attr]

        # Sync timeout uses real wallclock; see acquire_async for the
        # two-clock rationale.
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            client = self._get_sync_client()
            script = self._get_sync_script(client)
        except Exception:
            self._record_failure()
            if self._fallback is None:
                raise
            return self._fallback.acquire(tokens=tokens, timeout=timeout)

        while True:
            try:
                granted, _remaining = self._eval_sync(script, tokens)
            except Exception:
                self._record_failure()
                if self._fallback is None:
                    raise
                return self._fallback.acquire(tokens=tokens, timeout=timeout)
            self._record_success()
            if granted:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _eval_sync(self, script: Any, tokens: int) -> tuple[int, float]:
        now = self._monotonic()
        result = script(
            keys=[self._key, self._penalty_key],
            args=[
                self.refill_rate,
                self.capacity,
                float(tokens),
                now,
                self._ttl_ms,
            ],
        )
        granted_raw, remaining_raw = result
        granted = int(granted_raw)
        remaining = float(
            remaining_raw.decode()
            if isinstance(remaining_raw, bytes)
            else remaining_raw
        )
        return granted, remaining

    def _get_sync_client(self) -> Redis:
        if self._sync_client is not None:
            return self._sync_client
        from redis import Redis as _Redis

        if self._async_url is None:
            raise RuntimeError(
                "sync acquire requested but no sync client or URL configured"
            )
        self._sync_client = _Redis.from_url(self._async_url)
        return self._sync_client

    def _get_sync_script(self, client: Redis) -> Any:
        if self._sync_script is None:
            self._sync_script = client.register_script(_LUA_ACQUIRE)
        return self._sync_script

    # ── observability / maintenance ────────────────────────────────

    @property
    def available_tokens(self) -> float:
        """Return the current token depth as observed on Redis.

        Reads authoritative server state — does not cache. This
        triggers a refill computation by calling the acquire script
        with ``tokens=0`` (always granted; reports remaining).
        """
        client = self._get_sync_client()
        script = self._get_sync_script(client)
        _granted, remaining = self._eval_sync(script, tokens=0)
        return remaining

    def reset(self) -> None:
        """Drop the shared bucket and any active penalty. Next
        acquire starts at full capacity."""
        client = self._get_sync_client()
        client.delete(self._key, self._penalty_key)

    # ── penalty propagation ───────────────────────────────────────

    def penalize(self, delay_seconds: float) -> None:
        """Honour a server-issued Retry-After, visible to every
        worker sharing this scope.

        Writes an absolute deadline (``now + delay``) to the penalty
        key via a Lua script that takes the max of the stored and
        incoming deadlines, so a late short signal cannot shorten a
        long one already in flight. Key TTL matches the penalty
        duration so abandoned penalties self-garbage-collect.

        When Redis is unreachable and a fallback is configured, the
        penalty routes through the fallback (worker-local only —
        cross-worker coordination is already lost at that point).
        """
        if delay_seconds < 0:
            raise ValueError(f"delay_seconds must be non-negative, got {delay_seconds}")
        if delay_seconds == 0:
            return

        if self._should_use_fallback():
            if self._fallback is not None:
                self._fallback.penalize(delay_seconds=delay_seconds)
            return

        try:
            client = self._get_sync_client()
            script = self._get_sync_penalize_script(client)
            deadline = self._monotonic() + delay_seconds
            ttl_ms = int((delay_seconds + 1.0) * 1000)
            script(keys=[self._penalty_key], args=[deadline, ttl_ms])
        except Exception:
            self._record_failure()
            if self._fallback is None:
                raise
            self._fallback.penalize(delay_seconds=delay_seconds)
            return
        self._record_success()

    async def penalize_async(self, delay_seconds: float) -> None:
        """Async peer of :meth:`penalize`."""
        if delay_seconds < 0:
            raise ValueError(f"delay_seconds must be non-negative, got {delay_seconds}")
        if delay_seconds == 0:
            return
        client = await self._get_async_client()
        script = await self._get_async_penalize_script(client)
        deadline = self._monotonic() + delay_seconds
        ttl_ms = int((delay_seconds + 1.0) * 1000)
        await script(keys=[self._penalty_key], args=[deadline, ttl_ms])

    def _get_sync_penalize_script(self, client: Redis) -> Any:
        if self._sync_penalize is None:
            self._sync_penalize = client.register_script(_LUA_PENALIZE)
        return self._sync_penalize

    async def _get_async_penalize_script(self, client: AsyncRedis) -> Any:
        if self._async_penalize is None:
            self._async_penalize = client.register_script(_LUA_PENALIZE)
        return self._async_penalize

    # ── circuit breaker ───────────────────────────────────────────

    def _should_use_fallback(self) -> bool:
        """True when the breaker is open and the probe interval has
        not yet elapsed. When the probe interval has elapsed the
        breaker goes half-open — the next Redis call is attempted;
        success closes the breaker, failure re-opens it for another
        probe window.
        """
        if self._breaker_opened_at is None:
            return False
        if self._fallback is None:
            return False
        elapsed = self._monotonic() - self._breaker_opened_at
        if elapsed >= _BREAKER_PROBE_S:
            # Half-open: clear the state so the next call tries
            # Redis again. If it fails, _record_failure will
            # re-open.
            self._breaker_opened_at = None
            self._breaker_failures = 0
            # Reset memoised scripts so a rebuild is forced in case
            # the server lost its script cache.
            self._sync_script = None
            self._async_script = None
            self._sync_penalize = None
            self._async_penalize = None
            return False
        return True

    def _record_failure(self) -> None:
        self._breaker_failures += 1
        if self._breaker_failures >= _BREAKER_THRESHOLD:
            self._breaker_opened_at = self._monotonic()

    def _record_success(self) -> None:
        self._breaker_failures = 0
        self._breaker_opened_at = None
