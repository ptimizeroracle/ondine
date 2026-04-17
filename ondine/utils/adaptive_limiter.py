"""Adaptive asyncio concurrency limiter using the Gradient2 algorithm.

Why not ``asyncio.Semaphore``: the built-in semaphore has no public
``resize`` method and no way to signal "the server told us to back
off". Hand-rolling a condition-based limiter lets us:

* Shrink the in-flight cap in response to explicit 429s.
* Grow the cap only when the queue is saturated *and* RTT is near the
  observed no-load baseline — Netflix's Gradient2 rule, empirically
  the best fit for HTTPS clients bottlenecked on a remote rate
  limiter.
* Keep in-flight work running on shrink (no cancellation); we simply
  stop admitting new work until drain.

The algorithm tracks two RTT windows:

* ``rtt_noload`` — rolling minimum, our estimate of "no queueing"
* ``rtt_smoothed`` — exponential smoothing of recent observations

Gradient2 computes::

    gradient = max(0.5, min(1.0, TOLERANCE * rtt_noload / rtt_smoothed))
    target   = current_limit * gradient + sqrt(current_limit)
    limit    = limit * (1 - SMOOTHING) + target * SMOOTHING

and clamps to ``[min_limit, max_limit]``. On an explicit 429 the
current limit is multiplied by ``RATE_LIMIT_SHRINK`` (0.9) — a
gentler hit than TCP AIMD's 0.5 because a single 429 among many
successes is not a signal that half the capacity vanished.

Caller API is a slot context manager::

    async with limiter.slot(rtt_source=lambda: measured_rtt):
        ...

The RTT source is called on exit to feed the adapter. Keeping RTT
measurement in caller code avoids coupling the limiter to any
particular HTTP client.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

# ── tuning constants (Netflix concurrency-limits defaults) ─────────────
_TOLERANCE = 2.0
_SMOOTHING = 0.2
_RATE_LIMIT_SHRINK = 0.9
_RTT_NOLOAD_DECAY = 0.95  # slowly forgive the no-load estimate

# Minimum gap between effective shrink events. A single upstream
# overload typically produces a burst of 429s within a few hundred
# milliseconds — across retries of one request and across concurrent
# workers. Without a cooldown, N workers retrying M times each apply
# N*M shrinks for what is logically one capacity event, overshooting
# the adaptation. The cooldown still counts every 429 for
# observability; only the shrink arithmetic is deduplicated.
_RATE_LIMIT_COOLDOWN_S = 0.2


class AdaptiveLimiter:
    """Condition-based in-flight cap with Gradient2 adaptation.

    Thread-compatibility: asyncio-only. Not safe across threads.

    Attributes (read-only from outside):
        current_limit: The cap as it stands right now. Integer.
        in_flight: Number of active slot holders.
        peak_in_flight: High-water mark since construction.
    """

    def __init__(
        self,
        min_limit: int,
        max_limit: int,
        initial_limit: int,
        *,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not 1 <= min_limit <= max_limit:
            raise ValueError("require 1 <= min_limit <= max_limit")
        if not min_limit <= initial_limit <= max_limit:
            raise ValueError("require min_limit <= initial_limit <= max_limit")
        self._min = min_limit
        self._max = max_limit
        self._limit: float = float(initial_limit)
        self._in_flight = 0
        self._peak = 0
        self._rate_limit_hits = 0
        self._rtt_noload: float | None = None
        self._rtt_smoothed: float | None = None
        self._monotonic = monotonic or time.monotonic
        self._cond = asyncio.Condition()
        # Timestamp of the last effective shrink; used to debounce
        # bursts of 429s that represent one logical overload event.
        # Initialised to -inf so the very first 429 always shrinks.
        self._last_shrink_monotonic: float = float("-inf")

    # ── public properties ─────────────────────────────────────────

    @property
    def current_limit(self) -> int:
        return max(self._min, min(self._max, int(self._limit)))

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        return self._peak

    def snapshot(self) -> dict[str, float | int]:
        """Point-in-time metrics suitable for logs/traces/Prometheus."""
        return {
            "current_limit": self.current_limit,
            "in_flight": self._in_flight,
            "queue_depth": max(0, self._in_flight - self.current_limit),
            "rtt_noload_ms": (self._rtt_noload or 0.0) * 1000.0,
            "rtt_smoothed_ms": (self._rtt_smoothed or 0.0) * 1000.0,
            "rate_limit_hits": self._rate_limit_hits,
        }

    # ── core context manager ──────────────────────────────────────

    @contextlib.asynccontextmanager
    async def slot(
        self,
        rtt_source: Callable[[], float],
    ) -> AsyncIterator[None]:
        """Acquire an admission slot; release on exit with an RTT
        observation."""
        await self._acquire()
        start = self._monotonic()
        try:
            yield
            # Successful exit — feed RTT (prefer caller's measurement
            # if they tracked elapsed themselves, else use wallclock).
            rtt = _safe_rtt(rtt_source) or (self._monotonic() - start)
            self.observe(rtt_s=rtt, in_flight_at_sample=self._in_flight)
        finally:
            await self._release()

    # ── adaptation hooks ──────────────────────────────────────────

    def on_rate_limit(self, retry_after_s: float) -> None:
        """Signal that the server issued an explicit 429.

        Increments the 429 counter unconditionally for observability.
        Applies the ``_RATE_LIMIT_SHRINK`` multiplier only if the
        previous shrink was more than ``_RATE_LIMIT_COOLDOWN_S`` ago,
        so a burst of 429s from retries of a single request (or from
        concurrent workers hitting the same upstream overload) is
        treated as one logical capacity event instead of compounding
        shrinks per individual hit.

        ``retry_after_s`` is currently unused by the limiter itself
        (the token bucket owns that signal) but is accepted here for
        symmetry and future use.
        """
        del retry_after_s  # reserved; see docstring
        self._rate_limit_hits += 1
        now = self._monotonic()
        if now - self._last_shrink_monotonic < _RATE_LIMIT_COOLDOWN_S:
            return
        self._last_shrink_monotonic = now
        self._limit = max(
            float(self._min),
            self._limit * _RATE_LIMIT_SHRINK,
        )

    def observe(self, rtt_s: float, in_flight_at_sample: int) -> None:
        """Feed one RTT observation into the Gradient2 loop.

        The ``in_flight_at_sample`` is the concurrency that was
        active *when the sample was taken* — Gradient2 only grows
        when the queue is saturated (in_flight >= current_limit).
        """
        if rtt_s <= 0:
            return

        # Update no-load baseline (rolling minimum, slowly decays
        # upward so a transient low doesn't pin the baseline).
        if self._rtt_noload is None or rtt_s < self._rtt_noload:
            self._rtt_noload = rtt_s
        else:
            # Gentle drift back toward observed RTTs; prevents a
            # single unusually-fast sample from pinning the baseline
            # forever.
            self._rtt_noload = self._rtt_noload / _RTT_NOLOAD_DECAY
            self._rtt_noload = min(self._rtt_noload, rtt_s)

        # EWMA smoothing of recent RTT.
        if self._rtt_smoothed is None:
            self._rtt_smoothed = rtt_s
        else:
            self._rtt_smoothed = (
                self._rtt_smoothed * (1.0 - _SMOOTHING) + rtt_s * _SMOOTHING
            )

        # Gradient rule — only adapt once we have a stable baseline.
        if self._rtt_noload is None or self._rtt_smoothed is None:
            return
        gradient = max(
            0.5,
            min(1.0, _TOLERANCE * self._rtt_noload / self._rtt_smoothed),
        )

        # Grow only when saturated — Gradient2's queue-full condition.
        queue_bonus = (
            math.sqrt(self._limit)
            if (in_flight_at_sample >= self.current_limit)
            else 0.0
        )
        target = self._limit * gradient + queue_bonus

        self._limit = self._limit * (1.0 - _SMOOTHING) + target * _SMOOTHING
        self._limit = max(float(self._min), min(float(self._max), self._limit))

    # ── condition-based admission ────────────────────────────────

    async def _acquire(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self._in_flight < self.current_limit)
            self._in_flight += 1
            if self._in_flight > self._peak:
                self._peak = self._in_flight

    async def _release(self) -> None:
        async with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            # FIFO wake: notify one waiter so fairness is preserved
            # under shrink — notify_all causes thundering herd.
            self._cond.notify(1)


def _safe_rtt(source: Callable[[], float]) -> float | None:
    try:
        value = source()
    except Exception:
        return None
    if value is None or value <= 0:
        return None
    return float(value)
