"""Concurrency controller: bounded async execution + rate limiting.

Two modes, selected by the ``adaptive`` flag:

* ``adaptive=False`` (default, backwards-compatible): a fixed-size
  ``asyncio.Semaphore`` exactly as before.
* ``adaptive=True``: the Gradient2 ``AdaptiveLimiter``. The cap is
  bounded above by ``max_concurrent`` (the user's configured
  ceiling) and shrinks on 429 / RTT inflation.

Rate-limit propagation is a first-class operation via
:meth:`on_rate_limit`. Callers pass the parsed ``retry_after_s`` (or
``None`` when no header was available). The controller updates the
adaptive limiter and, when a delay was actually provided, drains the
shared token bucket so the signal is visible to every concurrent
caller — not just the one that saw the 429.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ondine.utils.adaptive_limiter import AdaptiveLimiter

if TYPE_CHECKING:
    from ondine.utils.rate_limiter import RateLimiter


class ConcurrencyController:
    """Bounded concurrent execution with optional adaptation."""

    def __init__(
        self,
        max_concurrent: int,
        rate_limiter: RateLimiter | None = None,
        *,
        adaptive: bool = False,
        min_concurrent: int = 1,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not 1 <= min_concurrent <= max_concurrent:
            raise ValueError("require 1 <= min_concurrent <= max_concurrent")

        self._rate_limiter = rate_limiter
        self._ceiling = max_concurrent
        self._adaptive = adaptive

        self._semaphore: asyncio.Semaphore | None = None
        self._limiter: AdaptiveLimiter | None = None
        if adaptive:
            self._limiter = AdaptiveLimiter(
                min_limit=min_concurrent,
                max_limit=max_concurrent,
                initial_limit=max_concurrent,
            )
        else:
            self._semaphore = asyncio.Semaphore(max_concurrent)

    # ── read-only properties ─────────────────────────────────────

    @property
    def adaptive(self) -> bool:
        return self._adaptive

    @property
    def max_concurrent(self) -> int:
        """Live in-flight cap. In adaptive mode this tracks the
        Gradient2 output; in fixed mode it returns the configured
        ceiling."""
        if self._limiter is not None:
            return self._limiter.current_limit
        return self._ceiling

    def snapshot(self) -> dict[str, float | int]:
        """Adaptive metrics snapshot; empty dict in fixed mode."""
        return self._limiter.snapshot() if self._limiter is not None else {}

    # ── adaptation hook ──────────────────────────────────────────

    def on_rate_limit(self, retry_after_s: float | None) -> None:
        """Record a 429 signal.

        Always shrinks the adaptive cap. When ``retry_after_s`` was
        actually provided by the server, also drains the token bucket
        so every caller observes the backoff.
        """
        if self._limiter is not None:
            self._limiter.on_rate_limit(retry_after_s=retry_after_s or 0.0)
        if retry_after_s and retry_after_s > 0 and self._rate_limiter:
            self._rate_limiter.penalize(delay_seconds=retry_after_s)

    # ── acquire / release ────────────────────────────────────────

    async def acquire(self) -> None:
        if self._limiter is not None:
            # Adaptive path: the slot() context manager handles the
            # observation. `acquire()` is a legacy shape callers can
            # still use, but they lose RTT feedback — prefer
            # throttle().
            await self._limiter_acquire_raw()
        else:
            assert self._semaphore is not None
            await self._semaphore.acquire()
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire_async()
        except Exception:
            if self._limiter is not None:
                await self._limiter_release_raw()
            else:
                assert self._semaphore is not None
                self._semaphore.release()
            raise

    def release(self) -> None:
        """Release a concurrency slot.

        Only valid in fixed-semaphore mode — the adaptive path uses
        :meth:`throttle` which handles release internally.
        """
        if self._limiter is not None:
            # Best-effort fire-and-forget for legacy callers.
            asyncio.ensure_future(self._limiter_release_raw())
            return
        assert self._semaphore is not None
        self._semaphore.release()

    @asynccontextmanager
    async def throttle(self):
        """Bounded-concurrency + rate-limit context manager.

        In adaptive mode, RTT is measured across the yield and fed
        into the Gradient2 loop.
        """
        if self._limiter is not None:
            async with self._limiter.slot(rtt_source=_noop_rtt):
                if self._rate_limiter:
                    await self._rate_limiter.acquire_async()
                yield
            return

        assert self._semaphore is not None
        await self._semaphore.acquire()
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire_async()
            yield
        finally:
            self._semaphore.release()

    # ── adaptive-mode internals ──────────────────────────────────

    async def _limiter_acquire_raw(self) -> None:
        assert self._limiter is not None
        # Re-use the limiter's condition-based admission without the
        # slot context manager. No RTT observation is recorded —
        # callers in this path must accept degraded adaptation.
        await self._limiter._acquire()  # noqa: SLF001

    async def _limiter_release_raw(self) -> None:
        assert self._limiter is not None
        await self._limiter._release()  # noqa: SLF001

    def __repr__(self) -> str:
        return (
            f"ConcurrencyController(max_concurrent={self._ceiling}, "
            f"adaptive={self._adaptive}, "
            f"rate_limiter={self._rate_limiter is not None})"
        )


def _noop_rtt() -> float:
    """RTT source used when callers don't supply one.

    Returning 0 signals "no measurement" and the limiter falls back
    to wallclock-elapsed inside ``slot()``.
    """
    return 0.0
