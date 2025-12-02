"""
Concurrency controller for managing parallel execution with rate limiting.

Provides a clean abstraction over asyncio.Semaphore and RateLimiter,
enabling controlled concurrent access to resources (e.g., LLM APIs).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.utils.rate_limiter import RateLimiter


class ConcurrencyController:
    """
    Manages concurrent execution with optional rate limiting.

    Combines semaphore-based concurrency control with token bucket rate limiting
    to prevent overwhelming downstream services while maximizing throughput.

    Example:
        controller = ConcurrencyController(max_concurrent=10, rate_limiter=limiter)

        async def process_item(item):
            async with controller.throttle():
                return await call_api(item)

        # Process 1000 items with max 10 concurrent, rate-limited calls
        results = await asyncio.gather(*[process_item(i) for i in items])
    """

    def __init__(
        self,
        max_concurrent: int,
        rate_limiter: "RateLimiter | None" = None,
    ):
        """
        Initialize concurrency controller.

        Args:
            max_concurrent: Maximum number of concurrent operations
            rate_limiter: Optional rate limiter for token bucket throttling
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = rate_limiter
        self._max_concurrent = max_concurrent

    @property
    def max_concurrent(self) -> int:
        """Return the maximum concurrency limit."""
        return self._max_concurrent

    async def acquire(self) -> None:
        """
        Acquire a concurrency slot and rate limit token.

        Blocks until both a semaphore slot is available and the rate limiter
        (if configured) permits the operation.
        """
        await self._semaphore.acquire()
        if self._rate_limiter:
            # RateLimiter.acquire() is sync, but we're in async context
            # This is fine - it's a quick token bucket check
            self._rate_limiter.acquire()

    def release(self) -> None:
        """Release a concurrency slot."""
        self._semaphore.release()

    @asynccontextmanager
    async def throttle(self):
        """
        Context manager for throttled execution.

        Acquires a slot on entry, releases on exit (even if exception occurs).

        Example:
            async with controller.throttle():
                result = await expensive_operation()
        """
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def __repr__(self) -> str:
        return (
            f"ConcurrencyController(max_concurrent={self._max_concurrent}, "
            f"rate_limiter={self._rate_limiter is not None})"
        )
