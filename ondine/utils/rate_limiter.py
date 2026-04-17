"""
Token bucket rate limiter for API calls.

Implements token bucket algorithm for rate limiting.
Supports both sync (threading.Lock) and async (asyncio.Lock) paths.
"""

import asyncio
import threading
import time


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request rates.

    Thread-safe implementation.
    """

    def __init__(self, requests_per_minute: int, burst_size: int | None = None):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (default: requests_per_minute)
        """
        self.rpm = requests_per_minute
        self.capacity = burst_size or requests_per_minute
        self.tokens = float(self.capacity)
        self.last_update = time.time()
        self.lock = threading.Lock()

        # Wallclock deadline before which the bucket stays empty —
        # used by penalize() so a server-issued "retry after Xs" is
        # enforced for every caller, not just the one that saw the
        # 429. Zero means "no active penalty".
        self._penalty_until = 0.0

        # Calculate refill rate (tokens per second)
        self.refill_rate = requests_per_minute / 60.0

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """
        Acquire tokens for making requests.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout

        Raises:
            ValueError: If tokens > capacity
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds capacity {self.capacity}"
            )

        deadline = None if timeout is None else time.time() + timeout

        while True:
            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # Check timeout
            if deadline is not None and time.time() >= deadline:
                return False

            # Sleep before retry
            time.sleep(0.1)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time.

        If a server-issued penalty is active, the bucket stays empty
        until the penalty deadline passes; refill resumes from there.
        """
        now = time.time()
        if now < self._penalty_until:
            # Keep the bucket drained and defer the refill clock to
            # the penalty deadline, so refill starts fresh once the
            # server-specified window closes.
            self.tokens = 0.0
            self.last_update = self._penalty_until
            return

        elapsed = now - self.last_update
        if elapsed <= 0:
            self.last_update = now
            return

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)

        self.last_update = now

    def penalize(self, delay_seconds: float) -> None:
        """Honour a server-issued "retry after" signal.

        After this call, the bucket is drained and no caller can
        acquire a token until ``delay_seconds`` have elapsed. If an
        existing penalty is longer, the longer one wins — a late
        short signal cannot shorten an earlier long one.

        Args:
            delay_seconds: How long to block acquisition for. Zero is
                a no-op. Negative values are rejected as programmer
                error.
        """
        if delay_seconds < 0:
            raise ValueError(f"delay_seconds must be non-negative, got {delay_seconds}")
        if delay_seconds == 0:
            return
        with self.lock:
            new_deadline = time.time() + delay_seconds
            if new_deadline > self._penalty_until:
                self._penalty_until = new_deadline
            # Drain tokens so any reader in the same moment sees empty.
            self.tokens = 0.0

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return self.tokens

    async def acquire_async(
        self, tokens: int = 1, timeout: float | None = None
    ) -> bool:
        """Acquire tokens asynchronously without blocking the event loop.

        Drop-in async replacement for acquire(). Uses asyncio.sleep()
        instead of time.sleep() so other coroutines can run while waiting.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum wait time in seconds (None = wait forever).

        Returns:
            True if tokens acquired, False if timeout.

        Raises:
            ValueError: If tokens > capacity.
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds capacity {self.capacity}"
            )

        deadline = None if timeout is None else time.time() + timeout

        while True:
            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # Check timeout
            if deadline is not None and time.time() >= deadline:
                return False

            # Yield to event loop instead of blocking thread
            await asyncio.sleep(0.05)

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        with self.lock:
            self.tokens = float(self.capacity)
            self.last_update = time.time()
