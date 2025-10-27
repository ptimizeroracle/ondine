"""
Token bucket rate limiter for API calls.

Implements token bucket algorithm for rate limiting.
"""

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
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)

        self.last_update = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return self.tokens

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        with self.lock:
            self.tokens = float(self.capacity)
            self.last_update = time.time()
