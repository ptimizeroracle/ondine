"""Unit tests for RateLimiter."""

import time

import pytest

from ondine.utils import RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60)

        assert limiter.rpm == 60
        assert limiter.capacity == 60
        assert limiter.refill_rate == 1.0  # 60/60 = 1 token/sec

    def test_acquire_tokens(self):
        """Test acquiring tokens."""
        limiter = RateLimiter(requests_per_minute=60)

        # Should succeed immediately
        success = limiter.acquire(tokens=1, timeout=1.0)
        assert success is True

    def test_multiple_acquires(self):
        """Test multiple token acquisitions."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Acquire 5 tokens
        for _ in range(5):
            assert limiter.acquire(tokens=1, timeout=0.1) is True

        # Should still have tokens
        assert limiter.available_tokens > 0

    def test_refill_over_time(self):
        """Test token refill mechanism."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Use all tokens
        limiter.acquire(tokens=10)
        assert limiter.available_tokens < 1

        # Wait for refill
        time.sleep(1.1)  # Should refill ~1 token

        # Should have refilled
        assert limiter.available_tokens >= 0.9

    def test_timeout(self):
        """Test timeout when tokens unavailable."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Use all tokens
        limiter.acquire(tokens=5)

        # Try to acquire more with short timeout
        success = limiter.acquire(tokens=5, timeout=0.1)
        assert success is False

    def test_reset(self):
        """Test resetting rate limiter."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Use tokens
        limiter.acquire(tokens=5)
        assert limiter.available_tokens < 10

        # Reset
        limiter.reset()
        assert limiter.available_tokens == 10.0

    def test_exceeds_capacity(self):
        """Test requesting more tokens than capacity."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        with pytest.raises(ValueError, match="exceeds capacity"):
            limiter.acquire(tokens=20)
