"""Unit tests for RetryHandler."""

import pytest

from ondine.utils import (
    NetworkError,
    RateLimitError,
    RetryHandler,
)


class TestRetryHandler:
    """Test suite for RetryHandler."""

    def test_successful_execution(self):
        """Test successful execution without retries."""
        handler = RetryHandler(max_attempts=3)

        call_count = 0

        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.execute(successful_func)

        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Test retry on retriable errors."""
        handler = RetryHandler(max_attempts=3, initial_delay=0.01)

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limit")
            return "success"

        result = handler.execute(flaky_func)

        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that max retries limit is respected."""
        handler = RetryHandler(max_attempts=2, initial_delay=0.01)

        def always_fails():
            raise NetworkError("Network error")

        with pytest.raises(NetworkError):
            handler.execute(always_fails)

    def test_non_retriable_error(self):
        """Test that non-retriable errors fail immediately."""
        handler = RetryHandler(max_attempts=3)

        def fails_with_value_error():
            raise ValueError("Not retriable")

        with pytest.raises(ValueError):
            handler.execute(fails_with_value_error)

    def test_delay_calculation(self):
        """Test exponential backoff calculation."""
        handler = RetryHandler(
            max_attempts=5,
            initial_delay=1.0,
            exponential_base=2,
            max_delay=32.0,
        )

        assert handler.calculate_delay(1) == 1.0
        assert handler.calculate_delay(2) == 2.0
        assert handler.calculate_delay(3) == 4.0
        assert handler.calculate_delay(4) == 8.0
        assert handler.calculate_delay(5) == 16.0
        assert handler.calculate_delay(6) == 32.0  # Max delay
        assert handler.calculate_delay(7) == 32.0  # Capped at max
