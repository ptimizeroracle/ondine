"""Tests for ConcurrencyController."""

import asyncio
from unittest.mock import MagicMock

import pytest

from ondine.orchestration.concurrency_controller import ConcurrencyController


class TestConcurrencyController:
    """Test ConcurrencyController functionality."""

    def test_init_with_valid_concurrency(self):
        """Test initialization with valid concurrency."""
        controller = ConcurrencyController(max_concurrent=5)
        assert controller.max_concurrent == 5

    def test_init_with_invalid_concurrency(self):
        """Test initialization with invalid concurrency raises error."""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            ConcurrencyController(max_concurrent=0)

    def test_init_with_rate_limiter(self):
        """Test initialization with rate limiter."""
        mock_limiter = MagicMock()
        controller = ConcurrencyController(max_concurrent=5, rate_limiter=mock_limiter)
        assert controller._rate_limiter is mock_limiter

    async def test_throttle_respects_concurrency_limit(self):
        """Test that throttle respects max concurrent limit."""
        controller = ConcurrencyController(max_concurrent=2)
        active_count = 0
        max_active = 0

        async def task():
            nonlocal active_count, max_active
            async with controller.throttle():
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1

        # Run 10 tasks with max 2 concurrent
        await asyncio.gather(*[task() for _ in range(10)])

        # Should never exceed 2 concurrent
        assert max_active <= 2

    async def test_throttle_releases_on_exception(self):
        """Test that throttle releases slot on exception."""
        controller = ConcurrencyController(max_concurrent=1)

        async def failing_task():
            async with controller.throttle():
                raise ValueError("Test error")

        # First task fails
        with pytest.raises(ValueError):
            await failing_task()

        # Second task should still be able to acquire
        acquired = False
        async with controller.throttle():
            acquired = True

        assert acquired

    async def test_acquire_release_manual(self):
        """Test manual acquire and release."""
        controller = ConcurrencyController(max_concurrent=1)

        await controller.acquire()
        # Should be able to release
        controller.release()

        # Should be able to acquire again
        await controller.acquire()
        controller.release()

    async def test_rate_limiter_called(self):
        """Test that rate limiter is called during acquire."""
        mock_limiter = MagicMock()
        controller = ConcurrencyController(max_concurrent=5, rate_limiter=mock_limiter)

        await controller.acquire()
        controller.release()

        mock_limiter.acquire.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        controller = ConcurrencyController(max_concurrent=5)
        repr_str = repr(controller)
        assert "ConcurrencyController" in repr_str
        assert "max_concurrent=5" in repr_str
        assert "rate_limiter=False" in repr_str

        mock_limiter = MagicMock()
        controller_with_limiter = ConcurrencyController(
            max_concurrent=10, rate_limiter=mock_limiter
        )
        repr_str = repr(controller_with_limiter)
        assert "rate_limiter=True" in repr_str
