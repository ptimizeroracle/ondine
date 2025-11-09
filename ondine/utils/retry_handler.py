"""
Retry handling with exponential backoff.

Provides robust retry logic for transient failures.
"""

from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should be retried."""

    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded error."""

    pass


class NetworkError(RetryableError):
    """Network-related error."""

    pass


class RetryHandler:
    """
    Request-level retry with exponential backoff (for transient errors).

    Scope: Single LLM API call or operation
    Use when: Transient errors (rate limits, network timeouts, API hiccups)
    NOT for: Row-level quality issues (use Pipeline.auto_retry_failed for that)

    Retry Strategy:
    - Exponential backoff (1s, 2s, 4s, 8s, ...)
    - Configurable max attempts (default: 3)
    - Only retries specific exception types

    Example:
        handler = RetryHandler(max_attempts=3, initial_delay=1.0)
        result = handler.execute(lambda: call_llm_api())

    See Also:
    - ErrorHandler: Orchestrates retry decisions based on policy
    - Pipeline._auto_retry_failed_rows(): Row-level quality retry
    - docs/architecture/decisions/ADR-006-retry-levels.md
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: int = 2,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """
        Initialize retry handler.

        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            retryable_exceptions: Exception types to retry
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

        if retryable_exceptions is None:
            self.retryable_exceptions = (
                RetryableError,
                RateLimitError,
                NetworkError,
            )
        else:
            self.retryable_exceptions = retryable_exceptions

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute function with retry logic.

        Only retries exceptions in self.retryable_exceptions tuple.
        NonRetryableError and its subclasses are re-raised immediately.

        Args:
            func: Function to execute

        Returns:
            Result from function

        Raises:
            NonRetryableError: Fatal errors (model not found, invalid API key, etc.)
            Exception: If all retries exhausted for retryable errors
        """
        retryer = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.initial_delay,
                max=self.max_delay,
                exp_base=self.exponential_base,
            ),
            retry=retry_if_exception_type(self.retryable_exceptions),
            reraise=True,
        )

        return retryer(func)

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        return min(delay, self.max_delay)
