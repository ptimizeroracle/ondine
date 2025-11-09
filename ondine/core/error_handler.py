"""
Error handling system with configurable policies.

Implements Strategy pattern for different error handling approaches.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ondine.core.exceptions import NonRetryableError
from ondine.core.specifications import ErrorPolicy
from ondine.utils import get_logger

logger = get_logger(__name__)


class ErrorAction(str, Enum):
    """Actions to take on errors."""

    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    USE_DEFAULT = "use_default"


@dataclass
class ErrorDecision:
    """Decision on how to handle an error."""

    action: ErrorAction
    default_value: Any = None
    retry_count: int = 0
    context: dict[str, Any] | None = None


class ErrorHandler:
    """
    Policy-based error handling (orchestrates retry/skip/fail decisions).

    Scope: Stage execution errors and pipeline-level error handling
    Policies: SKIP, FAIL, RETRY (delegates to RetryHandler for execution)
    Use when: Configuring how the pipeline handles errors

    Policy Behaviors:
    - SKIP: Log error and skip the row (continue processing)
    - FAIL: Raise error and stop pipeline
    - RETRY: Retry the operation (delegates to RetryHandler)
    - DEFAULT: Return a default value on error

    Example:
        handler = ErrorHandler(policy=ErrorPolicy.RETRY, max_retries=3)
        decision = handler.handle_error(exception, context)

    See Also:
    - RetryHandler: Executes the actual retry logic
    - Pipeline._auto_retry_failed_rows(): Row-level quality retry
    - docs/architecture/decisions/ADR-006-retry-levels.md

    Design Note:
        ErrorHandler decides WHAT to do (policy)
        RetryHandler decides HOW to do it (exponential backoff)
    """

    def __init__(
        self,
        policy: ErrorPolicy = ErrorPolicy.SKIP,
        max_retries: int = 3,
        default_value: Any = None,
        default_value_factory: Callable[[], Any] | None = None,
    ):
        """
        Initialize error handler.

        Args:
            policy: Error handling policy
            max_retries: Maximum retry attempts
            default_value: Static default value (or use default_value_factory)
            default_value_factory: Function to generate default values
        """
        self.policy = policy
        self.max_retries = max_retries
        self.default_value = default_value

        # If default_value_factory is provided, use it; otherwise use lambda returning default_value
        if default_value_factory is not None:
            self.default_value_factory = default_value_factory
        else:
            self.default_value_factory = lambda: default_value

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any],
        attempt: int = 1,
    ) -> ErrorDecision:
        """
        Decide how to handle an error.

        Args:
            error: The exception that occurred
            context: Error context (row_index, stage, etc.)
            attempt: Current attempt number

        Returns:
            ErrorDecision with action to take
        """
        # Get attempt from context if available, otherwise use parameter
        attempt = context.get("attempt", attempt)
        row_index = context.get("row_index", "unknown")
        stage = context.get("stage", "unknown")

        # Log the error
        logger.error(
            f"Error in {stage} at row {row_index}: {error}",
            exc_info=True,
        )

        # Check for NonRetryableError first (fail fast)
        if isinstance(error, NonRetryableError):
            logger.error(
                f"❌ NON-RETRYABLE ERROR: {error}\n"
                f"   Type: {type(error).__name__}\n"
                f"   This error cannot be recovered. Pipeline will terminate.\n"
                f"   Please fix the configuration and try again."
            )
            return ErrorDecision(
                action=ErrorAction.FAIL,
                context=context,
            )

        # Check for FATAL errors that should always fail immediately (legacy)
        if self._is_fatal_error(error):
            logger.error(
                f"❌ FATAL ERROR: {error}\n"
                f"   This error cannot be recovered. Pipeline will terminate."
            )
            return ErrorDecision(
                action=ErrorAction.FAIL,
                context=context,
            )

        # Apply policy
        if self.policy == ErrorPolicy.RETRY:
            if attempt < self.max_retries:
                logger.info(f"Retrying (attempt {attempt + 1}/{self.max_retries})")
                return ErrorDecision(
                    action=ErrorAction.RETRY,
                    retry_count=attempt + 1,
                    context=context,
                )
            logger.warning(f"Max retries ({self.max_retries}) exceeded, skipping")
            return ErrorDecision(
                action=ErrorAction.SKIP,
                context=context,
            )

        if self.policy == ErrorPolicy.SKIP:
            logger.info(f"Skipping row {row_index} due to error")
            return ErrorDecision(
                action=ErrorAction.SKIP,
                context=context,
            )

        if self.policy == ErrorPolicy.USE_DEFAULT:
            default = self.default_value_factory()
            logger.info(f"Using default value for row {row_index}: {default}")
            return ErrorDecision(
                action=ErrorAction.USE_DEFAULT,
                default_value=default,
                context=context,
            )

        if self.policy == ErrorPolicy.FAIL:
            logger.error("Failing pipeline due to error")
            return ErrorDecision(
                action=ErrorAction.FAIL,
                context=context,
            )

        # Unknown policy, default to fail
        return ErrorDecision(
            action=ErrorAction.FAIL,
            context=context,
        )

    def _is_fatal_error(self, error: Exception) -> bool:
        """
        Determine if error is fatal and should always fail immediately.

        Fatal errors are configuration/authentication issues that cannot
        be recovered by retrying or skipping rows.

        Args:
            error: The exception

        Returns:
            True if error is fatal
        """
        error_str = str(error).lower()

        # Fatal error patterns
        fatal_patterns = [
            "invalid api key",
            "invalid_api_key",
            "authentication failed",
            "401",
            "403",  # Forbidden
            "api key not found",
            "unauthorized",
            "invalid credentials",
            "permission denied",
        ]

        return any(pattern in error_str for pattern in fatal_patterns)

    def should_retry(self, error: Exception) -> bool:
        """
        Determine if error should be retried.

        Args:
            error: The exception

        Returns:
            True if retriable
        """
        # Don't retry fatal errors
        if self._is_fatal_error(error):
            return False

        retriable_keywords = [
            "rate limit",
            "timeout",
            "network",
            "connection",
            "503",
            "502",
            "429",
        ]

        error_str = str(error).lower()
        return any(keyword in error_str for keyword in retriable_keywords)
