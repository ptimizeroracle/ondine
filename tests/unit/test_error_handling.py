"""Unit tests for error handling."""

from ondine.core.error_handler import (
    ErrorAction,
    ErrorDecision,
    ErrorHandler,
)
from ondine.core.specifications import ErrorPolicy


class TestErrorHandler:
    """Test suite for ErrorHandler."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(
            policy=ErrorPolicy.RETRY,
            max_retries=3,
            default_value="fallback",
        )

        assert handler.policy == ErrorPolicy.RETRY
        assert handler.max_retries == 3
        assert handler.default_value == "fallback"

    def test_retry_policy(self):
        """Test RETRY error policy."""
        handler = ErrorHandler(policy=ErrorPolicy.RETRY, max_retries=3)
        error = ValueError("Test error")

        decision = handler.handle_error(error, context={"attempt": 1})

        assert decision.action == ErrorAction.RETRY
        assert decision.retry_count > 0

    def test_skip_policy(self):
        """Test SKIP error policy."""
        handler = ErrorHandler(policy=ErrorPolicy.SKIP)
        error = ValueError("Test error")

        decision = handler.handle_error(error, context={})

        assert decision.action == ErrorAction.SKIP
        assert decision.default_value is None

    def test_fail_policy(self):
        """Test FAIL error policy."""
        handler = ErrorHandler(policy=ErrorPolicy.FAIL)
        error = ValueError("Test error")

        decision = handler.handle_error(error, context={})

        assert decision.action == ErrorAction.FAIL

    def test_use_default_policy(self):
        """Test USE_DEFAULT error policy."""
        handler = ErrorHandler(
            policy=ErrorPolicy.USE_DEFAULT,
            default_value="fallback_value",
        )
        error = ValueError("Test error")

        decision = handler.handle_error(error, context={})

        assert decision.action == ErrorAction.USE_DEFAULT
        assert decision.default_value == "fallback_value"

    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        handler = ErrorHandler(policy=ErrorPolicy.RETRY, max_retries=3)
        error = ValueError("Test error")

        # Simulate max retries exceeded
        decision = handler.handle_error(error, context={"attempt": 4})

        # Should switch to FAIL after max retries
        assert decision.action in [ErrorAction.FAIL, ErrorAction.SKIP]

    def test_error_decision_attributes(self):
        """Test ErrorDecision attributes."""
        decision = ErrorDecision(
            action=ErrorAction.USE_DEFAULT,
            default_value="test",
            retry_count=2,
            context={"key": "value"},
        )

        assert decision.action == ErrorAction.USE_DEFAULT
        assert decision.default_value == "test"
        assert decision.retry_count == 2
        assert decision.context["key"] == "value"


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_retry_with_backoff(self):
        """Test retry with exponential backoff."""
        from ondine.utils import RetryHandler

        handler = RetryHandler(
            max_attempts=3,
            initial_delay=0.01,
            exponential_base=2,
            retryable_exceptions=(ValueError,),
        )

        attempt_count = 0

        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = handler.execute(flaky_function)

        assert result == "success"
        assert attempt_count == 3

    def test_skip_on_error(self):
        """Test skipping rows on error."""
        handler = ErrorHandler(policy=ErrorPolicy.SKIP)

        results = []
        errors = []

        for i in range(5):
            try:
                if i == 2:  # Simulate error on row 2
                    raise ValueError(f"Error on row {i}")
                results.append(f"Row {i}")
            except Exception as e:
                decision = handler.handle_error(e, context={"row": i})
                if decision.action == ErrorAction.SKIP:
                    errors.append(i)
                    continue

        assert len(results) == 4  # 4 successful rows
        assert len(errors) == 1  # 1 skipped row
        assert 2 in errors

    def test_use_default_on_error(self):
        """Test using default value on error."""
        handler = ErrorHandler(
            policy=ErrorPolicy.USE_DEFAULT,
            default_value="N/A",
        )

        results = []

        for i in range(5):
            try:
                if i == 2:
                    raise ValueError(f"Error on row {i}")
                results.append(f"Value {i}")
            except Exception as e:
                decision = handler.handle_error(e, context={"row": i})
                if decision.action == ErrorAction.USE_DEFAULT:
                    results.append(decision.default_value)

        assert len(results) == 5
        assert results[2] == "N/A"  # Default value used
        assert results[0] == "Value 0"  # Normal value
