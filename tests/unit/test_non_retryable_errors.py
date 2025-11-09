"""Tests for non-retryable error classification and handling."""

import pytest

from ondine.core.exceptions import (
    ConfigurationError,
    InvalidAPIKeyError,
    ModelNotFoundError,
    NonRetryableError,
    QuotaExceededError,
)
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.utils import NetworkError, RateLimitError, RetryHandler


class TestNonRetryableErrorHierarchy:
    """Test exception hierarchy."""

    def test_non_retryable_error_is_exception(self):
        """NonRetryableError should be an Exception."""
        error = NonRetryableError("test")
        assert isinstance(error, Exception)

    def test_model_not_found_is_non_retryable(self):
        """ModelNotFoundError should be a NonRetryableError."""
        error = ModelNotFoundError("model not found")
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, Exception)

    def test_invalid_api_key_is_non_retryable(self):
        """InvalidAPIKeyError should be a NonRetryableError."""
        error = InvalidAPIKeyError("invalid key")
        assert isinstance(error, NonRetryableError)

    def test_configuration_error_is_non_retryable(self):
        """ConfigurationError should be a NonRetryableError."""
        error = ConfigurationError("config error")
        assert isinstance(error, NonRetryableError)

    def test_quota_exceeded_is_non_retryable(self):
        """QuotaExceededError should be a NonRetryableError."""
        error = QuotaExceededError("quota exceeded")
        assert isinstance(error, NonRetryableError)


class TestErrorClassification:
    """Test error classification in LLMInvocationStage."""

    @pytest.fixture
    def mock_stage(self):
        """Create a mock LLM invocation stage for testing."""
        from unittest.mock import MagicMock

        return LLMInvocationStage(
            llm_client=MagicMock(),
            retry_handler=MagicMock(),
            error_policy="skip",
        )

    def test_classify_model_decommissioned_error(self, mock_stage):
        """Model decommissioned error should be classified as ModelNotFoundError."""
        error = Exception("The model `llama-3.1-70b-versatile` has been decommissioned")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, ModelNotFoundError)
        assert "model error" in str(classified).lower()

    def test_classify_model_not_found_error(self, mock_stage):
        """Model not found error should be classified as ModelNotFoundError."""
        error = Exception("Model 'gpt-5' does not exist")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, ModelNotFoundError)

    def test_classify_invalid_api_key_error(self, mock_stage):
        """Invalid API key should be classified as InvalidAPIKeyError."""
        error = Exception("Invalid API key provided")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, InvalidAPIKeyError)

    def test_classify_authentication_401_error(self, mock_stage):
        """401 authentication error should be classified as InvalidAPIKeyError."""
        error = Exception("HTTP 401 Unauthorized")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, InvalidAPIKeyError)

    def test_classify_quota_exceeded_error(self, mock_stage):
        """Quota exceeded should be classified as QuotaExceededError."""
        error = Exception("Quota exceeded for this account")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, QuotaExceededError)

    def test_classify_rate_limit_error(self, mock_stage):
        """Rate limit error should be classified as RateLimitError (retryable)."""
        error = Exception("Rate limit exceeded, please retry after 60s")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, RateLimitError)
        assert not isinstance(classified, NonRetryableError)

    def test_classify_429_error(self, mock_stage):
        """HTTP 429 should be classified as RateLimitError (retryable)."""
        error = Exception("HTTP 429 Too Many Requests")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, RateLimitError)

    def test_classify_network_timeout_error(self, mock_stage):
        """Network timeout should be classified as NetworkError (retryable)."""
        error = Exception("Connection timeout after 30s")
        classified = mock_stage._classify_error(error)

        assert isinstance(classified, NetworkError)
        assert not isinstance(classified, NonRetryableError)

    def test_classify_unknown_error_returns_original(self, mock_stage):
        """Unknown errors should return original exception (retryable by default)."""
        error = Exception("Some random error")
        classified = mock_stage._classify_error(error)

        assert classified is error
        assert not isinstance(classified, NonRetryableError)

    def test_classify_openai_auth_error(self, mock_stage):
        """OpenAI AuthenticationError should be classified as InvalidAPIKeyError."""
        try:
            from unittest.mock import MagicMock

            from openai import AuthenticationError

            # Mock the OpenAI exception with required attributes
            mock_response = MagicMock()
            mock_response.status_code = 401
            error = AuthenticationError(
                "Invalid API key", response=mock_response, body={"error": "invalid_key"}
            )
            classified = mock_stage._classify_error(error)

            assert isinstance(classified, InvalidAPIKeyError)
            assert "openai authentication failed" in str(classified).lower()
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_classify_openai_bad_request_model_error(self, mock_stage):
        """OpenAI BadRequestError with model issue should be ModelNotFoundError."""
        try:
            from unittest.mock import MagicMock

            from openai import BadRequestError

            # Mock the OpenAI exception with required attributes
            mock_response = MagicMock()
            mock_response.status_code = 400
            error = BadRequestError(
                "The model 'gpt-5' does not exist",
                response=mock_response,
                body={"error": "model_not_found"},
            )
            classified = mock_stage._classify_error(error)

            assert isinstance(classified, ModelNotFoundError)
        except ImportError:
            pytest.skip("OpenAI not installed")


class TestRetryHandlerWithNonRetryable:
    """Test RetryHandler behavior with NonRetryableError."""

    def test_non_retryable_error_not_retried(self):
        """NonRetryableError should be raised immediately without retry."""
        handler = RetryHandler(max_attempts=3)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ModelNotFoundError("Model not found")

        with pytest.raises(ModelNotFoundError):
            handler.execute(failing_func)

        # Should only be called once (no retries)
        assert call_count == 1

    def test_retryable_error_is_retried(self):
        """RateLimitError should be retried multiple times."""
        handler = RetryHandler(max_attempts=3, initial_delay=0.01)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Rate limit exceeded")

        with pytest.raises(RateLimitError):
            handler.execute(failing_func)

        # Should be called 3 times (initial + 2 retries)
        assert call_count == 3

    def test_network_error_is_retried(self):
        """NetworkError should be retried multiple times."""
        handler = RetryHandler(max_attempts=3, initial_delay=0.01)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Connection timeout")

        with pytest.raises(NetworkError):
            handler.execute(failing_func)

        # Should be called 3 times
        assert call_count == 3

    def test_generic_exception_not_retried_by_default(self):
        """Generic exceptions should not be retried (only RetryableError types)."""
        handler = RetryHandler(max_attempts=3)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Some error")

        with pytest.raises(ValueError):
            handler.execute(failing_func)

        # Should only be called once (ValueError not in retryable_exceptions)
        assert call_count == 1


class TestErrorHandlerWithNonRetryable:
    """Test ErrorHandler behavior with NonRetryableError."""

    def test_non_retryable_error_always_fails(self):
        """NonRetryableError should always result in FAIL action."""
        from ondine.core.error_handler import ErrorAction, ErrorHandler

        handler = ErrorHandler(policy="retry", max_retries=3)
        error = ModelNotFoundError("Model decommissioned")
        context = {"row_index": 0, "stage": "llm_invocation"}

        decision = handler.handle_error(error, context, attempt=1)

        assert decision.action == ErrorAction.FAIL

    def test_non_retryable_ignores_policy(self):
        """NonRetryableError should fail even with SKIP policy."""
        from ondine.core.error_handler import ErrorAction, ErrorHandler

        handler = ErrorHandler(policy="skip")
        error = InvalidAPIKeyError("Invalid API key")
        context = {"row_index": 0, "stage": "llm_invocation"}

        decision = handler.handle_error(error, context)

        # Should FAIL, not SKIP
        assert decision.action == ErrorAction.FAIL

    def test_retryable_error_respects_policy(self):
        """Retryable errors should respect the configured policy."""
        from ondine.core.error_handler import ErrorAction, ErrorHandler

        handler = ErrorHandler(policy="skip")
        error = RateLimitError("Rate limit")
        context = {"row_index": 0, "stage": "llm_invocation"}

        decision = handler.handle_error(error, context)

        # Should respect SKIP policy
        assert decision.action == ErrorAction.SKIP


class TestIntegration:
    """Integration tests for error classification flow."""

    def test_model_error_fails_pipeline_immediately(self):
        """Model error should fail pipeline without retries."""
        from unittest.mock import MagicMock

        # Mock LLM client that raises model error
        mock_client = MagicMock()
        mock_client.invoke.side_effect = Exception(
            "The model `invalid-model` has been decommissioned"
        )

        # Create stage with retry handler
        retry_handler = RetryHandler(max_attempts=3, initial_delay=0.01)

        stage = LLMInvocationStage(
            llm_client=mock_client,
            retry_handler=retry_handler,
            error_policy="retry",
            max_retries=3,
        )

        # Should raise ModelNotFoundError after 1 attempt
        with pytest.raises(ModelNotFoundError):
            stage._invoke_with_retry_and_ratelimit("test prompt", row_index=0)

        # Verify only called once (no retries)
        assert mock_client.invoke.call_count == 1

    def test_rate_limit_retries_then_fails(self):
        """Rate limit error should retry multiple times."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.invoke.side_effect = Exception("Rate limit exceeded")

        retry_handler = RetryHandler(max_attempts=3, initial_delay=0.01)

        stage = LLMInvocationStage(
            llm_client=mock_client,
            retry_handler=retry_handler,
            error_policy="retry",
        )

        # Should raise RateLimitError after 3 attempts
        with pytest.raises(RateLimitError):
            stage._invoke_with_retry_and_ratelimit("test prompt", row_index=0)

        # Verify called 3 times (retries)
        assert mock_client.invoke.call_count == 3
