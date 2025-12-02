"""Tests for non-retryable error classification and handling."""

from unittest.mock import MagicMock

import pytest

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.exceptions import (
    ConfigurationError,
    InvalidAPIKeyError,
    ModelNotFoundError,
    NonRetryableError,
    QuotaExceededError,
)
from ondine.core.specifications import LLMSpec
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


class TestClientErrorMapping:
    """Test error mapping in UnifiedLiteLLMClient."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        spec = LLMSpec(model="gpt-4", api_key="test")
        return UnifiedLiteLLMClient(spec)

    def test_map_model_decommissioned_error(self, client):
        """Model decommissioned error should be mapped to ModelNotFoundError."""
        error = Exception("The model `llama-3.1-70b-versatile` has been decommissioned")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, ModelNotFoundError)
        assert "model error" in str(mapped).lower()

    def test_map_model_not_found_error(self, client):
        """Model not found error should be mapped to ModelNotFoundError."""
        error = Exception("Model 'gpt-5' does not exist")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, ModelNotFoundError)

    def test_map_invalid_api_key_error(self, client):
        """Invalid API key should be mapped to InvalidAPIKeyError."""
        error = Exception("Invalid API key provided")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, InvalidAPIKeyError)

    def test_map_authentication_401_error(self, client):
        """401 authentication error should be mapped to InvalidAPIKeyError."""
        error = Exception("HTTP 401 Unauthorized")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, InvalidAPIKeyError)

    def test_map_quota_exceeded_error(self, client):
        """Quota exceeded should be mapped to QuotaExceededError."""
        error = Exception("Quota exceeded for this account")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, QuotaExceededError)

    def test_map_limit_exceeded_error(self, client):
        """Cerebras limit exceeded should be mapped to QuotaExceededError."""
        error = Exception("Tokens per hour limit exceeded")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, QuotaExceededError)

    def test_map_rate_limit_error(self, client):
        """Rate limit error should be mapped to RateLimitError (retryable)."""
        error = Exception("Rate limit exceeded, please retry after 60s")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, RateLimitError)
        assert not isinstance(mapped, NonRetryableError)

    def test_map_429_error(self, client):
        """HTTP 429 should be mapped to RateLimitError (retryable)."""
        error = Exception("HTTP 429 Too Many Requests")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, RateLimitError)

    def test_map_network_timeout_error(self, client):
        """Network timeout should be mapped to NetworkError (retryable)."""
        error = Exception("Connection timeout after 30s")
        mapped = client._map_provider_error(error)

        assert isinstance(mapped, NetworkError)
        assert not isinstance(mapped, NonRetryableError)

    def test_map_unknown_error_returns_original(self, client):
        """Unknown errors should return original exception."""
        error = Exception("Some random error")
        mapped = client._map_provider_error(error)

        assert mapped is error
        assert not isinstance(mapped, NonRetryableError)


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


class TestIntegration:
    """Integration tests for stage-level error handling."""

    def test_model_error_fails_pipeline_immediately(self):
        """Model error (mapped by client) should fail pipeline without retries."""
        # Mock LLM client that raises mapped ModelNotFoundError
        mock_client = MagicMock()
        mock_client.router = None
        # Client explicitly raises the Mapped Exception now
        mock_client.invoke.side_effect = ModelNotFoundError("Model decommissioned")

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
        """Rate limit error (mapped by client) should retry multiple times."""
        mock_client = MagicMock()
        # Client explicitly raises the Mapped Exception
        mock_client.invoke.side_effect = RateLimitError("Rate limit exceeded")

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
