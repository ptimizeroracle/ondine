"""
Unit tests for MLX client integration.

Tests MLX client for Apple Silicon local inference using Dependency Injection.
"""

import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from ondine.adapters.llm_client import MLXClient, create_llm_client
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMProvider, LLMSpec


class TestMLXClient:
    """Test suite for MLX client using dependency injection."""

    def test_raises_error_if_mlx_not_installed(self):
        """Should raise helpful error if MLX not installed."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        # Don't inject module - should try to import and fail
        # This test actually needs mlx_lm to NOT be importable
        # Skip in environments where mlx_lm is installed
        try:
            import mlx_lm  # noqa: F401

            pytest.skip("mlx_lm is installed, cannot test ImportError")
        except ImportError:
            # Good - mlx_lm not installed, test will work
            with pytest.raises(ImportError, match="MLX not installed"):
                MLXClient(spec)

    def test_loads_model_on_initialization(self):
        """Should load model once during initialization."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        # Use dependency injection (clean!)
        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)

        # Should load model once
        mock_mlx_lm.load.assert_called_once_with("mlx-community/Qwen3-1.7B-4bit")
        assert client.mlx_model is mock_model
        assert client.mlx_tokenizer is mock_tokenizer

    def test_caches_model_instance(self):
        """Should cache model and not reload on each invoke."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "Test response"

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)

        # Call invoke multiple times
        client.invoke("test1")
        client.invoke("test2")
        client.invoke("test3")

        # Model should only be loaded ONCE (in __init__)
        assert mock_mlx_lm.load.call_count == 1
        # But generate should be called 3 times
        assert mock_mlx_lm.generate.call_count == 3

    def test_invoke_returns_llm_response(self):
        """Should invoke MLX and return LLMResponse."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
            input_cost_per_1k_tokens=Decimal("0.0"),
            output_cost_per_1k_tokens=Decimal("0.0"),
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "The capital of France is Paris."

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        response = client.invoke("What is the capital of France?")

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text == "The capital of France is Paris."
        assert response.tokens_in == 5
        assert response.tokens_out == 5
        assert "MLX" in response.model
        assert response.cost == Decimal("0.0")  # Free local model
        assert response.latency_ms >= 0

    def test_estimate_tokens_uses_tokenizer(self):
        """Should estimate tokens using MLX tokenizer if available."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]  # 7 tokens
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        tokens = client.estimate_tokens("Hello, world!")

        assert tokens == 7
        mock_tokenizer.encode.assert_called_once()

    def test_estimate_tokens_fallback_on_error(self):
        """Should fallback to word count if tokenizer fails."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        tokens = client.estimate_tokens("Hello world test")

        # Should fallback to word count: 3 words
        assert tokens == 3

    def test_cost_tracking_for_local_model(self):
        """Should track cost as $0 for local models."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
            input_cost_per_1k_tokens=Decimal("0.0"),
            output_cost_per_1k_tokens=Decimal("0.0"),
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        cost = client.calculate_cost(tokens_in=1000, tokens_out=500)

        assert cost == Decimal("0.0")

    def test_handles_max_tokens_parameter(self):
        """Should pass max_tokens to MLX generate."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
            max_tokens=100,
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "Test response"

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        client.invoke("test prompt")

        # Verify max_tokens passed to generate
        mock_mlx_lm.generate.assert_called_once()
        call_kwargs = mock_mlx_lm.generate.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 100

    def test_includes_model_name_in_response(self):
        """Should include MLX and model name in response metadata."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "Response"

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        response = client.invoke("test")

        # Should identify as MLX in model field
        assert "MLX" in response.model

    def test_handles_verbose_parameter(self):
        """Should pass verbose=False to MLX generate to suppress output."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "Response"

        client = MLXClient(spec, _mlx_lm_module=mock_mlx_lm)
        client.invoke("test")

        # Should pass verbose=False
        call_kwargs = mock_mlx_lm.generate.call_args.kwargs
        assert call_kwargs.get("verbose") is False


class TestMLXClientFactory:
    """Test factory function with MLX provider."""

    def test_factory_creates_mlx_client(self):
        """Factory should create MLXClient for MLX provider."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="mlx-community/Qwen3-1.7B-4bit",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        # For factory, we need to patch at import time
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx_lm}):
            client = create_llm_client(spec)
            assert isinstance(client, MLXClient)

    def test_factory_backward_compatible(self):
        """Factory should still work with existing providers."""
        from ondine.adapters.llm_client import GroqClient

        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="test",  # pragma: allowlist secret
        )

        with patch("ondine.adapters.llm_client.Groq"):
            client = create_llm_client(spec)
            assert isinstance(client, GroqClient)


class TestMLXClientErrorHandling:
    """Test MLX client error scenarios."""

    def test_handles_model_load_failure(self):
        """Should raise clear error if model loading fails."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="invalid-model-name",
        )

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Failed to load MLX model"):
            MLXClient(spec, _mlx_lm_module=mock_mlx_lm)

    def test_provides_helpful_error_message(self):
        """Error message should include model name and guidance."""
        spec = LLMSpec(
            provider=LLMProvider.MLX,
            model="non-existent/model",
        )

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = Exception("404 Not Found")

        with pytest.raises(Exception) as exc_info:
            MLXClient(spec, _mlx_lm_module=mock_mlx_lm)

        error_msg = str(exc_info.value)
        assert "non-existent/model" in error_msg
        assert "HuggingFace" in error_msg or "access" in error_msg
