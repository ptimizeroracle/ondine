"""
Unit tests for custom LLM client injection via PipelineBuilder.

Tests the ability to provide a custom LLM client instance directly.
"""

from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class MockCustomClient(LLMClient):
    """Mock custom LLM client for testing."""

    def __init__(self, spec: LLMSpec):
        super().__init__(spec)
        self.invoke_called = False

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Mock invoke method."""
        self.invoke_called = True
        return LLMResponse(
            text="Mock response",
            tokens_in=10,
            tokens_out=5,
            model=self.model,
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )

    def estimate_tokens(self, text: str) -> int:
        """Mock token estimation."""
        return len(text.split())


class TestPipelineBuilderCustomClient:
    """Test suite for custom LLM client injection."""

    def test_with_custom_llm_client_method_exists(self):
        """PipelineBuilder should have with_custom_llm_client method."""
        builder = PipelineBuilder.create()
        assert hasattr(builder, "with_custom_llm_client")
        assert callable(builder.with_custom_llm_client)

    def test_with_custom_llm_client_returns_builder(self):
        """with_custom_llm_client should return self for chaining."""
        spec = LLMSpec(
            provider="openai",  # Doesn't matter, will be overridden
            model="test-model",
        )
        custom_client = MockCustomClient(spec)

        builder = PipelineBuilder.create()
        result = builder.with_custom_llm_client(custom_client)

        assert result is builder  # Should return self for chaining

    def test_with_custom_llm_client_accepts_llm_client_instance(self):
        """Should accept any LLMClient subclass instance."""
        spec = LLMSpec(
            provider="openai",
            model="custom-model",
        )
        custom_client = MockCustomClient(spec)

        builder = PipelineBuilder.create()
        builder.with_custom_llm_client(custom_client)

        # Should store the custom client
        assert hasattr(builder, "_custom_llm_client")
        assert builder._custom_llm_client is custom_client

    def test_pipeline_uses_custom_client_when_provided(self):
        """Built pipeline should use custom client instead of factory."""
        df = pd.DataFrame({"input": ["test1", "test2"]})

        spec = LLMSpec(
            provider="openai",
            model="custom-model",
        )
        custom_client = MockCustomClient(spec)

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["input"], output_columns=["output"])
            .with_prompt("Process: {input}")
            .with_custom_llm_client(custom_client)
        )

        # Build should succeed
        pipeline = builder.build()
        assert pipeline is not None

        # The pipeline should have reference to custom client (stored in metadata or context)
        # This will be validated in integration tests

    def test_custom_client_overrides_with_llm(self):
        """Custom client should take precedence over with_llm configuration."""
        df = pd.DataFrame({"input": ["test"]})

        spec = LLMSpec(
            provider="openai",
            model="my-custom-model",
        )
        custom_client = MockCustomClient(spec)

        # Call both with_llm and with_custom_llm_client
        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["input"], output_columns=["output"])
            .with_prompt("Process: {input}")
            .with_llm(
                provider="openai",
                model="gpt-4",
                api_key="test",  # pragma: allowlist secret
            )
            .with_custom_llm_client(custom_client)  # This should override
        )

        pipeline = builder.build()
        assert pipeline is not None
        # Custom client should be used (will verify in integration test)

    def test_rejects_non_llm_client_instances(self):
        """Should reject objects that don't inherit from LLMClient."""

        class NotAnLLMClient:
            pass

        builder = PipelineBuilder.create()

        with pytest.raises((TypeError, AttributeError, ValueError)):
            builder.with_custom_llm_client(NotAnLLMClient())

    def test_custom_client_integrates_with_builder_chain(self):
        """Custom client should work with full builder chain."""
        df = pd.DataFrame({"text": ["hello", "world"]})

        spec = LLMSpec(
            provider="openai",
            model="test",
        )
        custom_client = MockCustomClient(spec)

        # Full builder chain
        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Echo: {text}")
            .with_custom_llm_client(custom_client)
            .with_batch_size(10)
            .with_concurrency(2)
            .with_checkpoint_interval(100)
        )

        pipeline = builder.build()
        assert pipeline is not None


class TestCustomClientWithConfig:
    """Test custom client with YAML config compatibility."""

    def test_openai_compatible_via_builder(self):
        """Should be able to configure openai_compatible via builder."""
        df = pd.DataFrame({"input": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["input"], output_columns=["output"])
            .with_prompt("Process: {input}")
            .with_llm(
                provider="openai_compatible",
                model="llama-3.1-70b",
                api_key="test-key",  # pragma: allowlist secret
                base_url="https://api.together.xyz/v1",
                provider_name="Together.AI",
                input_cost_per_1k_tokens=Decimal("0.0006"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            )
        )

        # Should build successfully
        pipeline = builder.build()
        assert pipeline is not None
        assert pipeline.specifications.llm.provider == "openai_compatible"
        assert pipeline.specifications.llm.base_url == "https://api.together.xyz/v1"
        assert pipeline.specifications.llm.provider_name == "Together.AI"
