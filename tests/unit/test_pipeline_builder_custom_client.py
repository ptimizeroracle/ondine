"""
Unit tests for custom LLM client injection via PipelineBuilder.

Tests the ability to provide a custom LLM client instance directly.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import patch

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

    def structured_invoke(self, prompt: str, output_cls, **kwargs: Any) -> LLMResponse:
        """Mock structured invoke (required by abstract base)."""
        result = output_cls.model_validate({"field1": "test", "field2": 123})
        return LLMResponse(
            text=result.model_dump_json(),
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


class TestCustomClientIsActuallyUsed:
    """The builder stored the client; nothing ever used it (#230).

    Every test above asserts wiring — that the method exists, returns the
    builder, records the attribute. None ran a pipeline, so a no-op API passed
    CI for ondine's entire public life: the builder kept the client in
    metadata and both execution paths built a real client from the spec
    instead, calling the provider the caller was trying to replace.

    These tests assert the client is *invoked*, which is the only claim that
    matters and the only one the old tests could not make.
    """

    class _RecordingClient(LLMClient):
        """Answers every prompt and remembers it was asked."""

        def __init__(self, spec: LLMSpec):
            super().__init__(spec)
            self.prompts: list[str] = []
            self.token_estimates = 0

        def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
            self.prompts.append(prompt)
            return LLMResponse(
                text="from-custom-client",
                tokens_in=5,
                tokens_out=2,
                model="custom",
                cost=Decimal("0.001"),
                latency_ms=1.0,
                metadata={},
            )

        async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
            return self.invoke(prompt, **kwargs)

        def structured_invoke(self, prompt, output_cls, **kwargs):
            return self.invoke(prompt, **kwargs)

        async def structured_invoke_async(self, prompt, output_cls, **kwargs):
            return self.invoke(prompt, **kwargs)

        def estimate_tokens(self, text: str) -> int:
            self.token_estimates += 1
            return max(1, len(text) // 4)

        async def start(self) -> None: ...

        async def stop(self) -> None: ...

    def _pipeline(self, client):
        import pandas as pd

        return (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"t": ["a", "b"]}),
                input_columns=["t"],
                output_columns=["out"],
            )
            .with_prompt("Echo: {t}")
            .with_custom_llm_client(client)
            .with_batch_size(1)
            .build()
        )

    def test_execute_invokes_the_custom_client(self):
        """The whole point: a run must go through the injected client.

        The model name below is deliberately not a real one. Before the fix
        the pipeline called the provider with it and every row failed on
        "model does not exist" — the caller's client untouched.
        """
        client = self._RecordingClient(LLMSpec(provider="openai", model="not-a-model"))

        result = self._pipeline(client).execute()

        assert len(client.prompts) == 2, "custom client was never invoked"
        assert list(result.to_pandas()["out"]) == ["from-custom-client"] * 2

    def test_no_real_provider_client_is_built(self):
        """Nothing may fall back to create_llm_client when one was injected."""
        client = self._RecordingClient(LLMSpec(provider="openai", model="not-a-model"))

        with patch(
            "ondine.api.pipeline.create_llm_client",
            side_effect=AssertionError("built a real client despite an injected one"),
        ):
            self._pipeline(client).execute()

    def test_cost_estimation_uses_the_custom_client(self):
        """Estimation must price with the client the run will actually use."""
        client = self._RecordingClient(LLMSpec(provider="openai", model="not-a-model"))

        estimate = self._pipeline(client).estimate_cost()

        assert estimate.rows == 2
        assert client.token_estimates > 0, (
            "estimation built its own client instead of using the injected one"
        )

    @pytest.mark.asyncio
    async def test_streaming_uses_the_same_client_instance(self):
        """Chunks must share the injected instance, not deep copies of it.

        Sub-pipelines start from model_copy(deep=True), which copies whatever
        is in metadata. A copied client silently defeats the point of
        injecting one: any state it holds — a session, a token, a shared rate
        limiter — stops being shared, and a client wrapping something
        uncopyable would fail outright.
        """
        import pandas as pd

        client = self._RecordingClient(LLMSpec(provider="openai", model="not-a-model"))
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"t": [f"r{i}" for i in range(4)]}),
                input_columns=["t"],
                output_columns=["out"],
            )
            .with_prompt("Echo: {t}")
            .with_custom_llm_client(client)
            .with_batch_size(1)
            .build()
        )

        chunks = [c async for c in pipeline.execute_stream_async(chunk_size=2)]

        assert len(chunks) == 2
        assert len(client.prompts) == 4, (
            "chunks used copies of the client, not the injected instance"
        )

    def test_specifications_stay_serializable(self):
        """An injected client must not make the specs unserializable.

        The client used to be stored in specifications.metadata, which broke
        model_dump(mode="json") — the call MCP makes to snapshot a run
        (ondine/mcp/server.py). Specifications are configuration; a live
        client is runtime state and belongs on the Pipeline, not in them.
        """
        import pandas as pd

        client = self._RecordingClient(LLMSpec(provider="openai", model="not-a-model"))
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"t": ["a"]}), input_columns=["t"], output_columns=["out"]
            )
            .with_prompt("Echo: {t}")
            .with_custom_llm_client(client)
            .build()
        )

        dumped = pipeline.specifications.model_dump(mode="json")

        assert "custom_llm_client" not in dumped.get("metadata", {})
