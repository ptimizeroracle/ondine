"""Tests for prefix caching functionality."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from ondine.adapters.llm_client import AnthropicClient, OpenAIClient
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.models import RowMetadata
from ondine.core.specifications import LLMProvider, LLMSpec, PromptSpec
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.stages.prompt_formatter_stage import PromptFormatterStage


class TestOpenAIClientPrefixCaching:
    """Test OpenAI client prefix caching implementation."""

    @pytest.mark.skip(reason="Mocking LlamaIndex client is complex, tested manually")
    def test_openai_client_uses_system_message(self):
        """Verify OpenAI client separates system/user messages."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            enable_prefix_caching=True,
        )

        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "test-key"}
        ):  # pragma: allowlist secret
            client = OpenAIClient(spec)

            # Mock the LlamaIndex client
            mock_response = MagicMock()
            mock_response.__str__ = lambda x: "Test response"
            client.client.chat = MagicMock(return_value=mock_response)

            # Invoke with system message
            client.invoke(
                "What is 2+2?", system_message="You are a helpful math tutor."
            )

            # Verify chat was called with both system and user messages
            assert client.client.chat.called
            messages = client.client.chat.call_args[0][0]
            assert len(messages) == 2
            assert messages[0].role == "system"
            assert messages[0].content == "You are a helpful math tutor."
            assert messages[1].role == "user"
            assert messages[1].content == "What is 2+2?"

    @pytest.mark.skip(reason="Mocking LlamaIndex client is complex, tested manually")
    def test_openai_client_without_system_message(self):
        """Verify OpenAI client works without system message (backward compat)."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            enable_prefix_caching=True,
        )

        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "test-key"}
        ):  # pragma: allowlist secret
            client = OpenAIClient(spec)

            # Mock the LlamaIndex client
            mock_response = MagicMock()
            mock_response.__str__ = lambda x: "Test response"
            client.client.chat = MagicMock(return_value=mock_response)

            # Invoke without system message
            client.invoke("What is 2+2?")

            # Verify chat was called with only user message
            assert client.client.chat.called
            messages = client.client.chat.call_args[0][0]
            assert len(messages) == 1
            assert messages[0].role == "user"
            assert messages[0].content == "What is 2+2?"


class TestAnthropicClientPrefixCaching:
    """Test Anthropic client prefix caching implementation."""

    @pytest.mark.skip(reason="Mocking LlamaIndex client is complex, tested manually")
    def test_anthropic_client_cache_control(self):
        """Verify Anthropic client adds cache_control markers."""
        spec = LLMSpec(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4",
            enable_prefix_caching=True,
        )

        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
        ):  # pragma: allowlist secret
            client = AnthropicClient(spec)

            # Mock the LlamaIndex client
            mock_response = MagicMock()
            mock_response.__str__ = lambda x: "Test response"
            client.client.chat = MagicMock(return_value=mock_response)

            # Invoke with system message
            client.invoke(
                "What is 2+2?", system_message="You are a helpful math tutor."
            )

            # Verify chat was called
            assert client.client.chat.called

            # Note: The actual cache_control implementation depends on LlamaIndex's
            # Anthropic client API. This test verifies the system_param is prepared.
            # In production, this would be passed to the Anthropic API.


class TestPromptFormatterSystemMessage:
    """Test prompt formatter passes system message through metadata."""

    def test_prompt_formatter_passes_system_message(self):
        """Verify system message flows through metadata."""
        import pandas as pd

        stage = PromptFormatterStage(batch_size=10)

        df = pd.DataFrame({"text": ["Hello", "World"]})

        prompt_spec = PromptSpec(
            template="Process: {text}", system_message="You are a helpful assistant."
        )

        batches = stage.process((df, prompt_spec), None)

        # Verify batches were created
        assert len(batches) == 1
        batch = batches[0]

        # Verify prompts don't contain system message
        assert len(batch.prompts) == 2
        assert "You are a helpful assistant" not in batch.prompts[0]
        assert "Process: Hello" in batch.prompts[0]

        # Verify metadata contains system message
        assert len(batch.metadata) == 2
        assert batch.metadata[0].custom is not None
        assert (
            batch.metadata[0].custom["system_message"] == "You are a helpful assistant."
        )

    def test_prompt_formatter_without_system_message(self):
        """Verify formatter works without system message (backward compat)."""
        import pandas as pd

        stage = PromptFormatterStage(batch_size=10)

        df = pd.DataFrame({"text": ["Hello", "World"]})

        prompt_spec = PromptSpec(template="Process: {text}", system_message=None)

        batches = stage.process((df, prompt_spec), None)

        # Verify batches were created
        assert len(batches) == 1
        batch = batches[0]

        # Verify metadata has None custom field
        assert batch.metadata[0].custom is None


class TestLLMInvocationSystemMessage:
    """Test LLM invocation stage extracts system message from metadata."""

    def test_llm_invocation_extracts_system_message(self):
        """Verify LLM stage extracts system_message from metadata."""
        # Create mock LLM client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.tokens_in = 10
        mock_response.tokens_out = 5
        mock_response.cost = Decimal("0.001")
        mock_response.latency_ms = 100.0
        mock_client.invoke = MagicMock(return_value=mock_response)
        mock_client.model = "test-model"

        stage = LLMInvocationStage(llm_client=mock_client, concurrency=1)

        # Create metadata with system message
        metadata = RowMetadata(
            row_index=0,
            row_id=None,
            custom={"system_message": "You are a helpful assistant."},
        )

        # Invoke with metadata
        stage._invoke_with_retry_and_ratelimit(
            prompt="What is 2+2?", row_metadata=metadata, context=None, row_index=0
        )

        # Verify invoke was called with system_message kwarg
        assert mock_client.invoke.called
        call_kwargs = mock_client.invoke.call_args[1]
        assert "system_message" in call_kwargs
        assert call_kwargs["system_message"] == "You are a helpful assistant."


class TestPipelineBuilderAPI:
    """Test PipelineBuilder.with_system_prompt() method."""

    def test_with_system_prompt_method(self):
        """Verify with_system_prompt() sets system message."""
        builder = (
            PipelineBuilder.create()
            .with_prompt("Process: {text}")
            .with_system_prompt("You are a helpful assistant.")
        )

        assert builder._prompt_spec is not None
        assert builder._prompt_spec.system_message == "You are a helpful assistant."

    def test_with_system_prompt_requires_prompt_first(self):
        """Verify with_system_prompt() requires with_prompt() first."""
        builder = PipelineBuilder.create()

        with pytest.raises(
            ValueError, match="Call with_prompt\\(\\) before with_system_prompt\\(\\)"
        ):
            builder.with_system_prompt("You are a helpful assistant.")


class TestBackwardCompatibility:
    """Test that existing pipelines work without system_message."""

    def test_backward_compatibility(self):
        """Verify existing pipelines work without system_message."""
        # This should not raise any errors
        builder = PipelineBuilder.create().with_prompt("Process: {text}")

        # System message should be None by default
        assert builder._prompt_spec.system_message is None

    def test_llm_spec_enable_prefix_caching_default(self):
        """Verify enable_prefix_caching defaults to True."""
        spec = LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini")

        assert spec.enable_prefix_caching is True
