"""
Unit tests for structured output modes in UnifiedLiteLLMClient.

Tests Instructor integration with auto-detection and dual-path support.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMProvider, LLMSpec


# Test Pydantic models
class TestModel(BaseModel):
    """Simple test model."""

    field1: str
    field2: int


class BaconResult(BaseModel):
    """Bacon product model for testing."""

    cleaned_description: str = Field(description="Description")
    pack_size: float | None = Field(default=None)


class BatchItem(BaseModel):
    """Batch item wrapper."""

    id: int
    result: BaconResult


class BaconBatch(BaseModel):
    """Batch of bacon products."""

    items: list[BatchItem]


class TestAutoDetection:
    """Test auto-detection of structured output mode."""

    def test_groq_uses_json_mode(self):
        """Test that Groq auto-detects to JSON mode."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.JSON

    def test_openai_uses_tools_mode(self):
        """Test that OpenAI auto-detects to TOOLS mode."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.TOOLS

    def test_anthropic_uses_tools_mode(self):
        """Test that Anthropic auto-detects to TOOLS mode."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-5-haiku-20241022",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.TOOLS

    def test_azure_uses_tools_mode(self):
        """Test that Azure OpenAI auto-detects to TOOLS mode."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.AZURE_OPENAI,
            model="gpt-4",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="gpt-4",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.TOOLS


class TestExplicitModeSelection:
    """Test explicit mode configuration."""

    def test_force_json_mode(self):
        """Test forcing JSON mode via configuration."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.OPENAI,  # Would normally use TOOLS
            model="gpt-4o-mini",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
            structured_output_mode="instructor_json",  # Force JSON
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.JSON

    def test_force_tools_mode(self):
        """Test forcing TOOLS mode via configuration."""
        import instructor

        spec = LLMSpec(
            provider=LLMProvider.GROQ,  # Would normally use JSON
            model="llama-3.3-70b-versatile",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
            structured_output_mode="instructor_tools",  # Force TOOLS
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mode = client._get_structured_output_mode()
        assert mode == instructor.Mode.TOOLS


class TestStructuredInvokeMocked:
    """Test structured_invoke with mocked Instructor."""

    @pytest.mark.asyncio
    async def test_structured_invoke_async_returns_llm_response(self):
        """Test that structured_invoke_async returns proper LLMResponse."""
        from unittest.mock import AsyncMock

        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        # Mock Instructor
        mock_result = TestModel(field1="test", field2=123)

        with (
            patch("instructor.from_litellm") as mock_instructor,
            patch.object(client, "_calculate_cost_litellm", return_value=0.001),
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_instructor.return_value = mock_client

            response = await client.structured_invoke_async("Test prompt", TestModel)

        assert response.text == mock_result.model_dump_json()
        assert response.model == "gpt-4o-mini"
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_structured_invoke_async_with_system_message(self):
        """Test structured output with system message."""
        from unittest.mock import AsyncMock

        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mock_result = TestModel(field1="result", field2=456)

        with (
            patch("instructor.from_litellm") as mock_instructor,
            patch.object(client, "_calculate_cost_litellm", return_value=0.002),
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_instructor.return_value = mock_client

            await client.structured_invoke_async(
                "User prompt", TestModel, system_message="System instruction"
            )

            # Verify messages were built
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_structured_invoke_async_error_handling(self):
        """Test error handling in structured_invoke_async."""
        from unittest.mock import AsyncMock

        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        with (
            patch("instructor.from_litellm") as mock_instructor,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_instructor.return_value = mock_client

            with pytest.raises(ValueError, match="Structured prediction failed"):
                await client.structured_invoke_async("prompt", TestModel)

    def test_structured_invoke_sync_wraps_async(self):
        """Test that sync structured_invoke wraps async version."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        # Mock asyncio.run to verify it's called
        mock_result = MagicMock()
        with patch("asyncio.run", return_value=mock_result) as mock_run:
            result = client.structured_invoke("prompt", TestModel)

        assert result == mock_result
        mock_run.assert_called_once()


class TestInstructorMaxRetries:
    """Test that Instructor's built-in max_retries is used."""

    @pytest.mark.asyncio
    async def test_max_retries_configured(self):
        """Test that max_retries=3 is passed to Instructor."""
        from unittest.mock import AsyncMock

        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="test-key",  # pragma: allowlist secret  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mock_result = TestModel(field1="test", field2=789)

        with patch("instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_instructor.return_value = mock_client

            with patch.object(client, "_calculate_cost_litellm", return_value=0.001):
                await client.structured_invoke_async("prompt", TestModel)

            # Verify max_retries was passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["max_retries"] == 3
