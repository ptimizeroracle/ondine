"""
Unit tests for structured output invocation with Instructor.

Tests the UnifiedLiteLLMClient's structured_invoke method which uses
Instructor for Pydantic model validation and schema enforcement.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMProvider, LLMSpec


# Define a sample Pydantic model for testing
class TestModel(BaseModel):
    field1: str
    field2: int


class TestStructuredInvokeInstructor:
    """Test structured_invoke with Instructor integration."""

    def test_structured_invoke_sync_wrapper(self):
        """Test that sync structured_invoke wraps async version."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="dummy",  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        # Mock asyncio.run
        mock_response = MagicMock()
        mock_response.text = '{"field1": "test", "field2": 123}'

        with patch("asyncio.run", return_value=mock_response):
            response = client.structured_invoke("prompt", TestModel)

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_structured_invoke_async_uses_instructor(self):
        """Test that structured_invoke_async uses Instructor."""
        from unittest.mock import AsyncMock

        import instructor

        spec = LLMSpec(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_key="dummy",  # pragma: allowlist secret
        )

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        mock_result = TestModel(field1="groq", field2=789)

        with (
            patch("instructor.from_litellm") as mock_instructor,
            patch.object(client, "_calculate_cost_litellm", return_value=0.0005),
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_instructor.return_value = mock_client

            response = await client.structured_invoke_async("prompt", TestModel)

            # Verify JSON mode was used for Groq
            call_args = mock_instructor.call_args
            assert call_args.kwargs["mode"] == instructor.Mode.JSON

            # Verify response
            assert response.text == mock_result.model_dump_json()
