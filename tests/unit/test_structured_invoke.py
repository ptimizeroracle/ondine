"""
Unit tests for structured output invocation.

NOTE: Structured output tests moved to Phase 2 (Instructor integration).
This file kept as placeholder to avoid breaking test discovery.

Phase 2 will add:
- Instructor integration tests
- Dual-path tests (native function calling vs Instructor)
- Auto-detection tests
"""

import pytest
from pydantic import BaseModel

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMProvider, LLMSpec


# Define a sample Pydantic model for testing
class TestModel(BaseModel):
    field1: str
    field2: int


class TestStructuredInvokePlaceholder:
    """Placeholder tests for Phase 1 - structured output in Phase 2."""

    def test_structured_invoke_not_implemented_in_phase1(self):
        """Test that structured_invoke raises NotImplementedError in Phase 1."""
        spec = LLMSpec(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="dummy",  # pragma: allowlist secret
        )

        from unittest.mock import patch

        with patch("ondine.adapters.unified_litellm_client.os.environ", {}):
            client = UnifiedLiteLLMClient(spec)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            client.structured_invoke("prompt", TestModel)


# NOTE: Original structured_predict tests removed
# Rationale: Those tested LlamaIndex's structured_predict() which is removed.
# Phase 2 will add comprehensive Instructor tests with real API calls.
