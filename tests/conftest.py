"""
Pytest configuration and fixtures.

Provides reusable test fixtures and mocks for the test suite.
"""

from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMProvider,
    LLMSpec,
    PromptSpec,
)


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(self, spec: LLMSpec, mock_response: str = "Mock response"):
        """Initialize mock client."""
        super().__init__(spec)
        self.mock_response = mock_response
        self.call_count = 0

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return mock response."""
        self.call_count += 1

        return LLMResponse(
            text=self.mock_response,
            tokens_in=10,
            tokens_out=5,
            model=self.model,
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )

    def estimate_tokens(self, text: str) -> int:
        """Mock token estimation."""
        return len(text) // 4


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": ["Hello world", "Test data", "Sample text"],
            "category": ["A", "B", "A"],
        }
    )


@pytest.fixture
def dataset_spec():
    """Create sample DatasetSpec."""
    return DatasetSpec(
        source_type=DataSourceType.DATAFRAME,
        input_columns=["text"],
        output_columns=["processed"],
    )


@pytest.fixture
def prompt_spec():
    """Create sample PromptSpec."""
    return PromptSpec(
        template="Process: {text}",
        system_message="You are a helpful assistant.",
    )


@pytest.fixture
def llm_spec():
    """Create sample LLMSpec."""
    return LLMSpec(
        provider=LLMProvider.GROQ,
        model="llama-3.1-70b-versatile",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.00005"),
        output_cost_per_1k_tokens=Decimal("0.00008"),
    )


@pytest.fixture
def mock_llm_client(llm_spec):
    """Create mock LLM client."""
    return MockLLMClient(llm_spec)
