"""
Pytest configuration and fixtures.

Provides reusable test fixtures and mocks for the test suite.
"""

import logging
import warnings
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

# Suppress harmless warnings from LiteLLM and Pydantic GLOBALLY
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")  # Catch-all


def pytest_configure(config):
    """Configure pytest - suppress all warnings."""
    warnings.filterwarnings("ignore")
    logging.captureWarnings(True)


# Suppress harmless warnings from LiteLLM and Pydantic at pytest level
@pytest.fixture(scope="session", autouse=True)
def suppress_litellm_warnings():
    """Suppress harmless runtime warnings from LiteLLM and Pydantic."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*")
    warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
    warnings.filterwarnings(
        "ignore", message=".*PydanticSerializationUnexpectedValue.*"
    )
    warnings.filterwarnings("ignore", message=".*Expected.*fields but got.*")
    warnings.filterwarnings(
        "ignore", message=".*serialized value may not be as expected.*"
    )
    warnings.simplefilter("ignore")
    return


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

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Mock async invoke."""
        return self.invoke(prompt, **kwargs)

    def structured_invoke(self, prompt: str, output_cls, **kwargs: Any) -> LLMResponse:
        """Mock structured invoke (required by abstract base)."""
        from pydantic import BaseModel

        # Create mock instance with dummy data
        result = (
            output_cls.model_validate({"items": []})
            if hasattr(output_cls, "__fields__")
            else output_cls()
        )
        return LLMResponse(
            text=result.model_dump_json() if isinstance(result, BaseModel) else "{}",
            tokens_in=10,
            tokens_out=5,
            model=self.model,
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )

    async def structured_invoke_async(
        self, prompt: str, output_cls, **kwargs: Any
    ) -> LLMResponse:
        """Mock structured async invoke."""
        return self.structured_invoke(prompt, output_cls, **kwargs)

    async def start(self):
        """Mock start."""
        pass

    async def stop(self):
        """Mock stop."""
        pass

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
