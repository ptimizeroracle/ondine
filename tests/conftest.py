"""
Pytest configuration and fixtures.

Provides reusable test fixtures and mocks for the test suite.
"""

import logging
import math
import os
import re
import warnings
import zlib
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


@pytest.fixture(scope="session", autouse=True)
def _pandas_copy_on_write_toggle():
    """Enable pandas Copy-on-Write when ONDINE_TEST_COW=1.

    CoW becomes the unchangeable default in pandas 3.0. Running the full test
    suite under CoW surfaces latent SettingWithCopyWarning-style bugs *before*
    the 3.0 bump forces them. Disabled by default so it does not change normal
    CI semantics; enable in a dedicated CI job or locally via the env var.

    See DEPENDENCY_UPGRADE_ACTION_PLAN.md (Q3) for the rationale.
    """
    if os.getenv("ONDINE_TEST_COW") != "1":
        yield
        return

    pd.options.mode.copy_on_write = True
    try:
        yield
    finally:
        pd.options.mode.copy_on_write = False


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


class DeterministicEmbedder:
    """Bag-of-words embedder — no model, no network, sensible geometry.

    The knowledge tests used to embed with the real sentence-transformers
    model, which downloads ~90MB from huggingface.co on first use. That made a
    *unit* suite depend on a third-party host being reachable: when it was not,
    17 tests failed and the job burned ~57 minutes on HF's retry backoff before
    giving up (#221). A red suite that usually means "the network hiccuped" is
    worse than useless — it teaches everyone to re-run without reading.

    Each token is hashed into a fixed bucket and the vector L2-normalised, so
    cosine similarity tracks token overlap. That is enough geometry for the
    "best keyword match ranks first" assertions these tests actually make,
    while staying instant and hermetic. crc32 rather than hash() because the
    latter is salted per process, which would make rankings vary between runs.

    The real embedder is exercised where it is the subject, in the integration
    suite — not incidentally, in every test that happens to need a store.
    """

    DIMENSIONS = 64

    def __init__(self) -> None:
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def __repr__(self) -> str:
        return f"DeterministicEmbedder(dims={self.DIMENSIONS})"


@pytest.fixture
def deterministic_embedder():
    """An embedder that needs no model download. See DeterministicEmbedder."""
    return DeterministicEmbedder()


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
