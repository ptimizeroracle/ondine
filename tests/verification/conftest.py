"""Shared fixtures for claim verification tests."""

import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class CallCountingMockClient(LLMClient):
    """Mock LLM client that counts calls and returns configurable responses."""

    def __init__(
        self, response_text: str = "mock response", fail_indices: set[int] | None = None
    ):
        spec = LLMSpec(model="mock-model", provider="openai")
        super().__init__(spec)
        self.response_text = response_text
        self.call_count = 0
        self.fail_indices = fail_indices or set()
        self._invocations: list[str] = []

    def invoke(self, prompt: str, **kwargs):
        self.call_count += 1
        self._invocations.append(prompt)

        if self.call_count in self.fail_indices:
            raise RuntimeError(f"Simulated failure at call {self.call_count}")

        return LLMResponse(
            text=self.response_text,
            tokens_in=len(prompt.split()),
            tokens_out=len(self.response_text.split()),
            cost=Decimal("0.001"),
            model="mock-model",
            latency_ms=5.0,
        )

    def structured_invoke(self, prompt, output_cls, **kwargs):
        return self.invoke(prompt, **kwargs)

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def mock_client():
    """Basic mock LLM client."""
    return CallCountingMockClient()


@pytest.fixture
def sample_100_rows():
    """100-row DataFrame for batch testing."""
    return pd.DataFrame(
        {
            "text": [f"Sample text row {i}" for i in range(100)],
            "category": [f"cat_{i % 5}" for i in range(100)],
        }
    )


@pytest.fixture
def sample_1k_rows():
    """1000-row DataFrame for scale testing."""
    return pd.DataFrame(
        {
            "text": [f"Sample text row {i}" for i in range(1000)],
        }
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_csv(temp_dir):
    """Temp CSV file with sample data."""
    path = temp_dir / "sample.csv"
    df = pd.DataFrame({"text": ["hello", "world"], "label": ["a", "b"]})
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_excel(temp_dir):
    """Temp Excel file with sample data."""
    path = temp_dir / "sample.xlsx"
    df = pd.DataFrame({"text": ["hello", "world"], "label": ["a", "b"]})
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def sample_parquet(temp_dir):
    """Temp Parquet file with sample data."""
    path = temp_dir / "sample.parquet"
    df = pd.DataFrame({"text": ["hello", "world"], "label": ["a", "b"]})
    df.to_parquet(path, index=False)
    return path
