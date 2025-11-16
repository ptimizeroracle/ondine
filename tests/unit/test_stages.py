"""Unit tests for pipeline stages."""

from decimal import Decimal

import pandas as pd
import pytest

from ondine.core.models import LLMResponse, PromptBatch, RowMetadata
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    PromptSpec,
)
from ondine.orchestration import ExecutionContext
from ondine.stages import (
    DataLoaderStage,
    JSONParser,
    LLMInvocationStage,
    PromptFormatterStage,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParserStage,
)
from tests.conftest import MockLLMClient


class TestDataLoaderStage:
    """Test suite for DataLoaderStage."""

    def test_data_loader_with_dataframe(self):
        """Test loading data from DataFrame."""
        df = pd.DataFrame(
            {
                "text": ["sample1", "sample2", "sample3"],
                "value": [1, 2, 3],
            }
        )

        stage = DataLoaderStage(dataframe=df)
        context = ExecutionContext()

        spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["result"],
        )

        result = stage.process(spec, context)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "text" in result.columns

    def test_data_loader_validates_columns(self):
        """Test data loader validates required columns."""
        df = pd.DataFrame(
            {
                "text": ["sample"],
            }
        )

        stage = DataLoaderStage(dataframe=df)
        context = ExecutionContext()

        spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["missing_column"],
            output_columns=["result"],
        )

        with pytest.raises(ValueError, match="Missing columns"):
            stage.process(spec, context)

    def test_data_loader_cost_estimation(self):
        """Test data loader cost estimation."""
        df = pd.DataFrame({"text": ["test"] * 10})

        stage = DataLoaderStage(dataframe=df)
        ExecutionContext()

        spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["result"],
        )

        estimate = stage.estimate_cost(spec)

        assert estimate.rows == 10
        assert estimate.total_cost == Decimal("0.0")


class TestPromptFormatterStage:
    """Test suite for PromptFormatterStage."""

    def test_prompt_formatter_basic(self):
        """Test basic prompt formatting."""
        df = pd.DataFrame(
            {
                "text": ["hello", "world"],
            }
        )

        prompt_spec = PromptSpec(template="Process: {text}")

        stage = PromptFormatterStage(batch_size=10)
        context = ExecutionContext()

        batches = stage.process((df, prompt_spec), context)

        assert len(batches) > 0
        assert all(isinstance(b, PromptBatch) for b in batches)
        assert "Process: hello" in batches[0].prompts[0]

    def test_prompt_formatter_with_system_message(self):
        """Test prompt formatting with system message."""
        df = pd.DataFrame({"text": ["test"]})

        prompt_spec = PromptSpec(
            template="{text}",
            system_message="You are a helpful assistant.",
        )

        stage = PromptFormatterStage()
        context = ExecutionContext()

        batches = stage.process((df, prompt_spec), context)

        # System message should be in metadata, NOT in prompt
        assert "You are a helpful assistant" not in batches[0].prompts[0]
        assert batches[0].metadata[0].custom is not None
        assert (
            batches[0].metadata[0].custom["system_message"]
            == "You are a helpful assistant."
        )

    def test_prompt_formatter_batching(self):
        """Test prompt formatter creates batches correctly."""
        df = pd.DataFrame(
            {
                "text": [f"item{i}" for i in range(25)],
            }
        )

        prompt_spec = PromptSpec(template="{text}")

        stage = PromptFormatterStage(batch_size=10)
        context = ExecutionContext()

        batches = stage.process((df, prompt_spec), context)

        # Should create 3 batches (10 + 10 + 5)
        assert len(batches) == 3
        assert len(batches[0].prompts) == 10
        assert len(batches[1].prompts) == 10
        assert len(batches[2].prompts) == 5


class TestLLMInvocationStage:
    """Test suite for LLMInvocationStage."""

    def test_llm_invocation_with_mock_client(self):
        """Test LLM invocation with mock client."""
        from ondine.core.specifications import LLMProvider, LLMSpec

        mock_client = MockLLMClient(
            spec=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
            mock_response="Mocked response",
        )

        stage = LLMInvocationStage(llm_client=mock_client, concurrency=1)
        context = ExecutionContext()

        prompts = ["Test prompt 1", "Test prompt 2"]
        batch = PromptBatch(
            prompts=prompts,
            metadata=[
                RowMetadata(row_index=0, custom={"original_data": {"text": "test1"}}),
                RowMetadata(row_index=1, custom={"original_data": {"text": "test2"}}),
            ],
            batch_id=0,
        )

        response_batches = stage.process([batch], context)

        assert len(response_batches) == 1
        assert len(response_batches[0].responses) == 2
        assert mock_client.call_count == 2

    def test_llm_invocation_maintains_order(self):
        """Test LLM invocation maintains response order."""
        from ondine.core.specifications import LLMProvider, LLMSpec

        mock_client = MockLLMClient(
            spec=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        stage = LLMInvocationStage(llm_client=mock_client, concurrency=2)
        context = ExecutionContext()

        prompts = [f"Prompt {i}" for i in range(5)]
        batch = PromptBatch(
            prompts=prompts,
            metadata=[
                RowMetadata(row_index=i, custom={"original_data": {"text": f"test{i}"}})
                for i in range(5)
            ],
            batch_id=0,
        )

        response_batches = stage.process([batch], context)

        # Responses should be in same order as prompts
        assert len(response_batches[0].responses) == 5
        # Metadata should match original order
        for i, metadata in enumerate(response_batches[0].metadata):
            assert metadata.row_index == i

    def test_llm_invocation_concurrent_batches(self):
        """Test that multiple batches are processed concurrently (flatten-then-concurrent)."""
        import time

        from ondine.core.specifications import LLMProvider, LLMSpec

        # Create mock client with artificial delay
        class SlowMockClient(MockLLMClient):
            def generate(self, prompt, **kwargs):
                time.sleep(0.1)  # 100ms delay per call
                return super().generate(prompt, **kwargs)

        mock_client = SlowMockClient(
            spec=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
            mock_response="Test",
        )

        # Create 10 batches with 1 prompt each (simulates aggregated batches)
        batches = [
            PromptBatch(
                prompts=["test"],
                metadata=[RowMetadata(row_index=i)],
                batch_id=i,
            )
            for i in range(10)
        ]

        stage = LLMInvocationStage(llm_client=mock_client, concurrency=10)
        context = ExecutionContext()

        start = time.time()
        result = stage.process(batches, context)
        duration = time.time() - start

        # With concurrency=10, all 10 batches should process in parallel
        # Expected: ~0.1s (all parallel) + overhead
        # Without concurrency: ~1.0s (sequential)
        assert len(result) == 10
        assert duration < 0.5, (
            f"Expected <0.5s (parallel), got {duration:.2f}s (sequential!)"
        )
        assert mock_client.call_count == 10


class TestResponseParserStage:
    """Test suite for ResponseParserStage."""

    def test_raw_text_parser(self):
        """Test raw text parser."""
        parser = RawTextParser()
        result = parser.parse("  Hello World  ")

        assert result["output"] == "Hello World"

    def test_json_parser_valid(self):
        """Test JSON parser with valid JSON."""
        parser = JSONParser()
        result = parser.parse('{"name": "test", "value": 42}')

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_json_parser_with_markdown(self):
        """Test JSON parser extracts from markdown code blocks."""
        parser = JSONParser(strict=False)
        response = '```json\n{"result": "extracted"}\n```'

        result = parser.parse(response)

        assert result["result"] == "extracted"

    def test_json_parser_strict_mode(self):
        """Test JSON parser in strict mode fails on invalid JSON."""
        parser = JSONParser(strict=True)

        with pytest.raises(Exception):
            parser.parse("not valid json")

    def test_regex_parser(self):
        """Test regex parser."""
        parser = RegexParser(patterns={"number": r"\d+", "word": r"[A-Za-z]+"})

        result = parser.parse("The answer is 42 tests")

        assert result["number"] == "42"
        assert result["word"] == "The"

    def test_pydantic_parser(self):
        """Test pydantic parser."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        parser = PydanticParser(TestModel)
        result = parser.parse('{"name": "Alice", "age": 30}')

        assert isinstance(result, TestModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_response_parser_stage_with_json(self):
        """Test response parser stage with JSON parser."""
        from ondine.core.models import ResponseBatch

        parser = JSONParser()
        stage = ResponseParserStage(parser=parser, output_columns=["name", "value"])
        context = ExecutionContext()

        responses = [
            LLMResponse(
                text='{"name": "test1", "value": 10}',
                tokens_in=10,
                tokens_out=20,
                model="test",
                cost=Decimal("0.001"),
                latency_ms=100.0,
            ),
            LLMResponse(
                text='{"name": "test2", "value": 20}',
                tokens_in=10,
                tokens_out=20,
                model="test",
                cost=Decimal("0.001"),
                latency_ms=100.0,
            ),
        ]

        batch = ResponseBatch(
            responses=responses,
            metadata=[
                RowMetadata(row_index=0),
                RowMetadata(row_index=1),
            ],
            tokens_used=100,
            cost=Decimal("0.01"),
            batch_id=0,
        )

        result_df = stage.process([batch], context)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "name" in result_df.columns
        assert "value" in result_df.columns
        assert result_df.iloc[0]["name"] == "test1"
        assert result_df.iloc[1]["value"] == 20
