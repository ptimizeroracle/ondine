"""Unit tests for Pipeline class."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.api.pipeline import Pipeline
from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    LLMProvider,
    LLMSpec,
    PipelineSpecifications,
    ProcessingSpec,
    PromptSpec,
)
from ondine.orchestration import StreamingExecutor
from ondine.utils.budget_controller import BudgetExceededError


class TestPipeline:
    """Test suite for Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with specifications."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="Process: {text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)

        assert pipeline.id is not None
        assert pipeline.specifications == specs
        assert pipeline.observers == []

    def test_pipeline_with_dataframe(self):
        """Test pipeline with pre-loaded DataFrame."""
        df = pd.DataFrame({"text": ["test"]})
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs, dataframe=df)

        assert pipeline.dataframe is not None
        assert len(pipeline.dataframe) == 1

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        validation = pipeline.validate()

        assert validation.is_valid is True
        assert len(validation.errors) == 0

    def test_validate_missing_template_variable(self):
        """Test validation catches missing template variables."""
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{missing_var}"),  # Variable not in input
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        validation = pipeline.validate()

        assert validation.is_valid is False
        assert len(validation.errors) > 0

    def test_estimate_cost_with_sample(self):
        """Test cost estimation with sample data."""
        import os

        df = pd.DataFrame({"text": [f"Sample {i}" for i in range(100)]})

        # Set a dummy API key for cost estimation (doesn't need to be real)
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-estimation"

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                input_cost_per_1k_tokens=Decimal("0.00015"),
                output_cost_per_1k_tokens=Decimal("0.0006"),
            ),
        )

        try:
            pipeline = Pipeline(specs, dataframe=df)
            estimate = pipeline.estimate_cost()

            assert estimate.total_cost >= 0
            assert estimate.rows == 100
        finally:
            # Clean up
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_add_observer(self):
        """Test adding observers to pipeline."""
        from ondine.orchestration import LoggingObserver

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        pipeline = Pipeline(specs)
        observer = LoggingObserver()
        pipeline.add_observer(observer)

        assert len(pipeline.observers) == 1
        assert pipeline.observers[0] == observer

    def test_pipeline_with_executor(self):
        """Test pipeline with custom executor."""
        from ondine.orchestration import SyncExecutor

        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )

        executor = SyncExecutor()
        pipeline = Pipeline(specs, executor=executor)

        assert pipeline.executor == executor

    def test_execute_stream_uses_async_streaming_path(self):
        """Test execute_stream delegates to execute_stream_async instead of execute()."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
            metadata={
                "streaming": {
                    "enabled": True,
                    "chunk_size": 2,
                    "max_pending_chunks": 4,
                }
            },
        )
        pipeline = Pipeline(
            specs, dataframe=df, executor=StreamingExecutor(chunk_size=2)
        )

        chunk_result = ExecutionResult(
            data=pd.DataFrame({"text": ["a"], "result": ["A"]}),
            metrics=ProcessingStats(
                total_rows=1,
                processed_rows=1,
                failed_rows=0,
                skipped_rows=0,
            ),
            costs=CostEstimate(
                total_cost=Decimal("0.01"),
                total_tokens=10,
                input_tokens=5,
                output_tokens=5,
                rows=1,
                confidence="actual",
            ),
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
        )

        captured = {}

        async def fake_execute_stream_async(
            chunk_size: int, max_pending_chunks: int, output_path=None
        ):
            captured["chunk_size"] = chunk_size
            captured["max_pending_chunks"] = max_pending_chunks
            captured["output_path"] = output_path
            yield chunk_result

        with (
            patch.object(
                pipeline,
                "execute",
                side_effect=AssertionError("execute() should not be used"),
            ),
            patch.object(
                pipeline,
                "execute_stream_async",
                side_effect=fake_execute_stream_async,
            ),
        ):
            results = list(pipeline.execute_stream())

        assert results == [chunk_result]
        assert captured == {
            "chunk_size": 2,
            "max_pending_chunks": 4,
            "output_path": None,
        }

    @pytest.mark.asyncio
    async def test_execute_stream_async_enforces_budget_across_chunks(self):
        """Test streaming budget enforcement uses cumulative chunk cost."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
            processing=ProcessingSpec(max_budget=Decimal("0.50")),
        )
        pipeline = Pipeline(
            specs, dataframe=df, executor=StreamingExecutor(chunk_size=1)
        )

        chunk_frames = [
            pl.DataFrame({"text": ["a"]}),
            pl.DataFrame({"text": ["b"]}),
            pl.DataFrame({"text": ["c"]}),
        ]
        processed_chunks = []

        async def fake_stream_dataframe_chunks(dataframe, chunk_size):
            for chunk in chunk_frames:
                yield chunk

        async def fake_process_chunk_async(chunk_df, execution_id, chunk_index):
            processed_chunks.append(chunk_index)
            return ExecutionResult(
                data=pd.DataFrame(
                    {
                        "text": chunk_df["text"].tolist(),
                        "result": [
                            value.upper() for value in chunk_df["text"].tolist()
                        ],
                    }
                ),
                metrics=ProcessingStats(
                    total_rows=len(chunk_df),
                    processed_rows=len(chunk_df),
                    failed_rows=0,
                    skipped_rows=0,
                ),
                costs=CostEstimate(
                    total_cost=Decimal("0.30"),
                    total_tokens=10,
                    input_tokens=5,
                    output_tokens=5,
                    rows=len(chunk_df),
                    confidence="actual",
                ),
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
            )

        with (
            patch.object(
                pipeline,
                "_stream_dataframe_chunks",
                side_effect=fake_stream_dataframe_chunks,
            ),
            patch.object(
                pipeline,
                "_process_chunk_async",
                side_effect=fake_process_chunk_async,
            ),
            pytest.raises(BudgetExceededError, match="Budget exceeded"),
        ):
            async for _ in pipeline.execute_stream_async(chunk_size=1):
                pass

        assert processed_chunks == [0, 1]

    @pytest.mark.asyncio
    async def test_execute_stream_async_prefetches_using_max_pending_chunks(self):
        """Test async streaming prefetches source chunks up to the pending limit."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )
        pipeline = Pipeline(
            specs, dataframe=df, executor=StreamingExecutor(chunk_size=1)
        )

        chunk_frames = [
            pl.DataFrame({"text": ["a"]}),
            pl.DataFrame({"text": ["b"]}),
            pl.DataFrame({"text": ["c"]}),
        ]
        yielded_chunks = []
        release_first_chunk = asyncio.Event()

        async def fake_stream_dataframe_chunks(dataframe, chunk_size):
            for idx, chunk in enumerate(chunk_frames):
                yielded_chunks.append(idx)
                yield chunk

        async def fake_process_chunk_async(chunk_df, execution_id, chunk_index):
            if chunk_index == 0:
                await release_first_chunk.wait()
            return ExecutionResult(
                data=pd.DataFrame(
                    {
                        "text": chunk_df["text"].tolist(),
                        "result": [
                            value.upper() for value in chunk_df["text"].tolist()
                        ],
                    }
                ),
                metrics=ProcessingStats(
                    total_rows=len(chunk_df),
                    processed_rows=len(chunk_df),
                    failed_rows=0,
                    skipped_rows=0,
                ),
                costs=CostEstimate(
                    total_cost=Decimal("0.10"),
                    total_tokens=10,
                    input_tokens=5,
                    output_tokens=5,
                    rows=len(chunk_df),
                    confidence="actual",
                ),
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
            )

        with (
            patch.object(
                pipeline,
                "_stream_dataframe_chunks",
                side_effect=fake_stream_dataframe_chunks,
            ),
            patch.object(
                pipeline,
                "_process_chunk_async",
                side_effect=fake_process_chunk_async,
            ),
        ):
            stream = pipeline.execute_stream_async(chunk_size=1, max_pending_chunks=2)
            first_chunk_task = asyncio.create_task(anext(stream))

            for _ in range(20):
                if len(yielded_chunks) >= 2:
                    break
                await asyncio.sleep(0.01)

            assert len(yielded_chunks) >= 2

            release_first_chunk.set()
            first_chunk = await first_chunk_task
            remaining_chunks = [chunk async for chunk in stream]

        assert first_chunk.metrics.processed_rows == 1
        assert len(remaining_chunks) == 2

    @pytest.mark.asyncio
    async def test_execute_stream_async_writes_result_container_output(self, tmp_path):
        """Test streamed output writing handles ResultContainerImpl chunk data."""
        df = pd.DataFrame({"text": ["a", "b"]})
        output_path = tmp_path / "streamed.csv"
        specs = PipelineSpecifications(
            dataset=DatasetSpec(
                source_type=DataSourceType.DATAFRAME,
                input_columns=["text"],
                output_columns=["result"],
            ),
            prompt=PromptSpec(template="{text}"),
            llm=LLMSpec(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
        )
        pipeline = Pipeline(
            specs, dataframe=df, executor=StreamingExecutor(chunk_size=1)
        )

        async def fake_stream_dataframe_chunks(dataframe, chunk_size):
            yield pl.DataFrame({"text": ["a"]})
            yield pl.DataFrame({"text": ["b"]})

        async def fake_process_chunk_async(chunk_df, execution_id, chunk_index):
            rows = [
                {"text": value, "result": value.upper()}
                for value in chunk_df["text"].tolist()
            ]
            return ExecutionResult(
                data=ResultContainerImpl(rows, columns=["text", "result"]),
                metrics=ProcessingStats(
                    total_rows=len(rows),
                    processed_rows=len(rows),
                    failed_rows=0,
                    skipped_rows=0,
                ),
                costs=CostEstimate(
                    total_cost=Decimal("0.10"),
                    total_tokens=10,
                    input_tokens=5,
                    output_tokens=5,
                    rows=len(rows),
                    confidence="actual",
                ),
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
            )

        with (
            patch.object(
                pipeline,
                "_stream_dataframe_chunks",
                side_effect=fake_stream_dataframe_chunks,
            ),
            patch.object(
                pipeline,
                "_process_chunk_async",
                side_effect=fake_process_chunk_async,
            ),
        ):
            results = [
                chunk
                async for chunk in pipeline.execute_stream_async(
                    chunk_size=1, output_path=output_path
                )
            ]

        assert len(results) == 2
        written = pd.read_csv(output_path)
        assert written["result"].tolist() == ["A", "B"]
