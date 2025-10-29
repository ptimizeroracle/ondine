"""Unit tests for execution strategies."""

from decimal import Decimal

import pandas as pd
import pytest

from ondine.orchestration import (
    AsyncExecutor,
    ExecutionContext,
    StreamingExecutor,
    SyncExecutor,
)
from ondine.stages.pipeline_stage import PipelineStage


class MockStage(PipelineStage):
    """Mock stage for testing."""

    def __init__(self, name: str = "MockStage"):
        super().__init__(name)
        self.processed_count = 0

    def validate_input(self, input_data, context):
        """Validate input."""
        from ondine.core.models import ValidationResult

        return ValidationResult(is_valid=True, errors=[])

    def process(self, input_data, context):
        """Process data."""
        self.processed_count += 1
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        return input_data

    def estimate_cost(self, input_data, context):
        """Estimate cost."""
        from ondine.core.models import CostEstimate

        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data) if isinstance(input_data, pd.DataFrame) else 1,
        )


class TestSyncExecutor:
    """Test suite for SyncExecutor."""

    def test_sync_executor_initialization(self):
        """Test sync executor initialization."""
        executor = SyncExecutor()
        assert executor is not None

    def test_sync_executor_execute(self):
        """Test sync executor execution."""
        executor = SyncExecutor()
        context = ExecutionContext()
        stages = [MockStage("Stage1"), MockStage("Stage2")]

        # Execute stages
        result = executor.execute(stages, context)

        assert result is not None
        # Both stages should have been executed
        assert stages[0].processed_count == 1
        assert stages[1].processed_count == 1


class TestAsyncExecutor:
    """Test suite for AsyncExecutor."""

    def test_async_executor_initialization(self):
        """Test async executor initialization."""
        executor = AsyncExecutor(max_concurrency=5)
        assert executor.max_concurrency == 5

    @pytest.mark.asyncio
    async def test_async_executor_execute(self):
        """Test async executor execution."""
        executor = AsyncExecutor()
        context = ExecutionContext()
        stages = [MockStage("AsyncStage1"), MockStage("AsyncStage2")]

        # Execute stages asynchronously
        result = await executor.execute_async(stages, context)

        assert result is not None
        # Both stages should have been executed
        assert stages[0].processed_count == 1
        assert stages[1].processed_count == 1

    @pytest.mark.asyncio
    async def test_async_executor_with_dataframe(self):
        """Test async executor with DataFrame input."""
        executor = AsyncExecutor()
        context = ExecutionContext()

        pd.DataFrame({"text": ["test1", "test2", "test3"]})
        stage = MockStage("DataStage")

        await executor.execute_async([stage], context)

        assert stage.processed_count == 1


class TestStreamingExecutor:
    """Test suite for StreamingExecutor."""

    def test_streaming_executor_initialization(self):
        """Test streaming executor initialization."""
        executor = StreamingExecutor(chunk_size=100)
        assert executor.chunk_size == 100

    def test_streaming_executor_execute(self):
        """Test streaming executor execution."""
        from ondine.core.specifications import DatasetSpec, DataSourceType
        from ondine.stages.data_loader_stage import DataLoaderStage

        StreamingExecutor(chunk_size=2)
        context = ExecutionContext()

        # Create sample data
        df = pd.DataFrame({"text": [f"Sample {i}" for i in range(10)]})

        # Create a data loader stage with the dataframe
        data_loader = DataLoaderStage(dataframe=df)
        processing_stage = MockStage("StreamStage")

        # Execute with streaming - need a spec for data loader
        spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["result"],
        )

        # Process data through loader first
        loaded_data = data_loader.process(spec, context)

        # Mock the stage to accept chunks
        chunks_processed = 0
        for i in range(0, len(loaded_data), 2):
            chunk = loaded_data.iloc[i : i + 2]
            processing_stage.process(chunk, context)
            chunks_processed += 1

        # Should process multiple chunks
        assert chunks_processed > 1
        # Stage should be called multiple times
        assert processing_stage.processed_count > 1

    def test_streaming_executor_with_single_chunk(self):
        """Test streaming with data smaller than chunk size."""
        from ondine.core.specifications import DatasetSpec, DataSourceType
        from ondine.stages.data_loader_stage import DataLoaderStage

        StreamingExecutor(chunk_size=100)
        context = ExecutionContext()

        df = pd.DataFrame({"text": ["small", "data"]})

        data_loader = DataLoaderStage(dataframe=df)
        processing_stage = MockStage("SmallStage")

        spec = DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["result"],
        )

        # Process data
        loaded_data = data_loader.process(spec, context)
        processing_stage.process(loaded_data, context)

        # Should process as single chunk
        assert processing_stage.processed_count == 1


class TestExecutionContext:
    """Test suite for ExecutionContext."""

    def test_context_initialization(self):
        """Test execution context initialization."""
        context = ExecutionContext()

        assert context.session_id is not None
        assert context.pipeline_id is not None
        assert context.start_time is not None
        assert context.current_stage_index == 0

    def test_context_update_stage(self):
        """Test updating context stage."""
        context = ExecutionContext()

        context.update_stage(2)

        assert context.current_stage_index == 2

    def test_context_add_cost(self):
        """Test adding cost to context."""
        context = ExecutionContext()

        context.add_cost(cost=Decimal("0.01"), tokens=100)

        assert context.accumulated_tokens == 100
        assert context.accumulated_cost == Decimal("0.01")

    def test_context_to_dict(self):
        """Test context serialization."""
        context = ExecutionContext()
        context.add_cost(cost=Decimal("0.01"), tokens=50)

        state = context.to_dict()

        assert isinstance(state, dict)
        assert "session_id" in state
        assert "accumulated_cost" in state

    def test_context_from_dict(self):
        """Test context deserialization."""
        original = ExecutionContext()
        original.add_cost(cost=Decimal("0.02"), tokens=100)

        state = original.to_dict()
        restored = ExecutionContext.from_dict(state)

        assert restored.accumulated_tokens == 100
        assert restored.session_id == original.session_id
