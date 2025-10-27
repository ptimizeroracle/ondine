"""
Streaming execution strategy.

Provides memory-efficient processing for large datasets by processing
data in chunks.
"""

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal

import pandas as pd

from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats
from ondine.orchestration.execution_context import ExecutionContext
from ondine.orchestration.execution_strategy import ExecutionStrategy
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import get_logger

logger = get_logger(__name__)


class StreamingExecutor(ExecutionStrategy):
    """
    Streaming execution strategy.

    Processes data in chunks to maintain constant memory usage.
    Ideal for very large datasets (100K+ rows) that don't fit in memory.

    Benefits:
    - Constant memory footprint
    - Can process unlimited dataset sizes
    - Checkpoints at chunk boundaries
    - Early results available
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize streaming executor.

        Args:
            chunk_size: Number of rows per chunk
        """
        self.chunk_size = chunk_size
        self.logger = logger

    def execute(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> Iterator[pd.DataFrame]:
        """
        Execute stages in streaming mode.

        Args:
            stages: Pipeline stages
            context: Execution context

        Yields:
            DataFrames with processed chunks
        """
        self.logger.info(f"Starting streaming execution (chunk_size={self.chunk_size})")

        # Get data loader stage
        data_loader = stages[0]

        # Stream data in chunks
        chunk_index = 0
        total_rows_processed = 0

        # Read data in chunks
        for chunk in self._read_chunks(data_loader, context):
            self.logger.info(f"Processing chunk {chunk_index} ({len(chunk)} rows)")

            # Process chunk through remaining stages
            result_chunk = self._process_chunk(chunk, stages[1:], context)

            # Update context
            total_rows_processed += len(result_chunk)
            context.update_row(total_rows_processed - 1)

            # Yield result
            yield result_chunk

            chunk_index += 1

        self.logger.info(
            f"Streaming execution complete: {total_rows_processed} rows, "
            f"{chunk_index} chunks"
        )

    def _read_chunks(
        self,
        data_loader: PipelineStage,
        context: ExecutionContext,
    ) -> Iterator[pd.DataFrame]:
        """
        Read data in chunks.

        Uses pandas chunksize parameter for memory-efficient reading.
        """
        # Get data source from data loader
        # For now, this is a simplified implementation
        # In full implementation, would use data_loader's chunked reading

        # Placeholder: would integrate with DataLoaderStage's chunked reading
        yield pd.DataFrame()  # Placeholder

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> pd.DataFrame:
        """
        Process a single chunk through all stages.

        Args:
            chunk: Data chunk to process
            stages: Stages to apply (excluding data loader)
            context: Execution context

        Returns:
            Processed chunk
        """
        current_data = chunk

        for stage in stages:
            self.logger.debug(f"Applying stage: {stage.name}")
            current_data = stage.process(current_data, context)

        return current_data

    def supports_async(self) -> bool:
        """Streaming executor doesn't support async."""
        return False

    def supports_streaming(self) -> bool:
        """Streaming executor supports streaming."""
        return True

    # Alias for backward compatibility
    def execute_stream(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> Iterator[pd.DataFrame]:
        """Alias for execute() method."""
        return self.execute(stages, context)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "StreamingExecutor"


class StreamingResult:
    """
    Result container for streaming execution.

    Provides access to metrics after consuming the stream.
    """

    def __init__(self):
        """Initialize streaming result."""
        self.chunks_processed = 0
        self.total_rows = 0
        self.total_cost = Decimal("0.0")
        self.start_time = datetime.now()
        self.end_time = None

    def add_chunk(self, chunk: pd.DataFrame, cost: Decimal):
        """Add chunk statistics."""
        self.chunks_processed += 1
        self.total_rows += len(chunk)
        self.total_cost += cost

    def finalize(self) -> ExecutionResult:
        """Create final ExecutionResult."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        stats = ProcessingStats(
            total_rows=self.total_rows,
            processed_rows=self.total_rows,
            failed_rows=0,
            skipped_rows=0,
            rows_per_second=self.total_rows / duration if duration > 0 else 0,
            total_duration_seconds=duration,
        )

        costs = CostEstimate(
            total_cost=self.total_cost,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=self.total_rows,
            confidence="actual",
        )

        return ExecutionResult(
            data=pd.DataFrame(),  # Streaming doesn't return full data
            metrics=stats,
            costs=costs,
            start_time=self.start_time,
            end_time=self.end_time,
            success=True,
        )
