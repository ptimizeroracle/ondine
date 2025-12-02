"""Unit tests for StreamingProcessor."""

import asyncio
from collections.abc import AsyncIterator

import polars as pl
import pytest

from ondine.orchestration.streaming_processor import (
    ChunkResult,
    StreamingProcessor,
    StreamingStats,
)


async def async_iter(items: list[pl.DataFrame]) -> AsyncIterator[pl.DataFrame]:
    """Create an async iterator from a list."""
    for item in items:
        yield item
        await asyncio.sleep(0)


async def identity_processor(chunk: pl.DataFrame) -> pl.DataFrame:
    """Simple processor that returns the chunk unchanged."""
    return chunk


async def slow_processor(chunk: pl.DataFrame) -> pl.DataFrame:
    """Processor that simulates slow processing."""
    await asyncio.sleep(0.01)
    return chunk


async def failing_processor(chunk: pl.DataFrame) -> pl.DataFrame:
    """Processor that always fails."""
    raise ValueError("Simulated failure")


async def sometimes_failing_processor(chunk: pl.DataFrame) -> pl.DataFrame:
    """Processor that fails on chunks with id=1."""
    if chunk["id"][0] == 10:  # Second chunk starts at id=10
        raise ValueError("Chunk 2 failed")
    return chunk


class TestStreamingProcessor:
    """Tests for StreamingProcessor."""

    def test_init(self):
        """Test initialization."""
        processor = StreamingProcessor(max_pending_chunks=5, error_policy="fail")

        assert processor.max_pending_chunks == 5
        assert processor.error_policy == "fail"

    def test_init_invalid_max_pending(self):
        """Test initialization with invalid max_pending_chunks."""
        with pytest.raises(ValueError, match="max_pending_chunks must be at least 1"):
            StreamingProcessor(max_pending_chunks=0)

    @pytest.mark.asyncio
    async def test_process_stream_basic(self):
        """Test basic stream processing."""
        processor = StreamingProcessor(max_pending_chunks=2)

        chunks = [
            pl.DataFrame({"id": list(range(10))}),
            pl.DataFrame({"id": list(range(10, 20))}),
            pl.DataFrame({"id": list(range(20, 30))}),
        ]

        results = []
        async for result in processor.process_stream(
            async_iter(chunks), identity_processor
        ):
            results.append(result)

        assert len(results) == 3
        assert processor.stats.total_chunks == 3
        assert processor.stats.total_rows == 30
        assert processor.stats.successful_chunks == 3
        assert processor.stats.failed_chunks == 0

    @pytest.mark.asyncio
    async def test_process_stream_with_callback(self):
        """Test stream processing with chunk callback."""
        processor = StreamingProcessor()

        chunks = [
            pl.DataFrame({"id": [1, 2, 3]}),
            pl.DataFrame({"id": [4, 5, 6]}),
        ]

        callbacks_received = []

        async def on_complete(result: ChunkResult):
            callbacks_received.append(result)

        results = []
        async for result in processor.process_stream(
            async_iter(chunks), identity_processor, on_chunk_complete=on_complete
        ):
            results.append(result)

        assert len(callbacks_received) == 2
        assert callbacks_received[0].chunk_index == 0
        assert callbacks_received[0].rows_processed == 3
        assert callbacks_received[1].chunk_index == 1

    @pytest.mark.asyncio
    async def test_process_stream_skip_errors(self):
        """Test stream processing with skip error policy."""
        processor = StreamingProcessor(error_policy="skip")

        chunks = [
            pl.DataFrame({"id": list(range(10))}),
            pl.DataFrame({"id": list(range(10, 20))}),  # Will fail
            pl.DataFrame({"id": list(range(20, 30))}),
        ]

        results = []
        async for result in processor.process_stream(
            async_iter(chunks), sometimes_failing_processor
        ):
            results.append(result)

        # Should have 2 successful results (chunks 1 and 3)
        assert len(results) == 2
        assert processor.stats.successful_chunks == 2
        assert processor.stats.failed_chunks == 1

    @pytest.mark.asyncio
    async def test_process_stream_fail_on_error(self):
        """Test stream processing with fail error policy."""
        processor = StreamingProcessor(error_policy="fail")

        chunks = [
            pl.DataFrame({"id": list(range(10))}),
            pl.DataFrame({"id": list(range(10, 20))}),  # Will fail
        ]

        async def process():
            results = []
            async for result in processor.process_stream(
                async_iter(chunks), sometimes_failing_processor
            ):
                results.append(result)

        with pytest.raises(ValueError, match="Chunk 2 failed"):
            await process()

    @pytest.mark.asyncio
    async def test_backpressure(self):
        """Test that backpressure works with slow consumer."""
        processor = StreamingProcessor(max_pending_chunks=2)

        # Create many chunks
        chunks = [pl.DataFrame({"id": [i]}) for i in range(10)]

        # Track peak queue size
        results = []
        async for result in processor.process_stream(
            async_iter(chunks), slow_processor
        ):
            results.append(result)

        assert len(results) == 10
        # Queue should never exceed max_pending_chunks
        assert processor.stats.peak_queue_size <= processor.max_pending_chunks

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test processing empty stream."""
        processor = StreamingProcessor()

        async def empty_iter() -> AsyncIterator[pl.DataFrame]:
            return
            yield  # Make it a generator

        results = []
        async for result in processor.process_stream(empty_iter(), identity_processor):
            results.append(result)

        assert len(results) == 0
        assert processor.stats.total_chunks == 0

    @pytest.mark.asyncio
    async def test_stats_reset_between_runs(self):
        """Test that stats are reset between process_stream calls."""
        processor = StreamingProcessor()

        chunks = [pl.DataFrame({"id": [1, 2, 3]})]

        # First run
        async for _ in processor.process_stream(async_iter(chunks), identity_processor):
            pass

        assert processor.stats.total_rows == 3

        # Second run - stats should reset
        async for _ in processor.process_stream(async_iter(chunks), identity_processor):
            pass

        assert processor.stats.total_rows == 3  # Not 6

    def test_repr(self):
        """Test string representation."""
        processor = StreamingProcessor(max_pending_chunks=5, error_policy="skip")

        repr_str = repr(processor)

        assert "StreamingProcessor" in repr_str
        assert "max_pending_chunks=5" in repr_str
        assert "error_policy='skip'" in repr_str


class TestStreamingStats:
    """Tests for StreamingStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = StreamingStats()

        assert stats.total_chunks == 0
        assert stats.total_rows == 0
        assert stats.successful_chunks == 0
        assert stats.failed_chunks == 0
        assert stats.current_queue_size == 0
        assert stats.peak_queue_size == 0


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_creation(self):
        """Test creating a ChunkResult."""
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = ChunkResult(
            chunk_index=0,
            data=df,
            rows_processed=3,
        )

        assert result.chunk_index == 0
        assert result.rows_processed == 3
        assert result.error is None
        assert result.metadata == {}

    def test_with_error(self):
        """Test ChunkResult with error."""
        df = pl.DataFrame({"id": [1, 2, 3]})
        error = ValueError("test error")
        result = ChunkResult(
            chunk_index=1,
            data=df,
            rows_processed=0,
            error=error,
        )

        assert result.error is error
        assert result.rows_processed == 0

    def test_with_metadata(self):
        """Test ChunkResult with metadata."""
        df = pl.DataFrame({"id": [1]})
        result = ChunkResult(
            chunk_index=0,
            data=df,
            rows_processed=1,
            metadata={"processing_time": 0.5},
        )

        assert result.metadata["processing_time"] == 0.5
