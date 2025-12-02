"""Integration tests for streaming pipeline execution."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from ondine.adapters.streaming_loader import StreamingDataLoader
from ondine.adapters.streaming_writer import StreamingResultWriter
from ondine.api import PipelineBuilder
from ondine.orchestration.streaming_processor import StreamingProcessor


@pytest.fixture
def large_csv(tmp_path: Path) -> Path:
    """Create a larger CSV file for testing."""
    csv_path = tmp_path / "large_data.csv"

    # Create 1000 rows of test data
    df = pl.DataFrame(
        {
            "id": list(range(1000)),
            "text": [f"Sample text for row {i}" for i in range(1000)],
            "category": [f"cat_{i % 10}" for i in range(1000)],
        }
    )
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.model = "mock-model"
    client.temperature = 0.7
    client.max_tokens = 100

    # Mock async methods
    async def mock_ainvoke(prompt, **kwargs):
        return MagicMock(
            text="Mock response",
            tokens_in=10,
            tokens_out=5,
            cost=0.001,
            latency_ms=50,
        )

    client.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    client.start = AsyncMock()
    client.stop = AsyncMock()

    return client


class TestStreamingDataLoaderIntegration:
    """Integration tests for StreamingDataLoader."""

    def test_load_and_process_large_file(self, large_csv: Path):
        """Test loading and processing a large file in chunks."""
        loader = StreamingDataLoader(large_csv, chunk_size=100)

        total_rows = 0
        chunk_count = 0

        for chunk in loader.iter_chunks():
            total_rows += len(chunk)
            chunk_count += 1

        assert total_rows == 1000
        assert chunk_count == 10

    @pytest.mark.asyncio
    async def test_async_streaming(self, large_csv: Path):
        """Test async streaming of chunks."""
        loader = StreamingDataLoader(large_csv, chunk_size=200)

        chunks = []
        async for chunk in loader.stream_chunks():
            chunks.append(chunk)

        assert len(chunks) == 5
        assert sum(len(c) for c in chunks) == 1000


class TestStreamingProcessorIntegration:
    """Integration tests for StreamingProcessor."""

    @pytest.mark.asyncio
    async def test_process_stream_with_transformation(self, large_csv: Path):
        """Test stream processing with data transformation."""
        loader = StreamingDataLoader(large_csv, chunk_size=100)
        processor = StreamingProcessor(max_pending_chunks=3)

        async def transform(chunk: pl.DataFrame) -> pl.DataFrame:
            # Add a computed column
            return chunk.with_columns((pl.col("id") * 2).alias("id_doubled"))

        results = []
        async for result in processor.process_stream(loader.stream_chunks(), transform):
            results.append(result)

        assert len(results) == 10
        assert "id_doubled" in results[0].columns
        assert processor.stats.total_rows == 1000

    @pytest.mark.asyncio
    async def test_backpressure_with_slow_processing(self, large_csv: Path):
        """Test that backpressure works with slow processing."""
        loader = StreamingDataLoader(large_csv, chunk_size=100)
        processor = StreamingProcessor(max_pending_chunks=2)

        async def slow_transform(chunk: pl.DataFrame) -> pl.DataFrame:
            await asyncio.sleep(0.01)  # Simulate slow processing
            return chunk

        results = []
        async for result in processor.process_stream(
            loader.stream_chunks(), slow_transform
        ):
            results.append(result)

        assert len(results) == 10
        # Queue should never exceed max_pending_chunks
        assert processor.stats.peak_queue_size <= 2


class TestStreamingWriterIntegration:
    """Integration tests for StreamingResultWriter."""

    def test_write_large_file_incrementally(self, tmp_path: Path, large_csv: Path):
        """Test writing a large file incrementally."""
        output_path = tmp_path / "output.parquet"
        loader = StreamingDataLoader(large_csv, chunk_size=100)
        writer = StreamingResultWriter(output_path)

        for chunk in loader.iter_chunks():
            writer.append(chunk)

        stats = writer.finalize()

        assert stats["rows_written"] == 1000
        assert stats["chunks_written"] == 10

        # Verify output file
        result = pl.read_parquet(output_path)
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_async_write_with_processing(self, tmp_path: Path, large_csv: Path):
        """Test async writing with stream processing."""
        output_path = tmp_path / "processed.parquet"
        loader = StreamingDataLoader(large_csv, chunk_size=200)
        processor = StreamingProcessor()
        writer = StreamingResultWriter(output_path)

        async def add_column(chunk: pl.DataFrame) -> pl.DataFrame:
            return chunk.with_columns(pl.lit("processed").alias("status"))

        async for result in processor.process_stream(
            loader.stream_chunks(), add_column
        ):
            await writer.append_async(result)

        stats = writer.finalize()

        assert stats["rows_written"] == 1000

        # Verify processed data
        result = pl.read_parquet(output_path)
        assert "status" in result.columns
        assert result["status"][0] == "processed"


class TestEndToEndStreaming:
    """End-to-end streaming pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_streaming_pipeline(self, tmp_path: Path, large_csv: Path):
        """Test full streaming pipeline: load -> process -> write."""
        output_path = tmp_path / "final_output.parquet"

        # Load
        loader = StreamingDataLoader(large_csv, chunk_size=100, columns=["id", "text"])

        # Process
        processor = StreamingProcessor(max_pending_chunks=3)

        # Write
        writer = StreamingResultWriter(output_path)

        async def process_chunk(chunk: pl.DataFrame) -> pl.DataFrame:
            # Simulate LLM processing - just add a result column
            return chunk.with_columns(
                pl.col("text").str.len_chars().alias("text_length")
            )

        async for result in processor.process_stream(
            loader.stream_chunks(), process_chunk
        ):
            await writer.append_async(result)

        stats = writer.finalize()

        # Verify
        assert stats["rows_written"] == 1000
        assert processor.stats.successful_chunks == 10

        result = pl.read_parquet(output_path)
        assert len(result) == 1000
        assert "text_length" in result.columns

    def test_memory_estimate(self, large_csv: Path):
        """Test memory estimation for streaming."""
        loader = StreamingDataLoader(large_csv, chunk_size=100)

        estimate = loader.estimate_memory()

        assert estimate["total_rows"] == 1000
        assert estimate["chunk_size"] == 100
        assert estimate["num_chunks"] == 10
        assert estimate["memory_savings_ratio"] > 1


class TestPipelineBuilderStreaming:
    """Test PipelineBuilder streaming configuration."""

    def test_with_streaming_configuration(self, large_csv: Path):
        """Test that with_streaming configures the pipeline correctly."""
        pipeline = (
            PipelineBuilder.create()
            .from_csv(
                str(large_csv),
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Process: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .with_streaming(chunk_size=500, max_pending_chunks=2)
            .build()
        )

        # Verify streaming is configured in metadata
        metadata = pipeline.specifications.metadata
        assert metadata.get("streaming", {}).get("enabled") is True
        assert metadata.get("streaming", {}).get("chunk_size") == 500
        assert metadata.get("streaming", {}).get("max_pending_chunks") == 2


class TestMemoryBounded:
    """Tests to verify memory stays bounded during streaming."""

    @pytest.mark.asyncio
    async def test_memory_bounded_processing(self, tmp_path: Path):
        """Test that memory usage stays bounded during streaming."""
        import tracemalloc

        # Create a larger test file
        csv_path = tmp_path / "memory_test.csv"
        df = pl.DataFrame(
            {
                "id": list(range(10000)),
                "text": [f"Text content for row {i} " * 10 for i in range(10000)],
            }
        )
        df.write_csv(csv_path)

        tracemalloc.start()

        loader = StreamingDataLoader(csv_path, chunk_size=500)
        processor = StreamingProcessor(max_pending_chunks=2)
        output_path = tmp_path / "output.parquet"
        writer = StreamingResultWriter(output_path)

        async def identity(chunk: pl.DataFrame) -> pl.DataFrame:
            return chunk

        async for result in processor.process_stream(loader.stream_chunks(), identity):
            await writer.append_async(result)

        writer.finalize()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be bounded (rough check)
        # With 10K rows and chunking, should be much less than loading all at once
        # This is a sanity check - exact values depend on data
        peak_mb = peak / (1024 * 1024)

        # Verify output is complete
        result = pl.read_parquet(output_path)
        assert len(result) == 10000

        # Memory should be reasonable (less than 200MB for this test)
        # This is a loose bound to avoid flaky tests
        assert peak_mb < 200, f"Peak memory {peak_mb:.1f}MB exceeds expected bound"
