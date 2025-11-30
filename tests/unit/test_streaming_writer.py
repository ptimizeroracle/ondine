"""Unit tests for StreamingResultWriter."""

from pathlib import Path

import polars as pl
import pytest

from ondine.adapters.streaming_writer import (
    MultiFormatWriter,
    StreamingResultWriter,
)


@pytest.fixture
def sample_chunks() -> list[pl.DataFrame]:
    """Create sample DataFrame chunks for testing."""
    return [
        pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        pl.DataFrame({"id": [4, 5, 6], "name": ["d", "e", "f"]}),
        pl.DataFrame({"id": [7, 8], "name": ["g", "h"]}),
    ]


class TestStreamingResultWriter:
    """Tests for StreamingResultWriter."""

    def test_init_parquet(self, tmp_path: Path):
        """Test initialization with Parquet format."""
        writer = StreamingResultWriter(tmp_path / "output.parquet")

        assert writer.format == "parquet"
        assert writer.rows_written == 0
        assert writer.chunks_written == 0

    def test_init_csv(self, tmp_path: Path):
        """Test initialization with CSV format."""
        writer = StreamingResultWriter(tmp_path / "output.csv")

        assert writer.format == "csv"

    def test_init_ndjson(self, tmp_path: Path):
        """Test initialization with NDJSON format."""
        writer = StreamingResultWriter(tmp_path / "output.ndjson")

        assert writer.format == "ndjson"

    def test_init_explicit_format(self, tmp_path: Path):
        """Test initialization with explicit format override."""
        writer = StreamingResultWriter(tmp_path / "output.txt", format="csv")

        assert writer.format == "csv"

    def test_append_parquet(self, tmp_path: Path, sample_chunks: list[pl.DataFrame]):
        """Test appending chunks to Parquet file."""
        output_path = tmp_path / "output.parquet"
        writer = StreamingResultWriter(output_path)

        for chunk in sample_chunks:
            writer.append(chunk)

        assert writer.rows_written == 8
        assert writer.chunks_written == 3

        # Finalize and verify
        stats = writer.finalize()
        assert stats["rows_written"] == 8
        assert output_path.exists()

        # Read back and verify
        result = pl.read_parquet(output_path)
        assert len(result) == 8

    def test_append_csv(self, tmp_path: Path, sample_chunks: list[pl.DataFrame]):
        """Test appending chunks to CSV file."""
        output_path = tmp_path / "output.csv"
        writer = StreamingResultWriter(output_path)

        for chunk in sample_chunks:
            writer.append(chunk)

        stats = writer.finalize()
        assert stats["rows_written"] == 8

        # Read back and verify
        result = pl.read_csv(output_path)
        assert len(result) == 8
        assert list(result["id"]) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_append_ndjson(self, tmp_path: Path, sample_chunks: list[pl.DataFrame]):
        """Test appending chunks to NDJSON file."""
        output_path = tmp_path / "output.ndjson"
        writer = StreamingResultWriter(output_path)

        for chunk in sample_chunks:
            writer.append(chunk)

        stats = writer.finalize()
        assert stats["rows_written"] == 8

        # Read back and verify
        result = pl.read_ndjson(output_path)
        assert len(result) == 8

    def test_append_empty_chunk(self, tmp_path: Path):
        """Test that empty chunks are skipped."""
        output_path = tmp_path / "output.parquet"
        writer = StreamingResultWriter(output_path)

        writer.append(pl.DataFrame({"id": [1]}))
        writer.append(pl.DataFrame({"id": []}))  # Empty
        writer.append(pl.DataFrame({"id": [2]}))

        assert writer.chunks_written == 2  # Empty chunk not counted
        assert writer.rows_written == 2

    @pytest.mark.asyncio
    async def test_append_async(self, tmp_path: Path, sample_chunks: list[pl.DataFrame]):
        """Test async append."""
        output_path = tmp_path / "output.parquet"
        writer = StreamingResultWriter(output_path)

        for chunk in sample_chunks:
            await writer.append_async(chunk)

        assert writer.rows_written == 8

    def test_finalize_returns_stats(self, tmp_path: Path):
        """Test that finalize returns statistics."""
        output_path = tmp_path / "output.csv"
        writer = StreamingResultWriter(output_path)

        writer.append(pl.DataFrame({"id": [1, 2, 3]}))
        writer.append(pl.DataFrame({"id": [4, 5]}))

        stats = writer.finalize()

        assert stats["path"] == str(output_path)
        assert stats["format"] == "csv"
        assert stats["chunks_written"] == 2
        assert stats["rows_written"] == 5

    def test_repr(self, tmp_path: Path):
        """Test string representation."""
        writer = StreamingResultWriter(tmp_path / "output.parquet")
        writer.append(pl.DataFrame({"id": [1, 2, 3]}))

        repr_str = repr(writer)

        assert "StreamingResultWriter" in repr_str
        assert "format='parquet'" in repr_str
        assert "rows=3" in repr_str

    def test_csv_header_only_on_first_chunk(self, tmp_path: Path):
        """Test that CSV header is only written once."""
        output_path = tmp_path / "output.csv"
        writer = StreamingResultWriter(output_path)

        writer.append(pl.DataFrame({"id": [1], "name": ["a"]}))
        writer.append(pl.DataFrame({"id": [2], "name": ["b"]}))
        writer.finalize()

        # Read raw content and count header occurrences
        content = output_path.read_text()
        lines = content.strip().split("\n")

        assert lines[0] == "id,name"  # Header
        assert len(lines) == 3  # Header + 2 data rows

    def test_jsonl_extension(self, tmp_path: Path):
        """Test .jsonl extension is recognized as NDJSON."""
        writer = StreamingResultWriter(tmp_path / "output.jsonl")
        assert writer.format == "ndjson"

    def test_pq_extension(self, tmp_path: Path):
        """Test .pq extension is recognized as Parquet."""
        writer = StreamingResultWriter(tmp_path / "output.pq")
        assert writer.format == "parquet"


class TestMultiFormatWriter:
    """Tests for MultiFormatWriter."""

    def test_append_to_multiple(self, tmp_path: Path, sample_chunks: list[pl.DataFrame]):
        """Test appending to multiple writers."""
        parquet_path = tmp_path / "output.parquet"
        csv_path = tmp_path / "output.csv"

        writer = MultiFormatWriter([
            StreamingResultWriter(parquet_path),
            StreamingResultWriter(csv_path),
        ])

        for chunk in sample_chunks:
            writer.append(chunk)

        stats = writer.finalize()

        assert len(stats) == 2
        assert parquet_path.exists()
        assert csv_path.exists()

        # Verify both have same data
        parquet_data = pl.read_parquet(parquet_path)
        csv_data = pl.read_csv(csv_path)
        assert len(parquet_data) == len(csv_data) == 8

    @pytest.mark.asyncio
    async def test_append_async_to_multiple(
        self, tmp_path: Path, sample_chunks: list[pl.DataFrame]
    ):
        """Test async append to multiple writers."""
        parquet_path = tmp_path / "output.parquet"
        csv_path = tmp_path / "output.csv"

        writer = MultiFormatWriter([
            StreamingResultWriter(parquet_path),
            StreamingResultWriter(csv_path),
        ])

        for chunk in sample_chunks:
            await writer.append_async(chunk)

        stats = writer.finalize()
        assert len(stats) == 2

    def test_repr(self, tmp_path: Path):
        """Test string representation."""
        writer = MultiFormatWriter([
            StreamingResultWriter(tmp_path / "a.parquet"),
            StreamingResultWriter(tmp_path / "b.csv"),
        ])

        assert "MultiFormatWriter(writers=2)" in repr(writer)

