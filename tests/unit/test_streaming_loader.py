"""Unit tests for StreamingDataLoader."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from ondine.adapters.streaming_loader import StreamingDataLoader


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pl.DataFrame(
        {
            "id": list(range(100)),
            "name": [f"item_{i}" for i in range(100)],
            "value": [i * 1.5 for i in range(100)],
        }
    )
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample Parquet file for testing."""
    parquet_path = tmp_path / "test_data.parquet"
    df = pl.DataFrame(
        {
            "id": list(range(100)),
            "name": [f"item_{i}" for i in range(100)],
            "value": [i * 1.5 for i in range(100)],
        }
    )
    df.write_parquet(parquet_path)
    return parquet_path


class TestStreamingDataLoader:
    """Tests for StreamingDataLoader."""

    def test_init_csv(self, sample_csv: Path):
        """Test initialization with CSV file."""
        loader = StreamingDataLoader(sample_csv, chunk_size=10)

        assert loader.path == sample_csv
        assert loader.chunk_size == 10
        assert loader.columns is None

    def test_init_parquet(self, sample_parquet: Path):
        """Test initialization with Parquet file."""
        loader = StreamingDataLoader(sample_parquet, chunk_size=20)

        assert loader.path == sample_parquet
        assert loader.chunk_size == 20

    def test_init_with_columns(self, sample_csv: Path):
        """Test initialization with column selection."""
        loader = StreamingDataLoader(sample_csv, columns=["id", "name"])

        assert loader.columns == ["id", "name"]
        assert set(loader.column_names) == {"id", "name"}

    def test_row_count(self, sample_csv: Path):
        """Test row count property."""
        loader = StreamingDataLoader(sample_csv)

        assert loader.row_count == 100

    def test_schema(self, sample_csv: Path):
        """Test schema property."""
        loader = StreamingDataLoader(sample_csv)

        schema = loader.schema
        assert "id" in schema
        assert "name" in schema
        assert "value" in schema

    def test_column_names(self, sample_csv: Path):
        """Test column_names property."""
        loader = StreamingDataLoader(sample_csv)

        assert loader.column_names == ["id", "name", "value"]

    def test_iter_chunks(self, sample_csv: Path):
        """Test synchronous chunk iteration."""
        loader = StreamingDataLoader(sample_csv, chunk_size=30)

        chunks = list(loader.iter_chunks())

        # Should have 4 chunks: 30 + 30 + 30 + 10 = 100
        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10

        # Verify data integrity
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 100

    def test_iter_chunks_exact_multiple(self, tmp_path: Path):
        """Test chunking when rows are exact multiple of chunk_size."""
        csv_path = tmp_path / "exact.csv"
        df = pl.DataFrame({"id": list(range(50))})
        df.write_csv(csv_path)

        loader = StreamingDataLoader(csv_path, chunk_size=25)
        chunks = list(loader.iter_chunks())

        assert len(chunks) == 2
        assert len(chunks[0]) == 25
        assert len(chunks[1]) == 25

    @pytest.mark.asyncio
    async def test_stream_chunks(self, sample_csv: Path):
        """Test async chunk streaming."""
        loader = StreamingDataLoader(sample_csv, chunk_size=25)

        chunks = []
        async for chunk in loader.stream_chunks():
            chunks.append(chunk)

        assert len(chunks) == 4
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 100

    def test_to_pandas_chunks(self, sample_csv: Path):
        """Test Pandas chunk iteration."""
        loader = StreamingDataLoader(sample_csv, chunk_size=50)

        chunks = list(loader.to_pandas_chunks())

        assert len(chunks) == 2
        # Verify it's actually a Pandas DataFrame
        import pandas as pd

        assert isinstance(chunks[0], pd.DataFrame)
        assert len(chunks[0]) == 50

    @pytest.mark.asyncio
    async def test_stream_pandas_chunks(self, sample_csv: Path):
        """Test async Pandas chunk streaming."""
        import pandas as pd

        loader = StreamingDataLoader(sample_csv, chunk_size=50)

        chunks = []
        async for chunk in loader.stream_pandas_chunks():
            chunks.append(chunk)

        assert len(chunks) == 2
        assert isinstance(chunks[0], pd.DataFrame)

    def test_head(self, sample_csv: Path):
        """Test head preview."""
        loader = StreamingDataLoader(sample_csv)

        preview = loader.head(5)

        assert len(preview) == 5
        assert isinstance(preview, pl.DataFrame)

    def test_estimate_memory(self, sample_csv: Path):
        """Test memory estimation."""
        loader = StreamingDataLoader(sample_csv, chunk_size=10)

        estimate = loader.estimate_memory()

        assert estimate["total_rows"] == 100
        assert estimate["chunk_size"] == 10
        assert estimate["num_chunks"] == 10
        assert "estimated_full_load_mb" in estimate
        assert "estimated_streaming_mb" in estimate
        assert estimate["memory_savings_ratio"] > 1  # Streaming should save memory

    def test_repr(self, sample_csv: Path):
        """Test string representation."""
        loader = StreamingDataLoader(sample_csv, chunk_size=10, columns=["id"])

        repr_str = repr(loader)

        assert "StreamingDataLoader" in repr_str
        assert "chunk_size=10" in repr_str
        assert "columns=['id']" in repr_str

    def test_unsupported_format(self, tmp_path: Path):
        """Test error for unsupported file format."""
        fake_path = tmp_path / "data.xyz"
        fake_path.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported file format"):
            StreamingDataLoader(fake_path)

    def test_parquet_streaming(self, sample_parquet: Path):
        """Test Parquet file streaming."""
        loader = StreamingDataLoader(sample_parquet, chunk_size=25)

        chunks = list(loader.iter_chunks())

        assert len(chunks) == 4
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 100

    def test_column_selection_reduces_data(self, sample_csv: Path):
        """Test that column selection reduces loaded data."""
        loader_all = StreamingDataLoader(sample_csv)
        loader_subset = StreamingDataLoader(sample_csv, columns=["id"])

        all_cols = loader_all.column_names
        subset_cols = loader_subset.column_names

        assert len(all_cols) == 3
        assert len(subset_cols) == 1
        assert subset_cols == ["id"]

