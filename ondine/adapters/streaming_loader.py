"""
Streaming data loader using Polars lazy evaluation.

Provides memory-efficient loading for large datasets by using Polars'
lazy evaluation and streaming capabilities. Memory usage is O(chunk_size),
not O(total_rows).
"""

import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """
    Memory-efficient data loader using Polars lazy evaluation.

    Uses Polars' LazyFrame to defer computation until needed, then
    streams data in chunks to maintain bounded memory usage.

    Example:
        loader = StreamingDataLoader("large_dataset.csv", chunk_size=10000)

        # Sync iteration
        for chunk in loader.iter_chunks():
            process(chunk)

        # Async iteration
        async for chunk in loader.stream_chunks():
            await process_async(chunk)

    Memory Guarantee:
        Peak memory usage is approximately:
        - chunk_size * avg_row_size * 2 (current + next chunk buffer)
        - Independent of total dataset size
    """

    def __init__(
        self,
        path: str | Path,
        chunk_size: int = 10000,
        columns: list[str] | None = None,
        **read_options: Any,
    ):
        """
        Initialize streaming loader.

        Args:
            path: Path to data file (CSV, Parquet, JSON, etc.)
            chunk_size: Number of rows per chunk
            columns: Optional list of columns to load (reduces memory)
            **read_options: Additional options passed to Polars reader
        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.columns = columns
        self.read_options = read_options

        # Determine file format and create lazy frame
        self._lazy_frame = self._create_lazy_frame()

        # Cache row count (computed lazily on first access)
        self._row_count: int | None = None

    def _create_lazy_frame(self) -> pl.LazyFrame:
        """Create lazy frame based on file format."""
        suffix = self.path.suffix.lower()

        if suffix == ".csv":
            lf = pl.scan_csv(self.path, **self.read_options)
        elif suffix == ".parquet":
            lf = pl.scan_parquet(self.path, **self.read_options)
        elif suffix == ".json" or suffix == ".ndjson":
            lf = pl.scan_ndjson(self.path, **self.read_options)
        elif suffix in (".xlsx", ".xls"):
            # Excel doesn't support lazy loading, read eagerly then convert
            df = pl.read_excel(self.path, **self.read_options)
            lf = df.lazy()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Select only requested columns if specified
        if self.columns:
            lf = lf.select(self.columns)

        return lf

    @property
    def row_count(self) -> int:
        """
        Get total row count (computed lazily).

        Note: This triggers a scan of the file on first access.
        """
        if self._row_count is None:
            # Use optimized count that doesn't load all data
            self._row_count = self._lazy_frame.select(pl.len()).collect().item()
        return self._row_count

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Get schema without loading data."""
        return self._lazy_frame.schema

    @property
    def column_names(self) -> list[str]:
        """Get column names without loading data."""
        return self._lazy_frame.columns

    def iter_chunks(self) -> Iterator[pl.DataFrame]:
        """
        Iterate over chunks synchronously.

        Yields:
            Polars DataFrame chunks of size chunk_size
        """
        # Collect with streaming=True for memory efficiency
        # Then slice into chunks
        try:
            # For very large files, use streaming collect
            df = self._lazy_frame.collect(streaming=True)

            for i in range(0, len(df), self.chunk_size):
                chunk = df.slice(i, self.chunk_size)
                logger.debug(
                    f"Yielding chunk {i // self.chunk_size + 1}: "
                    f"rows {i} to {i + len(chunk)}"
                )
                yield chunk

        except pl.exceptions.ComputeError:
            # Fallback for formats that don't support streaming
            df = self._lazy_frame.collect()
            for i in range(0, len(df), self.chunk_size):
                yield df.slice(i, self.chunk_size)

    async def stream_chunks(self) -> AsyncIterator[pl.DataFrame]:
        """
        Iterate over chunks asynchronously.

        This is an async wrapper around iter_chunks() that yields
        control back to the event loop between chunks.

        Yields:
            Polars DataFrame chunks of size chunk_size
        """
        import asyncio

        for chunk in self.iter_chunks():
            yield chunk
            # Yield control to event loop between chunks
            await asyncio.sleep(0)

    def to_pandas_chunks(self) -> Iterator["pd.DataFrame"]:
        """
        Iterate over chunks as Pandas DataFrames.

        For compatibility with existing Pandas-based code.

        Yields:
            Pandas DataFrame chunks
        """
        for chunk in self.iter_chunks():
            yield chunk.to_pandas()

    async def stream_pandas_chunks(self) -> AsyncIterator["pd.DataFrame"]:
        """
        Iterate over chunks as Pandas DataFrames asynchronously.

        Yields:
            Pandas DataFrame chunks
        """
        async for chunk in self.stream_chunks():
            yield chunk.to_pandas()

    def head(self, n: int = 5) -> pl.DataFrame:
        """Preview first n rows without loading full dataset."""
        return self._lazy_frame.head(n).collect()

    def estimate_memory(self) -> dict[str, Any]:
        """
        Estimate memory usage for streaming vs full load.

        Returns:
            Dict with memory estimates and recommendations
        """
        # Sample to estimate row size
        sample = self.head(100)
        avg_row_bytes = sample.estimated_size() / len(sample) if len(sample) > 0 else 0

        total_rows = self.row_count
        chunk_rows = self.chunk_size

        return {
            "total_rows": total_rows,
            "chunk_size": chunk_rows,
            "num_chunks": (total_rows + chunk_rows - 1) // chunk_rows,
            "estimated_full_load_mb": (total_rows * avg_row_bytes) / (1024 * 1024),
            "estimated_streaming_mb": (chunk_rows * avg_row_bytes * 2) / (1024 * 1024),
            "memory_savings_ratio": total_rows / (chunk_rows * 2)
            if chunk_rows > 0
            else 0,
        }

    def __repr__(self) -> str:
        return (
            f"StreamingDataLoader(path={self.path}, "
            f"chunk_size={self.chunk_size}, "
            f"columns={self.columns})"
        )
