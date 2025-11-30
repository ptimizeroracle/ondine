"""
Streaming result writer for incremental output.

Appends results to output file as they are processed, avoiding
the need to hold all results in memory before writing.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class StreamingResultWriter:
    """
    Append results incrementally to output file.

    Supports multiple output formats (Parquet, CSV, JSON) with
    incremental writing to avoid memory accumulation.

    Example:
        writer = StreamingResultWriter("output.parquet")

        async for chunk in process_stream():
            await writer.append(chunk)

        writer.finalize()

    Formats:
        - Parquet: Best for large datasets (columnar, compressed)
        - CSV: Universal compatibility
        - NDJSON: Streaming-friendly JSON (one record per line)
    """

    def __init__(
        self,
        path: str | Path,
        format: str | None = None,
        **write_options: Any,
    ):
        """
        Initialize streaming writer.

        Args:
            path: Output file path
            format: Output format ("parquet", "csv", "ndjson"). Auto-detected if None.
            **write_options: Additional options passed to Polars writer
        """
        self.path = Path(path)
        self.format = format or self._detect_format()
        self.write_options = write_options

        self._initialized = False
        self._chunks_written = 0
        self._rows_written = 0
        self._temp_chunks: list[pl.DataFrame] = []

        # For Parquet, we need to collect all chunks then write at end
        # For CSV/NDJSON, we can append incrementally
        self._supports_append = self.format in ("csv", "ndjson")

    def _detect_format(self) -> str:
        """Detect format from file extension."""
        suffix = self.path.suffix.lower()
        format_map = {
            ".parquet": "parquet",
            ".pq": "parquet",
            ".csv": "csv",
            ".json": "ndjson",
            ".ndjson": "ndjson",
            ".jsonl": "ndjson",
        }
        return format_map.get(suffix, "parquet")

    def append(self, chunk: pl.DataFrame) -> None:
        """
        Append a chunk to the output.

        For formats that support appending (CSV, NDJSON), writes immediately.
        For Parquet, accumulates chunks and writes on finalize().

        Args:
            chunk: DataFrame chunk to append
        """
        if len(chunk) == 0:
            return

        self._chunks_written += 1
        self._rows_written += len(chunk)

        if self._supports_append:
            self._append_incremental(chunk)
        else:
            # Accumulate for batch write
            self._temp_chunks.append(chunk)

        logger.debug(
            f"Appended chunk {self._chunks_written}: {len(chunk)} rows "
            f"(total: {self._rows_written})"
        )

    async def append_async(self, chunk: pl.DataFrame) -> None:
        """
        Async wrapper for append.

        Note: Polars I/O is synchronous, so this just wraps append()
        and yields control to the event loop.
        """
        import asyncio

        self.append(chunk)
        await asyncio.sleep(0)  # Yield to event loop

    def _append_incremental(self, chunk: pl.DataFrame) -> None:
        """Append chunk to file incrementally."""
        if self.format == "csv":
            # First chunk includes header, subsequent chunks don't
            include_header = not self._initialized
            mode = "w" if not self._initialized else "a"

            # Polars doesn't have append mode, so we use Python file handling
            csv_str = chunk.write_csv(include_header=include_header)
            with open(self.path, mode) as f:
                f.write(csv_str)

        elif self.format == "ndjson":
            mode = "w" if not self._initialized else "a"
            ndjson_str = chunk.write_ndjson()
            with open(self.path, mode) as f:
                f.write(ndjson_str)

        self._initialized = True

    def finalize(self) -> dict[str, Any]:
        """
        Finalize the output file.

        For Parquet, this writes all accumulated chunks.
        For CSV/NDJSON, this is a no-op (already written).

        Returns:
            Dict with write statistics
        """
        if not self._supports_append and self._temp_chunks:
            # Combine all chunks and write
            combined = pl.concat(self._temp_chunks)
            self._write_final(combined)
            self._temp_chunks.clear()

        logger.info(
            f"Finalized output: {self._rows_written} rows in "
            f"{self._chunks_written} chunks -> {self.path}"
        )

        return {
            "path": str(self.path),
            "format": self.format,
            "chunks_written": self._chunks_written,
            "rows_written": self._rows_written,
        }

    def _write_final(self, df: pl.DataFrame) -> None:
        """Write final combined DataFrame."""
        if self.format == "parquet":
            df.write_parquet(self.path, **self.write_options)
        elif self.format == "csv":
            df.write_csv(self.path, **self.write_options)
        elif self.format == "ndjson":
            df.write_ndjson(self.path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    @property
    def rows_written(self) -> int:
        """Total rows written so far."""
        return self._rows_written

    @property
    def chunks_written(self) -> int:
        """Total chunks written so far."""
        return self._chunks_written

    def __repr__(self) -> str:
        return (
            f"StreamingResultWriter(path={self.path}, format={self.format!r}, "
            f"rows={self._rows_written})"
        )


class MultiFormatWriter:
    """
    Write to multiple output formats simultaneously.

    Useful for creating both Parquet (for analysis) and CSV (for sharing).

    Example:
        writer = MultiFormatWriter([
            StreamingResultWriter("output.parquet"),
            StreamingResultWriter("output.csv"),
        ])

        async for chunk in process_stream():
            await writer.append_async(chunk)

        writer.finalize()
    """

    def __init__(self, writers: list[StreamingResultWriter]):
        """
        Initialize multi-format writer.

        Args:
            writers: List of StreamingResultWriter instances
        """
        self.writers = writers

    def append(self, chunk: pl.DataFrame) -> None:
        """Append chunk to all writers."""
        for writer in self.writers:
            writer.append(chunk)

    async def append_async(self, chunk: pl.DataFrame) -> None:
        """Append chunk to all writers asynchronously."""
        for writer in self.writers:
            await writer.append_async(chunk)

    def finalize(self) -> list[dict[str, Any]]:
        """Finalize all writers."""
        return [writer.finalize() for writer in self.writers]

    def __repr__(self) -> str:
        return f"MultiFormatWriter(writers={len(self.writers)})"

