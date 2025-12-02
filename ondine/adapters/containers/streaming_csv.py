"""
Streaming CSV container - pure Python, O(1) memory.

This is the default container for ondine. It uses Python's built-in
csv module to stream rows without loading the entire file into memory.
"""

import csv
import logging
from collections.abc import Iterator
from pathlib import Path

from ondine.core.data_container import BaseDataContainer, Row

logger = logging.getLogger(__name__)


class StreamingCSVContainer(BaseDataContainer):
    """
    Memory-efficient CSV container using Python's csv module.

    This is the default container for ondine. It streams rows
    one at a time, keeping memory usage O(1) regardless of file size.

    Example:
        container = StreamingCSVContainer("large_file.csv")

        # Memory stays constant even for 5M rows
        for row in container:
            process(row)

    Features:
        - O(1) memory usage
        - Lazy row count (cached after first access)
        - Supports custom delimiters and encodings
        - Pure Python (no external dependencies)
    """

    def __init__(
        self,
        path: str | Path,
        delimiter: str = ",",
        encoding: str = "utf-8",
        columns: list[str] | None = None,
    ):
        """
        Initialize streaming CSV container.

        Args:
            path: Path to CSV file
            delimiter: Field delimiter (default: comma)
            encoding: File encoding (default: utf-8)
            columns: Optional column filter (only yield these columns)
        """
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self._column_filter = columns

        # Lazy-loaded properties
        self._columns: list[str] | None = None
        self._row_count: int | None = None
        self._schema: dict[str, type] | None = None

    def __iter__(self) -> Iterator[Row]:
        """
        Iterate over rows as dictionaries.

        Each iteration opens the file fresh, allowing multiple passes.
        Memory usage is O(1) - only one row in memory at a time.

        Yields:
            Row dictionaries
        """
        with open(self.path, encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)

            # Cache columns on first iteration
            if self._columns is None and reader.fieldnames:
                self._columns = list(reader.fieldnames)

            for row in reader:
                if self._column_filter:
                    yield {k: row[k] for k in self._column_filter if k in row}
                else:
                    yield dict(row)

    def __len__(self) -> int:
        """
        Get total row count.

        Note: First call requires scanning the entire file.
        Result is cached for subsequent calls.

        Returns:
            Number of data rows (excluding header)
        """
        if self._row_count is None:
            # Count lines efficiently
            with open(self.path, encoding=self.encoding) as f:
                # Skip header
                next(f, None)
                self._row_count = sum(1 for _ in f)
            logger.debug(f"Counted {self._row_count} rows in {self.path}")
        return self._row_count

    @property
    def columns(self) -> list[str]:
        """
        Get column names from CSV header.

        Returns:
            List of column names
        """
        if self._columns is None:
            with open(self.path, encoding=self.encoding, newline="") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                header = next(reader, [])
                self._columns = list(header)
        return self._column_filter or self._columns

    @property
    def schema(self) -> dict[str, type]:
        """
        Infer schema from first data row.

        Returns:
            Mapping of column names to inferred Python types
        """
        if self._schema is None:
            self._schema = {}
            for row in self:
                for key, value in row.items():
                    # Try to infer type
                    if value is None or value == "":
                        self._schema[key] = str
                    else:
                        self._schema[key] = self._infer_type(value)
                break  # Only need first row
        return self._schema

    def _infer_type(self, value: str) -> type:
        """Infer Python type from string value."""
        # Try int
        try:
            int(value)
            return int
        except ValueError:
            pass

        # Try float
        try:
            float(value)
            return float
        except ValueError:
            pass

        # Try bool
        if value.lower() in ("true", "false"):
            return bool

        return str

    def select(self, columns: list[str]) -> "StreamingCSVContainer":
        """
        Select specific columns.

        Returns a new container that only yields selected columns.
        This is still O(1) memory.

        Args:
            columns: Column names to select

        Returns:
            New StreamingCSVContainer with column filter
        """
        return StreamingCSVContainer(
            path=self.path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            columns=columns,
        )

    def head(self, n: int = 5) -> list[Row]:
        """
        Get first n rows efficiently.

        Args:
            n: Number of rows

        Returns:
            List of first n rows
        """
        rows = []
        for i, row in enumerate(self):
            if i >= n:
                break
            rows.append(row)
        return rows

    def sample(self, n: int = 10, seed: int | None = None) -> list[Row]:
        """
        Get random sample of rows.

        Note: This requires reading the entire file.

        Args:
            n: Number of rows to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled rows
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Reservoir sampling for memory efficiency
        reservoir: list[Row] = []
        for i, row in enumerate(self):
            if i < n:
                reservoir.append(row)
            else:
                j = random.randint(0, i)  # noqa: S311  # nosec B311
                if j < n:
                    reservoir[j] = row
        return reservoir

    def __repr__(self) -> str:
        return (
            f"StreamingCSVContainer(path={self.path}, "
            f"columns={len(self.columns)}, "
            f"delimiter={self.delimiter!r})"
        )


class StreamingParquetContainer(BaseDataContainer):
    """
    Memory-efficient Parquet container using PyArrow.

    Streams Parquet files row by row using PyArrow's
    record batch iteration.
    """

    def __init__(
        self,
        path: str | Path,
        columns: list[str] | None = None,
        batch_size: int = 10000,
    ):
        """
        Initialize streaming Parquet container.

        Args:
            path: Path to Parquet file
            columns: Optional column filter
            batch_size: Rows per batch when reading
        """
        self.path = Path(path)
        self._column_filter = columns
        self.batch_size = batch_size

        self._columns: list[str] | None = None
        self._row_count: int | None = None

    def __iter__(self) -> Iterator[Row]:
        """Iterate over rows using PyArrow batches."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "PyArrow is required for Parquet support. "
                "Install with: pip install pyarrow"
            )

        parquet_file = pq.ParquetFile(self.path)

        if self._columns is None:
            self._columns = parquet_file.schema.names

        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=self._column_filter,
        ):
            # Convert batch to list of dicts
            table = batch.to_pydict()
            num_rows = len(next(iter(table.values())))

            for i in range(num_rows):
                yield {k: v[i] for k, v in table.items()}

    def __len__(self) -> int:
        """Get row count from Parquet metadata."""
        if self._row_count is None:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError("PyArrow required for Parquet")

            parquet_file = pq.ParquetFile(self.path)
            self._row_count = parquet_file.metadata.num_rows
        return self._row_count

    @property
    def columns(self) -> list[str]:
        """Get column names from Parquet schema."""
        if self._columns is None:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError("PyArrow required for Parquet")

            parquet_file = pq.ParquetFile(self.path)
            self._columns = parquet_file.schema.names
        return self._column_filter or self._columns

    def __repr__(self) -> str:
        return f"StreamingParquetContainer(path={self.path})"


class StreamingJSONContainer(BaseDataContainer):
    """
    Memory-efficient JSON Lines (NDJSON) container.

    Streams JSON Lines files row by row.
    """

    def __init__(
        self,
        path: str | Path,
        encoding: str = "utf-8",
        columns: list[str] | None = None,
    ):
        """
        Initialize streaming JSON container.

        Args:
            path: Path to NDJSON file
            encoding: File encoding
            columns: Optional column filter
        """
        self.path = Path(path)
        self.encoding = encoding
        self._column_filter = columns

        self._columns: list[str] | None = None
        self._row_count: int | None = None

    def __iter__(self) -> Iterator[Row]:
        """Iterate over JSON lines."""
        import json

        with open(self.path, encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                # Cache columns from first row
                if self._columns is None:
                    self._columns = list(row.keys())

                if self._column_filter:
                    yield {k: row[k] for k in self._column_filter if k in row}
                else:
                    yield row

    def __len__(self) -> int:
        """Count lines in file."""
        if self._row_count is None:
            with open(self.path, encoding=self.encoding) as f:
                self._row_count = sum(1 for line in f if line.strip())
        return self._row_count

    @property
    def columns(self) -> list[str]:
        """Get columns from first row."""
        if self._columns is None:
            for row in self:
                break  # Just need first row to populate _columns
        return self._column_filter or self._columns or []

    def __repr__(self) -> str:
        return f"StreamingJSONContainer(path={self.path})"
