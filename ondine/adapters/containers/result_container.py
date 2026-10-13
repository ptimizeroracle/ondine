"""
Result container for pipeline output.

Provides a unified interface for accessing pipeline results
with conversion to various formats (Pandas, Polars, CSV, etc.).
"""

import csv
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ondine.core.data_container import BaseDataContainer, Row
from ondine.utils.optional_dependencies import raise_excel_extra_error

logger = logging.getLogger(__name__)


class ResultContainerImpl(BaseDataContainer):
    """
    Container for pipeline output with format conversion.

    This is the primary output type from pipeline execution.
    It provides methods to convert results to various formats
    and write to different file types.

    Example:
        result = pipeline.execute()

        # Access as iterator
        for row in result.data:
            print(row)

        # Convert to Pandas
        df = result.data.to_pandas()

        # Convert to Polars
        pl_df = result.data.to_polars()

        # Write to file
        result.data.to_csv("output.csv")
        result.data.to_parquet("output.parquet")

        # The everyday DataFrame idioms work directly
        print(result.data[["product", "category"]])
    """

    def __init__(
        self,
        data: list[Row] | None = None,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize result container.

        Args:
            data: List of result rows
            columns: Column names (inferred if not provided)
            metadata: Optional metadata about the results
        """
        self._data: list[Row] = data or []
        self._columns = columns
        self.metadata = metadata or {}

        # Infer columns from first row if not provided
        if self._columns is None and self._data:
            self._columns = list(self._data[0].keys())

    def __iter__(self) -> Iterator[Row]:
        """Iterate over result rows."""
        return iter(self._data)

    def __len__(self) -> int:
        """Get number of result rows."""
        return len(self._data)

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._columns or []

    def __getitem__(self, key: int | str | list[str]) -> Any:
        """
        Get a row by index, a column by name, or several columns by name.

        - ``container[0]`` -> the row at index 0
        - ``container["name"]`` -> that column, with pandas-like methods
        - ``container[["a", "b"]]`` -> a container holding only those columns,
          in that order, as ``DataFrame[[...]]`` does

        Raises:
            KeyError: A list names a column the container does not have.
        """
        if isinstance(key, list):
            unknown = [column for column in key if column not in self.columns]
            if unknown:
                raise KeyError(f"{unknown} not in columns; available: {self.columns}")
            return self.select(key)
        if isinstance(key, str):
            # Column access: return accessor with pandas-like methods
            values = [row.get(key) for row in self._data]
            return _ColumnAccessor(values)
        return self._data[key]

    @property
    def index(self) -> range:
        """
        Pandas-compatible index property.

        Returns:
            Range object representing row indices
        """
        return range(len(self._data))

    @property
    def iloc(self) -> "_ILocIndexer":
        """
        Pandas-compatible iloc indexer.

        Returns:
            Indexer that supports positional access
        """
        return _ILocIndexer(self._data)

    def iterrows(self) -> Iterator[tuple[int, Row]]:
        """
        Pandas-compatible row iteration.

        Yields:
            Tuples of (index, row_dict)
        """
        yield from enumerate(self._data)

    def to_list(self) -> list[Row]:
        """
        Get results as list of dictionaries.

        Returns:
            List of row dictionaries
        """
        return self._data

    def to_pandas(self) -> Any:
        """
        Convert to Pandas DataFrame.

        Returns:
            pandas.DataFrame

        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for to_pandas(). Install with: pip install pandas"
            )

        return pd.DataFrame(self._data)

    def to_polars(self) -> Any:
        """
        Convert to Polars DataFrame.

        Returns:
            polars.DataFrame

        Raises:
            ImportError: If polars is not installed
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for to_polars(). Install with: pip install polars"
            )

        return pl.DataFrame(self._data)

    def to_csv(
        self,
        path: str | Path,
        *,
        index: bool = False,
        header: bool = True,
        mode: str = "w",
        sep: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write results to CSV file.

        Takes the ``DataFrame.to_csv`` arguments callers reach for, so
        ``result.data.to_csv(path, index=False)`` and the chunk-appending
        ``to_csv(path, mode="a", header=False)`` work as they do on a frame.

        Args:
            path: Output file path
            index: Accepted for DataFrame compatibility. Rows carry no index,
                so none is written; ``True`` is rejected rather than ignored.
            header: Write the column names as the first line.
            mode: File mode: ``"w"`` to replace, ``"a"`` to append.
            sep: Field delimiter (the ``csv`` module's ``delimiter``).
            **kwargs: Additional arguments for csv.DictWriter
        """
        if index:
            raise ValueError(
                "to_csv(index=True) is not supported: result rows have no index. "
                "Use result.to_pandas().to_csv(...) to write one."
            )
        if sep is not None:
            kwargs.setdefault("delimiter", sep)
        path = Path(path)

        with open(path, mode, newline="", encoding="utf-8") as f:
            if not self._data:
                return

            writer = csv.DictWriter(f, fieldnames=self.columns, **kwargs)
            if header:
                writer.writeheader()
            writer.writerows(self._data)

        logger.info(f"Wrote {len(self)} rows to {path}")

    def to_parquet(self, path: str | Path, **kwargs: Any) -> None:
        """
        Write results to Parquet file.

        Uses Polars for efficient Parquet writing.

        Args:
            path: Output file path
            **kwargs: Additional arguments for write_parquet
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for Parquet output. "
                "Install with: pip install polars"
            )

        df = pl.DataFrame(self._data)
        df.write_parquet(path, **kwargs)
        logger.info(f"Wrote {len(self)} rows to {path}")

    def to_json(self, path: str | Path, lines: bool = True) -> None:
        """
        Write results to JSON file.

        Args:
            path: Output file path
            lines: If True, write as JSON Lines (NDJSON)
        """
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            if lines:
                for row in self._data:
                    f.write(json.dumps(row) + "\n")
            else:
                json.dump(self._data, f, indent=2)

        logger.info(f"Wrote {len(self)} rows to {path}")

    def to_excel(self, path: str | Path, sheet_name: str = "Results") -> None:
        """
        Write results to Excel file.

        Args:
            path: Output file path
            sheet_name: Name of the worksheet
        """
        df = self.to_pandas()
        try:
            df.to_excel(path, sheet_name=sheet_name, index=False)
        except ImportError as exc:
            raise_excel_extra_error("Writing Excel files", exc)
        logger.info(f"Wrote {len(self)} rows to {path}")

    def to_string(self, **kwargs: Any) -> str:
        """Render the rows as a text table.

        Args:
            **kwargs: Passed to ``pandas.DataFrame.to_string`` (for example
                ``index=False``).
        """
        return str(self.to_pandas().to_string(**kwargs))

    def to_dict(self, **kwargs: Any) -> Any:
        """Convert to a dictionary, shaped as ``pandas.DataFrame.to_dict`` shapes it.

        Args:
            **kwargs: Passed to ``pandas.DataFrame.to_dict`` (for example
                ``orient="records"``).
        """
        return self.to_pandas().to_dict(**kwargs)

    def head(self, n: int = 5) -> list[Row]:
        """Get first n rows."""
        return self._data[:n]

    def tail(self, n: int = 5) -> list[Row]:
        """Get last n rows."""
        return self._data[-n:]

    def select(self, columns: list[str]) -> "ResultContainerImpl":
        """
        Select specific columns.

        Args:
            columns: Column names to select

        Returns:
            New ResultContainer with selected columns
        """
        filtered_data = [{k: row.get(k) for k in columns} for row in self._data]
        return ResultContainerImpl(data=filtered_data, columns=columns)

    def filter(self, predicate: Callable[[Row], bool]) -> "ResultContainerImpl":
        """
        Filter rows by predicate.

        Args:
            predicate: Function that takes row and returns bool

        Returns:
            New ResultContainer with filtered rows
        """
        filtered_data = [row for row in self._data if predicate(row)]
        return ResultContainerImpl(data=filtered_data, columns=self._columns)

    def append(self, row: Row) -> None:
        """Append a result row."""
        self._data.append(row)

        if self._columns is None:
            self._columns = list(row.keys())

    def extend(self, rows: list[Row]) -> None:
        """Extend with multiple rows."""
        for row in rows:
            self.append(row)

    def merge_with(
        self,
        other: "ResultContainerImpl",
        on: str | list[str] | None = None,
    ) -> "ResultContainerImpl":
        """
        Merge with another result container.

        Args:
            other: Another ResultContainer to merge
            on: Column(s) to merge on (if None, concatenate)

        Returns:
            New merged ResultContainer
        """
        if on is None:
            # Simple concatenation
            merged_data = self._data + other._data
            merged_cols = list(set(self.columns + other.columns))
            return ResultContainerImpl(data=merged_data, columns=merged_cols)

        # Key-based merge
        on_cols = [on] if isinstance(on, str) else on

        # Build lookup from other
        other_lookup = {}
        for row in other:
            key = tuple(row.get(c) for c in on_cols)
            other_lookup[key] = row

        # Merge
        merged_data = []
        for row in self._data:
            key = tuple(row.get(c) for c in on_cols)
            merged_row = row.copy()
            if key in other_lookup:
                for k, v in other_lookup[key].items():
                    if k not in on_cols:
                        merged_row[k] = v
            merged_data.append(merged_row)

        return ResultContainerImpl(data=merged_data)

    @classmethod
    def from_pandas(cls, df: Any) -> "ResultContainerImpl":
        """
        Create from Pandas DataFrame.

        Args:
            df: Pandas DataFrame

        Returns:
            New ResultContainer
        """
        data = df.to_dict(orient="records")
        columns = list(df.columns)
        return cls(data=data, columns=columns)

    @classmethod
    def from_polars(cls, df: Any) -> "ResultContainerImpl":
        """
        Create from Polars DataFrame.

        Args:
            df: Polars DataFrame

        Returns:
            New ResultContainer
        """
        data = df.to_dicts()
        columns = df.columns
        return cls(data=data, columns=columns)

    def __repr__(self) -> str:
        return f"ResultContainer(rows={len(self)}, columns={self.columns})"

    def __str__(self) -> str:
        """The rows as a table, so ``print(result.data)`` shows the results."""
        return self.to_string()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ResultContainerImpl):
            return False
        return self._data == other._data


class _ILocIndexer:
    """
    Pandas-compatible iloc indexer for positional access.

    Supports:
    - container.iloc[0] -> first row
    - container.iloc[0:5] -> slice of rows
    - container.iloc[0]["column"] -> value at row 0, column "column"
    """

    def __init__(self, data: list[Row]):
        self._data = data

    def __getitem__(self, key: int | slice) -> Row | list[Row]:
        """Get row(s) by position."""
        if isinstance(key, slice):
            return self._data[key]
        return self._data[key]


class _ColumnAccessor(list):
    """
    Pandas-compatible column accessor.

    Extends list to add pandas Series methods for compatibility.
    """

    @property
    def iloc(self) -> "_ColumnILocIndexer":
        """Pandas-compatible iloc for column values."""
        return _ColumnILocIndexer(self)

    def tolist(self) -> list:
        """Pandas-compatible tolist()."""
        return list(self)

    def value_counts(self, **kwargs: Any) -> Any:
        """How often each value occurs, most common first, as a pandas Series.

        Args:
            **kwargs: Passed to ``pandas.Series.value_counts``.
        """
        import pandas as pd

        return pd.Series(list(self)).value_counts(**kwargs)

    def isna(self) -> list[bool]:
        """Check for null/None values."""
        return [v is None for v in self]

    def notna(self) -> list[bool]:
        """Check for non-null values."""
        return [v is not None for v in self]

    def notnull(self) -> list[bool]:
        """Alias for notna()."""
        return self.notna()

    def isnull(self) -> list[bool]:
        """Alias for isna()."""
        return self.isna()

    def nunique(self) -> int:
        """Count unique non-null values."""
        return len(set(v for v in self if v is not None))

    def sum(self) -> int:
        """Sum of boolean/numeric values."""
        return sum(1 for v in self if v)


class _ColumnILocIndexer:
    """iloc indexer for column values."""

    def __init__(self, data: list):
        self._data = data

    def __getitem__(self, key: int | slice) -> Any:
        """Get value(s) by position."""
        return self._data[key]
