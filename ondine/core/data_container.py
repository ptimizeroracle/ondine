"""
Abstract data container protocol - framework agnostic.

This module defines the core abstractions for data handling in ondine.
Instead of coupling to Pandas or Polars, we use a protocol-based approach
that allows any backend to be plugged in.

The key insight: LLM pipelines are fundamentally row-by-row operations.
We don't need complex tabular operations - just iteration over rows.
"""

from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol, TypeVar, runtime_checkable

# The universal row type - a simple dictionary
# This is what all stages work with internally
Row = dict[str, Any]

# Type variable for container implementations
T = TypeVar("T", bound="DataContainer")


@runtime_checkable
class DataContainer(Protocol):
    """
    Abstract protocol for data containers.

    This is the core abstraction that decouples ondine from any specific
    data framework (Pandas, Polars, etc.). Implementations can be:
    - StreamingCSVContainer (default, O(1) memory)
    - PolarsContainer (for Polars users)
    - PandasContainer (backward compatibility)
    - DictListContainer (in-memory)
    - Custom implementations

    Example:
        # Any of these work:
        container = StreamingCSVContainer("data.csv")
        container = PolarsContainer(pl.read_csv("data.csv"))
        container = PandasContainer(pd.read_csv("data.csv"))

        # Iterate uniformly
        for row in container:
            prompt = template.format(**row)
    """

    def __iter__(self) -> Iterator[Row]:
        """
        Iterate over rows as dictionaries.

        This is the primary interface for accessing data.
        Implementations should be memory-efficient where possible.

        Yields:
            Row dictionaries with column names as keys
        """
        ...

    def __len__(self) -> int:
        """
        Get total row count.

        Note: For streaming containers, this may require a full scan
        on first access. The result should be cached.

        Returns:
            Number of rows in the container
        """
        ...

    @property
    def columns(self) -> list[str]:
        """
        Get column names.

        Returns:
            List of column names in order
        """
        ...

    @property
    def schema(self) -> dict[str, type]:
        """
        Get column types (optional, may return empty dict).

        Returns:
            Mapping of column names to Python types
        """
        ...


class BaseDataContainer:
    """
    Base class for DataContainer implementations.

    Provides common functionality and sensible defaults.
    Subclasses must implement __iter__, __len__, and columns.
    """

    @property
    def schema(self) -> dict[str, type]:
        """Default schema implementation - infer from first row."""
        for row in self:
            return {k: type(v) for k, v in row.items()}
        return {}

    def to_list(self) -> list[Row]:
        """
        Materialize all rows into a list.

        Warning: This loads all data into memory. Use sparingly
        for large datasets.

        Returns:
            List of all rows as dictionaries
        """
        return list(self)

    def head(self, n: int = 5) -> list[Row]:
        """
        Get first n rows.

        Args:
            n: Number of rows to return

        Returns:
            List of first n rows
        """
        rows = []
        for i, row in enumerate(self):
            if i >= n:
                break
            rows.append(row)
        return rows

    def select(self, columns: list[str]) -> "DataContainer":
        """
        Select specific columns.

        Args:
            columns: Column names to select

        Returns:
            New container with only selected columns
        """
        from ondine.adapters.containers.dict_list import DictListContainer

        rows = [{k: row[k] for k in columns if k in row} for row in self]
        return DictListContainer(rows, columns=columns)

    def filter(self, predicate: callable) -> "DataContainer":
        """
        Filter rows by predicate.

        Args:
            predicate: Function that takes a row and returns bool

        Returns:
            New container with filtered rows
        """
        from ondine.adapters.containers.dict_list import DictListContainer

        rows = [row for row in self if predicate(row)]
        return DictListContainer(rows, columns=self.columns)

    def map(self, func: callable) -> "DataContainer":
        """
        Apply function to each row.

        Args:
            func: Function that takes a row and returns a modified row

        Returns:
            New container with transformed rows
        """
        from ondine.adapters.containers.dict_list import DictListContainer

        rows = [func(row) for row in self]
        return DictListContainer(rows)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(columns={self.columns}, len={len(self)})"


class AsyncDataContainer(Protocol):
    """
    Async version of DataContainer for streaming scenarios.

    Use this when data needs to be loaded asynchronously
    (e.g., from network, database, or very large files).
    """

    def __aiter__(self) -> AsyncIterator[Row]:
        """Async iterate over rows."""
        ...

    async def __len__(self) -> int:
        """Get row count (may require async scan)."""
        ...

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        ...


@runtime_checkable
class ResultContainer(Protocol):
    """
    Protocol for pipeline output containers.

    Similar to DataContainer but with additional methods
    for output serialization and format conversion.
    """

    def __iter__(self) -> Iterator[Row]:
        """Iterate over result rows."""
        ...

    def __len__(self) -> int:
        """Get row count."""
        ...

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        ...

    @abstractmethod
    def to_pandas(self) -> Any:
        """
        Convert to Pandas DataFrame.

        Returns:
            pandas.DataFrame
        """
        ...

    @abstractmethod
    def to_polars(self) -> Any:
        """
        Convert to Polars DataFrame.

        Returns:
            polars.DataFrame
        """
        ...

    @abstractmethod
    def to_list(self) -> list[Row]:
        """
        Convert to list of dictionaries.

        Returns:
            List of row dictionaries
        """
        ...

    @abstractmethod
    def to_csv(self, path: str) -> None:
        """Write to CSV file."""
        ...

    @abstractmethod
    def to_parquet(self, path: str) -> None:
        """Write to Parquet file."""
        ...

    @abstractmethod
    def to_json(self, path: str) -> None:
        """Write to JSON/NDJSON file."""
        ...


# Type aliases for clarity
RowIterator = Iterator[Row]
AsyncRowIterator = AsyncIterator[Row]

