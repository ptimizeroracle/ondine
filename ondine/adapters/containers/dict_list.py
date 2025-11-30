"""
In-memory dictionary list container.

Simple container backed by a list of dictionaries.
Useful for small datasets, test fixtures, and intermediate results.
"""

from collections.abc import Iterator
from typing import Any

from ondine.core.data_container import BaseDataContainer, Row


class DictListContainer(BaseDataContainer):
    """
    In-memory container backed by list of dictionaries.

    This is the simplest container implementation. Use it for:
    - Small datasets that fit in memory
    - Test fixtures
    - Intermediate pipeline results
    - Converting from other formats

    Example:
        # From list of dicts
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        container = DictListContainer(data)

        # Iterate
        for row in container:
            print(row["name"])
    """

    def __init__(
        self,
        data: list[Row] | None = None,
        columns: list[str] | None = None,
    ):
        """
        Initialize from list of dictionaries.

        Args:
            data: List of row dictionaries
            columns: Optional explicit column order
        """
        self._data: list[Row] = data or []
        self._columns = columns

        # Infer columns from first row if not provided
        if self._columns is None and self._data:
            self._columns = list(self._data[0].keys())

    def __iter__(self) -> Iterator[Row]:
        """Iterate over rows."""
        return iter(self._data)

    def __len__(self) -> int:
        """Get row count."""
        return len(self._data)

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._columns or []

    @property
    def schema(self) -> dict[str, type]:
        """Infer schema from first row."""
        if not self._data:
            return {}
        return {k: type(v) for k, v in self._data[0].items()}

    def __getitem__(self, idx: int) -> Row:
        """Get row by index."""
        return self._data[idx]

    def append(self, row: Row) -> None:
        """
        Append a row.

        Args:
            row: Row dictionary to append
        """
        self._data.append(row)

        # Update columns if needed
        if self._columns is None:
            self._columns = list(row.keys())
        else:
            # Add any new columns
            for key in row.keys():
                if key not in self._columns:
                    self._columns.append(key)

    def extend(self, rows: list[Row]) -> None:
        """
        Extend with multiple rows.

        Args:
            rows: List of row dictionaries
        """
        for row in rows:
            self.append(row)

    def to_list(self) -> list[Row]:
        """Return the underlying list (no copy)."""
        return self._data

    def copy(self) -> "DictListContainer":
        """Create a deep copy."""
        import copy

        return DictListContainer(
            data=copy.deepcopy(self._data),
            columns=self._columns.copy() if self._columns else None,
        )

    def select(self, columns: list[str]) -> "DictListContainer":
        """
        Select specific columns.

        Args:
            columns: Column names to select

        Returns:
            New container with only selected columns
        """
        filtered_data = [{k: row.get(k) for k in columns} for row in self._data]
        return DictListContainer(data=filtered_data, columns=columns)

    def filter(self, predicate: callable) -> "DictListContainer":
        """
        Filter rows by predicate.

        Args:
            predicate: Function that takes row and returns bool

        Returns:
            New container with filtered rows
        """
        filtered_data = [row for row in self._data if predicate(row)]
        return DictListContainer(data=filtered_data, columns=self._columns)

    def map(self, func: callable) -> "DictListContainer":
        """
        Apply function to each row.

        Args:
            func: Function that takes row and returns modified row

        Returns:
            New container with transformed rows
        """
        mapped_data = [func(row) for row in self._data]
        return DictListContainer(data=mapped_data)

    def sort(self, key: str, reverse: bool = False) -> "DictListContainer":
        """
        Sort rows by key.

        Args:
            key: Column name to sort by
            reverse: Sort descending if True

        Returns:
            New container with sorted rows
        """
        sorted_data = sorted(self._data, key=lambda r: r.get(key), reverse=reverse)
        return DictListContainer(data=sorted_data, columns=self._columns)

    def head(self, n: int = 5) -> list[Row]:
        """Get first n rows."""
        return self._data[:n]

    def tail(self, n: int = 5) -> list[Row]:
        """Get last n rows."""
        return self._data[-n:]

    @classmethod
    def from_records(cls, records: list[tuple], columns: list[str]) -> "DictListContainer":
        """
        Create from list of tuples with column names.

        Args:
            records: List of tuples (one per row)
            columns: Column names

        Returns:
            New DictListContainer
        """
        data = [dict(zip(columns, record, strict=False)) for record in records]
        return cls(data=data, columns=columns)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> "DictListContainer":
        """
        Create from column-oriented dictionary.

        Args:
            data: Dict mapping column names to lists of values

        Returns:
            New DictListContainer
        """
        columns = list(data.keys())
        if not columns:
            return cls(data=[], columns=[])

        num_rows = len(data[columns[0]])
        rows = [{col: data[col][i] for col in columns} for i in range(num_rows)]
        return cls(data=rows, columns=columns)

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Convert to column-oriented dictionary.

        Returns:
            Dict mapping column names to lists of values
        """
        if not self._data:
            return {col: [] for col in self.columns}

        result: dict[str, list[Any]] = {col: [] for col in self.columns}
        for row in self._data:
            for col in self.columns:
                result[col].append(row.get(col))
        return result

    def __repr__(self) -> str:
        return f"DictListContainer(rows={len(self)}, columns={self.columns})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DictListContainer):
            return False
        return self._data == other._data and self._columns == other._columns

