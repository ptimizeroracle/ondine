"""
Polars DataFrame container adapter.

Wraps Polars DataFrames to implement the DataContainer protocol.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ondine.core.data_container import BaseDataContainer, Row


class PolarsContainer(BaseDataContainer):
    """
    Container adapter for Polars DataFrames.

    Wraps a Polars DataFrame to implement the DataContainer protocol,
    allowing it to be used anywhere a DataContainer is expected.

    Example:
        import polars as pl

        df = pl.read_csv("data.csv")
        container = PolarsContainer(df)

        # Use in pipeline
        pipeline = PipelineBuilder.create().from_container(container).build()
    """

    def __init__(self, df: Any):
        """
        Initialize from Polars DataFrame.

        Args:
            df: Polars DataFrame (pl.DataFrame)
        """
        self._df = df
        self._validate_polars()

    def _validate_polars(self) -> None:
        """Validate that we have a Polars DataFrame."""
        try:
            import polars as pl

            if not isinstance(self._df, pl.DataFrame):
                raise TypeError(
                    f"Expected polars.DataFrame, got {type(self._df).__name__}"
                )
        except ImportError:
            raise ImportError(
                "Polars is required for PolarsContainer. "
                "Install with: pip install polars"
            )

    def __iter__(self) -> Iterator[Row]:
        """
        Iterate over rows as dictionaries.

        Uses Polars' iter_rows for efficient iteration.

        Yields:
            Row dictionaries
        """
        for row in self._df.iter_rows(named=True):
            yield dict(row)

    def __len__(self) -> int:
        """Get number of rows."""
        return len(self._df)

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._df.columns

    @property
    def schema(self) -> dict[str, type]:
        """
        Get schema with Python types.

        Maps Polars types to Python types.
        """
        import polars as pl

        type_map = {
            pl.Int8: int,
            pl.Int16: int,
            pl.Int32: int,
            pl.Int64: int,
            pl.UInt8: int,
            pl.UInt16: int,
            pl.UInt32: int,
            pl.UInt64: int,
            pl.Float32: float,
            pl.Float64: float,
            pl.Boolean: bool,
            pl.Utf8: str,
            pl.String: str,
        }

        result = {}
        for name, dtype in self._df.schema.items():
            result[name] = type_map.get(type(dtype), object)
        return result

    @property
    def dataframe(self) -> Any:
        """Get underlying Polars DataFrame."""
        return self._df

    def to_list(self) -> list[Row]:
        """Convert to list of dictionaries."""
        return self._df.to_dicts()

    def head(self, n: int = 5) -> list[Row]:
        """Get first n rows."""
        return self._df.head(n).to_dicts()

    def tail(self, n: int = 5) -> list[Row]:
        """Get last n rows."""
        return self._df.tail(n).to_dicts()

    def select(self, columns: list[str]) -> "PolarsContainer":
        """
        Select specific columns.

        Args:
            columns: Column names to select

        Returns:
            New PolarsContainer with selected columns
        """
        return PolarsContainer(self._df.select(columns))

    def filter(self, predicate: Any) -> "PolarsContainer":
        """
        Filter rows.

        Args:
            predicate: Polars expression or callable

        Returns:
            New PolarsContainer with filtered rows
        """
        if callable(predicate):
            # Convert to Polars-compatible filter
            mask = [predicate(row) for row in self]
            import polars as pl

            filtered = self._df.filter(pl.Series(mask))
            return PolarsContainer(filtered)
        else:
            # Assume it's a Polars expression
            return PolarsContainer(self._df.filter(predicate))

    def sample(self, n: int = 10, seed: int | None = None) -> "PolarsContainer":
        """
        Get random sample of rows.

        Args:
            n: Number of rows to sample
            seed: Random seed

        Returns:
            New PolarsContainer with sampled rows
        """
        return PolarsContainer(self._df.sample(n=n, seed=seed))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> "PolarsContainer":
        """
        Create from CSV file.

        Args:
            path: Path to CSV file
            columns: Optional column filter
            **kwargs: Additional arguments for pl.read_csv

        Returns:
            New PolarsContainer
        """
        import polars as pl

        df = pl.read_csv(path, columns=columns, **kwargs)
        return cls(df)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> "PolarsContainer":
        """
        Create from Parquet file.

        Args:
            path: Path to Parquet file
            columns: Optional column filter
            **kwargs: Additional arguments for pl.read_parquet

        Returns:
            New PolarsContainer
        """
        import polars as pl

        df = pl.read_parquet(path, columns=columns, **kwargs)
        return cls(df)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> "PolarsContainer":
        """
        Create from column-oriented dictionary.

        Args:
            data: Dict mapping column names to lists

        Returns:
            New PolarsContainer
        """
        import polars as pl

        df = pl.DataFrame(data)
        return cls(df)

    @classmethod
    def from_records(
        cls,
        records: list[Row],
        columns: list[str] | None = None,
    ) -> "PolarsContainer":
        """
        Create from list of dictionaries.

        Args:
            records: List of row dictionaries
            columns: Optional column order

        Returns:
            New PolarsContainer
        """
        import polars as pl

        df = pl.DataFrame(records)
        if columns:
            df = df.select(columns)
        return cls(df)

    def to_pandas(self) -> Any:
        """Convert to Pandas DataFrame."""
        return self._df.to_pandas()

    def to_polars(self) -> Any:
        """Return underlying Polars DataFrame."""
        return self._df

    def to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Write to CSV file."""
        self._df.write_csv(path, **kwargs)

    def to_parquet(self, path: str | Path, **kwargs: Any) -> None:
        """Write to Parquet file."""
        self._df.write_parquet(path, **kwargs)

    def to_json(self, path: str | Path) -> None:
        """Write to JSON Lines file."""
        self._df.write_ndjson(path)

    def __repr__(self) -> str:
        return f"PolarsContainer(rows={len(self)}, columns={self.columns})"

