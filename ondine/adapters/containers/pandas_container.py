"""
Pandas DataFrame container adapter.

Wraps Pandas DataFrames to implement the DataContainer protocol.
Provides backward compatibility for existing code using Pandas.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ondine.core.data_container import BaseDataContainer, Row


class PandasContainer(BaseDataContainer):
    """
    Container adapter for Pandas DataFrames.

    Wraps a Pandas DataFrame to implement the DataContainer protocol.
    This provides backward compatibility for existing code that uses Pandas.

    Example:
        import pandas as pd

        df = pd.read_csv("data.csv")
        container = PandasContainer(df)

        # Use in pipeline
        for row in container:
            print(row)

    Note:
        For new code, consider using PolarsContainer or StreamingCSVContainer
        for better performance and memory efficiency.
    """

    def __init__(self, df: Any):
        """
        Initialize from Pandas DataFrame.

        Args:
            df: Pandas DataFrame (pd.DataFrame)
        """
        self._df = df
        self._validate_pandas()

    def _validate_pandas(self) -> None:
        """Validate that we have a Pandas DataFrame."""
        try:
            import pandas as pd

            if not isinstance(self._df, pd.DataFrame):
                raise TypeError(
                    f"Expected pandas.DataFrame, got {type(self._df).__name__}"
                )
        except ImportError:
            raise ImportError(
                "Pandas is required for PandasContainer. "
                "Install with: pip install pandas"
            )

    def __iter__(self) -> Iterator[Row]:
        """
        Iterate over rows as dictionaries.

        Uses Pandas' itertuples for efficient iteration,
        then converts to dict.

        Yields:
            Row dictionaries
        """
        # itertuples is faster than iterrows
        for row in self._df.itertuples(index=False):
            yield dict(zip(self._df.columns, row, strict=False))

    def __len__(self) -> int:
        """Get number of rows."""
        return len(self._df)

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return list(self._df.columns)

    @property
    def schema(self) -> dict[str, type]:
        """
        Get schema with Python types.

        Maps Pandas/NumPy types to Python types.
        """
        import numpy as np

        type_map = {
            np.int8: int,
            np.int16: int,
            np.int32: int,
            np.int64: int,
            np.uint8: int,
            np.uint16: int,
            np.uint32: int,
            np.uint64: int,
            np.float16: float,
            np.float32: float,
            np.float64: float,
            np.bool_: bool,
            np.object_: str,
        }

        result = {}
        for col, dtype in self._df.dtypes.items():
            result[col] = type_map.get(dtype.type, object)
        return result

    @property
    def dataframe(self) -> Any:
        """Get underlying Pandas DataFrame."""
        return self._df

    def to_list(self) -> list[Row]:
        """Convert to list of dictionaries."""
        return self._df.to_dict(orient="records")

    def head(self, n: int = 5) -> list[Row]:
        """Get first n rows."""
        return self._df.head(n).to_dict(orient="records")

    def tail(self, n: int = 5) -> list[Row]:
        """Get last n rows."""
        return self._df.tail(n).to_dict(orient="records")

    def select(self, columns: list[str]) -> "PandasContainer":
        """
        Select specific columns.

        Args:
            columns: Column names to select

        Returns:
            New PandasContainer with selected columns
        """
        return PandasContainer(self._df[columns])

    def filter(self, predicate: Any) -> "PandasContainer":
        """
        Filter rows.

        Args:
            predicate: Boolean mask, callable, or query string

        Returns:
            New PandasContainer with filtered rows
        """
        if callable(predicate):
            mask = self._df.apply(predicate, axis=1)
            return PandasContainer(self._df[mask])
        elif isinstance(predicate, str):
            return PandasContainer(self._df.query(predicate))
        else:
            return PandasContainer(self._df[predicate])

    def sample(
        self,
        n: int = 10,
        seed: int | None = None,
    ) -> "PandasContainer":
        """
        Get random sample of rows.

        Args:
            n: Number of rows to sample
            seed: Random seed

        Returns:
            New PandasContainer with sampled rows
        """
        return PandasContainer(self._df.sample(n=n, random_state=seed))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> "PandasContainer":
        """
        Create from CSV file.

        Args:
            path: Path to CSV file
            columns: Optional column filter
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            New PandasContainer
        """
        import pandas as pd

        df = pd.read_csv(path, usecols=columns, **kwargs)
        return cls(df)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> "PandasContainer":
        """
        Create from Parquet file.

        Args:
            path: Path to Parquet file
            columns: Optional column filter
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            New PandasContainer
        """
        import pandas as pd

        df = pd.read_parquet(path, columns=columns, **kwargs)
        return cls(df)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> "PandasContainer":
        """
        Create from column-oriented dictionary.

        Args:
            data: Dict mapping column names to lists

        Returns:
            New PandasContainer
        """
        import pandas as pd

        df = pd.DataFrame(data)
        return cls(df)

    @classmethod
    def from_records(
        cls,
        records: list[Row],
        columns: list[str] | None = None,
    ) -> "PandasContainer":
        """
        Create from list of dictionaries.

        Args:
            records: List of row dictionaries
            columns: Optional column order

        Returns:
            New PandasContainer
        """
        import pandas as pd

        df = pd.DataFrame(records)
        if columns:
            df = df[columns]
        return cls(df)

    def to_pandas(self) -> Any:
        """Return underlying Pandas DataFrame."""
        return self._df

    def to_polars(self) -> Any:
        """Convert to Polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for to_polars(). "
                "Install with: pip install polars"
            )
        return pl.from_pandas(self._df)

    def to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Write to CSV file."""
        self._df.to_csv(path, index=False, **kwargs)

    def to_parquet(self, path: str | Path, **kwargs: Any) -> None:
        """Write to Parquet file."""
        self._df.to_parquet(path, index=False, **kwargs)

    def to_json(self, path: str | Path, orient: str = "records") -> None:
        """Write to JSON file."""
        self._df.to_json(path, orient=orient, lines=(orient == "records"))

    def to_excel(self, path: str | Path, **kwargs: Any) -> None:
        """Write to Excel file."""
        self._df.to_excel(path, index=False, **kwargs)

    def __repr__(self) -> str:
        return f"PandasContainer(rows={len(self)}, columns={self.columns})"

