"""
Data I/O adapters for reading and writing tabular data.

Provides unified interface for multiple data formats following the
Adapter pattern.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import polars as pl

from ondine.core.models import WriteConfirmation
from ondine.core.specifications import DataSourceType


class DataReader(ABC):
    """
    Abstract base class for data readers.

    Follows Open/Closed principle: open for extension via new readers,
    closed for modification.
    """

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """
        Read entire dataset.

        Returns:
            DataFrame with all data
        """
        pass

    @abstractmethod
    def read_chunked(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Read data in chunks for memory efficiency.

        Args:
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        pass


class CSVReader(DataReader):
    """CSV file reader implementation."""

    def __init__(
        self,
        file_path: Path,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ):
        """
        Initialize CSV reader.

        Args:
            file_path: Path to CSV file
            delimiter: Column delimiter
            encoding: File encoding
        """
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding

    def read(self) -> pd.DataFrame:
        """Read entire CSV file."""
        return pd.read_csv(
            self.file_path,
            delimiter=self.delimiter,
            encoding=self.encoding,
        )

    def read_chunked(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Read CSV in chunks."""
        yield from pd.read_csv(
            self.file_path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            chunksize=chunk_size,
        )


class ExcelReader(DataReader):
    """Excel file reader implementation."""

    def __init__(self, file_path: Path, sheet_name: str | int = 0):
        """
        Initialize Excel reader.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
        """
        self.file_path = file_path
        self.sheet_name = sheet_name

    def read(self) -> pd.DataFrame:
        """Read entire Excel file."""
        return pd.read_excel(self.file_path, sheet_name=self.sheet_name)

    def read_chunked(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Read Excel in chunks.

        Note: Excel doesn't support native chunking, so we load all
        and yield chunks.
        """
        df = self.read()
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size]


class ParquetReader(DataReader):
    """Parquet file reader implementation."""

    def __init__(self, file_path: Path):
        """
        Initialize Parquet reader.

        Args:
            file_path: Path to Parquet file
        """
        self.file_path = file_path

    def read(self) -> pd.DataFrame:
        """Read entire Parquet file."""
        return pd.read_parquet(self.file_path)

    def read_chunked(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Read Parquet in chunks using Polars for efficiency.
        """
        # Use Polars for efficient chunked reading
        lf = pl.scan_parquet(self.file_path)

        # Read in batches
        total_rows = lf.select(pl.len()).collect().item()

        for i in range(0, total_rows, chunk_size):
            chunk = lf.slice(i, chunk_size).collect().to_pandas()
            yield chunk


class DataFrameReader(DataReader):
    """In-memory DataFrame reader (pass-through)."""

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize DataFrame reader.

        Args:
            dataframe: Pandas DataFrame
        """
        self.dataframe = dataframe.copy()

    def read(self) -> pd.DataFrame:
        """Return DataFrame copy."""
        return self.dataframe.copy()

    def read_chunked(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks."""
        for i in range(0, len(self.dataframe), chunk_size):
            yield self.dataframe.iloc[i : i + chunk_size].copy()


class DataWriter(ABC):
    """
    Abstract base class for data writers.

    Follows Single Responsibility: only handles data persistence.
    """

    @abstractmethod
    def write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """
        Write data to destination.

        Args:
            data: DataFrame to write
            path: Destination path

        Returns:
            WriteConfirmation with details
        """
        pass

    @abstractmethod
    def atomic_write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """
        Write data atomically (with rollback on failure).

        Args:
            data: DataFrame to write
            path: Destination path

        Returns:
            WriteConfirmation with details
        """
        pass


class CSVWriter(DataWriter):
    """CSV file writer implementation."""

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        """
        Initialize CSV writer.

        Args:
            delimiter: Column delimiter
            encoding: File encoding
        """
        self.delimiter = delimiter
        self.encoding = encoding

    def write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to CSV file."""
        data.to_csv(
            path,
            sep=self.delimiter,
            encoding=self.encoding,
            index=False,
        )

        return WriteConfirmation(
            path=str(path),
            rows_written=len(data),
            success=True,
        )

    def atomic_write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to CSV atomically."""
        temp_path = path.with_suffix(".tmp")

        try:
            # Write to temp file
            data.to_csv(
                temp_path,
                sep=self.delimiter,
                encoding=self.encoding,
                index=False,
            )

            # Atomic rename
            temp_path.replace(path)

            return WriteConfirmation(
                path=str(path),
                rows_written=len(data),
                success=True,
            )
        except Exception as e:
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e


class ExcelWriter(DataWriter):
    """Excel file writer implementation."""

    def write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to Excel file."""
        data.to_excel(path, index=False)

        return WriteConfirmation(
            path=str(path),
            rows_written=len(data),
            success=True,
        )

    def atomic_write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to Excel atomically."""
        temp_path = path.with_suffix(".tmp")

        try:
            data.to_excel(temp_path, index=False)
            temp_path.replace(path)

            return WriteConfirmation(
                path=str(path),
                rows_written=len(data),
                success=True,
            )
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e


class ParquetWriter(DataWriter):
    """Parquet file writer implementation."""

    def write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to Parquet file."""
        data.to_parquet(path, index=False)

        return WriteConfirmation(
            path=str(path),
            rows_written=len(data),
            success=True,
        )

    def atomic_write(self, data: pd.DataFrame, path: Path) -> WriteConfirmation:
        """Write to Parquet atomically."""
        temp_path = path.with_suffix(".tmp")

        try:
            data.to_parquet(temp_path, index=False)
            temp_path.replace(path)

            return WriteConfirmation(
                path=str(path),
                rows_written=len(data),
                success=True,
            )
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e


def create_data_reader(
    source_type: DataSourceType,
    source_path: Path | None = None,
    dataframe: pd.DataFrame | None = None,
    **kwargs: any,
) -> DataReader:
    """
    Factory function to create appropriate data reader.

    Args:
        source_type: Type of data source
        source_path: Path to file (for file sources)
        dataframe: DataFrame (for DataFrame source)
        **kwargs: Additional reader-specific parameters

    Returns:
        Configured DataReader

    Raises:
        ValueError: If source type not supported or parameters invalid
    """
    if source_type == DataSourceType.CSV:
        if not source_path:
            raise ValueError("source_path required for CSV")
        return CSVReader(
            source_path,
            delimiter=kwargs.get("delimiter", ","),
            encoding=kwargs.get("encoding", "utf-8"),
        )
    if source_type == DataSourceType.EXCEL:
        if not source_path:
            raise ValueError("source_path required for Excel")
        return ExcelReader(source_path, sheet_name=kwargs.get("sheet_name", 0))
    if source_type == DataSourceType.PARQUET:
        if not source_path:
            raise ValueError("source_path required for Parquet")
        return ParquetReader(source_path)
    if source_type == DataSourceType.DATAFRAME:
        if dataframe is None:
            raise ValueError("dataframe required for DataFrame source")
        return DataFrameReader(dataframe)
    raise ValueError(f"Unsupported source type: {source_type}")


def create_data_writer(destination_type: DataSourceType) -> DataWriter:
    """
    Factory function to create appropriate data writer.

    Args:
        destination_type: Type of destination

    Returns:
        Configured DataWriter

    Raises:
        ValueError: If destination type not supported
    """
    if destination_type == DataSourceType.CSV:
        return CSVWriter()
    if destination_type == DataSourceType.EXCEL:
        return ExcelWriter()
    if destination_type == DataSourceType.PARQUET:
        return ParquetWriter()
    raise ValueError(f"Unsupported destination: {destination_type}")
