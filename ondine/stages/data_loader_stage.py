"""Data loading stage for reading tabular data."""

from decimal import Decimal
from pathlib import Path
from typing import Any

from ondine.core.data_container import DataContainer
from ondine.core.models import CostEstimate, ValidationResult
from ondine.core.specifications import DatasetSpec, DataSourceType
from ondine.stages.pipeline_stage import PipelineStage


def create_container(spec: DatasetSpec, dataframe: Any = None) -> DataContainer:
    """
    Create appropriate DataContainer based on spec.

    Args:
        spec: Dataset specification
        dataframe: Optional pre-loaded data (DataFrame or list of dicts)

    Returns:
        DataContainer implementation
    """
    from ondine.adapters.containers import (
        DictListContainer,
        PandasContainer,
        PolarsContainer,
        StreamingCSVContainer,
    )
    from ondine.adapters.containers.streaming_csv import (
        StreamingJSONContainer,
        StreamingParquetContainer,
    )

    # If dataframe provided, wrap it
    if dataframe is not None:
        # Check type and wrap appropriately
        if isinstance(dataframe, list):
            return DictListContainer(dataframe)

        # Check for Pandas DataFrame
        try:
            import pandas as pd

            if isinstance(dataframe, pd.DataFrame):
                return PandasContainer(dataframe)
        except ImportError:
            pass

        # Check for Polars DataFrame
        try:
            import polars as pl

            if isinstance(dataframe, pl.DataFrame):
                return PolarsContainer(dataframe)
        except ImportError:
            pass

        # Already a DataContainer
        if isinstance(dataframe, DataContainer):
            return dataframe

        raise TypeError(f"Unsupported dataframe type: {type(dataframe)}")

    # Create from file path
    if spec.source_path is None:
        raise ValueError("Either dataframe or source_path must be provided")

    path = Path(spec.source_path)
    suffix = path.suffix.lower()

    # Use streaming containers by default for memory efficiency
    if suffix == ".csv":
        return StreamingCSVContainer(
            path=path,
            delimiter=spec.delimiter or ",",
            encoding=spec.encoding or "utf-8",
            columns=spec.input_columns,
        )
    elif suffix == ".parquet" or suffix == ".pq":
        return StreamingParquetContainer(path=path, columns=spec.input_columns)
    elif suffix in (".json", ".jsonl", ".ndjson"):
        return StreamingJSONContainer(
            path=path,
            encoding=spec.encoding or "utf-8",
            columns=spec.input_columns,
        )
    elif suffix in (".xlsx", ".xls"):
        # Excel requires Pandas
        try:
            import pandas as pd

            df = pd.read_excel(path, sheet_name=spec.sheet_name)
            return PandasContainer(df)
        except ImportError:
            raise ImportError("Pandas required for Excel files")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


class DataLoaderStage(PipelineStage[DatasetSpec, DataContainer]):
    """
    Load data from source and validate schema.

    Now returns DataContainer instead of pd.DataFrame for
    framework-agnostic, memory-efficient processing.

    Responsibilities:
    - Read data from configured source
    - Validate required columns exist
    - Apply any filters
    - Update context with row count
    """

    def __init__(self, dataframe: Any = None):
        """
        Initialize data loader stage.

        Args:
            dataframe: Optional pre-loaded data (DataFrame, list of dicts, or DataContainer)
        """
        super().__init__("DataLoader")
        self.dataframe = dataframe

    def process(self, spec: DatasetSpec, context: Any) -> DataContainer:
        """Load data from source."""
        # Create container
        container = create_container(spec, self.dataframe)

        # Validate columns exist
        missing_cols = set(spec.input_columns) - set(container.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Apply filters if specified
        if spec.filters:
            container = self._apply_filters(container, spec.filters)

        # Update context with total rows
        context.total_rows = len(container)

        self.logger.info(f"Loaded {len(container)} rows from {spec.source_type}")

        return container

    def _apply_filters(
        self, container: DataContainer, filters: dict[str, Any]
    ) -> DataContainer:
        """Apply filters to container."""
        from ondine.adapters.containers import DictListContainer

        # Filter rows
        filtered_rows = []
        for row in container:
            match = True
            for column, value in filters.items():
                if column in row and row[column] != value:
                    match = False
                    break
            if match:
                filtered_rows.append(row)

        return DictListContainer(filtered_rows, columns=container.columns)

    def validate_input(self, spec: DatasetSpec) -> ValidationResult:
        """Validate dataset specification."""
        result = ValidationResult(is_valid=True)

        # Check file exists for file sources
        if spec.source_path and not spec.source_path.exists():
            result.add_error(f"Source file not found: {spec.source_path}")

        # Check input columns specified
        if not spec.input_columns:
            result.add_error("No input columns specified")

        # Check output columns specified
        if not spec.output_columns:
            result.add_error("No output columns specified")

        return result

    def estimate_cost(self, spec: DatasetSpec) -> CostEstimate:
        """Data loading has no LLM cost."""
        # Try to determine row count if dataframe is available
        row_count = 0
        if self.dataframe is not None:
            try:
                row_count = len(self.dataframe)
            except (TypeError, AttributeError):
                pass

        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=row_count,
        )
