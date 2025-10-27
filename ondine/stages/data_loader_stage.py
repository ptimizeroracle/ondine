"""Data loading stage for reading tabular data."""

from decimal import Decimal
from typing import Any

import pandas as pd

from ondine.adapters.data_io import create_data_reader
from ondine.core.models import CostEstimate, ValidationResult
from ondine.core.specifications import DatasetSpec
from ondine.stages.pipeline_stage import PipelineStage


class DataLoaderStage(PipelineStage[DatasetSpec, pd.DataFrame]):
    """
    Load data from source and validate schema.

    Responsibilities:
    - Read data from configured source
    - Validate required columns exist
    - Apply any filters
    - Update context with row count
    """

    def __init__(self, dataframe: pd.DataFrame | None = None):
        """
        Initialize data loader stage.

        Args:
            dataframe: Optional pre-loaded dataframe (for DataFrame source)
        """
        super().__init__("DataLoader")
        self.dataframe = dataframe

    def process(self, spec: DatasetSpec, context: Any) -> pd.DataFrame:
        """Load data from source."""
        # Create appropriate reader
        reader = create_data_reader(
            source_type=spec.source_type,
            source_path=spec.source_path,
            dataframe=self.dataframe,
            delimiter=spec.delimiter,
            encoding=spec.encoding,
            sheet_name=spec.sheet_name,
        )

        # Read data
        df = reader.read()

        # Validate columns exist
        missing_cols = set(spec.input_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Apply filters if specified
        if spec.filters:
            for column, value in spec.filters.items():
                if column in df.columns:
                    df = df[df[column] == value]

        # Update context with total rows
        context.total_rows = len(df)

        self.logger.info(f"Loaded {len(df)} rows from {spec.source_type}")

        return df

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
            row_count = len(self.dataframe)

        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=row_count,
        )
