"""Result writing stage for persisting output."""

from decimal import Decimal
from typing import Any

import pandas as pd

from ondine.adapters.data_io import create_data_writer
from ondine.core.models import (
    CostEstimate,
    ValidationResult,
)
from ondine.core.specifications import (
    MergeStrategy,
    OutputSpec,
)
from ondine.stages.pipeline_stage import PipelineStage


class ResultWriterStage(
    PipelineStage[tuple[pd.DataFrame, pd.DataFrame, OutputSpec], pd.DataFrame]
):
    """
    Write results to destination with merge support.

    Responsibilities:
    - Merge results with original data
    - Write to configured destination
    - Support atomic writes
    - Return merged DataFrame
    """

    def __init__(self):
        """Initialize result writer stage."""
        super().__init__("ResultWriter")

    def process(
        self,
        input_data: tuple[pd.DataFrame, pd.DataFrame, OutputSpec],
        context: Any,
    ) -> pd.DataFrame:
        """Write results to destination and return merged DataFrame."""
        original_df, results_df, output_spec = input_data

        # Merge results with original data
        merged_df = self._merge_results(
            original_df, results_df, output_spec.merge_strategy
        )

        # Write to destination
        if output_spec.destination_path:
            writer = create_data_writer(output_spec.destination_type)

            if output_spec.atomic_write:
                confirmation = writer.atomic_write(
                    merged_df, output_spec.destination_path
                )
            else:
                confirmation = writer.write(merged_df, output_spec.destination_path)

            self.logger.info(
                f"Wrote {confirmation.rows_written} rows to {confirmation.path}"
            )

        # Always return the merged DataFrame (needed for quality validation)
        return merged_df

    def _merge_results(
        self,
        original: pd.DataFrame,
        results: pd.DataFrame,
        strategy: MergeStrategy,
    ) -> pd.DataFrame:
        """Merge results with original data."""
        if strategy == MergeStrategy.REPLACE:
            # Replace existing columns or add new ones
            merged = original.copy()
            for col in results.columns:
                merged[col] = results[col]
            return merged

        if strategy == MergeStrategy.APPEND:
            # Add as new columns (error if exists)
            for col in results.columns:
                if col in original.columns:
                    raise ValueError(f"Column {col} already exists")
            return pd.concat([original, results], axis=1)

        if strategy == MergeStrategy.UPDATE:
            # Only update rows that changed
            merged = original.copy()
            for col in results.columns:
                if col in merged.columns:
                    # Update only non-null values
                    mask = results[col].notna()
                    merged.loc[mask, col] = results.loc[mask, col]
                else:
                    merged[col] = results[col]
            return merged

        raise ValueError(f"Unknown merge strategy: {strategy}")

    def validate_input(
        self,
        input_data: tuple[pd.DataFrame, pd.DataFrame, OutputSpec],
    ) -> ValidationResult:
        """Validate input data and output specification."""
        result = ValidationResult(is_valid=True)

        original_df, results_df, output_spec = input_data

        if original_df.empty:
            result.add_warning("Original DataFrame is empty")

        if results_df.empty:
            result.add_error("Results DataFrame is empty")

        # Check destination path if specified
        if output_spec.destination_path:
            dest_dir = output_spec.destination_path.parent
            if not dest_dir.exists():
                result.add_warning(f"Destination directory does not exist: {dest_dir}")

        return result

    def estimate_cost(
        self,
        input_data: tuple[pd.DataFrame, pd.DataFrame, OutputSpec],
    ) -> CostEstimate:
        """Result writing has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data[1]),
        )
