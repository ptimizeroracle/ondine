"""Result writing stage for persisting output."""

from decimal import Decimal
from pathlib import Path
from typing import Any

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.core.data_container import DataContainer, Row
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
    PipelineStage[tuple[DataContainer, ResultContainerImpl, OutputSpec], ResultContainerImpl]
):
    """
    Write results to destination with merge support.

    Now works with DataContainer and ResultContainerImpl instead of DataFrames.

    Responsibilities:
    - Merge results with original data
    - Write to configured destination
    - Support atomic writes
    - Return merged ResultContainer
    """

    def __init__(self):
        """Initialize result writer stage."""
        super().__init__("ResultWriter")

    def process(
        self,
        input_data: tuple[DataContainer, ResultContainerImpl, OutputSpec],
        context: Any,
    ) -> ResultContainerImpl:
        """Write results to destination and return merged ResultContainer."""
        original_container, results_container, output_spec = input_data

        # Merge results with original data
        merged = self._merge_results(
            original_container, results_container, output_spec.merge_strategy
        )

        # Write to destination
        if output_spec.destination_path:
            dest_path = Path(output_spec.destination_path)
            suffix = dest_path.suffix.lower()

            if suffix == ".csv":
                merged.to_csv(dest_path)
            elif suffix in (".parquet", ".pq"):
                merged.to_parquet(dest_path)
            elif suffix in (".json", ".jsonl", ".ndjson"):
                merged.to_json(dest_path)
            else:
                # Default to CSV
                merged.to_csv(dest_path)

            self.logger.info(f"Wrote {len(merged)} rows to {dest_path}")

        # Always return the merged container (needed for quality validation)
        return merged

    def _merge_results(
        self,
        original: DataContainer,
        results: ResultContainerImpl,
        strategy: MergeStrategy,
    ) -> ResultContainerImpl:
        """Merge results with original data."""
        # Build lookup from results by row_index
        results_lookup: dict[int, Row] = {}
        for row in results:
            row_idx = row.get("_row_index", row.get("row_index"))
            if row_idx is not None:
                results_lookup[row_idx] = row

        # Merge rows
        merged_rows: list[Row] = []
        result_columns = [c for c in results.columns if c != "_row_index"]

        for idx, orig_row in enumerate(original):
            merged_row = dict(orig_row)  # Copy original row

            if idx in results_lookup:
                result_row = results_lookup[idx]

                if strategy == MergeStrategy.REPLACE:
                    # Replace/add columns from results
                    for col in result_columns:
                        merged_row[col] = result_row.get(col)

                elif strategy == MergeStrategy.APPEND:
                    # Add as new columns (error if exists)
                    for col in result_columns:
                        if col in orig_row:
                            raise ValueError(f"Column {col} already exists")
                        merged_row[col] = result_row.get(col)

                elif strategy == MergeStrategy.UPDATE:
                    # Only update non-null values
                    for col in result_columns:
                        value = result_row.get(col)
                        if value is not None:
                            merged_row[col] = value

            merged_rows.append(merged_row)

        # Determine merged columns
        merged_columns = list(original.columns)
        for col in result_columns:
            if col not in merged_columns:
                merged_columns.append(col)

        return ResultContainerImpl(data=merged_rows, columns=merged_columns)

    def validate_input(
        self,
        input_data: tuple[DataContainer, ResultContainerImpl, OutputSpec],
    ) -> ValidationResult:
        """Validate input data and output specification."""
        result = ValidationResult(is_valid=True)

        original_container, results_container, output_spec = input_data

        if len(original_container) == 0:
            result.add_warning("Original container is empty")

        if len(results_container) == 0:
            result.add_error("Results container is empty")

        # Check destination path if specified
        if output_spec.destination_path:
            dest_dir = Path(output_spec.destination_path).parent
            if not dest_dir.exists():
                result.add_warning(f"Destination directory does not exist: {dest_dir}")

        return result

    def estimate_cost(
        self,
        input_data: tuple[DataContainer, ResultContainerImpl, OutputSpec],
    ) -> CostEstimate:
        """Result writing has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=len(input_data[1]),
        )
