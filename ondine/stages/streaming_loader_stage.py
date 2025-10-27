"""
Streaming data loader for memory-efficient processing of large files.

Implements streaming pattern for datasets that don't fit in memory.
"""

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pandas as pd

from ondine.adapters.data_io import create_data_reader
from ondine.core.models import CostEstimate, ValidationResult
from ondine.core.specifications import DatasetSpec
from ondine.stages.pipeline_stage import PipelineStage


class StreamingDataLoaderStage(PipelineStage[DatasetSpec, Iterator[pd.DataFrame]]):
    """
    Load data in chunks for memory-efficient processing.

    Use this for very large datasets (100K+ rows) that don't fit in memory.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize streaming data loader.

        Args:
            chunk_size: Number of rows per chunk
        """
        super().__init__("StreamingDataLoader")
        self.chunk_size = chunk_size

    def process(self, spec: DatasetSpec, context: Any) -> Iterator[pd.DataFrame]:
        """Load data as iterator of chunks."""
        # Create appropriate reader
        reader = create_data_reader(
            source_type=spec.source_type,
            source_path=spec.source_path,
            delimiter=spec.delimiter,
            encoding=spec.encoding,
            sheet_name=spec.sheet_name,
        )

        self.logger.info(f"Streaming data in chunks of {self.chunk_size} rows")

        # Return chunked iterator
        return reader.read_chunked(self.chunk_size)

    def validate_input(self, spec: DatasetSpec) -> ValidationResult:
        """Validate dataset specification."""
        result = ValidationResult(is_valid=True)

        # Check file exists for file sources
        if spec.source_path and not spec.source_path.exists():
            result.add_error(f"Source file not found: {spec.source_path}")

        # Check columns specified
        if not spec.input_columns:
            result.add_error("No input columns specified")

        return result

    def estimate_cost(self, spec: DatasetSpec) -> CostEstimate:
        """Streaming has no direct LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=0,  # Unknown until streaming starts
        )
