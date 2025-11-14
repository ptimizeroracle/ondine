"""Batch aggregator stage for multi-row processing.

This stage aggregates multiple prompts into a single batch prompt,
enabling 100× reduction in API calls.
"""

from decimal import Decimal
from typing import Any

from ondine.core.models import PromptBatch, RowMetadata
from ondine.stages.pipeline_stage import PipelineStage
from ondine.strategies.batch_formatting import BatchFormattingStrategy
from ondine.strategies.json_batch_strategy import JsonBatchStrategy
from ondine.strategies.models import BatchMetadata
from ondine.utils.logging_utils import get_logger
from ondine.utils.model_context_limits import validate_batch_size


class BatchAggregatorStage(PipelineStage):
    """Aggregate multiple prompts into batch prompts.

    Responsibility:
    - Group N prompts into chunks of batch_size
    - Use strategy to format each chunk as a single batch prompt
    - Preserve metadata for disaggregation

    Design Pattern: Strategy Pattern
    - Delegates formatting logic to BatchFormattingStrategy
    - Supports multiple formats (JSON, CSV) via strategy injection
    """

    def __init__(
        self,
        batch_size: int,
        strategy: BatchFormattingStrategy | None = None,
        model: str | None = None,
        validate_context_window: bool = True,
        name: str = "BatchAggregator",
    ):
        """Initialize batch aggregator stage.

        Args:
            batch_size: Number of prompts to aggregate per batch
            strategy: Formatting strategy (defaults to JsonBatchStrategy)
            model: Model name for context window validation (optional)
            validate_context_window: Whether to validate against context limits
            name: Stage name for logging
        """
        super().__init__(name=name)
        self.batch_size = batch_size
        self.strategy = strategy or JsonBatchStrategy()
        self.model = model
        self.validate_context_window = validate_context_window
        self.logger = get_logger(f"{__name__}.{name}")

    def process(self, batches: list[PromptBatch], context: Any) -> list[PromptBatch]:
        """Aggregate prompts into batch prompts.

        Args:
            batches: List of prompt batches (from PromptFormatterStage)
            context: Execution context

        Returns:
            List of aggregated prompt batches (1 prompt per batch_size rows)
        """
        aggregated_batches = []

        # Process each batch
        for batch in batches:
            # Group prompts into chunks of batch_size
            for i in range(0, len(batch.prompts), self.batch_size):
                chunk_prompts = batch.prompts[i : i + self.batch_size]
                chunk_metadata = batch.metadata[i : i + self.batch_size]

                # Extract row IDs from metadata
                row_ids = [m.row_index for m in chunk_metadata]

                # Create metadata for disaggregation
                metadata = BatchMetadata(
                    original_count=len(chunk_prompts),
                    row_ids=row_ids,
                    prompt_template=None,  # Not available in current structure
                )

                # Validate batch size against context window (if model provided)
                if self.validate_context_window and self.model:
                    # Estimate tokens for this batch
                    import tiktoken

                    tokenizer = tiktoken.get_encoding("cl100k_base")
                    avg_tokens = sum(
                        len(tokenizer.encode(p)) for p in chunk_prompts
                    ) // len(chunk_prompts)

                    is_valid, error_msg = validate_batch_size(
                        self.model, len(chunk_prompts), avg_tokens
                    )

                    if not is_valid:
                        self.logger.warning(
                            f"Batch size validation failed: {error_msg}. "
                            f"Consider reducing batch_size."
                        )

                # Use strategy to format batch prompt
                batch_prompt_text = self.strategy.format_batch(
                    chunk_prompts, metadata=metadata.model_dump()
                )

                # Create new PromptBatch with single mega-prompt
                # Use existing PromptBatch structure (list of strings)
                mega_metadata = RowMetadata(
                    row_index=chunk_metadata[0].row_index,
                    row_id=chunk_metadata[0].row_id,
                    custom={
                        "batch_metadata": metadata.model_dump(),
                        "is_batch": True,
                        "batch_size": len(chunk_prompts),
                    },
                )

                aggregated_batch = PromptBatch(
                    prompts=[batch_prompt_text],
                    metadata=[mega_metadata],
                    batch_id=batch.batch_id,
                )

                aggregated_batches.append(aggregated_batch)

        self.logger.info(
            f"Aggregated {sum(len(b.prompts) for b in batches)} prompts "
            f"into {len(aggregated_batches)} batch prompts "
            f"(batch_size={self.batch_size})"
        )

        return aggregated_batches

    def validate_input(self, input_data: list[PromptBatch]) -> Any:
        """Validate input batches.

        Args:
            input_data: List of PromptBatch objects

        Returns:
            ValidationResult
        """
        from ondine.core.models import ValidationResult

        if not input_data:
            return ValidationResult(is_valid=False, error="No input batches")

        return ValidationResult(is_valid=True)

    def estimate_cost(self, input_data: list[PromptBatch], context: Any) -> Any:
        """Estimate cost for batch aggregation.

        Args:
            input_data: List of PromptBatch objects
            context: Execution context

        Returns:
            CostEstimate (zero cost - aggregation is free)
        """
        from ondine.core.models import CostEstimate

        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=sum(len(b.prompts) for b in input_data),
            confidence="actual",
        )

    def validate(self, context: Any) -> bool:
        """Validate stage configuration.

        Args:
            context: Execution context

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.strategy is None:
            raise ValueError("strategy cannot be None")

        return True
