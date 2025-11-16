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
        validate_context_window: bool = False,  # Disabled by default (slow for large datasets)
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
        import time

        aggregated_batches = []

        # Calculate total prompts for progress tracking
        total_prompts = sum(len(b.prompts) for b in batches)
        if total_prompts == 0:
            self.logger.info("No prompts to aggregate")
            return aggregated_batches

        processed_prompts = 0
        start_time = time.time()
        last_log_time = start_time
        last_log_pct = 0

        expected_batches = (total_prompts + self.batch_size - 1) // self.batch_size
        self.logger.info(
            f"Creating {expected_batches:,} mega-prompts ({self.batch_size} rows each)"
        )

        # Process each batch
        for batch_idx, batch in enumerate(batches):
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

                # Validate batch size against context window (only once, not for every batch)
                if (
                    self.validate_context_window
                    and self.model
                    and len(aggregated_batches) == 0  # Only validate first batch
                ):
                    # Skip validation if chunk is empty
                    if not chunk_prompts:
                        self.logger.warning(
                            "Empty chunk encountered, skipping validation"
                        )
                        continue

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
                    else:
                        self.logger.info(
                            f"Batch size validation passed: {len(chunk_prompts)} rows, "
                            f"~{avg_tokens} tokens/row"
                        )

                # Use strategy to format batch prompt
                batch_prompt_text = self.strategy.format_batch(
                    chunk_prompts, metadata=metadata.model_dump()
                )

                # Create new PromptBatch with single mega-prompt
                # Preserve ALL custom fields from original metadata (especially system_message for caching!)
                original_custom = chunk_metadata[0].custom or {}

                mega_metadata = RowMetadata(
                    row_index=chunk_metadata[0].row_index,
                    row_id=chunk_metadata[0].row_id,
                    custom={
                        **original_custom,  # Preserve all custom fields (system_message, etc.)
                        "batch_metadata": metadata.model_dump(),  # Batch-specific fields override
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

                # Update progress
                processed_prompts += len(chunk_prompts)

                # Hybrid progress: Log every 10% OR every 30 seconds (only for slow operations)
                current_time = time.time()
                elapsed = current_time - start_time

                if total_prompts > 10000 and elapsed > 5:
                    current_pct = int((processed_prompts / total_prompts) * 100)

                    should_log = (
                        current_pct >= last_log_pct + 10 and current_pct <= 90
                    ) or (current_time - last_log_time >= 30)

                    if should_log:
                        elapsed = current_time - start_time
                        throughput = processed_prompts / elapsed if elapsed > 0 else 0
                        eta = (
                            (total_prompts - processed_prompts) / throughput
                            if throughput > 0
                            else 0
                        )

                        self.logger.info(
                            f"Aggregating: {current_pct}% ({processed_prompts:,}/{total_prompts:,}) | "
                            f"ETA: {eta:.0f}s"
                        )
                        last_log_time = current_time
                        last_log_pct = current_pct

        # Final summary
        elapsed = time.time() - start_time
        throughput = (
            len(aggregated_batches) / elapsed
            if elapsed > 0
            else len(aggregated_batches)
        )
        self.logger.info(
            f"✓ Created {len(aggregated_batches):,} mega-prompts in {elapsed:.1f}s "
            f"({throughput:.0f} batches/s)"
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
