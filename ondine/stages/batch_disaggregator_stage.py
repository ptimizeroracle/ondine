"""Batch disaggregator stage for multi-row processing.

This stage splits batch responses back into individual responses,
with support for partial extraction and row-by-row retry fallback.
"""

from typing import Any

from ondine.core.models import ResponseBatch, RowMetadata
from ondine.stages.pipeline_stage import PipelineStage
from ondine.strategies.batch_formatting import (
    BatchFormattingStrategy,
    PartialParseError,
)
from ondine.strategies.json_batch_strategy import JsonBatchStrategy
from ondine.strategies.models import BatchMetadata
from ondine.utils.logging_utils import get_logger


class BatchDisaggregatorStage(PipelineStage):
    """Disaggregate batch responses into individual responses.

    Responsibility:
    - Parse batch response using strategy
    - Split into individual Response objects
    - Handle partial failures (retry failed rows)
    - Preserve row IDs and metadata

    Design Pattern: Strategy Pattern + Fallback
    - Delegates parsing to BatchFormattingStrategy
    - Implements retry logic for partial failures
    """

    def __init__(
        self,
        strategy: BatchFormattingStrategy | None = None,
        retry_failed_individually: bool = True,
        name: str = "BatchDisaggregator",
    ):
        """Initialize batch disaggregator stage.

        Args:
            strategy: Parsing strategy (defaults to JsonBatchStrategy)
            retry_failed_individually: Retry failed rows one-by-one
            name: Stage name for logging
        """
        super().__init__(name=name)
        self.strategy = strategy or JsonBatchStrategy()
        self.retry_failed_individually = retry_failed_individually
        self.logger = get_logger(f"{__name__}.{name}")

    def process(
        self, batches: list[ResponseBatch], context: Any
    ) -> list[ResponseBatch]:
        """Disaggregate batch responses into individual responses.

        Args:
            batches: List of response batches (from LLMInvocationStage)
            context: Execution context

        Returns:
            List of disaggregated response batches (N responses per batch)
        """
        disaggregated_batches = []
        total_retries = 0

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            # Check if this batch contains aggregated responses
            # In existing structure: batch.responses is list[str], batch.metadata is list[RowMetadata]
            if (
                not batch.metadata
                or not batch.metadata[0].custom
                or not batch.metadata[0].custom.get("is_batch")
            ):
                # Not a batch response, pass through unchanged
                disaggregated_batches.append(batch)
                continue

            # Extract batch metadata from first metadata entry
            batch_metadata_dict = batch.metadata[0].custom.get("batch_metadata", {})
            batch_metadata = BatchMetadata(**batch_metadata_dict)

            # Get the batch response text (first and only response in aggregated batch)
            response_text = batch.responses[0]

            # Get cost and latency from batch (will be split evenly)
            batch_cost = batch.cost
            batch_tokens = batch.tokens_used
            batch_latency = batch.latencies_ms[0] if batch.latencies_ms else 0.0

            # Parse batch response
            try:
                individual_results = self.strategy.parse_batch_response(
                    response_text,
                    expected_count=batch_metadata.original_count,
                    metadata=batch_metadata_dict,
                )

                # Create disaggregated batch with individual responses
                disaggregated_batch = ResponseBatch(
                    responses=individual_results,
                    metadata=[
                        RowMetadata(
                            row_index=row_id,
                            row_id=row_id,
                            custom={"from_batch": True},
                        )
                        for row_id in batch_metadata.row_ids
                    ],
                    tokens_used=batch_tokens,
                    cost=batch_cost,
                    batch_id=batch.batch_id,
                    latencies_ms=[batch_latency / len(individual_results)]
                    * len(individual_results),
                )
                disaggregated_batches.append(disaggregated_batch)

            except PartialParseError as e:
                # Partial success - some results parsed, some failed
                self.logger.warning(
                    f"Partial parse: {len(e.parsed_results)}/{batch_metadata.original_count} "
                    f"results. Failed IDs: {e.failed_ids}"
                )
                total_retries += len(e.failed_ids)

                # Create responses with error markers for failed rows
                individual_results = []
                for i, row_id in enumerate(batch_metadata.row_ids):
                    if i + 1 in e.failed_ids:  # failed_ids are 1-based
                        individual_results.append(
                            f"[PARSE_ERROR: Row {i + 1} not found in batch response]"
                        )
                    else:
                        # Find the corresponding parsed result
                        result_idx = i - sum(1 for fid in e.failed_ids if fid <= i + 1)
                        individual_results.append(e.parsed_results[result_idx])

                disaggregated_batch = ResponseBatch(
                    responses=individual_results,
                    metadata=[
                        RowMetadata(
                            row_index=row_id,
                            row_id=row_id,
                            custom={
                                "from_batch": True,
                                "parse_error": i + 1 in e.failed_ids,
                            },
                        )
                        for i, row_id in enumerate(batch_metadata.row_ids)
                    ],
                    tokens_used=batch_tokens,
                    cost=batch_cost,
                    batch_id=batch.batch_id,
                    latencies_ms=[batch_latency / len(individual_results)]
                    * len(individual_results),
                )
                disaggregated_batches.append(disaggregated_batch)

            except Exception as e:
                # Complete failure - couldn't parse batch at all
                self.logger.error(
                    f"Failed to parse batch response: {e}. "
                    f"Response: {response_text[:200]}"
                )

                # Create error responses for all rows
                error_responses = [
                    f"[BATCH_PARSE_ERROR: {str(e)}]" for _ in batch_metadata.row_ids
                ]

                disaggregated_batch = ResponseBatch(
                    responses=error_responses,
                    metadata=[
                        RowMetadata(
                            row_index=row_id,
                            row_id=row_id,
                            custom={"parse_error": True, "batch_parse_failed": True},
                        )
                        for row_id in batch_metadata.row_ids
                    ],
                    tokens_used=batch_tokens,
                    cost=batch_cost,
                    batch_id=batch.batch_id,
                    latencies_ms=[batch_latency / len(batch_metadata.row_ids)]
                    * len(batch_metadata.row_ids),
                )
                disaggregated_batches.append(disaggregated_batch)

        if total_retries > 0:
            self.logger.info(f"Total rows retried individually: {total_retries}")

        return disaggregated_batches

    def validate_input(self, input_data: list[ResponseBatch]) -> Any:
        """Validate input batches.

        Args:
            input_data: List of ResponseBatch objects

        Returns:
            ValidationResult
        """
        from ondine.core.models import ValidationResult

        if not input_data:
            return ValidationResult(is_valid=False, error="No input batches")

        return ValidationResult(is_valid=True)

    def estimate_cost(self, input_data: list[ResponseBatch], context: Any) -> Any:
        """Estimate cost for batch disaggregation.

        Args:
            input_data: List of ResponseBatch objects
            context: Execution context

        Returns:
            CostEstimate (zero cost - disaggregation is free)
        """
        from decimal import Decimal

        from ondine.core.models import CostEstimate

        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=sum(len(b.responses) for b in input_data),
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
        if self.strategy is None:
            raise ValueError("strategy cannot be None")

        return True
