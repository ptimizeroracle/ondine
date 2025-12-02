"""
Batch processor for flattening and reconstructing prompt/response batches.

Provides utilities for converting between batched and flat representations
of prompts and responses, enabling concurrent processing while preserving
batch structure.
"""

from dataclasses import dataclass, field

from ondine.core.models import LLMResponse, PromptBatch, ResponseBatch, RowMetadata


@dataclass
class PromptItem:
    """A single prompt with its metadata and batch context."""

    prompt: str
    metadata: RowMetadata
    batch_id: str


@dataclass
class BatchMap:
    """
    Maps flat indices to batch positions.

    Tracks the relationship between flattened prompt list indices
    and their original (batch_idx, prompt_idx_in_batch) positions.
    """

    _mappings: list[tuple[int, int]] = field(default_factory=list)

    def add(self, batch_idx: int, prompt_idx: int) -> None:
        """Add a mapping from flat index to batch position."""
        self._mappings.append((batch_idx, prompt_idx))

    def get(self, flat_idx: int) -> tuple[int, int]:
        """Get (batch_idx, prompt_idx) for a flat index."""
        return self._mappings[flat_idx]

    def __len__(self) -> int:
        return len(self._mappings)

    def __iter__(self):
        return iter(self._mappings)


class BatchProcessor:
    """
    Handles batch flattening and reconstruction for concurrent processing.

    The LLM invocation stage needs to process prompts concurrently regardless
    of batch boundaries, but must reconstruct the original batch structure
    for downstream stages. This class provides:

    1. flatten(): Convert list of PromptBatch -> flat list of PromptItem + BatchMap
    2. reconstruct(): Convert flat responses + BatchMap -> list of ResponseBatch

    Example:
        processor = BatchProcessor()

        # Flatten for concurrent processing
        items, batch_map = processor.flatten(prompt_batches)

        # Process all items concurrently
        responses = await process_all(items)

        # Reconstruct original batch structure
        response_batches = processor.reconstruct(responses, prompt_batches, batch_map)
    """

    @staticmethod
    def flatten(batches: list[PromptBatch]) -> tuple[list[PromptItem], BatchMap]:
        """
        Flatten batches into individual items with mapping.

        Args:
            batches: List of PromptBatch objects

        Returns:
            Tuple of (items, batch_map) where:
            - items: List of PromptItem objects
            - batch_map: BatchMap tracking original positions
        """
        items: list[PromptItem] = []
        batch_map = BatchMap()

        for batch_idx, batch in enumerate(batches):
            for prompt_idx, (prompt, metadata) in enumerate(
                zip(batch.prompts, batch.metadata, strict=False)
            ):
                items.append(PromptItem(prompt, metadata, batch.batch_id))
                batch_map.add(batch_idx, prompt_idx)

        return items, batch_map

    @staticmethod
    def reconstruct(
        responses: list[LLMResponse],
        original_batches: list[PromptBatch],
        batch_map: BatchMap,
    ) -> list[ResponseBatch]:
        """
        Reconstruct response batches from flat responses.

        Args:
            responses: Flat list of LLMResponse objects (same order as flattened items)
            original_batches: Original PromptBatch objects (for metadata)
            batch_map: BatchMap from flatten() call

        Returns:
            List of ResponseBatch objects in original batch order
        """
        # Group responses by batch
        batch_responses: dict[int, list[tuple[int, LLMResponse]]] = {
            i: [] for i in range(len(original_batches))
        }

        for response_idx, (batch_idx, prompt_idx) in enumerate(batch_map):
            batch_responses[batch_idx].append((prompt_idx, responses[response_idx]))

        # Create ResponseBatch objects in original order
        response_batches: list[ResponseBatch] = []

        for batch_idx, original_batch in enumerate(original_batches):
            # Sort by prompt index to maintain order within batch
            sorted_responses = sorted(batch_responses[batch_idx], key=lambda x: x[0])
            batch_llm_responses = [r for _, r in sorted_responses]

            # Calculate batch metrics
            total_tokens = sum(r.tokens_in + r.tokens_out for r in batch_llm_responses)
            total_cost = sum(r.cost for r in batch_llm_responses)
            latencies = [r.latency_ms for r in batch_llm_responses]

            response_batch = ResponseBatch(
                responses=batch_llm_responses,  # Keep full LLMResponse objects (preserves structured_result!)
                metadata=original_batch.metadata,
                tokens_used=total_tokens,
                cost=total_cost,
                batch_id=original_batch.batch_id,
                latencies_ms=latencies,
            )
            response_batches.append(response_batch)

        return response_batches

    @staticmethod
    def calculate_total_rows(batches: list[PromptBatch]) -> int:
        """
        Calculate total rows accounting for aggregated batches.

        Args:
            batches: List of PromptBatch objects

        Returns:
            Total number of logical rows (accounting for batch aggregation)
        """
        total = 0

        for batch in batches:
            if not batch.metadata:
                continue

            for metadata in batch.metadata:
                if metadata.custom and metadata.custom.get("is_batch"):
                    # Aggregated batch: use batch_size from metadata
                    total += metadata.custom.get("batch_size", 1)
                else:
                    # Non-aggregated: count as 1 row
                    total += 1

        return total

    @staticmethod
    def get_batch_size(metadata: RowMetadata) -> int:
        """
        Get the logical batch size from metadata.

        For aggregated batches, returns the batch_size from custom metadata.
        For regular rows, returns 1.

        Args:
            metadata: RowMetadata for the prompt

        Returns:
            Number of logical rows this prompt represents
        """
        if metadata.custom and metadata.custom.get("is_batch"):
            return metadata.custom.get("batch_size", 1)
        return 1
