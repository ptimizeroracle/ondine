"""Batch formatting strategy interface for multi-row processing.

This module defines the abstract interface for batch formatting strategies,
which are responsible for:
1. Formatting multiple prompts into a single batch prompt
2. Parsing batch responses back into individual results

The Strategy pattern allows different formatting approaches (JSON, CSV, XML)
without modifying the core batch processing stages.
"""

from abc import ABC, abstractmethod
from typing import Any


class PartialParseError(Exception):
    """Raised when batch response is partially parsed.

    Attributes:
        parsed_results: Successfully parsed results
        failed_ids: IDs of rows that failed to parse
        original_response: The original response text
    """

    def __init__(
        self,
        message: str,
        parsed_results: list[str],
        failed_ids: list[int],
        original_response: str,
    ):
        """Initialize partial parse error.

        Args:
            message: Error description
            parsed_results: Successfully parsed results
            failed_ids: IDs of rows that failed to parse
            original_response: The original response text
        """
        super().__init__(message)
        self.parsed_results = parsed_results
        self.failed_ids = failed_ids
        self.original_response = original_response


class BatchFormattingStrategy(ABC):
    """Abstract strategy for batch prompt formatting and response parsing.

    This interface defines the contract for batch processing strategies.
    Implementations must handle:
    - Formatting N prompts into 1 batch prompt
    - Parsing 1 batch response into N individual results
    - Partial failure handling (some results parsed, some failed)

    Design Pattern: Strategy Pattern
    - Allows different formatting approaches (JSON, CSV, XML)
    - Stages depend on this abstraction (Dependency Inversion)
    - New strategies can be added without modifying stages (Open/Closed)
    """

    @abstractmethod
    def format_batch(
        self,
        prompts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format multiple prompts into a single batch prompt.

        Args:
            prompts: List of individual prompts to batch together
            metadata: Optional metadata (e.g., row IDs, column names)

        Returns:
            Single batch prompt containing all inputs

        Example:
            prompts = ["Classify: Product A", "Classify: Product B"]
            result = strategy.format_batch(prompts)
            # Returns: "Classify these 2 items: 1. Product A, 2. Product B..."
        """
        pass

    @abstractmethod
    def parse_batch_response(
        self,
        response: str,
        expected_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Parse batch response into individual results.

        Args:
            response: Single batch response from LLM
            expected_count: Expected number of results
            metadata: Optional metadata from format_batch

        Returns:
            List of individual results (length must equal expected_count)

        Raises:
            PartialParseError: If some results parsed but not all
            ValueError: If response format is completely invalid

        Example:
            response = '[{"id": 1, "result": "positive"}, {"id": 2, "result": "negative"}]'
            results = strategy.parse_batch_response(response, expected_count=2)
            # Returns: ["positive", "negative"]
        """
        pass

    def estimate_batch_tokens(
        self,
        prompts: list[str],
        tokenizer: Any,
    ) -> int:
        """Estimate total tokens for batch prompt (optional override).

        Default implementation: Sum individual prompt tokens + overhead.

        Args:
            prompts: List of prompts to batch
            tokenizer: Tokenizer for counting (e.g., tiktoken)

        Returns:
            Estimated token count for batch prompt
        """
        # Default: sum of individual prompts + 10% overhead for formatting
        total = sum(len(tokenizer.encode(p)) for p in prompts)
        return int(total * 1.1)
