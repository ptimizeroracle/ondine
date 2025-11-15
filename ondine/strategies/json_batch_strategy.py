"""JSON-based batch formatting strategy using LlamaIndex structured output.

This strategy formats multiple prompts as a JSON array and uses LlamaIndex's
structured_predict() for reliable parsing with Pydantic validation.
"""

import json
import re
from typing import Any

from ondine.strategies.batch_formatting import (
    BatchFormattingStrategy,
    PartialParseError,
)
from ondine.strategies.models import BatchItem, BatchResult
from ondine.utils.logging_utils import get_logger


class JsonBatchStrategy(BatchFormattingStrategy):
    """JSON-based batch formatting strategy.

    Formats prompts as JSON array and parses responses using LlamaIndex's
    structured_predict() for reliable Pydantic-validated parsing.

    Design:
    - Uses structured output (Pydantic models) for type safety
    - Leverages LlamaIndex for automatic retry on malformed JSON
    - Supports partial extraction (parse what works, report failures)
    """

    def __init__(self):
        """Initialize JSON batch strategy."""
        self.logger = get_logger(f"{__name__}.JsonBatchStrategy")

    def format_batch(
        self,
        prompts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format multiple prompts as JSON array.

        Args:
            prompts: List of prompts to batch
            metadata: Optional metadata (row_ids, etc.)

        Returns:
            JSON-formatted batch prompt with instructions

        Example:
            Input: ["Classify: Product A", "Classify: Product B"]
            Output:
            '''
            Process these 2 items and return a JSON array:

            INPUT:
            [
              {"id": 1, "input": "Classify: Product A"},
              {"id": 2, "input": "Classify: Product B"}
            ]

            OUTPUT FORMAT (return ONLY this JSON, nothing else):
            [
              {"id": 1, "result": "positive"},
              {"id": 2, "result": "negative"}
            ]
            '''
        """
        # Create BatchItem objects
        items = [
            BatchItem(id=i + 1, input=prompt, result=None)
            for i, prompt in enumerate(prompts)
        ]

        # Format as prompt
        items_json = json.dumps(
            [{"id": item.id, "input": item.input} for item in items], indent=2
        )

        return f"""Process these {len(prompts)} items and return a JSON array.

INPUT:
{items_json}

CRITICAL OUTPUT REQUIREMENTS:
1. Return a JSON array with {len(prompts)} objects
2. Each object must have "id" (number) and "result" (string) fields
3. IDs must match the input IDs (1 to {len(prompts)})
4. Return ONLY the JSON array, no explanations or markdown

OUTPUT FORMAT:
[
  {{"id": 1, "result": "your result here"}},
  {{"id": 2, "result": "your result here"}},
  ...
  {{"id": {len(prompts)}, "result": "your result here"}}
]

JSON Array:"""

    def parse_batch_response(
        self,
        response: str,
        expected_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Parse JSON batch response into individual results.

        Uses manual JSON parsing with fallback to partial extraction.
        LlamaIndex structured_predict() is used by the LLM client, not here.

        Args:
            response: LLM response containing JSON array
            expected_count: Expected number of results
            metadata: Optional metadata

        Returns:
            List of result strings, sorted by ID

        Raises:
            PartialParseError: If some results parsed but not all
            ValueError: If response is completely invalid
        """
        # Extract JSON array from response (handle markdown code blocks)
        json_text = self._extract_json(response)

        if not json_text:
            raise ValueError(
                f"No JSON array found in response. Response: {response[:200]}"
            )

        # Parse JSON
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Validate it's a list
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data)}")

        # Parse into BatchResult for validation
        try:
            items = [BatchItem(**item) for item in data]
            batch_result = BatchResult(results=items)
        except Exception as e:
            raise ValueError(f"Invalid batch result format: {e}") from e

        # Check if we got all expected results
        missing_ids = batch_result.get_missing_ids(expected_count)

        if missing_ids:
            # Partial success - raise PartialParseError
            self.logger.warning(
                f"Partial parse: got {len(batch_result.results)}/{expected_count} results. "
                f"Missing IDs: {missing_ids}"
            )
            raise PartialParseError(
                message=f"Missing {len(missing_ids)} results: IDs {missing_ids}",
                parsed_results=batch_result.to_list(),
                failed_ids=missing_ids,
                original_response=response,
            )

        # Full success - return sorted results
        return batch_result.to_list()

    def _extract_json(self, response: str) -> str | None:
        """Extract JSON array from response text.

        Handles:
        - Plain JSON: [{"id": 1, ...}]
        - Markdown code blocks: ```json [...] ```
        - Markdown without language: ``` [...] ```
        - JSON object (wraps in array): {"id": 1, ...}

        Args:
            response: Response text

        Returns:
            Extracted JSON string or None if not found
        """
        # Try to find JSON array (handles nested objects)
        # Pattern: [ ... ] with proper nesting
        json_pattern = r"\[(?:[^[\]]|\[(?:[^[\]]|\[[^\[\]]*\])*\])*\]"
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            return match.group(0)

        # Fallback: Try to find JSON object and wrap in array
        # Pattern: { ... }
        obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(obj_pattern, response, re.DOTALL)

        if match:
            # Wrap single object in array
            return f"[{match.group(0)}]"

        return None

    def estimate_batch_tokens(
        self,
        prompts: list[str],
        tokenizer: Any,
    ) -> int:
        """Estimate tokens for JSON batch prompt.

        JSON adds overhead:
        - Array structure: ~50 tokens
        - Per-item overhead: ~20 tokens (id, input, result fields)
        - Instructions: ~150 tokens

        Args:
            prompts: List of prompts
            tokenizer: Tokenizer (e.g., tiktoken)

        Returns:
            Estimated token count
        """
        # Base instruction overhead
        overhead = 200

        # Per-item overhead (JSON structure)
        per_item_overhead = 25

        # Sum prompt tokens
        prompt_tokens = sum(len(tokenizer.encode(p)) for p in prompts)

        # Total
        return overhead + (len(prompts) * per_item_overhead) + prompt_tokens
