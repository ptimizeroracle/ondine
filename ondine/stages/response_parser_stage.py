"""Response parsing stage for structured output extraction."""

import json
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.core.models import (
    CostEstimate,
    ResponseBatch,
    ValidationResult,
)
from ondine.stages.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from ondine.core.data_container import Row  # noqa: TC001


class ResponseParser(ABC):
    """Abstract base for response parsers (Strategy pattern)."""

    @abstractmethod
    def parse(self, response: str) -> dict[str, Any]:
        """Parse response into structured data."""
        pass


class RawTextParser(ResponseParser):
    """Parser that returns raw text."""

    def parse(self, response: str) -> dict[str, Any]:
        """Return response as-is, after cleaning chat format artifacts."""
        cleaned = response.strip()

        # Strip common chat format prefixes (assistant:, user:, system:)
        for prefix in ["assistant:", "user:", "system:"]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return {"output": cleaned}


class JSONParser(ResponseParser):
    """Parser that extracts JSON from response."""

    def __init__(self, strict: bool = False):
        """
        Initialize JSON parser.

        Args:
            strict: If True, fail on invalid JSON; if False, try to extract
        """
        self.strict = strict

    def parse(self, response: str) -> dict[str, Any]:
        """Parse JSON from response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            if self.strict:
                raise

            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            if "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            # Return as raw text if can't parse
            return {"output": response.strip()}


class PydanticParser(ResponseParser):
    """
    Parser that validates responses against Pydantic models.

    Provides type-safe extraction with automatic validation.
    """

    def __init__(self, model: type[BaseModel], strict: bool = True):
        """
        Initialize Pydantic parser.

        Args:
            model: Pydantic model class for validation
            strict: If True, fail on validation errors
        """
        self.model = model
        self.strict = strict

    def parse(self, response: str) -> BaseModel:
        """Parse and validate response with Pydantic model."""
        try:
            # Try to parse as JSON first
            json_parser = JSONParser(strict=False)
            data = json_parser.parse(response)

            # Validate with Pydantic and return the model instance
            return self.model(**data)

        except ValidationError as e:
            if self.strict:
                raise ValueError(f"Pydantic validation failed: {e}")
            # Return raw data if validation fails
            return {"output": response.strip(), "validation_error": str(e)}


class RegexParser(ResponseParser):
    """
    Parser that extracts data using regex patterns.

    Useful for extracting specific fields from structured text.
    """

    def __init__(self, patterns: dict[str, str]):
        """
        Initialize regex parser.

        Args:
            patterns: Dict mapping field names to regex patterns
        """
        self.patterns = {key: re.compile(pattern) for key, pattern in patterns.items()}

    def parse(self, response: str) -> dict[str, Any]:
        """Extract fields using regex patterns."""
        result = {}

        for field_name, pattern in self.patterns.items():
            match = pattern.search(response)
            if match:
                # Use first group if groups exist, else full match
                if match.groups():
                    result[field_name] = match.group(1)
                else:
                    result[field_name] = match.group(0)
            else:
                result[field_name] = None

        return result


class ResponseParserStage(
    PipelineStage[tuple[list[ResponseBatch], list[str]], ResultContainerImpl]
):
    """
    Parse LLM responses into structured ResultContainer.

    Now returns ResultContainerImpl instead of pd.DataFrame for
    framework-agnostic output.

    Responsibilities:
    - Parse responses using configured parser
    - Map parsed data to output columns
    - Handle parse errors gracefully
    - Return ResultContainer with results
    """

    def __init__(
        self,
        parser: ResponseParser | None = None,
        output_columns: list[str] | None = None,
    ):
        """
        Initialize response parser stage.

        Args:
            parser: Response parser (default: RawTextParser)
            output_columns: Output column names
        """
        super().__init__("ResponseParser")
        self.parser = parser or RawTextParser()
        self.output_columns = output_columns or ["output"]

    def process(
        self,
        input_data: tuple[list[ResponseBatch], list[str]] | list[ResponseBatch],
        context: Any,
    ) -> ResultContainerImpl:
        """Parse responses into ResultContainer."""
        # Handle both tuple (batches, output_cols) and list [batches] for backward compatibility
        if isinstance(input_data, tuple):
            batches, output_cols = input_data
            # Use output_cols from input_data (overrides self.output_columns if provided)
            if not output_cols:
                output_cols = self.output_columns
        else:
            # Backward compatibility: input_data is just the list of batches
            batches = input_data
            output_cols = self.output_columns

        # Initialize result storage - use dict to maintain row_index ordering
        results: dict[int, Row] = {}

        # Parse all responses
        for batch in batches:
            for response, metadata in zip(
                batch.responses, batch.metadata, strict=False
            ):
                try:
                    # Parse response text
                    response_text = (
                        response.text if hasattr(response, "text") else str(response)
                    )
                    parsed = self.parser.parse(response_text)

                    # Handle None result (e.g. from "null" input for retry)
                    if parsed is None:
                        results[metadata.row_index] = {
                            "_row_index": metadata.row_index,
                            **{col: None for col in output_cols},
                        }
                        continue

                    # Map to output columns
                    row_data: Row = {"_row_index": metadata.row_index}

                    if len(output_cols) == 1:
                        # Single output column
                        if isinstance(parsed, dict) and output_cols[0] in parsed:
                            # 1. Exact match for column name
                            row_data[output_cols[0]] = parsed[output_cols[0]]
                        elif isinstance(parsed, dict) and "output" in parsed:
                            # 2. "output" field (standard fallback)
                            row_data[output_cols[0]] = parsed["output"]
                        elif isinstance(parsed, dict):
                            # 3. Use first value (risky but sometimes needed)
                            row_data[output_cols[0]] = next(iter(parsed.values()))
                        else:
                            row_data[output_cols[0]] = parsed
                    else:
                        # Multiple output columns
                        for col in output_cols:
                            row_data[col] = (
                                parsed.get(col, None)
                                if isinstance(parsed, dict)
                                else None
                            )

                    results[metadata.row_index] = row_data

                except Exception as e:
                    self.logger.error(
                        f"Failed to parse response at row {metadata.row_index}: {e}"
                    )
                    # Store None for failed parses
                    results[metadata.row_index] = {
                        "_row_index": metadata.row_index,
                        **{col: None for col in output_cols},
                    }

        # Convert to list sorted by row_index
        sorted_results = [results[idx] for idx in sorted(results.keys())]

        # Create ResultContainer
        container = ResultContainerImpl(
            data=sorted_results,
            columns=["_row_index"] + list(output_cols),
        )

        self.logger.info(f"Parsed {len(results)} responses")

        return container

    def validate_input(
        self, input_data: tuple[list[ResponseBatch], list[str]]
    ) -> ValidationResult:
        """Validate response batches."""
        result = ValidationResult(is_valid=True)

        batches, output_cols = input_data

        if not batches:
            result.add_error("No response batches provided")

        if not output_cols:
            result.add_error("No output columns specified")

        return result

    def estimate_cost(
        self, input_data: tuple[list[ResponseBatch], list[str]]
    ) -> CostEstimate:
        """Response parsing has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=sum(len(b.responses) for b in input_data[0]),
        )
