"""Pydantic models for batch processing.

These models define the structure for batch requests and responses,
enabling type-safe batch processing with validation.
"""

from pydantic import BaseModel, Field, field_validator


class BatchItem(BaseModel):
    """Single item in a batch request or response.

    Attributes:
        id: Unique identifier for the item (1-based index)
        input: Input text for this item (request only)
        result: Output result for this item (response only)
    """

    id: int = Field(..., description="Unique item ID (1-based)", ge=1)
    input: str | None = Field(None, description="Input text (request)")
    result: str | None = Field(None, description="Output result (response)")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: int) -> int:
        """Validate ID is positive."""
        if v < 1:
            raise ValueError("ID must be >= 1")
        return v


class BatchRequest(BaseModel):
    """Batch request containing multiple items.

    Used for formatting batch prompts with structured data.

    Attributes:
        items: List of items to process
        task: Task description (e.g., "Classify sentiment")
        output_format: Expected output format description
    """

    items: list[BatchItem] = Field(..., description="Items to process")
    task: str | None = Field(None, description="Task description")
    output_format: str | None = Field(None, description="Output format instructions")

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: list[BatchItem]) -> list[BatchItem]:
        """Validate items list is not empty."""
        if not v:
            raise ValueError("Batch must contain at least 1 item")
        return v


class BatchResult(BaseModel):
    """Batch response containing multiple results.

    Used for parsing LLM responses with structured_predict().

    Attributes:
        results: List of results (one per input item)
    """

    results: list[BatchItem] = Field(..., description="Processing results")

    @field_validator("results")
    @classmethod
    def validate_results(cls, v: list[BatchItem]) -> list[BatchItem]:
        """Validate results list is not empty and all have IDs."""
        if not v:
            raise ValueError("Batch result must contain at least 1 result")

        # Validate all items have IDs
        for item in v:
            if item.id is None:
                raise ValueError("All result items must have an ID")

        return v

    def to_list(self) -> list[str]:
        """Convert to list of result strings, sorted by ID.

        Returns:
            List of result strings in ID order

        Raises:
            ValueError: If any result is missing
        """
        # Sort by ID
        sorted_results = sorted(self.results, key=lambda x: x.id)

        # Extract result strings
        return [item.result or "" for item in sorted_results]

    def get_missing_ids(self, expected_count: int) -> list[int]:
        """Get IDs that are missing from results.

        Args:
            expected_count: Expected number of results

        Returns:
            List of missing IDs (1-based)
        """
        present_ids = {item.id for item in self.results}
        expected_ids = set(range(1, expected_count + 1))
        return sorted(expected_ids - present_ids)


class BatchMetadata(BaseModel):
    """Metadata for batch processing.

    Tracks information needed for disaggregation and error handling.

    Attributes:
        original_count: Number of items in the batch
        row_ids: Original row IDs from the dataset
        prompt_template: Original prompt template used
    """

    original_count: int = Field(..., description="Number of items in batch", ge=1)
    row_ids: list[int] = Field(..., description="Original row IDs")
    prompt_template: str | None = Field(None, description="Original prompt template")

    @field_validator("row_ids")
    @classmethod
    def validate_row_ids(cls, v: list[int], info) -> list[int]:
        """Validate row_ids matches original_count."""
        if "original_count" in info.data and len(v) != info.data["original_count"]:
            raise ValueError(
                f"row_ids length ({len(v)}) must match original_count ({info.data['original_count']})"
            )
        return v
