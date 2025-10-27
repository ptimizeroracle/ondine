"""
Core data models for execution results and metadata.

These models represent the outputs and state information from pipeline
execution with type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pandas as pd


@dataclass
class LLMResponse:
    """Response from a single LLM invocation."""

    text: str
    tokens_in: int
    tokens_out: int
    model: str
    cost: Decimal
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostEstimate:
    """Cost estimation for pipeline execution."""

    total_cost: Decimal
    total_tokens: int
    input_tokens: int
    output_tokens: int
    rows: int
    breakdown_by_stage: dict[str, Decimal] = field(default_factory=dict)
    confidence: str = "estimate"  # estimate, sample-based, actual


@dataclass
class ProcessingStats:
    """Statistics from pipeline execution."""

    total_rows: int
    processed_rows: int
    failed_rows: int
    skipped_rows: int
    rows_per_second: float
    total_duration_seconds: float
    stage_durations: dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Information about an error during processing."""

    row_index: int
    stage_name: str
    error_type: str
    error_message: str
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Complete result from pipeline execution."""

    data: pd.DataFrame
    metrics: ProcessingStats
    costs: CostEstimate
    errors: list[ErrorInfo] = field(default_factory=list)
    execution_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.metrics.total_rows == 0:
            return 0.0
        return (self.metrics.failed_rows / self.metrics.total_rows) * 100

    def validate_output_quality(self, output_columns: list[str]) -> "QualityReport":
        """
        Validate the quality of output data by checking for null/empty values.

        Args:
            output_columns: List of output column names to check

        Returns:
            QualityReport with quality metrics and warnings
        """
        total = len(self.data)

        # Count null and empty values across output columns
        null_count = 0
        empty_count = 0

        for col in output_columns:
            if col in self.data.columns:
                # Count nulls (None, NaN, NaT)
                null_count += self.data[col].isna().sum()
                # Count empty strings (only for string columns)
                if self.data[col].dtype == "object":
                    empty_count += (self.data[col].astype(str).str.strip() == "").sum()

        # Calculate per-column metrics (exclude both nulls and empties)
        valid_outputs = total - null_count - empty_count
        success_rate = (valid_outputs / total * 100) if total > 0 else 0.0

        # Determine quality score
        if success_rate >= 95.0:
            quality_score = "excellent"
        elif success_rate >= 80.0:
            quality_score = "good"
        elif success_rate >= 50.0:
            quality_score = "poor"
        else:
            quality_score = "critical"

        # Generate warnings and issues
        warnings = []
        issues = []

        if success_rate < 70.0:
            issues.append(
                f"⚠️  LOW SUCCESS RATE: Only {success_rate:.1f}% of outputs are valid "
                f"({valid_outputs}/{total} rows)"
            )

        if null_count > total * 0.3:  # > 30% nulls
            issues.append(
                f"⚠️  HIGH NULL RATE: {null_count} null values found "
                f"({null_count / total * 100:.1f}% of rows)"
            )

        if empty_count > total * 0.1:  # > 10% empty
            warnings.append(
                f"Empty outputs detected: {empty_count} rows "
                f"({empty_count / total * 100:.1f}%)"
            )

        # Check if reported metrics match actual data quality
        if self.metrics.failed_rows == 0 and null_count > 0:
            issues.append(
                f"⚠️  METRICS MISMATCH: Pipeline reported 0 failures but "
                f"{null_count} rows have null outputs. This may indicate silent errors."
            )

        return QualityReport(
            total_rows=total,
            valid_outputs=valid_outputs,
            null_outputs=null_count,
            empty_outputs=empty_count,
            success_rate=success_rate,
            quality_score=quality_score,
            warnings=warnings,
            issues=issues,
        )


@dataclass
class ValidationResult:
    """Result from validation checks."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class QualityReport:
    """Quality assessment of pipeline output."""

    total_rows: int
    valid_outputs: int
    null_outputs: int
    empty_outputs: int
    success_rate: float
    quality_score: str  # "excellent", "good", "poor", "critical"
    warnings: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable (>= 70% success)."""
        return self.success_rate >= 70.0

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.issues) > 0 or len(self.warnings) > 0


@dataclass
class WriteConfirmation:
    """Confirmation of successful data write."""

    path: str
    rows_written: int
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    session_id: UUID
    checkpoint_path: str
    row_index: int
    stage_index: int
    timestamp: datetime
    size_bytes: int


@dataclass
class RowMetadata:
    """Metadata for a single row during processing."""

    row_index: int
    row_id: Any | None = None
    batch_id: int | None = None
    attempt: int = 1
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBatch:
    """Batch of prompts for processing."""

    prompts: list[str]
    metadata: list[RowMetadata]
    batch_id: int


@dataclass
class ResponseBatch:
    """Batch of responses from LLM."""

    responses: list[str]
    metadata: list[RowMetadata]
    tokens_used: int
    cost: Decimal
    batch_id: int
    latencies_ms: list[float] = field(default_factory=list)
