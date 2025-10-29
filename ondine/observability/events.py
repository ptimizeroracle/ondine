"""
Event models for pipeline observability.

These event dataclasses are emitted at key points during pipeline execution
and dispatched to all registered observers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


@dataclass
class PipelineStartEvent:
    """
    Emitted when pipeline execution starts.

    Contains pipeline configuration and metadata for the entire run.
    """

    pipeline_id: UUID
    run_id: UUID
    timestamp: datetime
    trace_id: str
    span_id: str
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_rows: int = 0


@dataclass
class StageStartEvent:
    """
    Emitted when a pipeline stage begins execution.

    Tracks which stage is starting and when.
    """

    pipeline_id: UUID
    run_id: UUID
    stage_name: str
    stage_type: str
    timestamp: datetime
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallEvent:
    """
    Emitted on every LLM invocation.

    This is the MOST IMPORTANT event for LLM observability.
    Contains full prompt/completion text, tokens, cost, and optional RAG metadata.

    Observers can choose to truncate or sanitize prompts based on their needs.
    """

    # Request context (required fields first)
    pipeline_id: UUID
    run_id: UUID
    stage_name: str
    row_index: int
    timestamp: datetime
    trace_id: str
    span_id: str

    # LLM Request (required fields)
    prompt: str
    model: str
    provider: str
    temperature: float
    completion: str

    # Optional fields with defaults
    parent_span_id: str | None = None
    max_tokens: int | None = None
    system_message: str | None = None
    finish_reason: str = "stop"

    # Metadata with defaults
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    latency_ms: float = 0.0

    # RAG Context (optional, for future RAG integration)
    rag_context: str | None = None
    rag_sources: list[dict] | None = None
    rag_technique: str | None = None
    retrieval_latency_ms: float | None = None

    # Prompt Engineering (optional)
    prompt_template_id: str | None = None
    prompt_version: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageEndEvent:
    """
    Emitted when a pipeline stage completes successfully or with errors.

    Contains stage-level metrics and success status.
    """

    pipeline_id: UUID
    run_id: UUID
    stage_name: str
    success: bool
    timestamp: datetime
    trace_id: str
    span_id: str
    duration_ms: float = 0.0
    rows_processed: int = 0
    error: Exception | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """
    Emitted when errors occur during pipeline execution.

    Captures error context for debugging and alerting.
    """

    pipeline_id: UUID
    run_id: UUID
    timestamp: datetime
    trace_id: str
    span_id: str
    stage_name: str | None = None
    row_index: int | None = None
    error: Exception = None
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEndEvent:
    """
    Emitted when pipeline execution completes.

    Contains final metrics for the entire pipeline run.
    """

    pipeline_id: UUID
    run_id: UUID
    success: bool
    timestamp: datetime
    trace_id: str
    span_id: str
    total_duration_ms: float = 0.0
    rows_processed: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    rows_skipped: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
