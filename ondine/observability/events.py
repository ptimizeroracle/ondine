"""
Event models for pipeline observability.

These event dataclasses are emitted at key points during pipeline execution
and dispatched to all registered observers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
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
    parent_span_id: Optional[str] = None
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
    parent_span_id: Optional[str] = None
    max_tokens: Optional[int] = None
    system_message: Optional[str] = None
    finish_reason: str = "stop"

    # Metadata with defaults
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    latency_ms: float = 0.0

    # RAG Context (optional, for future RAG integration)
    rag_context: Optional[str] = None
    rag_sources: Optional[list[dict]] = None
    rag_technique: Optional[str] = None
    retrieval_latency_ms: Optional[float] = None

    # Prompt Engineering (optional)
    prompt_template_id: Optional[str] = None
    prompt_version: Optional[str] = None

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
    error: Optional[Exception] = None
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
    stage_name: Optional[str] = None
    row_index: Optional[int] = None
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
