"""
Execution context for carrying runtime state between stages.

Implements Memento pattern for checkpoint serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from ondine.core.models import ProcessingStats

if TYPE_CHECKING:
    from ondine.observability.dispatcher import ObserverDispatcher
    from ondine.orchestration.observers import ExecutionObserver


@dataclass
class ExecutionContext:
    """
    Lightweight orchestration state (passed between pipeline stages).

    Scope: Runtime execution state and progress tracking
    Pattern: Memento (serializable for checkpointing)

    Cost Tracking in ExecutionContext:
    - Simple accumulation for orchestration purposes
    - Used by: Executors to track overall progress
    - NOT for: Detailed accounting (use CostTracker for that)

    Why separate from CostTracker?
    - ExecutionContext = orchestration state (stage progress, session ID, timing)
    - CostTracker = detailed accounting (per-stage breakdowns, thread-safe entries, metrics)
    - Different concerns, different use cases

    ExecutionContext is:
    - Passed between stages in the pipeline
    - Serialized for checkpointing
    - Focused on execution orchestration

    CostTracker is:
    - Used within LLMInvocationStage for detailed tracking
    - Thread-safe for concurrent operations
    - Focused on cost reporting and analytics

    See Also:
    - CostTracker: For detailed cost accounting with breakdowns
    - docs/TECHNICAL_REFERENCE.md: Cost tracking architecture

    Carries shared state between stages and tracks progress.
    Immutable for most fields to prevent accidental modification.
    """

    session_id: UUID = field(default_factory=uuid4)
    pipeline_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Progress tracking
    current_stage_index: int = 0
    last_processed_row: int = 0
    total_rows: int = 0

    # Cost tracking
    accumulated_cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    accumulated_tokens: int = 0

    # Intermediate data storage
    intermediate_data: dict[str, Any] = field(default_factory=dict)

    # Statistics
    failed_rows: int = 0
    skipped_rows: int = 0

    # Observers for progress notifications (backward compatibility)
    observers: list["ExecutionObserver"] = field(default_factory=list)

    # New observability system
    observer_dispatcher: Optional["ObserverDispatcher"] = None

    # Distributed tracing
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4()))

    def update_stage(self, stage_index: int) -> None:
        """Update current stage."""
        self.current_stage_index = stage_index

    def update_row(self, row_index: int) -> None:
        """Update last processed row."""
        self.last_processed_row = row_index

    def add_cost(self, cost: Decimal, tokens: int) -> None:
        """Add cost and token usage."""
        self.accumulated_cost += cost
        self.accumulated_tokens += tokens

    def notify_progress(self) -> None:
        """Notify all observers of progress update."""
        for observer in self.observers:
            try:
                observer.on_progress_update(self)
            except Exception:  # nosec B110
                # Silently ignore observer errors to not break pipeline
                # Observers are non-critical, pipeline should continue even if they fail
                pass

    def get_progress(self) -> float:
        """Get completion percentage."""
        if self.total_rows == 0:
            return 0.0
        return (self.last_processed_row / self.total_rows) * 100

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        duration = (
            (datetime.now() - self.start_time).total_seconds()
            if self.end_time is None
            else (self.end_time - self.start_time).total_seconds()
        )

        # last_processed_row is 0-based index, so add 1 for count
        actual_processed = (
            self.last_processed_row + 1 if self.last_processed_row >= 0 else 0
        )

        rows_per_second = actual_processed / duration if duration > 0 else 0.0

        return ProcessingStats(
            total_rows=self.total_rows,
            processed_rows=actual_processed,
            failed_rows=self.failed_rows,
            skipped_rows=self.skipped_rows,
            rows_per_second=rows_per_second,
            total_duration_seconds=duration,
        )

    def to_checkpoint(self) -> dict[str, Any]:
        """
        Serialize to checkpoint dictionary (Memento pattern).

        Returns:
            Dictionary representation for persistence
        """
        return {
            "session_id": str(self.session_id),
            "pipeline_id": str(self.pipeline_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage_index": self.current_stage_index,
            "last_processed_row": self.last_processed_row,
            "total_rows": self.total_rows,
            "accumulated_cost": str(self.accumulated_cost),
            "accumulated_tokens": self.accumulated_tokens,
            "intermediate_data": self.intermediate_data,
            "failed_rows": self.failed_rows,
            "skipped_rows": self.skipped_rows,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "ExecutionContext":
        """
        Deserialize from checkpoint dictionary.

        Args:
            data: Checkpoint data

        Returns:
            Restored ExecutionContext
        """
        return cls(
            session_id=UUID(data["session_id"]),
            pipeline_id=UUID(data["pipeline_id"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            current_stage_index=data["current_stage_index"],
            last_processed_row=data["last_processed_row"],
            total_rows=data["total_rows"],
            accumulated_cost=Decimal(data["accumulated_cost"]),
            accumulated_tokens=data["accumulated_tokens"],
            intermediate_data=data.get("intermediate_data", {}),
            failed_rows=data.get("failed_rows", 0),
            skipped_rows=data.get("skipped_rows", 0),
        )

    # Aliases for backward compatibility
    def to_dict(self) -> dict[str, Any]:
        """Alias for to_checkpoint()."""
        return self.to_checkpoint()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        """Alias for from_checkpoint()."""
        return cls.from_checkpoint(data)
