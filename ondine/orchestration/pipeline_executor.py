"""
Pipeline executor for orchestrating stage execution.

Implements the complete execution flow with state machine management
as specified in the design document.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from ondine.core.models import ExecutionResult
from ondine.orchestration.execution_context import ExecutionContext
from ondine.orchestration.observers import ExecutionObserver
from ondine.orchestration.state_manager import StateManager
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import get_logger

logger = get_logger(__name__)


class ExecutionState(str, Enum):
    """Pipeline execution states."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineExecutor:
    """
    Orchestrates pipeline execution with state management.

    Implements Command and Mediator patterns for coordinating
    stages, observers, and state management.

    State Machine:
        IDLE → INITIALIZING → EXECUTING → [PAUSED ↔ EXECUTING] → COMPLETED
                                 ↓
                              FAILED
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        state_manager: StateManager,
        observers: list[ExecutionObserver] | None = None,
    ):
        """
        Initialize pipeline executor.

        Args:
            stages: Ordered list of processing stages
            state_manager: State manager for checkpointing
            observers: Optional execution observers
        """
        self.execution_id = uuid4()
        self.stages = stages
        self.state_manager = state_manager
        self.observers = observers or []
        self.state = ExecutionState.IDLE
        self.context: ExecutionContext | None = None
        self.logger = get_logger(f"{__name__}.{self.execution_id}")

    def add_observer(self, observer: ExecutionObserver) -> "PipelineExecutor":
        """
        Add execution observer.

        Args:
            observer: Observer to add

        Returns:
            Self for chaining
        """
        self.observers.append(observer)
        return self

    def execute(self, pipeline: Any) -> ExecutionResult:
        """
        Execute pipeline end-to-end.

        Args:
            pipeline: Pipeline instance to execute

        Returns:
            ExecutionResult with data and metrics

        Raises:
            RuntimeError: If pipeline in invalid state
        """
        if self.state not in [ExecutionState.IDLE, ExecutionState.FAILED]:
            raise RuntimeError(f"Cannot execute from state: {self.state}")

        try:
            # Initialize
            self.state = ExecutionState.INITIALIZING
            self.context = self._initialize_context()

            # Check for existing checkpoint
            if self.state_manager.can_resume(self.context.session_id):
                self.logger.info("Found existing checkpoint, resuming...")
                self.context = self.state_manager.load_checkpoint(
                    self.context.session_id
                )

            # Notify observers
            self._notify_pipeline_start(pipeline)

            # Execute stages
            self.state = ExecutionState.EXECUTING
            result_data = self._execute_all_stages(pipeline)

            # Mark completion
            self.state = ExecutionState.COMPLETED
            self.context.end_time = datetime.now()

            # Create result
            result = self._create_execution_result(result_data)

            # Cleanup checkpoints
            self.state_manager.cleanup_checkpoints(self.context.session_id)

            # Notify observers
            self._notify_pipeline_complete(result)

            return result

        except Exception as e:
            self.state = ExecutionState.FAILED
            self._notify_pipeline_error(e)

            # Save checkpoint on failure
            if self.context:
                self.state_manager.save_checkpoint(self.context)

            raise

    def pause(self) -> None:
        """
        Gracefully pause execution.

        Finishes current batch and saves checkpoint.
        """
        if self.state != ExecutionState.EXECUTING:
            raise RuntimeError(f"Cannot pause from state: {self.state}")

        self.logger.info("Pausing execution...")
        self.state = ExecutionState.PAUSED

        # Save checkpoint
        if self.context:
            self.state_manager.save_checkpoint(self.context)

    def resume(self, session_id: UUID) -> ExecutionResult:
        """
        Resume from saved checkpoint.

        Args:
            session_id: Session ID to resume

        Returns:
            ExecutionResult

        Raises:
            ValueError: If no checkpoint found
        """
        if not self.state_manager.can_resume(session_id):
            raise ValueError(f"No checkpoint found for session {session_id}")

        self.context = self.state_manager.load_checkpoint(session_id)
        if not self.context:
            raise ValueError("Failed to load checkpoint")

        self.logger.info(f"Resuming from row {self.context.last_processed_row}")

        # Continue execution
        # Note: Would need to reconstruct pipeline and skip processed rows
        raise NotImplementedError("Resume functionality coming soon")

    def cancel(self) -> None:
        """
        Immediately stop and save checkpoint.
        """
        self.logger.info("Cancelling execution...")

        # Save checkpoint
        if self.context:
            self.state_manager.save_checkpoint(self.context)

        self.state = ExecutionState.IDLE

    def _initialize_context(self) -> ExecutionContext:
        """Initialize execution context."""
        return ExecutionContext(session_id=self.execution_id)

    def _execute_all_stages(self, pipeline: Any) -> pd.DataFrame:
        """
        Execute all pipeline stages sequentially.

        Args:
            pipeline: Pipeline instance

        Returns:
            Final DataFrame result
        """
        # This will be implemented with actual stage orchestration
        # For now, delegate to pipeline's execution logic
        # In a future iteration, we'll move all execution here
        raise NotImplementedError(
            "Stage orchestration implemented in Pipeline.execute() currently"
        )

    def _create_execution_result(self, data: pd.DataFrame) -> ExecutionResult:
        """
        Create execution result from context and data.

        Args:
            data: Final processed data

        Returns:
            ExecutionResult
        """
        if not self.context:
            raise RuntimeError("No execution context available")

        return ExecutionResult(
            data=data,
            metrics=self.context.get_stats(),
            costs=self.context.accumulated_cost,
            execution_id=self.context.session_id,
            start_time=self.context.start_time,
            end_time=self.context.end_time,
            success=True,
        )

    def _notify_pipeline_start(self, pipeline: Any) -> None:
        """Notify all observers of pipeline start."""
        if self.context:
            for observer in self.observers:
                try:
                    observer.on_pipeline_start(pipeline, self.context)
                except Exception as e:
                    self.logger.error(
                        f"Observer {observer.__class__.__name__} failed: {e}"
                    )

    def _notify_pipeline_complete(self, result: ExecutionResult) -> None:
        """Notify all observers of pipeline completion."""
        if self.context:
            for observer in self.observers:
                try:
                    observer.on_pipeline_complete(self.context, result)
                except Exception as e:
                    self.logger.error(
                        f"Observer {observer.__class__.__name__} failed: {e}"
                    )

    def _notify_pipeline_error(self, error: Exception) -> None:
        """Notify all observers of pipeline error."""
        if self.context:
            for observer in self.observers:
                try:
                    observer.on_pipeline_error(self.context, error)
                except Exception as e:
                    self.logger.error(
                        f"Observer {observer.__class__.__name__} failed: {e}"
                    )
