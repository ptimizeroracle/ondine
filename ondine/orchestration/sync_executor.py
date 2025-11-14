"""
Synchronous execution strategy.

Default executor that maintains current behavior using ThreadPoolExecutor
for concurrent LLM calls.
"""

from datetime import datetime

import pandas as pd

from ondine.core.models import (
    CostEstimate,
    ExecutionResult,
    ProcessingStats,
)
from ondine.orchestration.execution_context import ExecutionContext
from ondine.orchestration.execution_strategy import ExecutionStrategy
from ondine.stages.pipeline_stage import PipelineStage
from ondine.utils import get_logger

logger = get_logger(__name__)


class SyncExecutor(ExecutionStrategy):
    """
    Synchronous execution strategy.

    Uses ThreadPoolExecutor for concurrent LLM calls while maintaining
    sequential stage execution. This is the default strategy that preserves
    current behavior.
    """

    def __init__(self):
        """Initialize synchronous executor."""
        self.logger = logger

    def execute(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute stages synchronously.

        Args:
            stages: Pipeline stages
            context: Execution context

        Returns:
            ExecutionResult with data and metrics
        """
        start_time = datetime.now()

        try:
            # Execute stages sequentially
            result_data = self._execute_stages(stages, context)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Calculate stats
            stats = ProcessingStats(
                total_rows=context.total_rows,
                processed_rows=context.last_processed_row + 1,
                failed_rows=context.total_rows - (context.last_processed_row + 1),
                skipped_rows=0,
                rows_per_second=context.total_rows / duration if duration > 0 else 0,
                total_duration_seconds=duration,
            )

            # Get cost estimate (leverage LlamaIndex token counts from intermediate_data)
            token_tracking = context.intermediate_data.get("token_tracking", {})
            input_tokens = token_tracking.get("input_tokens", 0)
            output_tokens = token_tracking.get("output_tokens", 0)

            cost_estimate = CostEstimate(
                total_cost=context.accumulated_cost,
                total_tokens=context.accumulated_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                rows=context.total_rows,
                confidence="actual",
            )

            return ExecutionResult(
                data=result_data,
                metrics=stats,
                costs=cost_estimate,
                execution_id=context.session_id,
                start_time=start_time,
                end_time=end_time,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

    def _execute_stages(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> pd.DataFrame:
        """Execute all stages sequentially."""
        data = None

        for stage in stages:
            self.logger.info(f"Executing stage: {stage.name}")
            context.current_stage = stage.name

            # Process data through stage
            data = stage.process(data, context)

        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def supports_async(self) -> bool:
        """Sync executor doesn't support async."""
        return False

    def supports_streaming(self) -> bool:
        """Sync executor doesn't support streaming."""
        return False

    @property
    def name(self) -> str:
        """Strategy name."""
        return "SyncExecutor"
