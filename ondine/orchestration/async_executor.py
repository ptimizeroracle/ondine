"""
Asynchronous execution strategy.

Provides async/await support for non-blocking execution, ideal for
integration with FastAPI, aiohttp, and other async frameworks.
"""

import asyncio
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


class AsyncExecutor(ExecutionStrategy):
    """
    Asynchronous execution strategy.

    Uses asyncio for true non-blocking execution. Leverages LlamaIndex's
    async methods (acomplete) for concurrent LLM calls without threads.

    Benefits:
    - Non-blocking (works with FastAPI, aiohttp)
    - Better resource utilization
    - Higher concurrency without thread overhead
    - Ideal for I/O-bound operations
    """

    def __init__(self, max_concurrency: int = 10):
        """
        Initialize async executor.

        Args:
            max_concurrency: Maximum concurrent async tasks
        """
        self.max_concurrency = max_concurrency
        self.logger = logger
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute stages asynchronously.

        Args:
            stages: Pipeline stages
            context: Execution context

        Returns:
            ExecutionResult with data and metrics
        """
        start_time = datetime.now()

        try:
            # Execute stages with async/await
            result_data = await self._execute_stages_async(stages, context)

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
            self.logger.error(f"Async pipeline execution failed: {e}")
            raise

    async def _execute_stages_async(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> pd.DataFrame:
        """
        Execute stages asynchronously.

        For stages that support async (have process_async method), use it.
        For sync-only stages, run in thread pool to avoid blocking.
        """
        current_data = None

        for stage in stages:
            self.logger.info(f"Starting async stage: {stage.name}")

            # Check if stage has async support
            if hasattr(stage, "process_async"):
                # Use native async
                current_data = await stage.process_async(current_data, context)
            else:
                # Run sync stage in thread to avoid blocking
                current_data = await asyncio.to_thread(
                    stage.process, current_data, context
                )

            self.logger.info(f"Completed async stage: {stage.name}")

        return current_data

    async def _invoke_llm_batch_async(self, prompts: list[str], llm_client):
        """
        Invoke LLM for multiple prompts concurrently with semaphore.

        Uses asyncio.gather for true parallelism without thread overhead.
        """

        async def _invoke_one(prompt: str):
            async with self.semaphore:
                # Use LlamaIndex async method
                if hasattr(llm_client, "acomplete"):
                    return await llm_client.acomplete(prompt)
                # Fallback to sync in thread
                return await asyncio.to_thread(llm_client.invoke, prompt)

        # Execute all prompts concurrently
        tasks = [_invoke_one(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def supports_async(self) -> bool:
        """Async executor supports async."""
        return True

    def supports_streaming(self) -> bool:
        """Async executor doesn't support streaming."""
        return False

    # Alias for backward compatibility
    async def execute_async(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Alias for execute() method."""
        return await self.execute(stages, context)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "AsyncExecutor"
