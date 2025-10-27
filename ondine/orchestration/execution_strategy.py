"""
Execution strategy abstraction for different execution modes.

Implements Strategy pattern to support sync, async, and streaming execution
without modifying core pipeline logic.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

import pandas as pd

from ondine.core.models import ExecutionResult
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.pipeline_stage import PipelineStage


class ExecutionStrategy(ABC):
    """
    Abstract base for execution strategies.

    Follows Strategy pattern: defines interface for executing pipeline stages
    in different modes (sync, async, streaming).
    """

    @abstractmethod
    def execute(
        self,
        stages: list[PipelineStage],
        context: ExecutionContext,
    ) -> ExecutionResult | Iterator[pd.DataFrame] | AsyncIterator[pd.DataFrame]:
        """
        Execute pipeline stages.

        Args:
            stages: List of pipeline stages to execute
            context: Execution context for state management

        Returns:
            ExecutionResult or iterator for streaming
        """
        pass

    @abstractmethod
    def supports_async(self) -> bool:
        """Whether this strategy supports async execution."""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this strategy supports streaming."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass
