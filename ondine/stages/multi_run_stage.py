"""
Multi-run stage for executing stages multiple times with aggregation.

Implements Decorator pattern to wrap any stage and run it multiple times.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Generic, TypeVar

from ondine.core.models import CostEstimate, ValidationResult
from ondine.stages.pipeline_stage import PipelineStage, TInput, TOutput

T = TypeVar("T")


class AggregationStrategy(ABC, Generic[T]):
    """
    Abstract base for aggregation strategies.

    Follows Strategy pattern for different ways to aggregate results.
    """

    @abstractmethod
    def aggregate(self, results: list[T]) -> T:
        """
        Aggregate multiple results into one.

        Args:
            results: List of results from multiple runs

        Returns:
            Aggregated result
        """
        pass


class ConsensusStrategy(AggregationStrategy[str]):
    """Returns most common result (consensus voting)."""

    def aggregate(self, results: list[str]) -> str:
        """Return most frequent result."""
        if not results:
            return ""

        # Count occurrences
        counter = Counter(results)
        return counter.most_common(1)[0][0]


class FirstSuccessStrategy(AggregationStrategy[T]):
    """Returns first successful (non-None) result."""

    def aggregate(self, results: list[T]) -> T:
        """Return first non-None result."""
        for result in results:
            if result is not None:
                return result
        return results[0] if results else None


class AllStrategy(AggregationStrategy[T]):
    """Returns all results as list (no aggregation)."""

    def aggregate(self, results: list[T]) -> list[T]:
        """Return all results."""
        return results


class AverageStrategy(AggregationStrategy[float]):
    """Returns average of numeric results."""

    def aggregate(self, results: list[float]) -> float:
        """Return average."""
        if not results:
            return 0.0
        return sum(results) / len(results)


class MultiRunStage(PipelineStage[TInput, TOutput]):
    """
    Decorator stage that runs wrapped stage multiple times.

    Use cases:
    - Run LLM 3 times, take consensus (reduce hallucinations)
    - Retry until success
    - Collect multiple responses for analysis

    Example:
        multi_llm = MultiRunStage(
            wrapped=LLMInvocationStage(...),
            num_runs=3,
            aggregation=ConsensusStrategy()
        )
    """

    def __init__(
        self,
        wrapped_stage: PipelineStage[TInput, TOutput],
        num_runs: int = 3,
        aggregation_strategy: AggregationStrategy | None = None,
    ):
        """
        Initialize multi-run stage.

        Args:
            wrapped_stage: Stage to execute multiple times
            num_runs: Number of times to run
            aggregation_strategy: Strategy for aggregating results
        """
        super().__init__(f"MultiRun({wrapped_stage.name})")
        self.wrapped_stage = wrapped_stage
        self.num_runs = num_runs
        self.aggregation_strategy = aggregation_strategy or ConsensusStrategy()

    def process(self, input_data: TInput, context: Any) -> TOutput:
        """Execute wrapped stage multiple times and aggregate."""
        results: list[TOutput] = []

        self.logger.info(f"Running {self.wrapped_stage.name} {self.num_runs} times")

        for run_num in range(self.num_runs):
            try:
                result = self.wrapped_stage.process(input_data, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Run {run_num + 1}/{self.num_runs} failed: {e}")
                # Continue with other runs
                continue

        if not results:
            raise RuntimeError(
                f"All {self.num_runs} runs failed for {self.wrapped_stage.name}"
            )

        # Aggregate results
        aggregated = self.aggregation_strategy.aggregate(results)

        self.logger.info(
            f"Aggregated {len(results)} results using "
            f"{self.aggregation_strategy.__class__.__name__}"
        )

        return aggregated

    def validate_input(self, input_data: TInput) -> ValidationResult:
        """Delegate validation to wrapped stage."""
        return self.wrapped_stage.validate_input(input_data)

    def estimate_cost(self, input_data: TInput) -> CostEstimate:
        """Estimate cost as num_runs × wrapped stage cost."""
        single_run_cost = self.wrapped_stage.estimate_cost(input_data)

        return CostEstimate(
            total_cost=single_run_cost.total_cost * self.num_runs,
            total_tokens=single_run_cost.total_tokens * self.num_runs,
            input_tokens=single_run_cost.input_tokens * self.num_runs,
            output_tokens=single_run_cost.output_tokens * self.num_runs,
            rows=single_run_cost.rows,
            confidence=f"{single_run_cost.confidence} × {self.num_runs} runs",
        )
