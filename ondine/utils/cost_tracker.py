"""
Cost tracking for LLM API calls.

Provides accurate cost tracking with thread safety and detailed breakdowns.
"""

import threading
from dataclasses import dataclass
from decimal import Decimal

from ondine.core.models import CostEstimate


@dataclass
class CostEntry:
    """Single cost tracking entry."""

    tokens_in: int
    tokens_out: int
    cost: Decimal
    model: str
    timestamp: float


class CostTracker:
    """
    Detailed cost accounting with thread-safety and per-stage breakdowns.

    Scope: Detailed financial tracking and reporting
    Pattern: Accumulator with thread-safe operations

    Use CostTracker for:
    - Stage-by-stage cost breakdowns
    - Detailed entry logging (timestamp, model, tokens)
    - Thread-safe accumulation in concurrent execution
    - Cost reporting and analytics
    - Budget enforcement (via BudgetController)

    NOT for:
    - Simple orchestration state (use ExecutionContext for that)

    Why separate from ExecutionContext?
    - CostTracker = detailed accounting system (entries, breakdowns, thread-safety)
    - ExecutionContext = orchestration state (progress, session, timing)
    - Different concerns: accounting vs execution control

    Thread Safety:
    - All methods protected by threading.Lock
    - Safe for concurrent LLM invocations

    Example:
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.00015"),
            output_cost_per_1k=Decimal("0.0006")
        )
        cost = tracker.add(tokens_in=1000, tokens_out=500, model="gpt-4o-mini")
        breakdown = tracker.get_stage_costs()  # {"llm_invocation": Decimal("0.00045")}

    See Also:
    - ExecutionContext: For orchestration-level state
    - BudgetController: For cost limit enforcement
    - docs/TECHNICAL_REFERENCE.md: Cost tracking architecture
    """

    def __init__(
        self,
        input_cost_per_1k: Decimal | None = None,
        output_cost_per_1k: Decimal | None = None,
    ):
        """
        Initialize cost tracker.

        Args:
            input_cost_per_1k: Input token cost per 1K tokens
            output_cost_per_1k: Output token cost per 1K tokens
        """
        self.input_cost_per_1k = input_cost_per_1k or Decimal("0.0")
        self.output_cost_per_1k = output_cost_per_1k or Decimal("0.0")

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = Decimal("0.0")
        self._entries: list[CostEntry] = []
        self._stage_costs: dict[str, Decimal] = {}
        self._lock = threading.Lock()

    def add(
        self,
        tokens_in: int,
        tokens_out: int,
        model: str,
        timestamp: float,
        stage: str | None = None,
    ) -> Decimal:
        """
        Add cost entry.

        Args:
            tokens_in: Input tokens used
            tokens_out: Output tokens used
            model: Model identifier
            timestamp: Timestamp of request
            stage: Optional stage name

        Returns:
            Cost for this entry
        """
        cost = self.calculate_cost(tokens_in, tokens_out)

        with self._lock:
            entry = CostEntry(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                model=model,
                timestamp=timestamp,
            )
            self._entries.append(entry)

            self._total_input_tokens += tokens_in
            self._total_output_tokens += tokens_out
            self._total_cost += cost

            if stage:
                self._stage_costs[stage] = (
                    self._stage_costs.get(stage, Decimal("0.0")) + cost
                )

        return cost

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        """
        Calculate cost for given token counts.

        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Total cost
        """
        from ondine.utils.cost_calculator import CostCalculator

        return CostCalculator.calculate(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            input_cost_per_1k=self.input_cost_per_1k,
            output_cost_per_1k=self.output_cost_per_1k,
        )

    @property
    def total_cost(self) -> Decimal:
        """Get total accumulated cost."""
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        with self._lock:
            return self._total_input_tokens + self._total_output_tokens

    @property
    def input_tokens(self) -> int:
        """Get total input tokens."""
        with self._lock:
            return self._total_input_tokens

    @property
    def output_tokens(self) -> int:
        """Get total output tokens."""
        with self._lock:
            return self._total_output_tokens

    def get_estimate(self, rows: int = 0) -> CostEstimate:
        """
        Get cost estimate.

        Args:
            rows: Number of rows processed

        Returns:
            CostEstimate object
        """
        with self._lock:
            total_tokens = self._total_input_tokens + self._total_output_tokens
            return CostEstimate(
                total_cost=self._total_cost,
                total_tokens=total_tokens,
                input_tokens=self._total_input_tokens,
                output_tokens=self._total_output_tokens,
                rows=rows,
                breakdown_by_stage=dict(self._stage_costs),
                confidence="actual",
            )

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost = Decimal("0.0")
            self._entries.clear()
            self._stage_costs.clear()

    def get_stage_costs(self) -> dict[str, Decimal]:
        """Get costs breakdown by stage."""
        with self._lock:
            return dict(self._stage_costs)
