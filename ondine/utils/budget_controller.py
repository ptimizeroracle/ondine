"""
Budget control and enforcement for LLM costs.

Implements cost monitoring with threshold warnings and hard limits.
"""

from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""

    pass


class BudgetController:
    """
    Controls and enforces budget limits during execution.

    Follows Single Responsibility: only handles budget management.
    """

    def __init__(
        self,
        max_budget: Decimal | None = None,
        warn_at_75: bool = True,
        warn_at_90: bool = True,
        fail_on_exceed: bool = True,
    ):
        """
        Initialize budget controller.

        Args:
            max_budget: Maximum allowed budget in USD
            warn_at_75: Warn at 75% of budget
            warn_at_90: Warn at 90% of budget
            fail_on_exceed: Raise error if budget exceeded
        """
        self.max_budget = max_budget
        self.warn_at_75 = warn_at_75
        self.warn_at_90 = warn_at_90
        self.fail_on_exceed = fail_on_exceed

        self._warned_75 = False
        self._warned_90 = False

    def check_budget(self, current_cost: Decimal) -> None:
        """
        Check if budget is within limits.

        Args:
            current_cost: Current accumulated cost

        Raises:
            BudgetExceededError: If budget exceeded and fail_on_exceed=True
        """
        if self.max_budget is None:
            return

        usage_ratio = float(current_cost / self.max_budget)

        # 75% warning
        if self.warn_at_75 and not self._warned_75 and usage_ratio >= 0.75:
            logger.warning(
                f"Budget warning: 75% used "
                f"(${current_cost:.4f} / ${self.max_budget:.2f})"
            )
            self._warned_75 = True

        # 90% warning
        if self.warn_at_90 and not self._warned_90 and usage_ratio >= 0.90:
            logger.warning(
                f"Budget warning: 90% used "
                f"(${current_cost:.4f} / ${self.max_budget:.2f})"
            )
            self._warned_90 = True

        # Budget exceeded
        if current_cost > self.max_budget:
            error_msg = f"Budget exceeded: ${current_cost:.4f} > ${self.max_budget:.2f}"
            logger.error(error_msg)

            if self.fail_on_exceed:
                raise BudgetExceededError(error_msg)

    def get_remaining(self, current_cost: Decimal) -> Decimal | None:
        """
        Get remaining budget.

        Args:
            current_cost: Current accumulated cost

        Returns:
            Remaining budget or None if no limit
        """
        if self.max_budget is None:
            return None
        return self.max_budget - current_cost

    def get_usage_percentage(self, current_cost: Decimal) -> float | None:
        """
        Get budget usage as percentage.

        Args:
            current_cost: Current accumulated cost

        Returns:
            Usage percentage or None if no limit
        """
        if self.max_budget is None:
            return None
        return float(current_cost / self.max_budget) * 100

    def can_afford(self, estimated_cost: Decimal, current_cost: Decimal) -> bool:
        """
        Check if estimated additional cost is within budget.

        Args:
            estimated_cost: Estimated cost for next operation
            current_cost: Current accumulated cost

        Returns:
            True if within budget
        """
        if self.max_budget is None:
            return True
        return (current_cost + estimated_cost) <= self.max_budget
