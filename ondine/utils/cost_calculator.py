"""
Centralized cost calculation utilities.

Provides single source of truth for LLM cost calculation formula,
eliminating duplication across LLMClient and CostTracker.
"""

from decimal import Decimal


class CostCalculator:
    """
    Centralized cost calculation for LLM API usage.

    Single Responsibility: Calculate cost from token counts and pricing.
    Used by: LLMClient, CostTracker, and any component needing cost calculation.

    Design Decision: Centralize the cost formula in one place to ensure
    consistency and make future changes (e.g., tiered pricing) easier.
    """

    @staticmethod
    def calculate(
        tokens_in: int,
        tokens_out: int,
        input_cost_per_1k: Decimal,
        output_cost_per_1k: Decimal,
    ) -> Decimal:
        """
        Calculate cost from token counts and pricing.

        Formula:
            cost = (tokens_in / 1000) * input_cost_per_1k +
                   (tokens_out / 1000) * output_cost_per_1k

        Args:
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            input_cost_per_1k: Cost per 1000 input tokens
            output_cost_per_1k: Cost per 1000 output tokens

        Returns:
            Total cost as Decimal (exact precision for financial calculations)

        Example:
            >>> from decimal import Decimal
            >>> cost = CostCalculator.calculate(
            ...     tokens_in=1000,
            ...     tokens_out=500,
            ...     input_cost_per_1k=Decimal("0.00005"),
            ...     output_cost_per_1k=Decimal("0.00008")
            ... )
            >>> cost
            Decimal('0.00009')
        """
        input_cost = (Decimal(tokens_in) / 1000) * input_cost_per_1k
        output_cost = (Decimal(tokens_out) / 1000) * output_cost_per_1k
        return input_cost + output_cost
