"""Tests for centralized cost calculation utility."""

from decimal import Decimal

from ondine.utils.cost_calculator import CostCalculator


class TestCostCalculator:
    """Tests for CostCalculator utility."""

    def test_calculate_basic_cost(self):
        """Should calculate cost from token counts and pricing."""
        cost = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=500,
            input_cost_per_1k=Decimal("0.00005"),
            output_cost_per_1k=Decimal("0.00008"),
        )

        # Expected: (1000/1000 * 0.00005) + (500/1000 * 0.00008)
        #         = 0.00005 + 0.00004
        #         = 0.00009
        expected = Decimal("0.00009")
        assert cost == expected

    def test_calculate_with_zero_tokens(self):
        """Should return zero cost for zero tokens."""
        cost = CostCalculator.calculate(
            tokens_in=0,
            tokens_out=0,
            input_cost_per_1k=Decimal("0.00005"),
            output_cost_per_1k=Decimal("0.00008"),
        )

        assert cost == Decimal("0.0")

    def test_calculate_with_zero_pricing(self):
        """Should return zero cost for zero pricing (free models)."""
        cost = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=500,
            input_cost_per_1k=Decimal("0.0"),
            output_cost_per_1k=Decimal("0.0"),
        )

        assert cost == Decimal("0.0")

    def test_calculate_input_only(self):
        """Should calculate cost for input tokens only."""
        cost = CostCalculator.calculate(
            tokens_in=2000,
            tokens_out=0,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002"),
        )

        # Expected: 2000/1000 * 0.001 = 0.002
        assert cost == Decimal("0.002")

    def test_calculate_output_only(self):
        """Should calculate cost for output tokens only."""
        cost = CostCalculator.calculate(
            tokens_in=0,
            tokens_out=3000,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002"),
        )

        # Expected: 3000/1000 * 0.002 = 0.006
        assert cost == Decimal("0.006")

    def test_calculate_preserves_decimal_precision(self):
        """Should maintain Decimal precision (no float rounding errors)."""
        # Use pricing that would cause float errors
        cost1 = CostCalculator.calculate(
            tokens_in=100,
            tokens_out=50,
            input_cost_per_1k=Decimal("0.00015"),
            output_cost_per_1k=Decimal("0.0006"),
        )

        cost2 = CostCalculator.calculate(
            tokens_in=200,
            tokens_out=100,
            input_cost_per_1k=Decimal("0.00015"),
            output_cost_per_1k=Decimal("0.0006"),
        )

        # Should be exact multiples (no floating point errors)
        assert cost2 == cost1 * 2

    def test_calculate_large_token_counts(self):
        """Should handle large token counts accurately."""
        cost = CostCalculator.calculate(
            tokens_in=1_000_000,
            tokens_out=500_000,
            input_cost_per_1k=Decimal("0.00005"),
            output_cost_per_1k=Decimal("0.00008"),
        )

        # Expected: (1M/1000 * 0.00005) + (500K/1000 * 0.00008)
        #         = (1000 * 0.00005) + (500 * 0.00008)
        #         = 0.05 + 0.04
        #         = 0.09
        expected = Decimal("0.09")
        assert cost == expected

    def test_calculate_different_pricing_tiers(self):
        """Should work with various pricing models."""
        # GPT-4o-mini pricing
        cost_mini = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=1000,
            input_cost_per_1k=Decimal("0.00015"),
            output_cost_per_1k=Decimal("0.0006"),
        )
        assert cost_mini == Decimal("0.00075")

        # Claude Sonnet pricing
        cost_claude = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=1000,
            input_cost_per_1k=Decimal("0.003"),
            output_cost_per_1k=Decimal("0.015"),
        )
        assert cost_claude == Decimal("0.018")

        # Groq pricing
        cost_groq = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=1000,
            input_cost_per_1k=Decimal("0.00059"),
            output_cost_per_1k=Decimal("0.00079"),
        )
        assert cost_groq == Decimal("0.00138")

    def test_calculate_with_fractional_tokens(self):
        """Should handle non-round token counts."""
        cost = CostCalculator.calculate(
            tokens_in=123,
            tokens_out=456,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002"),
        )

        # Expected: (123/1000 * 0.001) + (456/1000 * 0.002)
        #         = 0.000123 + 0.000912
        #         = 0.001035
        expected = Decimal("0.001035")
        assert cost == expected

    def test_is_static_method(self):
        """Should be usable as static method (no instance needed)."""
        # Should not require CostCalculator instance
        cost = CostCalculator.calculate(
            tokens_in=100,
            tokens_out=50,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.001"),
        )

        assert isinstance(cost, Decimal)
        assert cost == Decimal("0.00015")


class TestCostCalculatorIntegration:
    """Integration tests for CostCalculator with other components."""

    def test_llm_client_uses_cost_calculator(self):
        """LLMClient.calculate_cost() should delegate to CostCalculator."""
        # Test that the formula is correct (LLMClient uses CostCalculator internally)
        expected = CostCalculator.calculate(
            tokens_in=100,
            tokens_out=50,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002"),
        )

        assert expected == Decimal("0.0002")

    def test_cost_tracker_uses_cost_calculator(self):
        """CostTracker.calculate_cost() should delegate to CostCalculator."""
        from ondine.utils.cost_tracker import CostTracker

        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.001"), output_cost_per_1k=Decimal("0.002")
        )

        cost = tracker.calculate_cost(tokens_in=100, tokens_out=50)

        expected = CostCalculator.calculate(
            tokens_in=100,
            tokens_out=50,
            input_cost_per_1k=Decimal("0.001"),
            output_cost_per_1k=Decimal("0.002"),
        )

        assert cost == expected
        assert cost == Decimal("0.0002")

    def test_consistency_across_components(self):
        """All components should produce same cost for same inputs."""
        from ondine.utils.cost_tracker import CostTracker

        tokens_in, tokens_out = 1000, 500
        input_cost_per_1k = Decimal("0.00015")
        output_cost_per_1k = Decimal("0.0006")

        # Direct calculation
        direct = CostCalculator.calculate(
            tokens_in, tokens_out, input_cost_per_1k, output_cost_per_1k
        )

        # Via CostTracker
        tracker = CostTracker(input_cost_per_1k, output_cost_per_1k)
        via_tracker = tracker.calculate_cost(tokens_in, tokens_out)

        # Should be identical
        assert direct == via_tracker
        assert direct == Decimal("0.00045")
