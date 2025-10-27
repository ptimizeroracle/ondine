"""Unit tests for CostTracker."""

import time
from decimal import Decimal

from ondine.utils import CostTracker


class TestCostTracker:
    """Test suite for CostTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        assert tracker.total_cost == Decimal("0.0")
        assert tracker.total_tokens == 0
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0

    def test_cost_calculation(self):
        """Test cost calculation."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        cost = tracker.calculate_cost(1000, 500)

        # 1000 tokens * $0.01/1k = $0.01
        # 500 tokens * $0.02/1k = $0.01
        # Total = $0.02
        assert cost == Decimal("0.02")

    def test_add_cost_entry(self):
        """Test adding cost entries."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        # Add first entry
        cost1 = tracker.add(100, 50, "gpt-4o-mini", time.time())
        assert cost1 == Decimal("0.002")

        # Add second entry
        cost2 = tracker.add(200, 100, "gpt-4o-mini", time.time())
        assert cost2 == Decimal("0.004")

        # Check totals
        assert tracker.total_cost == Decimal("0.006")
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 150

    def test_stage_cost_tracking(self):
        """Test tracking costs by stage."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        # Add costs for different stages
        tracker.add(100, 50, "gpt-4o", time.time(), stage="stage1")
        tracker.add(200, 100, "gpt-4o", time.time(), stage="stage1")
        tracker.add(150, 75, "gpt-4o", time.time(), stage="stage2")

        stage_costs = tracker.get_stage_costs()

        assert "stage1" in stage_costs
        assert "stage2" in stage_costs
        assert stage_costs["stage1"] > stage_costs["stage2"]

    def test_get_estimate(self):
        """Test getting cost estimate."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        tracker.add(1000, 500, "gpt-4o", time.time())

        estimate = tracker.get_estimate(rows=10)

        assert estimate.total_cost == Decimal("0.02")
        assert estimate.total_tokens == 1500
        assert estimate.rows == 10
        assert estimate.confidence == "actual"

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker(
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.02"),
        )

        tracker.add(1000, 500, "gpt-4o", time.time())
        assert tracker.total_cost > 0

        tracker.reset()

        assert tracker.total_cost == Decimal("0.0")
        assert tracker.total_tokens == 0
