"""Unit tests for BudgetController."""

from decimal import Decimal

import pytest

from ondine.utils import BudgetController, BudgetExceededError


class TestBudgetController:
    """Test suite for BudgetController."""

    def test_initialization(self):
        """Test budget controller initialization."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        assert controller.max_budget == Decimal("10.0")
        assert controller.warn_at_75 is True
        assert controller.warn_at_90 is True

    def test_no_budget_limit(self):
        """Test with no budget limit."""
        controller = BudgetController(max_budget=None)

        # Should not raise or warn
        controller.check_budget(Decimal("1000.0"))

    def test_within_budget(self):
        """Test cost within budget."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        # Should not raise
        controller.check_budget(Decimal("5.0"))

    def test_budget_exceeded_with_enforcement(self):
        """Test budget exceeded raises error."""
        controller = BudgetController(max_budget=Decimal("10.0"), fail_on_exceed=True)

        with pytest.raises(BudgetExceededError, match="Budget exceeded"):
            controller.check_budget(Decimal("15.0"))

    def test_budget_exceeded_without_enforcement(self):
        """Test budget exceeded without enforcement."""
        controller = BudgetController(max_budget=Decimal("10.0"), fail_on_exceed=False)

        # Should not raise
        controller.check_budget(Decimal("15.0"))

    def test_75_percent_warning(self, caplog):
        """Test warning at 75% budget usage."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        controller.check_budget(Decimal("7.5"))

        assert controller._warned_75 is True

    def test_90_percent_warning(self, caplog):
        """Test warning at 90% budget usage."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        controller.check_budget(Decimal("9.0"))

        assert controller._warned_90 is True

    def test_get_remaining(self):
        """Test getting remaining budget."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        remaining = controller.get_remaining(Decimal("6.0"))

        assert remaining == Decimal("4.0")

    def test_get_usage_percentage(self):
        """Test getting usage percentage."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        usage = controller.get_usage_percentage(Decimal("7.5"))

        assert usage == 75.0

    def test_can_afford(self):
        """Test checking if can afford estimated cost."""
        controller = BudgetController(max_budget=Decimal("10.0"))

        # Can afford
        assert controller.can_afford(Decimal("2.0"), Decimal("5.0")) is True

        # Cannot afford
        assert controller.can_afford(Decimal("6.0"), Decimal("5.0")) is False
