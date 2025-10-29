"""Tests for quality validation and auto-retry functionality."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pandas as pd

from ondine.core.models import (
    CostEstimate,
    ExecutionResult,
    ProcessingStats,
    QualityReport,
)


class TestQualityReport:
    """Test QualityReport dataclass."""

    def test_is_acceptable_at_70_percent(self):
        """Should be acceptable at exactly 70%."""
        report = QualityReport(
            total_rows=100,
            valid_outputs=70,
            null_outputs=30,
            empty_outputs=0,
            success_rate=70.0,
            quality_score="good",
        )
        assert report.is_acceptable is True

    def test_not_acceptable_below_70_percent(self):
        """Should not be acceptable below 70%."""
        report = QualityReport(
            total_rows=100,
            valid_outputs=69,
            null_outputs=31,
            empty_outputs=0,
            success_rate=69.0,
            quality_score="poor",
        )
        assert report.is_acceptable is False

    def test_has_issues_when_issues_present(self):
        """Should detect issues."""
        report = QualityReport(
            total_rows=100,
            valid_outputs=50,
            null_outputs=50,
            empty_outputs=0,
            success_rate=50.0,
            quality_score="poor",
            issues=["Low success rate"],
        )
        assert report.has_issues is True

    def test_has_no_issues_when_clean(self):
        """Should have no issues when clean."""
        report = QualityReport(
            total_rows=100,
            valid_outputs=100,
            null_outputs=0,
            empty_outputs=0,
            success_rate=100.0,
            quality_score="excellent",
        )
        assert report.has_issues is False


class TestValidateOutputQuality:
    """Test ExecutionResult.validate_output_quality method."""

    def test_detects_null_outputs(self):
        """Should count null values."""
        df = pd.DataFrame({"output": [None, "valid", None, "valid", None]})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(5, 5, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 5),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.total_rows == 5
        assert quality.null_outputs == 3
        assert quality.valid_outputs == 2
        assert quality.success_rate == 40.0
        assert quality.quality_score == "critical"

    def test_detects_empty_strings(self):
        """Should count empty strings."""
        df = pd.DataFrame({"output": ["valid", "", "  ", "valid", ""]})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(5, 5, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 5),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.empty_outputs == 3  # '', '  ', ''
        assert quality.valid_outputs == 2

    def test_excellent_quality_score(self):
        """Should assign excellent for 95%+ success."""
        df = pd.DataFrame({"output": ["valid"] * 96 + [None] * 4})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 96.0
        assert quality.quality_score == "excellent"

    def test_good_quality_score(self):
        """Should assign good for 80-94% success."""
        df = pd.DataFrame({"output": ["valid"] * 85 + [None] * 15})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 85.0
        assert quality.quality_score == "good"

    def test_poor_quality_score(self):
        """Should assign poor for 50-79% success."""
        df = pd.DataFrame({"output": ["valid"] * 60 + [None] * 40})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 60.0
        assert quality.quality_score == "poor"

    def test_critical_quality_score(self):
        """Should assign critical for <50% success."""
        df = pd.DataFrame({"output": ["valid"] * 30 + [None] * 70})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 30.0
        assert quality.quality_score == "critical"

    def test_detects_metrics_mismatch(self):
        """Should detect when reported failures don't match nulls."""
        df = pd.DataFrame({"output": ["valid"] * 50 + [None] * 50})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(
                total_rows=100,
                processed_rows=100,
                failed_rows=0,  # Reports 0 failures!
                skipped_rows=0,
                rows_per_second=1.0,
                total_duration_seconds=10.0,
            ),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        # Should detect mismatch: 0 reported failures but 50 nulls
        assert quality.null_outputs == 50
        assert len(quality.issues) > 0
        assert any("MISMATCH" in issue for issue in quality.issues)

    def test_generates_warnings_for_high_null_rate(self):
        """Should warn when >30% nulls."""
        df = pd.DataFrame({"output": ["valid"] * 60 + [None] * 40})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.null_outputs == 40
        assert len(quality.issues) > 0
        assert any("NULL RATE" in issue for issue in quality.issues)

    def test_no_warnings_for_excellent_quality(self):
        """Should have no warnings/issues for 100% success."""
        df = pd.DataFrame({"output": ["valid"] * 100})

        result = ExecutionResult(
            data=df,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 100.0
        assert len(quality.warnings) == 0
        assert len(quality.issues) == 0
        assert quality.is_acceptable  # Use == for numpy bool compatibility


class TestAutoRetryLogic:
    """Test auto-retry detection logic (unit tests for the logic, not full pipeline)."""

    def test_identifies_null_rows(self):
        """Should identify rows with null outputs."""
        df = pd.DataFrame(
            {
                "output": ["valid", None, "valid", None, "valid"],
                "index_col": [0, 1, 2, 3, 4],
            }
        )

        # Simulate retry logic
        null_mask = df["output"].isna()
        failed_indices = df[null_mask].index.tolist()

        assert failed_indices == [1, 3]

    def test_identifies_empty_rows(self):
        """Should identify rows with empty outputs."""
        df = pd.DataFrame({"output": ["valid", "", "valid", "  ", "valid"]})

        # Simulate retry logic
        empty_mask = df["output"].astype(str).str.strip() == ""
        failed_indices = df[empty_mask].index.tolist()

        assert failed_indices == [1, 3]

    def test_identifies_both_null_and_empty(self):
        """Should identify both null and empty rows."""
        df = pd.DataFrame({"output": ["valid", None, "", "valid", "  ", None]})

        # Simulate retry logic
        null_mask = df["output"].isna()
        empty_mask = df["output"].astype(str).str.strip() == ""
        failed_mask = null_mask | empty_mask
        failed_indices = df[failed_mask].index.tolist()

        assert set(failed_indices) == {1, 2, 4, 5}
        assert len(failed_indices) == 4
