"""Tests for quality validation and auto-retry functionality."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from ondine.adapters.containers import ResultContainerImpl
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
        data = ResultContainerImpl([
            {"output": None},
            {"output": "valid"},
            {"output": None},
            {"output": "valid"},
            {"output": None},
        ])

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl([
            {"output": "valid"},
            {"output": ""},
            {"output": "  "},
            {"output": "valid"},
            {"output": ""},
        ])

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl(
            [{"output": "valid"}] * 96 + [{"output": None}] * 4
        )

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl(
            [{"output": "valid"}] * 85 + [{"output": None}] * 15
        )

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl(
            [{"output": "valid"}] * 60 + [{"output": None}] * 40
        )

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl(
            [{"output": "valid"}] * 30 + [{"output": None}] * 70
        )

        result = ExecutionResult(
            data=data,
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
        data = ResultContainerImpl(
            [{"output": "valid"}] * 50 + [{"output": None}] * 50
        )

        result = ExecutionResult(
            data=data,
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
        assert any("METRICS MISMATCH" in issue for issue in quality.issues)

    def test_generates_warnings_for_high_null_rate(self):
        """Should warn when null rate exceeds 30%."""
        data = ResultContainerImpl(
            [{"output": "valid"}] * 60 + [{"output": None}] * 40
        )

        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(100, 100, 40, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        # 40% nulls should trigger HIGH NULL RATE warning
        assert any("HIGH NULL RATE" in issue for issue in quality.issues)

    def test_no_warnings_for_excellent_quality(self):
        """Should have no warnings for excellent quality."""
        data = ResultContainerImpl([{"output": "valid"}] * 100)

        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(100, 100, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 100),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["output"])

        assert quality.success_rate == 100.0
        assert quality.quality_score == "excellent"
        assert len(quality.issues) == 0
        assert len(quality.warnings) == 0


class TestMultipleOutputColumns:
    """Test quality validation with multiple output columns."""

    def test_counts_nulls_across_columns(self):
        """Should count nulls across all output columns."""
        data = ResultContainerImpl([
            {"col1": "valid", "col2": None},
            {"col1": None, "col2": "valid"},
            {"col1": "valid", "col2": "valid"},
        ])

        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(3, 3, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 3),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["col1", "col2"])

        # 2 null cells out of 6 total cells (3 rows × 2 columns)
        assert quality.null_outputs == 2
        # All 3 rows have at least one valid column
        assert quality.valid_outputs == 3
        assert quality.success_rate == 100.0

    def test_row_invalid_when_all_columns_null(self):
        """Row is invalid only when ALL output columns are null."""
        data = ResultContainerImpl([
            {"col1": None, "col2": None},  # Invalid row
            {"col1": "valid", "col2": None},  # Valid (has col1)
            {"col1": None, "col2": "valid"},  # Valid (has col2)
        ])

        result = ExecutionResult(
            data=data,
            metrics=ProcessingStats(3, 3, 0, 0, 1.0, 10.0),
            costs=CostEstimate(Decimal("0.01"), 100, 50, 50, 3),
            execution_id=uuid4(),
            start_time=datetime.now(),
        )

        quality = result.validate_output_quality(["col1", "col2"])

        # Only 2 rows are valid (have at least one non-null column)
        assert quality.valid_outputs == 2
        # 4 null cells out of 6 total
        assert quality.null_outputs == 4
