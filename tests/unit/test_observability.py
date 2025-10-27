"""Tests for observability module (TDD approach - tests written first)."""

from unittest.mock import Mock, patch

import pytest


class TestTracingSetup:
    """Test tracing enable/disable functionality."""

    def test_tracing_disabled_by_default(self):
        """Tracing should be disabled by default (opt-in)."""
        from ondine.observability import is_tracing_enabled

        assert is_tracing_enabled() is False

    def test_enable_tracing_with_console_exporter(self):
        """Should enable tracing with console exporter."""
        from ondine.observability import (
            disable_tracing,
            enable_tracing,
            is_tracing_enabled,
        )

        try:
            enable_tracing(exporter="console")
            assert is_tracing_enabled() is True
        finally:
            disable_tracing()

    def test_enable_tracing_with_jaeger_exporter(self):
        """Should enable tracing with Jaeger exporter."""
        from ondine.observability import (
            disable_tracing,
            enable_tracing,
            is_tracing_enabled,
        )

        try:
            enable_tracing(
                exporter="jaeger", endpoint="http://localhost:14268/api/traces"
            )
            assert is_tracing_enabled() is True
        finally:
            disable_tracing()

    def test_disable_tracing_cleanup(self):
        """Should properly cleanup when disabling tracing."""
        from ondine.observability import (
            disable_tracing,
            enable_tracing,
            is_tracing_enabled,
        )

        enable_tracing(exporter="console")
        assert is_tracing_enabled() is True

        disable_tracing()
        assert is_tracing_enabled() is False


class TestPIISanitization:
    """Test PII sanitization functionality."""

    def test_pii_sanitization_enabled_by_default(self):
        """Prompts should be sanitized by default (not exposed in spans)."""
        from ondine.observability.sanitizer import sanitize_prompt

        prompt = "User email: john@example.com, process this"
        sanitized = sanitize_prompt(prompt, include_prompts=False)

        # Should NOT contain original prompt
        assert "john@example.com" not in sanitized
        # Should contain hash or placeholder
        assert "sanitized" in sanitized.lower()

    def test_pii_sanitization_opt_in(self):
        """Should include prompts when explicitly enabled."""
        from ondine.observability.sanitizer import sanitize_prompt

        prompt = "User email: john@example.com, process this"
        not_sanitized = sanitize_prompt(prompt, include_prompts=True)

        # Should contain original prompt when opted in
        assert not_sanitized == prompt

    def test_sanitize_response(self):
        """Response sanitization should work same as prompt sanitization."""
        from ondine.observability.sanitizer import sanitize_response

        response = "Sensitive data: SSN 123-45-6789"

        # Default: sanitized
        sanitized = sanitize_response(response, include_prompts=False)
        assert "123-45-6789" not in sanitized

        # Opt-in: not sanitized
        not_sanitized = sanitize_response(response, include_prompts=True)
        assert not_sanitized == response


class TestTracingObserver:
    """Test TracingObserver integration with ExecutionObserver pattern."""

    def test_tracing_observer_creation(self):
        """Should create TracingObserver instance."""
        from ondine.observability import TracingObserver

        observer = TracingObserver(include_prompts=False)
        assert observer is not None
        assert hasattr(observer, "on_pipeline_start")
        assert hasattr(observer, "on_stage_start")

    def test_tracing_observer_with_prompts_enabled(self):
        """Should support include_prompts flag."""
        from ondine.observability import TracingObserver

        observer = TracingObserver(include_prompts=True)
        assert observer._include_prompts is True

    @patch("ondine.observability.observer.is_tracing_enabled")
    @patch("ondine.observability.observer.get_tracer")
    def test_observer_creates_no_spans_when_disabled(
        self, mock_get_tracer, mock_is_enabled
    ):
        """Observer should not create spans when tracing is disabled."""
        from ondine.observability import TracingObserver

        mock_is_enabled.return_value = False

        observer = TracingObserver()
        # Mock pipeline and context
        mock_pipeline = Mock()
        mock_context = Mock()

        # Should not call get_tracer if tracing is disabled
        observer.on_pipeline_start(mock_pipeline, mock_context)

        mock_get_tracer.assert_not_called()

    @patch("ondine.observability.observer.is_tracing_enabled")
    @patch("ondine.observability.observer.get_tracer")
    def test_observer_creates_span_when_enabled(self, mock_get_tracer, mock_is_enabled):
        """Observer should create spans when tracing is enabled."""
        from ondine.observability import TracingObserver

        mock_is_enabled.return_value = True
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_get_tracer.return_value = mock_tracer

        observer = TracingObserver()
        mock_pipeline = Mock()
        mock_context = Mock()
        mock_context.total_rows = 100

        observer.on_pipeline_start(mock_pipeline, mock_context)

        # Should have called tracer to create span
        mock_get_tracer.assert_called_once()


class TestStageExecutionTracing:
    """Test stage execution creates appropriate spans."""

    @patch("ondine.observability.observer.is_tracing_enabled")
    @patch("ondine.observability.observer.get_tracer")
    def test_stage_execution_creates_span(self, mock_get_tracer, mock_is_enabled):
        """Each stage execution should create a span."""
        from ondine.observability import TracingObserver

        mock_is_enabled.return_value = True
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        observer = TracingObserver()
        mock_stage = Mock()
        mock_stage.__class__.__name__ = "TestStage"
        mock_context = Mock()
        mock_context.last_processed_row = 0  # Fix: mock the attribute properly

        observer.on_stage_start(mock_stage, mock_context)

        # Should create span with stage name
        mock_tracer.start_span.assert_called_once_with("stage.TestStage")

    @patch("ondine.observability.observer.is_tracing_enabled")
    @patch("ondine.observability.observer.get_tracer")
    def test_stage_error_records_exception(self, mock_get_tracer, mock_is_enabled):
        """Stage errors should be recorded in span."""
        from ondine.observability import TracingObserver

        mock_is_enabled.return_value = True
        mock_tracer = Mock()
        mock_span = Mock()
        mock_get_tracer.return_value = mock_tracer

        observer = TracingObserver()
        observer._spans = {"test_stage": mock_span}  # Simulate active span

        mock_stage = Mock()
        mock_stage.__class__.__name__ = "test_stage"
        mock_context = Mock()
        error = ValueError("Test error")

        observer.on_stage_error(mock_stage, mock_context, error)

        # Should record exception in span
        mock_span.record_exception.assert_called_once_with(error)


class TestExportFailureHandling:
    """Test graceful degradation when trace export fails."""

    @patch("ondine.observability.tracer.BatchSpanProcessor")
    def test_export_failure_does_not_break_pipeline(self, mock_processor):
        """Pipeline should continue even if trace export fails."""
        from ondine.observability import disable_tracing, enable_tracing

        # Simulate export failure
        mock_processor.side_effect = Exception("Export failed")

        try:
            # Should not raise exception
            enable_tracing(exporter="console")
        except Exception:
            pytest.fail("enable_tracing should not raise on export failure")
        finally:
            disable_tracing()


class TestLLMInvocationTracing:
    """Test LLM invocation creates spans with proper attributes."""

    def test_llm_invoke_span_attributes_placeholder(self):
        """
        Placeholder test for LLM invocation tracing.

        Will be implemented in Phase 3 after observer is working.
        This ensures we don't forget to add LLM instrumentation.
        """
        # This test will be implemented when we instrument LLMClient
        pytest.skip("LLM instrumentation not yet implemented (Phase 3)")


# Integration test placeholder
class TestTracingIntegration:
    """Integration tests for full pipeline tracing."""

    def test_full_pipeline_trace_placeholder(self):
        """
        Placeholder for full pipeline trace integration test.

        Will be implemented after basic components are working.
        """
        pytest.skip("Integration test will be in tests/integration/test_tracing_e2e.py")
