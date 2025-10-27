"""End-to-end integration tests for tracing functionality."""

from unittest.mock import patch

import pandas as pd
import pytest


class TestFullPipelineTracing:
    """Test complete pipeline execution with tracing enabled."""

    @pytest.mark.skip(reason="Will implement after basic components work")
    def test_full_pipeline_creates_trace_hierarchy(self):
        """
        Full pipeline execution should create proper span hierarchy.

        Expected hierarchy:
        pipeline.execute (root)
        ├── stage.DataLoader
        ├── stage.PromptFormatter
        ├── stage.LLMInvocation
        │   └── llm.invoke (nested)
        ├── stage.ResponseParser
        └── stage.ResultWriter
        """
        from ondine import PipelineBuilder
        from ondine.observability import (
            disable_tracing,
            enable_tracing,
        )

        try:
            # Enable tracing with console exporter
            enable_tracing(exporter="console")

            # Create simple pipeline
            data = pd.DataFrame({"text": ["test1", "test2"]})

            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(data, input_columns=["text"], output_columns=["result"])
                .with_prompt("Process: {text}")
                .with_llm(provider="openai", model="gpt-4o-mini")
                .build()
            )

            # TODO: Add mechanism to attach observer to pipeline
            # observer = TracingObserver(include_prompts=False)
            # pipeline.add_observer(observer)

            # Execute pipeline
            with patch("ondine.adapters.llm_client.OpenAIClient.invoke") as mock_invoke:
                # Mock LLM response
                from decimal import Decimal

                from ondine.core.models import LLMResponse

                mock_invoke.return_value = LLMResponse(
                    text="result",
                    tokens_in=10,
                    tokens_out=5,
                    model="gpt-4o-mini",
                    cost=Decimal("0.001"),
                    latency_ms=100.0,
                )

                result = pipeline.execute()

            # Verify execution succeeded
            assert result is not None
            assert len(result.data) == 2

            # TODO: Verify span hierarchy was created
            # TODO: Verify span attributes (model, tokens, cost)

        finally:
            disable_tracing()

    @pytest.mark.skip(reason="Will implement after basic components work")
    def test_trace_includes_cost_and_token_metrics(self):
        """Trace should include LLM cost and token usage metrics."""
        # TODO: Implement after LLM instrumentation is done
        pass

    @pytest.mark.skip(reason="Will implement after basic components work")
    def test_trace_sanitizes_pii_by_default(self):
        """Trace should not include prompts/responses by default (PII safety)."""
        # TODO: Implement PII verification
        pass
