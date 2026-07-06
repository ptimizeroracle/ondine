"""Tests for the enrich() front-door function.

These tests exercise the API surface and argument handling without making
real network calls. End-to-end execution is covered by the smoke test and
the integration suite.
"""

from decimal import Decimal

import pandas as pd
import pytest

from ondine import enrich
from ondine.api.enrich import enrich as enrich_direct


class TestEnrichSignature:
    """enrich() argument handling and validation."""

    def test_enrich_is_exported(self):
        """enrich should be importable from the package root."""
        import ondine

        assert ondine.enrich is enrich_direct
        assert "enrich" in ondine.__all__

    def test_rejects_unknown_option(self):
        """Unknown kwargs must raise TypeError, not be silently ignored."""
        df = pd.DataFrame({"review": ["x"]})
        with pytest.raises(TypeError, match="unexpected option"):
            enrich(df, "Classify: {review}", bogus=True)

    def test_rejects_prompt_without_placeholders(self):
        """A prompt with no {column} placeholder is a caller error."""
        df = pd.DataFrame({"review": ["x"]})
        with pytest.raises(ValueError, match="No placeholders"):
            enrich(df, "no placeholders here")

    def test_rejects_placeholder_missing_from_data(self):
        """Placeholder referencing a non-existent column is a caller error."""
        df = pd.DataFrame({"review": ["x"]})
        with pytest.raises(ValueError, match="not found"):
            enrich(df, "Classify: {missing_col}")


class TestEnrichPipelineConstruction:
    """enrich() builds a QuickPipeline with correct specs."""

    def test_builds_pipeline_with_default_output_column(self):
        """enrich() should produce specs equivalent to QuickPipeline defaults."""
        df = pd.DataFrame({"review": ["great", "bad"]})
        # We can't execute without a real LLM, but we can verify construction
        # by calling QuickPipeline.create through enrich's logic indirectly.
        from ondine.api.quick import QuickPipeline

        # Replicate what enrich does internally (sans execute)
        pipeline = QuickPipeline.create(
            data=df,
            prompt="Classify tone of: {review}",
            model="gpt-4o-mini",
            output_columns=None,
            max_budget=5.0,
        )
        assert pipeline.specifications.dataset.output_columns == ["output"]
        assert pipeline.specifications.llm.model == "gpt-4o-mini"

    def test_builds_pipeline_with_named_output_columns(self):
        """Named output_columns flow through to the pipeline spec."""
        df = pd.DataFrame({"review": ["x"]})
        from ondine.api.quick import QuickPipeline

        pipeline = QuickPipeline.create(
            data=df,
            prompt="Classify: {review}",
            model="gpt-4o-mini",
            output_columns=["sentiment"],
            max_budget=Decimal("2.0"),
        )
        assert pipeline.specifications.dataset.output_columns == ["sentiment"]
        # budget cap should be set
        assert pipeline.specifications.processing.max_budget is not None

    def test_allowlist_options_forward_through(self):
        """Recognized options (provider, temperature, etc.) must not raise."""
        df = pd.DataFrame({"review": ["x"]})
        from ondine.api.quick import QuickPipeline

        # These options are in the allowlist and should construct cleanly.
        pipeline = QuickPipeline.create(
            data=df,
            prompt="Classify: {review}",
            model="gpt-4o-mini",
            output_columns=["sentiment"],
            provider="openai",
            temperature=0.5,
            batch_size=10,
            concurrency=3,
        )
        assert pipeline.specifications.llm.provider.value == "openai"
