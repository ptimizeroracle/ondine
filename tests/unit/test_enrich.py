"""Tests for ondine.enrich() front-door facade.

The facade orchestrates two collaborators: QuickPipeline.create (builds the
pipeline) and Pipeline.execute (runs it). These tests mock at that boundary
and assert on the orchestration contract — what the facade forwards, what it
injects, and what it returns — without dragging in the real pipeline machinery.
"""

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine.api.enrich import enrich
from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats


class _StubPipeline:
    """Minimal stand-in for a built Pipeline.

    Records the specifications object the facade mutates (for schema injection
    assertions) and returns a canned ExecutionResult from execute().
    """

    def __init__(self, result_df):
        self.specifications = type(
            "Specs", (), {"metadata": {}}
        )()  # simple namespace w/ metadata dict
        self._result_df = result_df
        self.execute_calls = 0

    def execute(self):
        self.execute_calls += 1
        return ExecutionResult(
            data=self._result_df,
            metrics=ProcessingStats(
                total_rows=len(self._result_df),
                processed_rows=len(self._result_df),
                failed_rows=0,
                skipped_rows=0,
            ),
            costs=CostEstimate(
                total_cost=Decimal("0.01"),
                total_tokens=10,
                input_tokens=5,
                output_tokens=5,
                rows=len(self._result_df),
                confidence="actual",
            ),
            success=True,
        )


def _patch_create(monkeypatch, result_df=None, capture=None):
    """Replace QuickPipeline.create with a stub-returning fake.

    Args:
        result_df: DataFrame the stub's execute() will surface.
        capture: dict that will be filled with the kwargs passed to create().
    """
    df = result_df if result_df is not None else pd.DataFrame({"output": ["ok"]})
    stub_holder = {}

    def fake_create(*args, **kwargs):
        if capture is not None:
            capture["args"] = args
            capture["kwargs"] = kwargs
        stub = _StubPipeline(df)
        stub_holder["pipeline"] = stub
        return stub

    monkeypatch.setattr("ondine.api.enrich.QuickPipeline.create", fake_create)
    return stub_holder


class TestEnrich:
    """Behavioral tests for the enrich() facade."""

    def test_enrich_returns_result_dataframe(self, monkeypatch):
        """enrich() executes the pipeline and returns its data as a DataFrame."""
        out = pd.DataFrame({"text": ["a", "b"], "category": ["x", "y"]})
        stub_holder = _patch_create(monkeypatch, result_df=out)

        result = enrich(
            pd.DataFrame({"text": ["a", "b"]}),
            prompt="Categorize: {text}",
        )

        assert isinstance(result, pd.DataFrame)
        assert "category" in result.columns
        assert stub_holder["pipeline"].execute_calls == 1

    def test_enrich_forwards_core_positional_args(self, monkeypatch):
        """data, prompt, output_columns reach QuickPipeline.create unchanged."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)

        df = pd.DataFrame({"text": ["a"]})
        enrich(df, prompt="Process: {text}", output_columns=["label"])

        assert captured["kwargs"]["data"] is df
        assert captured["kwargs"]["prompt"] == "Process: {text}"
        assert captured["kwargs"]["output_columns"] == ["label"]

    def test_enrich_forwards_model(self, monkeypatch):
        """model keyword reaches QuickPipeline.create."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)

        enrich(pd.DataFrame({"text": ["a"]}), prompt="P: {text}", model="claude-3-sonnet")

        assert captured["kwargs"]["model"] == "claude-3-sonnet"

    def test_enrich_forwards_budget_to_create(self, monkeypatch):
        """budget is forwarded as max_budget (QuickPipeline's param name)."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)

        enrich(
            pd.DataFrame({"text": ["a"]}),
            prompt="P: {text}",
            budget=Decimal("5.0"),
        )

        assert captured["kwargs"]["max_budget"] == Decimal("5.0")

    def test_enrich_budget_accepts_float_and_int(self, monkeypatch):
        """budget accepts plain numeric types, not just Decimal."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)
        enrich(pd.DataFrame({"text": ["a"]}), prompt="P: {text}", budget=5.0)
        assert captured["kwargs"]["max_budget"] == 5.0

    def test_enrich_injects_schema_into_metadata(self, monkeypatch):
        """schema is injected into the built pipeline's metadata for structured output."""

        class MySchema(BaseModel):
            label: str
            score: float

        stub_holder = _patch_create(monkeypatch)

        enrich(
            pd.DataFrame({"text": ["a"]}),
            prompt="P: {text}",
            output_columns=["label", "score"],
            schema=MySchema,
        )

        metadata = stub_holder["pipeline"].specifications.metadata
        assert metadata["structured_output_model"] is MySchema

    def test_enrich_no_schema_leaves_metadata_clean(self, monkeypatch):
        """Without schema, metadata is not polluted with structured-output keys."""
        stub_holder = _patch_create(monkeypatch)

        enrich(pd.DataFrame({"text": ["a"]}), prompt="P: {text}")

        metadata = stub_holder["pipeline"].specifications.metadata
        assert "structured_output_model" not in metadata

    def test_enrich_forwards_allowed_options(self, monkeypatch):
        """Recognized **options are forwarded to QuickPipeline.create."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)

        enrich(
            pd.DataFrame({"text": ["a"]}),
            prompt="P: {text}",
            temperature=0.7,
            max_tokens=100,
            batch_size=25,
            concurrency=10,
            provider="groq",
        )

        kw = captured["kwargs"]
        assert kw["temperature"] == 0.7
        assert kw["max_tokens"] == 100
        assert kw["batch_size"] == 25
        assert kw["concurrency"] == 10
        assert kw["provider"] == "groq"

    def test_enrich_rejects_unknown_option(self, monkeypatch):
        """Unknown **options raise TypeError (explicit allowlist, no getattr magic)."""
        _patch_create(monkeypatch)

        with pytest.raises(TypeError, match="unexpected"):
            enrich(
                pd.DataFrame({"text": ["a"]}),
                prompt="P: {text}",
                bogus_param=123,
            )

    def test_enrich_rejects_reserved_option_names(self, monkeypatch):
        """First-class params (model/schema/budget) can't be re-passed via **options."""
        _patch_create(monkeypatch)

        with pytest.raises(TypeError, match="unexpected"):
            enrich(
                pd.DataFrame({"text": ["a"]}),
                prompt="P: {text}",
                max_budget=5.0,  # internal name — must be rejected
            )

    def test_enrich_executes_exactly_once(self, monkeypatch):
        """enrich() runs the pipeline once per call (no double execution)."""
        stub_holder = _patch_create(monkeypatch)

        enrich(pd.DataFrame({"text": ["a"]}), prompt="P: {text}")

        assert stub_holder["pipeline"].execute_calls == 1


class TestEnrichAcceptsFilePaths:
    """enrich() passes data through to QuickPipeline, which handles file paths."""

    def test_enrich_forwards_file_path(self, monkeypatch, tmp_path):
        """A path-like data argument is forwarded as-is (QuickPipeline loads it)."""
        captured = {}
        _patch_create(monkeypatch, capture=captured)

        csv = tmp_path / "data.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(csv, index=False)

        enrich(csv, prompt="P: {text}")

        assert captured["kwargs"]["data"] == csv
