"""Tests for the enrich() front-door function.

These tests exercise the API surface, option forwarding, and orchestration
contract without making real network calls: `Pipeline.execute` is monkeypatched
to short-circuit the actual LLM invocation, while `QuickPipeline.create` and
`PipelineBuilder` run for real so the specifications they build (parser
injection, schema metadata, budget, etc.) are genuine, not simulated.
"""

from decimal import Decimal

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

from ondine import enrich
from ondine.api.enrich import enrich as enrich_direct
from ondine.api.pipeline import Pipeline
from ondine.api.quick import QuickPipeline
from ondine.core.models import CostEstimate, ExecutionResult, ProcessingStats
from ondine.stages.response_parser_stage import JSONParser


def _canned_result(data) -> ExecutionResult:
    """Build a real ExecutionResult wrapping *data*, as execute() would."""
    return ExecutionResult(
        data=data,
        metrics=ProcessingStats(
            total_rows=len(data), processed_rows=len(data), failed_rows=0
        ),
        costs=CostEstimate(
            total_cost=Decimal("0.01"),
            total_tokens=10,
            input_tokens=5,
            output_tokens=5,
            rows=len(data),
            confidence="actual",
        ),
        success=True,
    )


def _patch_execute(monkeypatch, result_df=None, capture=None):
    """Stub Pipeline.execute so no real LLM call happens.

    Records the pipeline instance (and call count) in *capture* so tests can
    assert on the specifications the real builder chain produced.
    """
    df = result_df if result_df is not None else pd.DataFrame({"output": ["ok"]})
    calls = {"count": 0}

    def fake_execute(self, resume_from=None):
        calls["count"] += 1
        if capture is not None:
            capture["pipeline"] = self
            capture["execute_calls"] = calls["count"]
        return _canned_result(df)

    monkeypatch.setattr(Pipeline, "execute", fake_execute)
    return calls


def _patch_create(monkeypatch, capture=None):
    """Wrap QuickPipeline.create to record the kwargs it was called with.

    The real implementation still runs underneath, so the returned Pipeline
    (and its specifications) are genuine.
    """
    original_create = QuickPipeline.create

    def wrapped_create(*args, **kwargs):
        if capture is not None:
            capture["kwargs"] = kwargs
        return original_create(*args, **kwargs)

    monkeypatch.setattr(QuickPipeline, "create", wrapped_create)


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

    def test_rejects_reserved_option_names(self):
        """Internal names (e.g. max_budget) can't be smuggled in via **options."""
        df = pd.DataFrame({"review": ["x"]})
        with pytest.raises(TypeError, match="unexpected option"):
            enrich(df, "Classify: {review}", max_budget=5.0)

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


class TestEnrichOrchestration:
    """enrich() forwards arguments correctly and executes exactly once."""

    def test_enrich_returns_result_dataframe(self, monkeypatch):
        """enrich() executes the pipeline and returns its data as a DataFrame."""
        out = pd.DataFrame({"review": ["a", "b"], "sentiment": ["pos", "neg"]})
        capture = {}
        _patch_execute(monkeypatch, result_df=out, capture=capture)

        result = enrich(
            pd.DataFrame({"review": ["a", "b"]}),
            prompt="Classify: {review}",
            output_columns=["sentiment"],
        )

        assert isinstance(result, pd.DataFrame)
        assert "sentiment" in result.columns
        assert capture["execute_calls"] == 1

    def test_enrich_forwards_core_args(self, monkeypatch):
        """data, prompt, output_columns, model reach QuickPipeline.create."""
        captured = {}
        _patch_execute(monkeypatch)
        _patch_create(monkeypatch, capture=captured)

        df = pd.DataFrame({"review": ["a"]})
        enrich(
            df,
            prompt="Classify: {review}",
            output_columns=["sentiment"],
            model="claude-3-5-sonnet",
        )

        assert captured["kwargs"]["data"] is df
        assert captured["kwargs"]["prompt"] == "Classify: {review}"
        assert captured["kwargs"]["output_columns"] == ["sentiment"]
        assert captured["kwargs"]["model"] == "claude-3-5-sonnet"

    def test_enrich_forwards_budget_as_max_budget(self, monkeypatch):
        """budget is forwarded as max_budget (QuickPipeline's parameter name)."""
        captured = {}
        _patch_execute(monkeypatch)
        _patch_create(monkeypatch, capture=captured)

        enrich(
            pd.DataFrame({"review": ["a"]}),
            prompt="Classify: {review}",
            budget=Decimal("5.0"),
        )

        assert captured["kwargs"]["max_budget"] == Decimal("5.0")

    def test_enrich_budget_accepts_float(self, monkeypatch):
        """budget accepts plain numeric types, not just Decimal."""
        captured = {}
        _patch_execute(monkeypatch)
        _patch_create(monkeypatch, capture=captured)

        enrich(pd.DataFrame({"review": ["a"]}), prompt="Classify: {review}", budget=5.0)

        assert captured["kwargs"]["max_budget"] == 5.0

    def test_enrich_forwards_allowed_options(self, monkeypatch):
        """Recognized **options are forwarded to QuickPipeline.create."""
        captured = {}
        _patch_execute(monkeypatch)
        _patch_create(monkeypatch, capture=captured)

        enrich(
            pd.DataFrame({"review": ["a"]}),
            prompt="Classify: {review}",
            temperature=0.7,
            max_tokens=100,
            batch_size=25,
            concurrency=10,
            provider="groq",
            model="llama-3.3-70b-versatile",
        )

        kw = captured["kwargs"]
        assert kw["temperature"] == 0.7
        assert kw["max_tokens"] == 100
        assert kw["batch_size"] == 25
        assert kw["concurrency"] == 10
        assert kw["provider"] == "groq"

    def test_enrich_forwards_file_path(self, monkeypatch, tmp_path):
        """A path-like data argument is forwarded as-is (QuickPipeline loads it)."""
        captured = {}
        _patch_execute(monkeypatch)
        _patch_create(monkeypatch, capture=captured)

        csv = tmp_path / "data.csv"
        pd.DataFrame({"review": ["a"]}).to_csv(csv, index=False)

        enrich(csv, prompt="Classify: {review}")

        assert captured["kwargs"]["data"] == csv

    def test_enrich_executes_exactly_once(self, monkeypatch):
        """enrich() runs the pipeline once per call (no double execution)."""
        capture = {}
        _patch_execute(monkeypatch, capture=capture)

        enrich(pd.DataFrame({"review": ["a"]}), prompt="Classify: {review}")

        assert capture["execute_calls"] == 1


class TestEnrichStructuredOutput:
    """schema= must go through PipelineBuilder.with_structured_output, which is
    the whole reason this implementation (over the metadata-poking alternative)
    was chosen: it also auto-injects a JSONParser when no parser is configured.
    """

    def test_schema_configures_structured_output_and_json_parser(self, monkeypatch):
        """schema= reaches the built pipeline's metadata AND a JSONParser is
        auto-injected, proving the builder path (not raw metadata assignment)
        was used.
        """

        class Sentiment(BaseModel):
            label: str
            score: float

        capture = {}
        _patch_execute(monkeypatch, capture=capture)

        enrich(
            pd.DataFrame({"review": ["a"]}),
            prompt="Classify: {review}",
            output_columns=["label", "score"],
            schema=Sentiment,
        )

        metadata = capture["pipeline"].specifications.metadata
        assert metadata["structured_output_model"] is Sentiment
        assert isinstance(metadata["custom_parser"], JSONParser)

    def test_no_schema_leaves_metadata_without_structured_output(self, monkeypatch):
        """Without schema, no structured-output metadata or parser is injected."""
        capture = {}
        _patch_execute(monkeypatch, capture=capture)

        enrich(pd.DataFrame({"review": ["a"]}), prompt="Classify: {review}")

        metadata = capture["pipeline"].specifications.metadata
        assert "structured_output_model" not in metadata
        assert "custom_parser" not in metadata


class TestEnrichInputTypePreservation:
    """enrich() preserves the caller's DataFrame flavor (ARCHITECTURE_PROPOSAL §1)."""

    def test_pandas_in_pandas_out(self, monkeypatch):
        """A pandas DataFrame in gets a pandas DataFrame back."""
        out = pd.DataFrame({"review": ["a"], "sentiment": ["pos"]})
        _patch_execute(monkeypatch, result_df=out)

        result = enrich(
            pd.DataFrame({"review": ["a"]}),
            prompt="Classify: {review}",
            output_columns=["sentiment"],
        )

        assert isinstance(result, pd.DataFrame)

    def test_polars_in_polars_out(self, monkeypatch):
        """A Polars DataFrame in gets a Polars DataFrame back, with no network
        call: Pipeline.execute is mocked at the LLM boundary.
        """
        out = pd.DataFrame({"review": ["a", "b"], "sentiment": ["pos", "neg"]})
        capture = {}
        _patch_execute(monkeypatch, result_df=out, capture=capture)

        pl_df = pl.DataFrame({"review": ["a", "b"]})
        result = enrich(
            pl_df,
            prompt="Classify: {review}",
            output_columns=["sentiment"],
        )

        assert isinstance(result, pl.DataFrame)
        assert result["sentiment"].to_list() == ["pos", "neg"]
        assert capture["execute_calls"] == 1

    def test_path_in_pandas_out(self, monkeypatch, tmp_path):
        """A file path in gets pandas back (never Polars)."""
        out = pd.DataFrame({"review": ["a"], "sentiment": ["pos"]})
        _patch_execute(monkeypatch, result_df=out)

        csv = tmp_path / "data.csv"
        pd.DataFrame({"review": ["a"]}).to_csv(csv, index=False)

        result = enrich(csv, prompt="Classify: {review}", output_columns=["sentiment"])

        assert isinstance(result, pd.DataFrame)
