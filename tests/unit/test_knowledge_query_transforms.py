"""Tests for QueryTransformer implementations and factory.

Uses mocking to avoid real LLM calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from ondine.knowledge.protocols import QueryTransformer
from ondine.knowledge.query import (
    HyDETransformer,
    MultiQueryTransformer,
    StepBackTransformer,
    resolve_query_transform,
)


class TestMultiQueryTransformer:
    def test_satisfies_protocol(self):
        assert isinstance(MultiQueryTransformer(), QueryTransformer)

    @patch("litellm.completion")
    def test_returns_original_plus_variants(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='["What is RAG?", "How does retrieval work?", "Define RAG"]'
                    )
                )
            ]
        )

        t = MultiQueryTransformer(n=3)
        queries = t.transform("What is RAG architecture?")

        assert queries[0] == "What is RAG architecture?"
        assert len(queries) == 4  # original + 3 variants
        mock_completion.assert_called_once()

    @patch("litellm.completion", side_effect=Exception("API error"))
    def test_fallback_on_error(self, mock_completion):
        t = MultiQueryTransformer()
        queries = t.transform("test query")
        assert queries == ["test query"]

    def test_repr(self):
        assert "multi" in repr(MultiQueryTransformer()).lower() or "Multi" in repr(
            MultiQueryTransformer()
        )


class TestHyDETransformer:
    def test_satisfies_protocol(self):
        assert isinstance(HyDETransformer(), QueryTransformer)

    @patch("litellm.completion")
    def test_returns_original_plus_hypothesis(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="RAG combines retrieval with generation to reduce hallucination."
                    )
                )
            ]
        )

        t = HyDETransformer()
        queries = t.transform("What is RAG?")

        assert len(queries) == 2
        assert queries[0] == "What is RAG?"
        assert "RAG" in queries[1]

    @patch("litellm.completion", side_effect=Exception("API error"))
    def test_fallback_on_error(self, mock_completion):
        t = HyDETransformer()
        queries = t.transform("test")
        assert queries == ["test"]


class TestStepBackTransformer:
    def test_satisfies_protocol(self):
        assert isinstance(StepBackTransformer(), QueryTransformer)

    @patch("litellm.completion")
    def test_returns_original_plus_stepback(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="What are the main approaches to reducing LLM hallucination?"
                    )
                )
            ]
        )

        t = StepBackTransformer()
        queries = t.transform("How does RAG reduce hallucination in GPT-4?")

        assert len(queries) == 2
        assert queries[0] == "How does RAG reduce hallucination in GPT-4?"
        assert "hallucination" in queries[1].lower()

    @patch("litellm.completion", side_effect=Exception("API error"))
    def test_fallback_on_error(self, mock_completion):
        t = StepBackTransformer()
        queries = t.transform("test")
        assert queries == ["test"]


class TestResolveQueryTransform:
    def test_none_returns_none(self):
        assert resolve_query_transform(None) is None

    def test_multi_query_string(self):
        r = resolve_query_transform("multi-query")
        assert isinstance(r, MultiQueryTransformer)

    def test_hyde_string(self):
        r = resolve_query_transform("hyde")
        assert isinstance(r, HyDETransformer)

    def test_step_back_string(self):
        r = resolve_query_transform("step-back")
        assert isinstance(r, StepBackTransformer)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            resolve_query_transform("nonexistent")

    def test_passthrough_existing_transformer(self):
        class _Custom:
            def transform(self, query):
                return [query]

        custom = _Custom()
        assert resolve_query_transform(custom) is custom
