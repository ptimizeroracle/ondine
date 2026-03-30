"""Tests for LLMJudge and eval factory.

Uses mocking to avoid real LLM calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from ondine.knowledge.eval import LLMJudge, resolve_scorer
from ondine.knowledge.protocols import EvalResult, RetrievalScorer


class TestLLMJudge:
    def test_satisfies_protocol(self):
        assert isinstance(LLMJudge(), RetrievalScorer)

    @patch("litellm.completion")
    def test_score_returns_eval_result(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"faithfulness": 0.9, "relevancy": 0.85, "context_precision": 0.7}'
                    )
                )
            ]
        )

        judge = LLMJudge()
        result = judge.score(
            query="What is RAG?",
            answer="RAG combines retrieval with generation.",
            contexts=["RAG is a technique...", "Retrieval augmented..."],
        )

        assert isinstance(result, EvalResult)
        assert result.faithfulness == pytest.approx(0.9)
        assert result.relevancy == pytest.approx(0.85)
        assert result.context_precision == pytest.approx(0.7)
        assert result.metadata["model"] == "openai/gpt-4o-mini"

    @patch("litellm.completion")
    def test_score_with_json_code_block(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='```json\n{"faithfulness": 1.0, "relevancy": 1.0, "context_precision": 1.0}\n```'
                    )
                )
            ]
        )

        judge = LLMJudge()
        result = judge.score("q", "a", ["c"])
        assert result.faithfulness == pytest.approx(1.0)

    @patch("litellm.completion", side_effect=Exception("API error"))
    def test_graceful_fallback(self, mock_completion):
        judge = LLMJudge()
        result = judge.score("q", "a", ["c"])

        assert isinstance(result, EvalResult)
        assert result.faithfulness == 0.0
        assert "error" in result.metadata

    @patch("litellm.completion")
    def test_passes_api_key(self, mock_completion):
        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"faithfulness": 0.5, "relevancy": 0.5, "context_precision": 0.5}'
                    )
                )
            ]
        )

        judge = LLMJudge(api_key="sk-test")  # pragma: allowlist secret
        judge.score("q", "a", ["c"])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"  # pragma: allowlist secret

    def test_repr(self):
        assert "gpt-4o-mini" in repr(LLMJudge())


class TestResolveScorer:
    def test_none_returns_none(self):
        assert resolve_scorer(None) is None

    def test_false_returns_none(self):
        assert resolve_scorer(False) is None

    def test_true_returns_llm_judge(self):
        r = resolve_scorer(True)
        assert isinstance(r, LLMJudge)

    def test_string_model_returns_judge(self):
        r = resolve_scorer("anthropic/claude-3-haiku")
        assert isinstance(r, LLMJudge)
        assert r._model == "anthropic/claude-3-haiku"

    def test_passthrough_existing_scorer(self):
        class _Custom:
            def score(self, query, answer, contexts):
                return EvalResult()

        custom = _Custom()
        assert resolve_scorer(custom) is custom
