"""End-to-end tests: PipelineBuilder.with_knowledge_base() with new kwargs.

Verifies that the builder correctly stores new config options and that
they flow through to the knowledge stage.
"""

import pytest

from ondine.api.pipeline_builder import PipelineBuilder


class TestWithKnowledgeBaseBuilder:
    def _builder_with_kb(self, **kb_kwargs):
        """Helper: create a minimal builder with KB configured."""
        import pandas as pd

        from ondine.knowledge.store import KnowledgeStore

        kb = KnowledgeStore(":memory:")
        kb.ingest_text("Test document for pipeline integration")

        df = pd.DataFrame({"question": ["What is a test?"]})

        return (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["question"], output_columns=["answer"])
            .with_knowledge_base(kb, **kb_kwargs)
            .with_prompt("Context: {_kb_context}\n\nQ: {question}\nA:")
            .with_llm(model="openai/gpt-4o-mini")
        )

    def test_default_config_preserved(self):
        builder = self._builder_with_kb()
        config = builder._custom_metadata["knowledge_config"]
        assert config["top_k"] == 3
        assert config["rerank"] is False
        assert config["query_transform"] is None
        assert config["evaluate"] is False

    def test_query_transform_stored(self):
        builder = self._builder_with_kb(query_transform="hyde")
        config = builder._custom_metadata["knowledge_config"]
        assert config["query_transform"] == "hyde"

    def test_evaluate_stored(self):
        builder = self._builder_with_kb(
            evaluate=True, eval_model="anthropic/claude-3-haiku"
        )
        config = builder._custom_metadata["knowledge_config"]
        assert config["evaluate"] is True
        assert config["eval_model"] == "anthropic/claude-3-haiku"

    def test_reranker_model_upgraded_default(self):
        builder = self._builder_with_kb(rerank=True)
        config = builder._custom_metadata["knowledge_config"]
        assert "L-12" in config["reranker_model"]

    def test_all_kwargs_combined(self):
        builder = self._builder_with_kb(
            top_k=10,
            rerank=True,
            reranker_model="jina-reranker-v2",
            query_transform="multi-query",
            evaluate=True,
            eval_model="openai/gpt-4o",
        )
        config = builder._custom_metadata["knowledge_config"]
        assert config["top_k"] == 10
        assert config["rerank"] is True
        assert config["reranker_model"] == "jina-reranker-v2"
        assert config["query_transform"] == "multi-query"
        assert config["evaluate"] is True
        assert config["eval_model"] == "openai/gpt-4o"

    def test_store_required(self):
        with pytest.raises(ValueError, match="KnowledgeStore"):
            PipelineBuilder.create().with_knowledge_base(None)

    def test_build_succeeds(self):
        builder = self._builder_with_kb(query_transform="hyde", evaluate=True)
        pipeline = builder.build()
        assert pipeline is not None
