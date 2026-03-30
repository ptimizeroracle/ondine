"""Tests for KnowledgeRetrievalStage — verifies row augmentation.

Regressions caught:
- Stage adds _kb_context column to each row (core contract)
- Empty KB produces empty context strings (graceful degradation)
- Query is built from the configured columns (column dispatch)
- Validation catches missing columns (early error detection)
"""

import pytest

from ondine.adapters.containers import DictListContainer
from ondine.knowledge.store import KnowledgeStore
from ondine.stages.knowledge_retrieval_stage import KnowledgeRetrievalStage


def _make_container(rows: list[dict], columns: list[str]) -> DictListContainer:
    return DictListContainer(data=rows, columns=columns)


class TestKnowledgeRetrievalStage:
    @pytest.fixture
    def populated_store(self):
        store = KnowledgeStore(":memory:")
        store.ingest_text("Organic cereals contain whole grains and fiber")
        store.ingest_text("Frozen vegetables are flash-frozen at harvest")
        return store

    def test_augments_rows_with_kb_context(self, populated_store):
        stage = KnowledgeRetrievalStage(
            store=populated_store,
            query_columns=["question"],
            top_k=2,
        )

        container = _make_container(
            [{"question": "What are organic cereals?"}],
            ["question"],
        )

        result = stage.process(container, context=None)
        row = list(result)[0]
        assert "_kb_context" in row
        assert len(row["_kb_context"]) > 0

    def test_empty_store_produces_empty_context(self):
        store = KnowledgeStore(":memory:")
        stage = KnowledgeRetrievalStage(store=store, query_columns=["q"], top_k=3)

        container = _make_container([{"q": "anything"}], ["q"])
        result = stage.process(container, context=None)
        row = list(result)[0]
        assert row["_kb_context"] == ""

    def test_multiple_query_columns_concatenated(self, populated_store):
        stage = KnowledgeRetrievalStage(
            store=populated_store,
            query_columns=["col_a", "col_b"],
            top_k=1,
        )

        container = _make_container(
            [{"col_a": "organic", "col_b": "cereals"}],
            ["col_a", "col_b"],
        )

        result = stage.process(container, context=None)
        row = list(result)[0]
        assert "_kb_context" in row
        assert len(row["_kb_context"]) > 0

    def test_validation_catches_missing_columns(self, populated_store):
        stage = KnowledgeRetrievalStage(
            store=populated_store,
            query_columns=["nonexistent"],
            top_k=1,
        )

        container = _make_container([{"question": "test"}], ["question"])
        validation = stage.validate_input(container)
        assert not validation.is_valid

    def test_kb_context_added_to_columns(self, populated_store):
        stage = KnowledgeRetrievalStage(
            store=populated_store,
            query_columns=["q"],
            top_k=1,
        )

        container = _make_container([{"q": "organic"}], ["q"])
        result = stage.process(container, context=None)
        assert "_kb_context" in result.columns
