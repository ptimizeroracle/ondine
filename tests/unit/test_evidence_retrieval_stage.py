"""Unit tests for EvidenceRetrievalStage."""

from __future__ import annotations

from ondine.adapters.containers import DictListContainer
from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import EvidenceRecord
from ondine.stages.evidence_retrieval_stage import EvidenceRetrievalStage


def _make_container(rows: list[dict], columns: list[str]) -> DictListContainer:
    return DictListContainer(data=rows, columns=columns)


def _store_with_evidence(texts: list[str]) -> InMemoryContextStore:
    store = InMemoryContextStore()
    for text in texts:
        store.store(
            EvidenceRecord(
                text=text,
                source_ref="test_source",
                claim_type="factual",
                source_type="llm_response",
                asserted_by="test",
            )
        )
    return store


class TestEvidenceRetrievalStage:
    def test_adds_both_output_columns(self):
        store = _store_with_evidence(["Corn Flakes is a cereal product."])
        stage = EvidenceRetrievalStage(store, query_columns=["product"], min_score=0.0)
        container = _make_container([{"product": "Corn Flakes"}], columns=["product"])
        result = stage.process(container, context=None)
        assert "_evidence_context" in result.columns
        assert "_evidence_count" in result.columns

    def test_empty_store_produces_zero_count(self):
        store = InMemoryContextStore()
        stage = EvidenceRetrievalStage(store, query_columns=["product"])
        container = _make_container([{"product": "Corn Flakes"}], columns=["product"])
        result = stage.process(container, context=None)
        row = list(result)[0]
        assert row["_evidence_context"] == ""
        assert row["_evidence_count"] == 0

    def test_matching_evidence_includes_score_and_source(self):
        store = _store_with_evidence(["Corn Flakes is a cereal."])
        stage = EvidenceRetrievalStage(
            store, query_columns=["product"], top_k=1, min_score=0.0
        )
        container = _make_container(
            [{"product": "Corn Flakes cereal"}], columns=["product"]
        )
        result = stage.process(container, context=None)
        row = list(result)[0]
        # Should include the evidence text
        assert "Corn Flakes" in row["_evidence_context"]
        # Should include score attribution
        assert "[score=" in row["_evidence_context"]
        # Should include source ref
        assert "test_source" in row["_evidence_context"]
        assert row["_evidence_count"] == 1

    def test_min_score_filters_low_relevance_results(self):
        """Results below min_score should be discarded — this is the noise filter."""
        store = _store_with_evidence(["completely unrelated astrophysics content"])
        stage = EvidenceRetrievalStage(store, query_columns=["product"], min_score=0.5)
        container = _make_container(
            [{"product": "Corn Flakes cereal"}], columns=["product"]
        )
        result = stage.process(container, context=None)
        row = list(result)[0]
        # TF-IDF similarity between "Corn Flakes cereal" and "astrophysics content"
        # should be near zero, below 0.5 threshold
        assert row["_evidence_context"] == ""
        assert row["_evidence_count"] == 0

    def test_top_k_limits_retrieved_evidence(self):
        store = _store_with_evidence(
            [
                "Corn Flakes is a cereal.",
                "Corn Flakes are breakfast foods.",
                "Corn Flakes are made of corn.",
            ]
        )
        stage = EvidenceRetrievalStage(
            store, query_columns=["product"], top_k=2, min_score=0.0
        )
        container = _make_container([{"product": "Corn Flakes"}], columns=["product"])
        result = stage.process(container, context=None)
        row = list(result)[0]
        assert row["_evidence_count"] <= 2

    def test_empty_query_produces_zero_count(self):
        store = _store_with_evidence(["Corn Flakes is a cereal."])
        stage = EvidenceRetrievalStage(store, query_columns=["product"])
        container = _make_container([{"product": ""}], columns=["product"])
        result = stage.process(container, context=None)
        row = list(result)[0]
        assert row["_evidence_context"] == ""
        assert row["_evidence_count"] == 0

    def test_multi_column_query_concatenates(self):
        store = _store_with_evidence(["frozen chocolate is a dessert."])
        stage = EvidenceRetrievalStage(
            store, query_columns=["name", "description"], min_score=0.0
        )
        container = _make_container(
            [{"name": "frozen chocolate", "description": "dessert"}],
            columns=["name", "description"],
        )
        result = stage.process(container, context=None)
        row = list(result)[0]
        assert "_evidence_context" in row
        assert "_evidence_count" in row

    def test_preserves_existing_columns(self):
        store = InMemoryContextStore()
        stage = EvidenceRetrievalStage(store, query_columns=["product"])
        container = _make_container(
            [{"product": "x", "category": "food"}],
            columns=["product", "category"],
        )
        result = stage.process(container, context=None)
        assert "product" in result.columns
        assert "category" in result.columns

    def test_validate_input_flags_missing_columns(self):
        store = InMemoryContextStore()
        stage = EvidenceRetrievalStage(store, query_columns=["nonexistent"])
        container = _make_container([{"product": "x"}], columns=["product"])
        validation = stage.validate_input(container)
        assert not validation.is_valid

    def test_validate_input_passes_for_valid_columns(self):
        store = InMemoryContextStore()
        stage = EvidenceRetrievalStage(store, query_columns=["product"])
        container = _make_container([{"product": "x"}], columns=["product"])
        validation = stage.validate_input(container)
        assert validation.is_valid


class TestEvidencePrimingBuilder:
    """Tests that with_evidence_priming() correctly wires metadata."""

    def test_sets_metadata_with_all_params(self):
        from ondine.api.pipeline_builder import PipelineBuilder

        builder = (
            PipelineBuilder.create()
            .from_csv("dummy.csv", input_columns=["product"], output_columns=["cat"])
            .with_prompt("Classify: {product}")
            .with_evidence_priming(query_columns=["product"], top_k=5, min_score=0.3)
        )
        cfg = builder._custom_metadata["evidence_priming"]
        assert cfg["query_columns"] == ["product"]
        assert cfg["top_k"] == 5
        assert cfg["min_score"] == 0.3
        assert "context_store" in builder._custom_metadata

    def test_auto_creates_context_store(self):
        from ondine.api.pipeline_builder import PipelineBuilder
        from ondine.context.protocol import ContextStore

        builder = (
            PipelineBuilder.create()
            .from_csv("dummy.csv", input_columns=["product"], output_columns=["cat"])
            .with_prompt("Classify: {product}")
            .with_evidence_priming()
        )
        assert isinstance(builder._custom_metadata["context_store"], ContextStore)

    def test_default_params(self):
        from ondine.api.pipeline_builder import PipelineBuilder

        builder = (
            PipelineBuilder.create()
            .from_csv("dummy.csv", input_columns=["product"], output_columns=["cat"])
            .with_prompt("Classify: {product}")
            .with_evidence_priming()
        )
        cfg = builder._custom_metadata["evidence_priming"]
        assert cfg["top_k"] == 3
        assert cfg["min_score"] == 0.1
        assert cfg["query_columns"] is None  # falls back to input_columns at runtime

    def test_respects_explicit_store(self):
        from ondine.api.pipeline_builder import PipelineBuilder

        store = InMemoryContextStore()
        builder = (
            PipelineBuilder.create()
            .from_csv("dummy.csv", input_columns=["product"], output_columns=["cat"])
            .with_prompt("Classify: {product}")
            .with_context_store(store)
            .with_evidence_priming(query_columns=["product"])
        )
        assert builder._custom_metadata["context_store"] is store
