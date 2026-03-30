"""Tests for context store builder methods on PipelineBuilder."""

import pytest

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import ContextStore
from ondine.context.rust_store import RustContextStore


class TestBuilderContextMethods:
    def _base_builder(self) -> PipelineBuilder:
        return (
            PipelineBuilder.create()
            .from_csv(
                "dummy.csv",
                input_columns=["text"],
                output_columns=["result"],
            )
            .with_prompt("Process: {text}")
        )

    def test_with_context_store_explicit(self):
        store = InMemoryContextStore()
        builder = self._base_builder().with_context_store(store)
        assert builder._custom_metadata["context_store"] is store

    def test_with_context_store_auto_detect(self):
        builder = self._base_builder().with_context_store()
        assert isinstance(builder._custom_metadata["context_store"], ContextStore)

    def test_with_grounding_sets_metadata(self):
        builder = self._base_builder().with_grounding(threshold=0.5, action="skip")
        assert builder._custom_metadata["grounding"]["threshold"] == 0.5
        assert builder._custom_metadata["grounding"]["action"] == "skip"
        assert "context_store" in builder._custom_metadata

    def test_with_grounding_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action must be one of"):
            self._base_builder().with_grounding(action="explode")

    def test_with_contradiction_detection_sets_metadata(self):
        builder = self._base_builder().with_contradiction_detection(
            key_columns=["product_id"],
            value_columns=["category"],
        )
        cfg = builder._custom_metadata["contradiction_detection"]
        assert cfg["key_columns"] == ["product_id"]
        assert cfg["value_columns"] == ["category"]
        assert "context_store" in builder._custom_metadata

    def test_with_confidence_scoring_sets_metadata(self):
        builder = self._base_builder().with_confidence_scoring(include_in_output=False)
        cfg = builder._custom_metadata["confidence_scoring"]
        assert cfg["include_in_output"] is False
        assert "context_store" in builder._custom_metadata

    def test_with_confidence_scoring_sigmoid_mode(self):
        builder = self._base_builder().with_confidence_scoring(scoring_mode="sigmoid")
        cfg = builder._custom_metadata["confidence_scoring"]
        assert cfg["scoring_mode"] == "sigmoid"

    def test_with_confidence_scoring_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="scoring_mode must be one of"):
            self._base_builder().with_confidence_scoring(scoring_mode="bad")

    def test_with_confidence_scoring_default_mode(self):
        builder = self._base_builder().with_confidence_scoring()
        cfg = builder._custom_metadata["confidence_scoring"]
        assert cfg["scoring_mode"] == "default"

    def test_with_contradiction_tolerance_sets_metadata(self):
        builder = self._base_builder().with_contradiction_detection(
            key_columns=["product_id"],
            value_columns=["score"],
            tolerance=1,
        )
        cfg = builder._custom_metadata["contradiction_detection"]
        assert cfg["tolerance"] == 1

    def test_with_contradiction_tolerance_default_is_none(self):
        builder = self._base_builder().with_contradiction_detection()
        cfg = builder._custom_metadata["contradiction_detection"]
        assert cfg["tolerance"] is None

    def test_with_grounding_embed_fn_sets_metadata(self):
        def mock_fn(texts: list) -> list:
            return [[0.0] * 10] * len(texts)

        builder = self._base_builder().with_grounding(embed_fn=mock_fn)
        assert builder._custom_metadata["grounding"]["embed_fn"] is mock_fn

    def test_with_grounding_retry_action_rejected(self):
        with pytest.raises(ValueError, match="action must be one of"):
            self._base_builder().with_grounding(action="retry")

    def test_chaining_all_context_methods(self):
        store = RustContextStore(":memory:")
        builder = (
            self._base_builder()
            .with_context_store(store)
            .with_grounding(threshold=0.4, action="flag")
            .with_contradiction_detection(key_columns=["id"])
            .with_confidence_scoring()
        )
        assert builder._custom_metadata["context_store"] is store
        assert builder._custom_metadata["grounding"]["threshold"] == 0.4
        assert builder._custom_metadata["contradiction_detection"]["key_columns"] == [
            "id"
        ]
        assert (
            builder._custom_metadata["confidence_scoring"]["include_in_output"] is True
        )
