"""Tests for Embedder implementations and factory.

Uses mocking to avoid loading real models or calling real APIs.
"""

from unittest.mock import MagicMock, patch

import pytest

from ondine.knowledge.embedders import (
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    resolve_embedder,
)
from ondine.knowledge.protocols import Embedder


class TestSentenceTransformerEmbedder:
    def test_satisfies_protocol(self):
        assert isinstance(SentenceTransformerEmbedder(), Embedder)

    @patch("ondine.knowledge.embedders.SentenceTransformerEmbedder._load")
    def test_embed_delegates_to_model(self, mock_load):
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_load.return_value = fake_model

        embedder = SentenceTransformerEmbedder("test-model")
        result = embedder.embed(["hello", "world"])

        assert len(result) == 2
        assert isinstance(result[0], list)
        assert result[0] == pytest.approx([0.1, 0.2])
        fake_model.encode.assert_called_once()

    def test_default_model_is_bge(self):
        e = SentenceTransformerEmbedder()
        assert "bge-base-en" in e._model_name

    def test_repr(self):
        e = SentenceTransformerEmbedder("my-model")
        assert "my-model" in repr(e)


class TestOpenAIEmbedder:
    def test_satisfies_protocol(self):
        assert isinstance(OpenAIEmbedder(), Embedder)

    @patch("litellm.embedding")
    def test_embed_calls_litellm(self, mock_embedding):
        mock_embedding.return_value = MagicMock(
            data=[
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        )

        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        result = embedder.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        mock_embedding.assert_called_once()

    @patch("litellm.embedding")
    def test_passes_api_key(self, mock_embedding):
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1]}])

        embedder = OpenAIEmbedder(api_key="sk-test")  # pragma: allowlist secret
        embedder.embed(["x"])

        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"  # pragma: allowlist secret

    def test_repr(self):
        e = OpenAIEmbedder("my-embedding-model")
        assert "my-embedding-model" in repr(e)


class TestResolveEmbedder:
    def test_none_returns_sentence_transformer_or_none(self):
        result = resolve_embedder(None)
        # Either a SentenceTransformerEmbedder or None depending on env
        assert result is None or isinstance(result, SentenceTransformerEmbedder)

    def test_string_model_name_returns_sentence_transformer(self):
        result = resolve_embedder("BAAI/bge-base-en-v1.5")
        assert isinstance(result, SentenceTransformerEmbedder)
        assert result._model_name == "BAAI/bge-base-en-v1.5"

    def test_openai_string_returns_openai_embedder(self):
        result = resolve_embedder("text-embedding-3-small")
        assert isinstance(result, OpenAIEmbedder)

    def test_openai_prefix_returns_openai_embedder(self):
        result = resolve_embedder("openai/text-embedding-3-small")
        assert isinstance(result, OpenAIEmbedder)

    def test_passthrough_existing_embedder(self):
        class _Custom:
            def embed(self, texts):
                return [[0.0] for _ in texts]

        custom = _Custom()
        assert resolve_embedder(custom) is custom
