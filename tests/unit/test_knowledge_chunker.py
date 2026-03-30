"""Tests for SemanticChunker — verifies chunk splitting behavior.

Each test catches a specific regression:
- Empty text produces no chunks (boundary condition)
- Single sentence produces exactly one chunk (no spurious splits)
- Text exceeding max_chunk_tokens is split (overflow protection)
- Chunk IDs are unique (identity correctness)
- Source/metadata propagation to chunks (provenance integrity)
"""

import pytest

from ondine.knowledge.chunker import Chunk, SemanticChunker, _sentence_split


class TestSentenceSplit:
    def test_splits_on_period_boundary(self):
        result = _sentence_split("Hello world. How are you? Fine!")
        assert result == ["Hello world.", "How are you?", "Fine!"]

    def test_empty_string_produces_empty_list(self):
        assert _sentence_split("") == []

    def test_single_sentence_no_split(self):
        result = _sentence_split("Just one sentence")
        assert result == ["Just one sentence"]


class TestSemanticChunkerFixedFallback:
    """Test chunking in fixed-size mode (no sentence-transformers installed)."""

    @pytest.fixture
    def chunker(self):
        c = SemanticChunker(max_chunk_tokens=10)
        c._model = None
        c._try_embed = lambda _: None  # force fallback
        return c

    def test_empty_text_produces_no_chunks(self, chunker):
        assert chunker.chunk("", "src.txt") == []

    def test_single_sentence_produces_one_chunk(self, chunker):
        chunks = chunker.chunk("Hello world.", "src.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].source == "src.txt"

    def test_long_text_is_split_at_token_boundary(self, chunker):
        text = ". ".join(f"Word{i} is here" for i in range(20)) + "."
        chunks = chunker.chunk(text, "test.md")
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text.split()) <= 12  # some slack for boundary

    def test_chunk_ids_are_unique(self, chunker):
        text = "First sentence. Second sentence. Third sentence."
        # Use very small max to force multiple chunks
        chunker._max_tokens = 3
        chunks = chunker.chunk(text, "s")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_propagated_to_chunks(self, chunker):
        meta = {"page": 1, "format": "pdf"}
        chunks = chunker.chunk("Some text here.", "doc.pdf", base_metadata=meta)
        assert len(chunks) == 1
        assert chunks[0].metadata == meta


class TestChunkDataclass:
    def test_chunk_is_frozen(self):
        chunk = Chunk(chunk_id="abc", text="t", source="s")
        with pytest.raises(AttributeError):
            chunk.text = "modified"
