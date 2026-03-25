"""Tests for pure Python TF-IDF fallback (should match Rust output)."""

import pytest

from ondine.context.text import term_frequency, tfidf_cosine_similarity, tokenize


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        tokens = tokenize("organic, premium cereals!")
        assert "organic" in tokens
        assert "premium" in tokens
        assert "cereals" in tokens
        # Hyphens inside a word are preserved (matches Rust behavior)
        assert tokenize("premium-grade") == ["premium-grade"]

    def test_empty(self):
        assert tokenize("") == []

    def test_numbers(self):
        assert tokenize("item 42") == ["item", "42"]


class TestTermFrequency:
    def test_uniform(self):
        tf = term_frequency(["a", "b", "c"])
        assert abs(tf["a"] - 1 / 3) < 1e-10

    def test_repeated(self):
        tf = term_frequency(["a", "a", "b"])
        assert abs(tf["a"] - 2 / 3) < 1e-10
        assert abs(tf["b"] - 1 / 3) < 1e-10

    def test_empty(self):
        assert term_frequency([]) == {}


class TestTfidfCosineSimilarity:
    def test_identical_is_one(self):
        sim = tfidf_cosine_similarity("hello world", "hello world")
        assert abs(sim - 1.0) < 1e-10

    def test_disjoint_is_zero(self):
        sim = tfidf_cosine_similarity("cat dog", "fish bird")
        assert abs(sim) < 1e-10

    def test_empty_both_is_one(self):
        sim = tfidf_cosine_similarity("", "")
        assert abs(sim - 1.0) < 1e-10

    def test_one_empty_is_zero(self):
        sim = tfidf_cosine_similarity("hello", "")
        assert abs(sim) < 1e-10

    def test_partial_overlap(self):
        sim = tfidf_cosine_similarity("organic cereals", "organic premium snacks")
        assert 0.0 < sim < 1.0

    def test_matches_rust_engine(self):
        """Python fallback should produce same results as Rust engine."""
        try:
            from ondine._engine import tfidf_similarity as rust_sim
        except ImportError:
            pytest.skip("Rust engine not available")

        pairs = [
            ("organic cereals", "organic cereals premium"),
            ("hello world", "hello world"),
            ("cat dog", "fish bird"),
            ("the quick brown fox", "a lazy brown dog"),
        ]
        for a, b in pairs:
            py_val = tfidf_cosine_similarity(a, b)
            rs_val = rust_sim(a, b)
            assert abs(py_val - rs_val) < 1e-10, (
                f"Mismatch for ({a!r}, {b!r}): py={py_val}, rs={rs_val}"
            )
