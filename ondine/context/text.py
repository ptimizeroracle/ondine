"""Pure Python TF-IDF implementation (fallback when Rust engine unavailable).

Port of crates/ondine-core/src/text/mod.rs. Used only by InMemoryContextStore
and as fallback when the compiled Rust extension is not available.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Split on whitespace, strip non-alphanumeric edges, lowercase, drop len<=1."""
    tokens = []
    for word in text.split():
        clean = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word).lower()
        if clean and len(clean) > 1:
            tokens.append(clean)
    return tokens


def term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term frequency for a token list."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two texts.

    Uses a two-document corpus with smoothed IDF: ln(N/df) + 1.
    Matches the Rust implementation in crates/ondine-core/src/text/mod.rs.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = term_frequency(tokens_a)
    tf_b = term_frequency(tokens_b)

    all_terms = set(tf_a.keys()) | set(tf_b.keys())

    num_docs = 2.0
    idf = {}
    for term in all_terms:
        df = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
        idf[term] = math.log(num_docs / df) + 1.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term in all_terms:
        tfidf_a = tf_a.get(term, 0.0) * idf[term]
        tfidf_b = tf_b.get(term, 0.0) * idf[term]
        dot += tfidf_a * tfidf_b
        norm_a += tfidf_a * tfidf_a
        norm_b += tfidf_b * tfidf_b

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0

    return dot / denom
