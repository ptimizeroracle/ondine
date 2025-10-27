"""
Input preprocessing for LLM prompts.

Best practices: Remove noise, normalize whitespace, control length.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing operation."""

    rows_processed: int
    chars_before: int
    chars_after: int
    truncated_count: int
    null_count: int

    @property
    def reduction_pct(self) -> float:
        """Calculate character reduction percentage."""
        if self.chars_before == 0:
            return 0.0
        return (self.chars_before - self.chars_after) / self.chars_before * 100


class TextCleaner(Protocol):
    """Protocol for text cleaning strategies."""

    def clean(self, text: str) -> str:
        """Clean text according to strategy."""
        ...


class UnicodeNormalizer:
    """Normalize Unicode to canonical form (NFC)."""

    def clean(self, text: str) -> str:
        """Normalize Unicode: é vs e + ´ → consistent form."""
        return unicodedata.normalize("NFC", text)


class ControlCharRemover:
    """Remove control characters that confuse tokenizers."""

    def clean(self, text: str) -> str:
        """Replace control chars with space (preserves word boundaries)."""
        return "".join(
            char if unicodedata.category(char)[0] != "C" else " " for char in text
        )


class SpecialCharCleaner:
    """Remove noise characters while preserving semantic punctuation."""

    def __init__(self, preserve: str = r",\-/\.\(\)&"):
        self.preserve = preserve

    def clean(self, text: str) -> str:
        """Remove ®™© and excessive special chars."""
        # Remove trademark symbols
        text = re.sub(r"[®™©℗℠]", "", text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # Remove zero-width characters
        text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)

        # Keep only: alphanumeric, whitespace, specified punctuation
        pattern = f"[^\\w\\s{self.preserve}]"
        return re.sub(pattern, " ", text)


class WhitespaceNormalizer:
    """Collapse multiple spaces/tabs/newlines."""

    def clean(self, text: str) -> str:
        """Replace tabs/newlines with spaces, collapse multiples."""
        text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", text).strip()


class TextTruncator:
    """Intelligently truncate at word boundaries."""

    def __init__(self, max_length: int = 500):
        self.max_length = max_length

    def clean(self, text: str) -> str:
        """Truncate respecting word boundaries."""
        if len(text) <= self.max_length:
            return text

        # Reserve space for ellipsis
        limit = self.max_length - 3
        min_keep = int(self.max_length * 0.7)

        # Try delimiters first: |, ;, ' - '
        for delim in ["|", ";", " - ", "  "]:
            pos = text.rfind(delim, 0, limit)
            if pos > min_keep:
                return text[:pos].strip() + "..."

        # Fall back to last space
        pos = text.rfind(" ", 0, limit)
        if pos > min_keep:
            return text[:pos].strip() + "..."

        # Hard truncate if no boundary found
        return text[:limit].strip() + "..."


class TextPreprocessor:
    """
    Composable text preprocessor following Chain of Responsibility.

    Single Responsibility: Orchestrate cleaning steps.
    Open/Closed: Extensible via cleaners list.
    Dependency Inversion: Depends on Protocol, not concrete classes.
    """

    def __init__(self, max_length: int = 500):
        """Initialize with default cleaning pipeline."""
        self.cleaners: list[TextCleaner] = [
            UnicodeNormalizer(),
            ControlCharRemover(),
            SpecialCharCleaner(),
            WhitespaceNormalizer(),
            TextTruncator(max_length),
        ]

    def process(self, text: str) -> str:
        """Apply all cleaners in sequence."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        for cleaner in self.cleaners:
            text = cleaner.clean(text)

        return text

    def add_cleaner(self, cleaner: TextCleaner) -> None:
        """Extend pipeline with custom cleaner."""
        self.cleaners.append(cleaner)


def preprocess_dataframe(
    df: pd.DataFrame,
    input_columns: list[str],
    max_length: int = 500,
) -> tuple[pd.DataFrame, PreprocessingStats]:
    """
    Preprocess input columns in dataframe.

    Args:
        df: Input dataframe
        input_columns: Columns to clean
        max_length: Max chars per field

    Returns:
        (cleaned_df, stats)
    """
    result = df.copy()
    preprocessor = TextPreprocessor(max_length)

    chars_before = 0
    chars_after = 0
    truncated = 0
    nulls = 0

    for col in input_columns:
        if col not in result.columns:
            continue

        for idx in result.index:
            original = result.at[idx, col]

            if pd.isna(original):
                nulls += 1
                continue

            original_str = str(original)
            chars_before += len(original_str)

            cleaned = preprocessor.process(original_str)
            chars_after += len(cleaned)

            if len(original_str) > max_length:
                truncated += 1

            result.at[idx, col] = cleaned

    stats = PreprocessingStats(
        rows_processed=len(result),
        chars_before=chars_before,
        chars_after=chars_after,
        truncated_count=truncated,
        null_count=nulls,
    )

    return result, stats
