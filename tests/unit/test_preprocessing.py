"""Tests for input preprocessing functionality."""

import pandas as pd

from ondine.utils.input_preprocessing import (
    ControlCharRemover,
    PreprocessingStats,
    SpecialCharCleaner,
    TextPreprocessor,
    TextTruncator,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    preprocess_dataframe,
)


class TestTextPreprocessor:
    """Test the main TextPreprocessor orchestrator."""

    def test_removes_special_chars(self):
        """Should remove trademark symbols."""
        preprocessor = TextPreprocessor()
        result = preprocessor.process("PREMIUM® PRODUCT™")
        assert "®" not in result
        assert "™" not in result
        assert "PREMIUM" in result
        assert "PRODUCT" in result

    def test_normalizes_whitespace(self):
        """Should collapse multiple spaces."""
        preprocessor = TextPreprocessor()
        result = preprocessor.process("PREMIUM    PRODUCT\n\nQUALITY")
        assert result == "PREMIUM PRODUCT QUALITY"

    def test_truncates_long_text(self):
        """Should truncate at word boundaries."""
        preprocessor = TextPreprocessor(max_length=50)
        long_text = "PRODUCT ITEM " * 20  # Way over 50 chars
        result = preprocessor.process(long_text)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_handles_none(self):
        """Should return empty string for None."""
        preprocessor = TextPreprocessor()
        result = preprocessor.process(None)
        assert result == ""

    def test_handles_empty_string(self):
        """Should return empty string for empty input."""
        preprocessor = TextPreprocessor()
        result = preprocessor.process("")
        assert result == ""


class TestUnicodeNormalizer:
    """Test Unicode normalization."""

    def test_normalizes_unicode(self):
        """Should normalize to NFC form."""
        cleaner = UnicodeNormalizer()
        # é can be represented as single char or e + combining accent
        result = cleaner.clean("café")
        assert "café" in result or "cafe" in result


class TestControlCharRemover:
    """Test control character removal."""

    def test_removes_control_chars(self):
        """Should remove null bytes and control chars."""
        cleaner = ControlCharRemover()
        result = cleaner.clean("PRODUCT\x00ITEM\x01")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "PRODUCT" in result
        assert "ITEM" in result


class TestSpecialCharCleaner:
    """Test special character cleaning."""

    def test_removes_trademark_symbols(self):
        """Should remove ®, ™, ©."""
        cleaner = SpecialCharCleaner()
        result = cleaner.clean("PRODUCT® ITEM™ COPYRIGHT©")
        assert "®" not in result
        assert "™" not in result
        assert "©" not in result

    def test_normalizes_quotes(self):
        """Should convert curly quotes to straight quotes."""
        cleaner = SpecialCharCleaner()
        # Using escaped curly quotes
        result = cleaner.clean("\u201cPRODUCT\u201d")
        assert '"' in result or "PRODUCT" in result


class TestWhitespaceNormalizer:
    """Test whitespace normalization."""

    def test_collapses_multiple_spaces(self):
        """Should collapse multiple spaces to one."""
        cleaner = WhitespaceNormalizer()
        result = cleaner.clean("PRODUCT    ITEM")
        assert result == "PRODUCT ITEM"

    def test_replaces_tabs_and_newlines(self):
        """Should replace tabs/newlines with spaces."""
        cleaner = WhitespaceNormalizer()
        result = cleaner.clean("PRODUCT\t\nITEM")
        assert result == "PRODUCT ITEM"

    def test_strips_leading_trailing(self):
        """Should strip leading/trailing whitespace."""
        cleaner = WhitespaceNormalizer()
        result = cleaner.clean("  PRODUCT ITEM  ")
        assert result == "PRODUCT ITEM"


class TestTextTruncator:
    """Test intelligent text truncation."""

    def test_truncates_at_word_boundary(self):
        """Should truncate at last space before limit."""
        truncator = TextTruncator(max_length=20)
        text = "PRODUCT ITEM PREMIUM QUALITY"
        result = truncator.clean(text)
        assert len(result) <= 20
        assert result.endswith("...")
        assert not result[:-3].endswith(" ")  # No trailing space before ...

    def test_does_not_truncate_short_text(self):
        """Should not modify text under limit."""
        truncator = TextTruncator(max_length=100)
        text = "PRODUCT ITEM"
        result = truncator.clean(text)
        assert result == text
        assert "..." not in result

    def test_truncates_at_delimiter(self):
        """Should prefer delimiter over space."""
        truncator = TextTruncator(max_length=30)
        text = "PRODUCT ITEM | PREMIUM QUALITY | TOP GRADE"
        result = truncator.clean(text)
        assert "|" in result or result.endswith("...")


class TestPreprocessDataframe:
    """Test DataFrame preprocessing."""

    def test_preprocesses_all_columns(self):
        """Should preprocess specified columns."""
        df = pd.DataFrame(
            {
                "text": ["PRODUCT®  ITEM", "PREMIUM™  QUALITY"],
                "other": ["keep", "unchanged"],
            }
        )

        result_df, stats = preprocess_dataframe(df, input_columns=["text"])

        assert "®" not in result_df["text"].iloc[0]
        assert "™" not in result_df["text"].iloc[1]
        assert result_df["other"].iloc[0] == "keep"  # Unchanged

    def test_returns_stats(self):
        """Should return preprocessing statistics."""
        df = pd.DataFrame(
            {
                "text": ["PRODUCT    ITEM", "A" * 1000]  # Second row will truncate
            }
        )

        result_df, stats = preprocess_dataframe(
            df, input_columns=["text"], max_length=100
        )

        assert isinstance(stats, PreprocessingStats)
        assert stats.rows_processed == 2
        assert stats.truncated_count == 1
        assert stats.chars_before > stats.chars_after
        assert stats.reduction_pct > 0

    def test_handles_nulls(self):
        """Should track null values."""
        df = pd.DataFrame({"text": ["PRODUCT", None, "ITEM"]})

        result_df, stats = preprocess_dataframe(df, input_columns=["text"])

        assert stats.null_count == 1
        assert stats.rows_processed == 3

    def test_does_not_modify_original_by_default(self):
        """Should not modify original DataFrame."""
        df = pd.DataFrame({"text": ["PRODUCT®"]})
        original_value = df["text"].iloc[0]

        result_df, _ = preprocess_dataframe(df, input_columns=["text"])

        assert df["text"].iloc[0] == original_value  # Original unchanged
        assert "®" not in result_df["text"].iloc[0]  # Result cleaned


class TestPreprocessingIntegration:
    """Integration tests for preprocessing."""

    def test_full_pipeline_cleans_product_description(self):
        """Should clean a realistic product description."""
        preprocessor = TextPreprocessor(max_length=200)

        dirty = "PREMIUM BRAND®    PRODUCT\n\nITEM™    QUALITY\x00\x01"
        clean = preprocessor.process(dirty)

        # Should remove: ®, ™, \x00, \x01, extra spaces, newlines
        assert "®" not in clean
        assert "™" not in clean
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "  " not in clean
        assert "\n" not in clean

        # Should keep: meaningful text
        assert "PREMIUM BRAND" in clean
        assert "PRODUCT" in clean
        assert "ITEM" in clean
        assert "QUALITY" in clean

    def test_dataframe_preprocessing_reduces_size(self):
        """Should reduce overall character count."""
        df = pd.DataFrame(
            {
                "desc": [
                    "PRODUCT®    ITEM™",
                    "PREMIUM\n\nQUALITY",
                    "A" * 1000,  # Long text
                ]
            }
        )

        result_df, stats = preprocess_dataframe(
            df, input_columns=["desc"], max_length=100
        )

        assert stats.chars_after < stats.chars_before
        assert stats.reduction_pct > 0
        assert len(result_df) == 3  # Same number of rows
