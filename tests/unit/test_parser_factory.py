"""Unit tests for parser factory."""

import pytest

from ondine.core.specifications import PromptSpec
from ondine.stages import (
    JSONParser,
    RawTextParser,
    RegexParser,
    create_response_parser,
)


class TestParserFactory:
    """Test suite for parser factory."""

    def test_create_raw_parser_default(self):
        """Test creating default raw text parser."""
        spec = PromptSpec(
            template="Process: {text}",
            response_format="raw",
        )

        parser = create_response_parser(spec, ["output"])

        assert isinstance(parser, RawTextParser)

    def test_create_raw_parser_implicit(self):
        """Test raw parser is default when not specified."""
        spec = PromptSpec(
            template="Process: {text}",
            # response_format defaults to "raw"
        )

        parser = create_response_parser(spec, ["output"])

        assert isinstance(parser, RawTextParser)

    def test_create_json_parser(self):
        """Test creating JSON parser."""
        spec = PromptSpec(
            template="Return JSON: {text}",
            response_format="json",
            json_fields=["field1", "field2"],
        )

        parser = create_response_parser(spec, ["field1", "field2"])

        assert isinstance(parser, JSONParser)

    def test_create_json_parser_without_fields(self):
        """Test JSON parser without explicit field list."""
        spec = PromptSpec(
            template="Return JSON: {text}",
            response_format="json",
        )

        parser = create_response_parser(spec, ["result"])

        assert isinstance(parser, JSONParser)

    def test_create_regex_parser(self):
        """Test creating regex parser."""
        spec = PromptSpec(
            template="Return structured: {text}",
            response_format="regex",
            regex_patterns={
                "score": r"SCORE:\s*(\d+)",
                "explanation": r"EXPLANATION:\s*(.+)",
            },
        )

        parser = create_response_parser(spec, ["score", "explanation"])

        assert isinstance(parser, RegexParser)

    def test_regex_parser_missing_patterns(self):
        """Test regex parser fails without patterns."""
        spec = PromptSpec(
            template="Return structured: {text}",
            response_format="regex",
            # Missing regex_patterns
        )

        with pytest.raises(ValueError, match="requires regex_patterns"):
            create_response_parser(spec, ["output"])

    def test_regex_parser_incomplete_patterns(self):
        """Test regex parser fails with incomplete patterns."""
        spec = PromptSpec(
            template="Return structured: {text}",
            response_format="regex",
            regex_patterns={
                "score": r"SCORE:\s*(\d+)",
                # Missing "explanation" pattern
            },
        )

        with pytest.raises(ValueError, match="Missing regex patterns"):
            create_response_parser(spec, ["score", "explanation"])

    def test_invalid_response_format(self):
        """Test validation catches invalid response format."""
        with pytest.raises(ValueError, match="response_format must be"):
            PromptSpec(
                template="Process: {text}",
                response_format="invalid",
            )


class TestParserIntegration:
    """Integration tests for parser factory with actual parsing."""

    def test_json_parser_multi_output(self):
        """Test JSON parser extracts multiple fields."""
        spec = PromptSpec(
            template="Analyze: {text}",
            response_format="json",
        )

        parser = create_response_parser(spec, ["similarity", "explanation"])

        # Simulate LLM response
        response = '{"similarity": "95%", "explanation": "Very similar"}'
        result = parser.parse(response)

        assert result["similarity"] == "95%"
        assert result["explanation"] == "Very similar"

    def test_json_parser_handles_malformed(self):
        """Test JSON parser handles malformed JSON gracefully."""
        spec = PromptSpec(
            template="Analyze: {text}",
            response_format="json",
        )

        parser = create_response_parser(spec, ["output"])

        # Malformed JSON
        response = "This is not JSON"
        result = parser.parse(response)

        # Should fallback to raw text
        assert "output" in result

    def test_regex_parser_extraction(self):
        """Test regex parser extracts fields correctly."""
        spec = PromptSpec(
            template="Analyze: {text}",
            response_format="regex",
            regex_patterns={
                "score": r"SCORE:\s*(\d+)%?",
                "reason": r"REASON:\s*(.+)",
            },
        )

        parser = create_response_parser(spec, ["score", "reason"])

        # Simulate LLM response
        response = "SCORE: 95%\nREASON: Both are similar"
        result = parser.parse(response)

        assert result["score"] == "95"
        assert result["reason"] == "Both are similar"

    def test_raw_parser_single_output(self):
        """Test raw parser returns single output."""
        spec = PromptSpec(
            template="Process: {text}",
            response_format="raw",
        )

        parser = create_response_parser(spec, ["result"])

        response = "Processed result"
        result = parser.parse(response)

        assert result["output"] == "Processed result"


class TestBackwardCompatibility:
    """Test backward compatibility with existing configs."""

    def test_old_config_still_works(self):
        """Test configs without response_format still work."""
        spec = PromptSpec(
            template="Process: {text}",
            system_message="You are a helper",
            # No response_format specified
        )

        parser = create_response_parser(spec, ["output"])

        # Should default to RawTextParser
        assert isinstance(parser, RawTextParser)

        # Should parse normally
        result = parser.parse("Some output")
        assert result["output"] == "Some output"

    def test_multiple_outputs_with_raw_parser_warns(self):
        """Test raw parser with multiple outputs logs warning."""
        spec = PromptSpec(
            template="Process: {text}",
            response_format="raw",
        )

        # Multiple output columns with raw parser
        # Should create parser but log warning
        parser = create_response_parser(spec, ["output1", "output2", "output3"])

        assert isinstance(parser, RawTextParser)
