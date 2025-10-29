"""Unit tests for response parsers."""

import pytest
from pydantic import BaseModel

from ondine.stages import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
)


class TestRawTextParser:
    """Test suite for RawTextParser."""

    def test_parse_simple_text(self):
        """Test parsing raw text."""
        parser = RawTextParser()
        result = parser.parse("Hello world")

        assert result == {"output": "Hello world"}

    def test_parse_strips_whitespace(self):
        """Test that parser strips whitespace."""
        parser = RawTextParser()
        result = parser.parse("  Hello world  \n")

        assert result == {"output": "Hello world"}


class TestJSONParser:
    """Test suite for JSONParser."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        parser = JSONParser()
        json_str = '{"name": "John", "age": 30}'

        result = parser.parse(json_str)

        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_code_block(self):
        """Test extracting JSON from markdown code block."""
        parser = JSONParser(strict=False)
        response = '```json\n{"name": "John"}\n```'

        result = parser.parse(response)

        assert result == {"name": "John"}

    def test_parse_invalid_json_strict(self):
        """Test strict mode fails on invalid JSON."""
        parser = JSONParser(strict=True)

        with pytest.raises(Exception):
            parser.parse("not json")

    def test_parse_invalid_json_fallback(self):
        """Test fallback to raw text on invalid JSON."""
        parser = JSONParser(strict=False)
        result = parser.parse("not json")

        assert result == {"output": "not json"}


class TestPydanticParser:
    """Test suite for PydanticParser."""

    class ProductModel(BaseModel):
        """Sample Pydantic model."""

        name: str
        price: float
        category: str

    def test_parse_valid_response(self):
        """Test parsing valid Pydantic response."""
        parser = PydanticParser(self.ProductModel, strict=True)
        json_str = '{"name": "iPhone", "price": 999.99, "category": "Electronics"}'

        result = parser.parse(json_str)

        # Result is a Pydantic model instance, use attribute access
        assert result.name == "iPhone"
        assert result.price == 999.99
        assert result.category == "Electronics"

    def test_parse_invalid_strict(self):
        """Test strict mode fails on validation error."""
        parser = PydanticParser(self.ProductModel, strict=True)
        json_str = '{"name": "iPhone"}'  # Missing required fields

        with pytest.raises(ValueError, match="Pydantic validation failed"):
            parser.parse(json_str)

    def test_parse_invalid_non_strict(self):
        """Test non-strict mode handles validation errors."""
        parser = PydanticParser(self.ProductModel, strict=False)
        json_str = '{"name": "iPhone"}'

        result = parser.parse(json_str)

        assert "output" in result or "validation_error" in result


class TestRegexParser:
    """Test suite for RegexParser."""

    def test_parse_with_patterns(self):
        """Test extracting fields with regex patterns."""
        parser = RegexParser(
            {
                "price": r"\$(\d+\.?\d*)",
                "brand": r"Brand:\s*(\w+)",
            }
        )

        response = "Brand: Apple\nPrice: $999.99"
        result = parser.parse(response)

        assert result["price"] == "999.99"
        assert result["brand"] == "Apple"

    def test_parse_missing_field(self):
        """Test handling missing fields."""
        parser = RegexParser(
            {
                "price": r"\$(\d+\.?\d*)",
                "brand": r"Brand:\s*(\w+)",
            }
        )

        response = "Price: $999.99"  # No brand
        result = parser.parse(response)

        assert result["price"] == "999.99"
        assert result["brand"] is None

    def test_parse_no_groups(self):
        """Test pattern without capture groups."""
        parser = RegexParser(
            {
                "found": r"test",
            }
        )

        response = "this is a test string"
        result = parser.parse(response)

        assert result["found"] == "test"
