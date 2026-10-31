r"""Factory for creating response parsers based on configuration."""

from collections.abc import Mapping, Sequence
from typing import Any

from ondine.core.specifications import PromptSpec
from ondine.stages.response_parser_stage import (
    JSONParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
)
from ondine.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_response_parser(
    prompt_spec: PromptSpec,
    output_columns: list[str],
) -> ResponseParser:
    r"""
    Create appropriate response parser based on prompt specification.

    This factory enables configuration-driven parser selection, supporting:
    - Raw text output (default, backward compatible)
    - JSON structured output (for multiple fields)
    - Regex pattern extraction (for formatted text)

    Args:
        prompt_spec: Prompt specification with response_format
        output_columns: Expected output column names

    Returns:
        Configured ResponseParser instance

    Raises:
        ValueError: If response_format is invalid or required config missing

    Example:
        # JSON mode
        parser = create_response_parser(
            prompt_spec=PromptSpec(
                template="...",
                response_format="json",
                json_fields=["score", "explanation"]
            ),
            output_columns=["score", "explanation"]
        )

        # Regex mode
        parser = create_response_parser(
            prompt_spec=PromptSpec(
                template="...",
                response_format="regex",
                regex_patterns={
                    "score": r"SCORE:\s*(\d+)",
                    "explanation": r"EXPLANATION:\s*(.+)"
                }
            ),
            output_columns=["score", "explanation"]
        )
    """
    parse_mode = validate_response_format(
        prompt_spec.response_format,
        output_columns,
        prompt_spec.regex_patterns,
        json_fields=prompt_spec.json_fields,
    )
    if parse_mode == "regex":
        logger.info("Creating RegexParser for pattern-based extraction")
        return RegexParser(patterns=prompt_spec.regex_patterns or {})
    if parse_mode == "json":
        logger.info("Creating JSONParser for multi-field extraction")
        return JSONParser(strict=False)
    logger.info("Creating RawTextParser (default, backward compatible)")
    return RawTextParser()


def validate_response_format(
    response_format: str,
    output_columns: Sequence[str],
    regex_patterns: Mapping[str, Any] | None = None,
    *,
    json_fields: Sequence[str] | None = None,
) -> str:
    """Check a response format against the output columns; return how to read it.

    The one place that knows the response-format rules. The parser factory
    builds its parser from the result, and the pipeline calls it to fail a
    run the way the factory always failed it.

    Args:
        response_format: "raw", "json" or "regex", in any case.
        output_columns: The columns each answer fills.
        regex_patterns: For "regex": a pattern per output column.
        json_fields: For "json": the fields the prompt asks for. A mismatch
            with the output columns is only logged; the columns win.

    Returns:
        "text" (one column of raw text), "json" (an object; also "raw" with
        several columns, since raw text can fill only one) or "regex".

    Raises:
        ValueError: An unknown format, or a regex format without a pattern
            for every output column.
    """
    fmt = response_format.lower()

    if fmt == "json":
        if json_fields:
            json_set = set(json_fields)
            output_set = set(output_columns)
            if json_set != output_set:
                logger.warning(
                    f"JSON fields {json_set} don't match output columns {output_set}. "
                    f"Using output_columns as authoritative."
                )
        return "json"

    if fmt == "regex":
        if not regex_patterns:
            raise ValueError(
                "response_format='regex' requires regex_patterns to be specified"
            )
        missing = set(output_columns) - set(regex_patterns.keys())
        if missing:
            raise ValueError(f"Missing regex patterns for output columns: {missing}")
        return "regex"

    if fmt == "raw":
        if len(output_columns) > 1:
            # Raw text fills one column named "output" — none of the several
            # asked for — so every cell came back empty from a run that
            # reported success. Several columns can only come from a structured
            # answer; read it as JSON. A reply that is not JSON still lands in
            # the first column rather than being discarded.
            logger.info(
                f"{len(output_columns)} output columns: reading each response as "
                f"a JSON object with keys {list(output_columns)}. Ask for that "
                f"shape in the prompt, or set response_format='regex' / "
                f"structured output."
            )
            return "json"
        return "text"

    # Should never reach here due to Pydantic validation
    raise ValueError(
        f"Invalid response_format: '{response_format}'. "
        f"Must be 'raw', 'json', or 'regex'."
    )
