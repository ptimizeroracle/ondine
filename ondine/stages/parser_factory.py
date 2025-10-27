r"""Factory for creating response parsers based on configuration."""

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
    response_format = prompt_spec.response_format.lower()

    if response_format == "json":
        logger.info("Creating JSONParser for multi-field extraction")

        # Validate JSON fields match output columns
        if prompt_spec.json_fields:
            json_set = set(prompt_spec.json_fields)
            output_set = set(output_columns)
            if json_set != output_set:
                logger.warning(
                    f"JSON fields {json_set} don't match output columns {output_set}. "
                    f"Using output_columns as authoritative."
                )

        return JSONParser(strict=False)

    if response_format == "regex":
        logger.info("Creating RegexParser for pattern-based extraction")

        if not prompt_spec.regex_patterns:
            raise ValueError(
                "response_format='regex' requires regex_patterns to be specified"
            )

        # Validate regex patterns cover all output columns
        pattern_cols = set(prompt_spec.regex_patterns.keys())
        output_set = set(output_columns)
        missing = output_set - pattern_cols
        if missing:
            raise ValueError(f"Missing regex patterns for output columns: {missing}")

        return RegexParser(patterns=prompt_spec.regex_patterns)

    if response_format == "raw":
        logger.info("Creating RawTextParser (default, backward compatible)")

        if len(output_columns) > 1:
            logger.warning(
                f"Using RawTextParser with {len(output_columns)} output columns. "
                f"Only the first column will be populated. "
                f"Consider using response_format='json' or 'regex' for multi-column output."
            )

        return RawTextParser()

    # Should never reach here due to Pydantic validation
    raise ValueError(
        f"Invalid response_format: '{response_format}'. "
        f"Must be 'raw', 'json', or 'regex'."
    )
