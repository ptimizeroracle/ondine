"""Wording shared by contract problems and judge reasons.

A client reads both: a contract problem at build time ("allowed value 'x' is
not an integer") and a judge reason at run time ("'x' is not an integer"). The
two must name a type the same way, so the wording lives here, once, and both
modules read it. A second copy would drift the first time someone rewords one.
"""

from __future__ import annotations

# How a reason names each declared type. "any" has no phrase: nothing is ever
# checked against it.
TYPE_PHRASES = {
    "text": "text",
    "integer": "an integer",
    "number": "a number",
    "boolean": "true or false",
}

_SCHEMA_PREFIXES = ("Value error, ", "Assertion failed, ")


def type_phrase(field_type: str) -> str:
    """How a sentence names ``field_type``; "a value" for an unknown type."""
    return TYPE_PHRASES.get(field_type, "a value")


def schema_reason_text(message: str) -> str:
    """A Pydantic error message with its generic prefix removed.

    A validator raising ``ValueError("refunds need an order id")`` surfaces as
    "Value error, refunds need an order id"; the client wrote the second half,
    and that is the reason worth showing.
    """
    for prefix in _SCHEMA_PREFIXES:
        if message.startswith(prefix):
            return message[len(prefix) :]
    return message
