"""How a person's typed label cell is read.

One cell is read by the field's contract rules, the reading gold gets, never a
looser one: :func:`read_cell` returns the answers it accepts, or every part the
rules refuse, as data. The sheet (:mod:`ondine.ruler.sheet`) locates the
problems; this module knows nothing of files.

* one answer: read by the rules;
* ``a | b``: any of these is right (the whole cell is read first, so an allowed
  value that contains ``|`` still works);
* ``?`` alone: cannot tell; empty: not labelled;
* ``-``: no value, for an optional field only.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ondine.ruler.ruler import declaration_of, read_reference
from ondine.ruler.source import ABSENT, ReferenceProblem

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ondine.harness.contract import OutputContract

#: A whole cell of these means "cannot tell" and "no value".
UNSURE = "?"
NO_VALUE = "-"

_COMMA_DECIMAL = re.compile(r"[+-]?\d+,\d+")
_COMMA_HINT = "use a dot, not a comma, for decimals"
_EMPTY_PART = "an empty answer after a '|'"
_UNSURE_PART = "'?' must be the whole cell, not one of several answers"


@dataclass(frozen=True)
class Cell:
    """One label cell, read.

    Attributes:
        state: "empty" (not labelled), "unsure" (``?``) or "answer".
        raw: For an answer: the text, a list of acceptable answers, or
            :data:`~ondine.ruler.source.ABSENT`; what ``Reference.answers``
            holds. None when the cell has problems.
        problems: Every part the field's rules refuse, as data.
        hint: A hint for the commonest slip (a decimal comma).
        suggestions: Typed answer -> the allowed value it nearly is.
    """

    state: Literal["empty", "unsure", "answer"]
    raw: Any = None
    problems: tuple[ReferenceProblem, ...] = ()
    hint: str | None = None
    suggestions: Mapping[str, str] = field(default_factory=dict)


def read_cell(contract: OutputContract, name: str, text: str, row: int) -> Cell:
    """Read what a person typed in one label cell, by the field's contract rules.

    The whole cell is read first, so an allowed value that contains ``|``
    still works; only if that fails is the cell split on ``|`` into several
    acceptable answers. Every answer goes through
    :func:`~ondine.ruler.ruler.read_reference`, the reading gold gets, never a
    looser one. ``?`` alone and an empty cell are never an answer.
    """
    text = text.strip()
    if not text:
        return Cell("empty")
    if text == UNSURE:
        return Cell("unsure")
    if text == NO_VALUE:
        return _checked(contract, name, text, row, ABSENT, [])
    _, problems = read_reference(contract, name, text, row)
    if not problems:
        return Cell("answer", text)
    if "|" not in text:
        return _failed(contract, name, text, problems)
    own: list[ReferenceProblem] = []
    parts: list[Any] = []
    for part in (piece.strip() for piece in text.split("|")):
        if not part:
            own.append(ReferenceProblem(name, row, part, (_EMPTY_PART,)))
        elif part == UNSURE:
            own.append(ReferenceProblem(name, row, part, (_UNSURE_PART,)))
        else:
            parts.append(ABSENT if part == NO_VALUE else part)
    return _checked(contract, name, text, row, parts, own)


def _checked(
    contract: OutputContract,
    name: str,
    text: str,
    row: int,
    raw: Any,
    problems: list[ReferenceProblem],
) -> Cell:
    _, found = read_reference(contract, name, raw, row)
    problems = [*problems, *found]
    if problems:
        return _failed(contract, name, text, problems)
    return Cell("answer", raw)


def _failed(
    contract: OutputContract, name: str, text: str, problems: Sequence[ReferenceProblem]
) -> Cell:
    declaration = declaration_of(contract, name)
    suggestions = {
        str(problem.raw): close
        for problem in problems
        if isinstance(problem.raw, str)
        and (close := _suggest(declaration, problem.raw)) is not None
    }
    hint = None
    if declaration.type in ("integer", "number") and _COMMA_DECIMAL.fullmatch(text):
        hint = _COMMA_HINT
    return Cell("answer", None, tuple(problems), hint, suggestions)


def _suggest(declaration: Any, typed: str) -> str | None:
    """The allowed value or alias ``typed`` is closest to, ignoring case."""
    pool = [str(value) for value in declaration.allowed or ()]
    pool += [str(alias) for alias in declaration.aliases]
    folded = {entry.casefold(): entry for entry in pool}
    close = difflib.get_close_matches(typed.casefold(), list(folded), n=1)
    return folded[close[0]] if close else None
