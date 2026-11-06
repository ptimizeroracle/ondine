"""Re-ask: which rows are asked again, and the correction each one is sent.

A row whose latest answer failed a check is asked again with the prompt it
was first asked, followed by a correction: the answer it gave and why it was
not accepted. The pipeline runs the pass (``Pipeline._re_ask_failing_rows``);
the ledger decides the outcome by its usual rule, so a passing re-ask makes
the row ``repaired`` and a failing one leaves it as it was.

Design: ``.claude/designs/re-ask.md``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ondine.harness.status import RowStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ondine.harness.contract import OutputContract
    from ondine.harness.ledger import LedgerSnapshot, RowOutcome

#: Longest previous answer quoted back; the rest is cut and its length given.
MAX_ANSWER_CHARS = 2000

#: Most allowed values a correction lists by name; past this it refers to
#: them, since the prompt that offered that many values already lists them.
MAX_LISTED_VALUES = 100

_GENERIC = "Answer again. Reply with only the answer, in the format asked for above."

#: The hidden row column the prompt formatter appends after rendering.
REASK_COLUMN = "_reask_context"

_RE_ASKED = (RowStatus.REJECTED, RowStatus.UNVERIFIED)


def failing_rows(snapshot: LedgerSnapshot) -> list[RowOutcome]:
    """Rows whose latest answer failed a check, in input order.

    Rejected rows, and unverified ones (delivered flagged). Never a failed
    row: it has no answer to correct, and auto-retry re-runs it.
    """
    return [outcome for outcome in snapshot.rows if outcome.status in _RE_ASKED]


def last_failure(
    outcome: RowOutcome,
) -> tuple[Any | None, tuple[str, ...], tuple[str, ...]]:
    """The latest judged answer of ``outcome``, why it failed, and which fields.

    The answer is None when the ledger no longer holds it (past its cap).
    """
    if not outcome.verdicts:
        return None, outcome.failed_checks, ()
    attempt, judgement = outcome.verdicts[-1]
    answer = next(
        (a.raw_answer for a in outcome.attempts if a.attempt == attempt), None
    )
    failed = tuple(field.name for field in judgement.fields if field.failed)
    return answer, judgement.failure_reasons, failed


def instruction(contract: OutputContract, failed_fields: Sequence[str]) -> str:
    """What would be accepted, from the contract, for the fields that failed.

    A re-ask that only says an answer was not accepted gets the same kind of
    answer back: asked "is it consistent?", a model answers "No", and asked
    again, "No" again. So the allowed values are named.
    """
    declarations = {field.name: field for field in contract.fields}
    allowed = {
        name: list(declarations[name].allowed or ())
        for name in failed_fields
        if name in declarations and declarations[name].allowed
    }
    if len(contract.output_columns) == 1:
        values = next(iter(allowed.values()), None)
        if not values:
            return "Reply with only the answer, and nothing else."
        if len(values) > MAX_LISTED_VALUES:
            return f"Reply with only one of the {len(values)} allowed values, and nothing else."
        return (
            f"Reply with only one of these values, and nothing else: {_listed(values)}."
        )
    parts = ["Reply with only the answer, in the format asked for above."]
    for name, values in allowed.items():
        if len(values) > MAX_LISTED_VALUES:
            parts.append(f"{name} must be one of its {len(values)} allowed values.")
        else:
            parts.append(f"{name} must be one of: {_listed(values)}.")
    return " ".join(parts)


def _listed(values: Sequence[Any]) -> str:
    return ", ".join(str(value) for value in values)


def correction(
    answer: Any | None, reasons: Sequence[str], *, instruction: str = _GENERIC
) -> str:
    """The text appended to a re-asked row's prompt.

    Args:
        answer: The answer that failed, or None when it is not held.
        reasons: Why it failed, one per failed check.
        instruction: What to reply with (:func:`instruction`); the last line.
    """
    why = "; ".join(dict.fromkeys(reason.rstrip(".") for reason in reasons))
    closing = instruction
    if answer is None:
        return f"Your previous answer was not accepted: {why}.\n{closing}"
    return (
        f"Your previous answer was:\n{_quoted(answer)}\n"
        f"It was not accepted: {why}.\n{closing}"
    )


def _quoted(answer: Any) -> str:
    text = answer if isinstance(answer, str) else json.dumps(answer, default=str)
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    return (
        f"{text[:MAX_ANSWER_CHARS]} [… {len(text) - MAX_ANSWER_CHARS} more characters]"
    )
