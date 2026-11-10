"""The error a run raises when its first rows fail the contract too often.

Early stop is a policy decision, like a budget stop, so it surfaces the same
way: the run raises, and ``pipeline.row_ledger`` and ``pipeline.run_report``
show what it reached. The message says what was counted, the most common
reason, and the ways out. Design: ``.claude/designs/early-stop.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.harness.delivery import RejectionReason

#: Reasons the message names; the full list is in the run report.
_REASONS_IN_MESSAGE = 1


class EarlyStopError(RuntimeError):
    """More than ``fraction`` of the first ``window`` rows were rejected.

    Counted on each row's first answer (before any re-ask). Rows the run never
    asked about are ``failed`` in the ledger, "run stopped before this row:
    early stop".

    Attributes:
        rejected: Rows of the window whose first answer was rejected.
        window: Rows counted: the policy's ``early_stop_window``, or fewer
            when the input is shorter.
        fraction: The policy's ``early_stop_fraction``.
        reasons: The blocking checks of the rows the run judged, most
            frequent first, as in the run report.
    """

    def __init__(
        self,
        rejected: int,
        window: int,
        fraction: float,
        reasons: tuple[RejectionReason, ...] = (),
        *,
        resume_hint: str | None = None,
    ) -> None:
        self.rejected = rejected
        self.window = window
        self.fraction = fraction
        self.reasons = reasons
        super().__init__(_message(rejected, window, fraction, reasons, resume_hint))


def _message(
    rejected: int,
    window: int,
    fraction: float,
    reasons: tuple[RejectionReason, ...],
    resume_hint: str | None,
) -> str:
    lines = [
        f"Early stop: {rejected} of the first {window} rows were rejected on "
        f"their first answer ({rejected / window:.0%}), above "
        f"{fraction * 100:g}%. The prompt or the contract is likely broken."
    ]
    for reason in reasons[:_REASONS_IN_MESSAGE]:
        rule = (
            f"{reason.field}.{reason.rule}"
            if reason.field is not None and reason.rule is not None
            else "unknown rule"
        )
        lines.append(f"Most common: {reason.count} x {rule} (e.g. {reason.example}).")
    lines.append("Fix the prompt or the contract and run again.")
    if resume_hint is not None:
        lines.append(f"To finish this run anyway: {resume_hint}.")
    lines.append(
        "To turn early stop off: with_contract_policy(early_stop_fraction=None)."
    )
    return "\n".join(lines)


__all__ = ["EarlyStopError"]
