"""A reference model's answers for a run's sampled rows: the audit's source.

:class:`ReferenceAnswers` turns into the Ruler's :class:`~ondine.ruler.source.Reference`.

The rows are asked in chunks through an injected ``ask``: the pipeline runs
each chunk as an isolated pass of the run's own prompt, on the reference
model, judged by the run's contract with one re-ask. This module knows which
rows to ask, what that may cost, and what each row's answer, or reason for
having none, is. It never sees the pipeline.

Cost: before each chunk, only the leading rows whose estimate fits what is
left of the cap are asked. A row's static estimate is two calls (an answer and
its re-ask) by :func:`ondine.harness.pricing.call_cost`, with the margin; it
errs high, and the first chunk is priced at it. Later chunks are priced at the
observed ratio of what the reference charged to what the static estimate
charged those rows, times a safety factor of 2, never above the static
estimate. So the cap is respected unless later rows cost more than twice as
much, relative to their size, as the rows already asked. Rows left over are
unresolved, "over the cost cap".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ondine.harness.pricing import COST_MARGIN, call_cost
from ondine.ruler.delivered import CertificationError, _cell
from ondine.ruler.source import NO_ANSWER, Reference

if TYPE_CHECKING:
    import pandas as pd

    from ondine.core.models import ExecutionResult
    from ondine.harness.ledger import RowOutcome
    from ondine.ruler.sample import Sample

#: Rows sent per pass; the cap is checked against actual spending between.
CHUNK_ROWS = 50
#: Calls a row may take: its answer, and one re-ask when that breaks the contract.
CALLS_PER_ROW = 2
#: Later rows may cost up to this many times what the asked ones did, relative
#: to their size, and still fit the cap.
SAFETY = Decimal("2")

OVER_CAP = "over the cost cap"
#: Why a row has no reference answer, by the status its pass gave it.
_CAUSES = {"failed": "provider", "rejected": "broke the contract"}

#: Runs one chunk: the rows' input columns, in sample order. Returns the
#: pass's result, one row per input row, in the same order.
Ask = Callable[["pd.DataFrame"], "ExecutionResult"]


@dataclass(frozen=True)
class ReferenceAnswers:
    """What a reference model answered for one run's sample.

    Attributes:
        model: The reference model.
        prompt_fingerprint: The prompt it was asked with.
        run_id: The run whose rows were asked.
        fingerprint: The run's delivered values the sample is bound to.
        rows: Every sampled row this audit covers, answered or not.
        values: Field -> row -> the reference's value, read by the contract.
        unresolved: Row -> why it has no value for some field.
        cost: Everything paid for these answers, reused ones included.
    """

    model: str
    prompt_fingerprint: str
    run_id: str
    fingerprint: str
    rows: tuple[int, ...]
    values: dict[str, dict[int, Any]] = field(repr=False)
    unresolved: dict[int, str]
    cost: Decimal

    def as_reference(self) -> Reference:
        """These answers as the Ruler's reference.

        A row's cause (it is kept per row) applies to every field the row has
        no answer for.
        """
        causes = {
            name: {
                row: self.unresolved.get(row, NO_ANSWER)
                for row in self.rows
                if row not in answers
            }
            for name, answers in self.values.items()
        }
        return Reference(
            kind="model", answers=self.values, causes=causes, model=self.model
        )


@dataclass(frozen=True)
class AuditEstimate:
    """The price of an audit, and what would leave, before anything is sent.

    Attributes:
        model: The reference model.
        provider: Its provider.
        rows: Rows that would be asked.
        reused: Rows a previous audit already answered (not asked, not paid).
        columns: The input columns the prompt reads, sent for each row.
        carries: What else the rendered prompt sends, e.g. "a system prompt".
        cost: The high estimate: the most the audit would spend uncapped.
    """

    model: str
    provider: str
    rows: int
    reused: int
    columns: tuple[str, ...]
    carries: tuple[str, ...]
    cost: Decimal

    def render(self) -> str:
        extra = f", with {', '.join(self.carries)}" if self.carries else ""
        reused = f" ({self.reused} reused, not asked again)" if self.reused else ""
        return (
            f"Audit estimate: {self.model} ({self.provider})\n"
            f"  Sends: {self.rows} rows{reused}; columns {', '.join(self.columns)}{extra}\n"
            f"  Price: at most ${self.cost:.4f} (estimate errs high; the audit "
            "stops at its max_cost)"
        )


def row_estimate(outcome: RowOutcome, client: Any) -> Decimal:
    """The most one row's reference answer is expected to cost."""
    return call_cost(outcome, client) * CALLS_PER_ROW * COST_MARGIN


def check_previous(
    previous: ReferenceAnswers,
    *,
    run_id: str,
    fingerprint: str,
    model: str,
    prompt_fingerprint: str,
) -> None:
    """Refuse answers given for another run, values, model or prompt."""
    if previous.run_id != run_id:
        raise CertificationError(
            f"the previous audit is of another run ({previous.run_id[:8]}…, "
            f"not {run_id[:8]}…)"
        )
    if previous.fingerprint != fingerprint:
        raise CertificationError(
            "the delivered values changed since the previous audit; audit again "
            "without it"
        )
    if previous.model != model:
        raise CertificationError(
            f"the previous audit asked {previous.model}, not {model}; its answers "
            "are not this model's"
        )
    if previous.prompt_fingerprint != prompt_fingerprint:
        raise CertificationError(
            "the previous audit asked with another prompt; audit again without it"
        )


def reusable(previous: ReferenceAnswers | None, rows: Sequence[int]) -> set[int]:
    """The rows of ``rows`` a previous audit already answered."""
    if previous is None:
        return set()
    covered = set(previous.rows) - set(previous.unresolved)
    return {row for row in rows if row in covered}


def ask_model(
    sample: Sample,
    fields: Sequence[str],
    outcomes: Mapping[int, RowOutcome],
    *,
    client: Any,
    model: str,
    prompt_fingerprint: str,
    max_cost: Decimal,
    ask: Ask,
    previous: ReferenceAnswers | None = None,
) -> ReferenceAnswers:
    """The reference's answers for ``sample``'s rows, within ``max_cost``.

    Args:
        sample: The rows to ask, with their input columns.
        fields: The fields compared; a row lacking any of them is unresolved.
        outcomes: The run's ledger outcome per row, for pricing.
        client: The reference client (its prices).
        max_cost: The most this call may spend; answers reused cost nothing.
            The first chunk is priced at the static high estimate; later
            chunks at the observed ratio of spending to estimate with a
            safety factor of :data:`SAFETY`, never above the static estimate.
            The cap is respected unless later rows cost more than twice as
            much, relative to their size, as the rows already asked.
        ask: Runs one chunk (see :data:`Ask`).
        previous: Answers to reuse, already checked by :func:`check_previous`.
    """
    values: dict[str, dict[int, Any]] = {name: {} for name in fields}
    unresolved: dict[int, str] = {}
    reused = reusable(previous, sample.rows)
    if previous is not None:
        for row in sorted(reused):
            for name in fields:
                if row in previous.values.get(name, {}):
                    values[name][row] = previous.values[name][row]

    todo = [row for row in sample.rows if row not in reused]
    spent = Decimal("0")
    estimated = Decimal("0")  # what the static estimate charged the rows asked
    while todo:
        factor = _factor(spent, estimated)
        chunk = _affordable(
            todo[:CHUNK_ROWS], outcomes, client, max_cost - spent, factor
        )
        if not chunk:
            break
        result = ask(sample.inputs.loc[chunk].reset_index(drop=True))
        spent += Decimal(str(result.costs.total_cost))
        estimated += sum(
            (row_estimate(outcomes[row], client) for row in chunk), Decimal("0")
        )
        _read(result, chunk, fields, values, unresolved)
        todo = todo[len(chunk) :]
    for row in todo:
        unresolved[row] = OVER_CAP

    return ReferenceAnswers(
        model=model,
        prompt_fingerprint=prompt_fingerprint,
        run_id=sample.run_id,
        fingerprint=sample.fingerprint,
        rows=tuple(sample.rows),
        values=values,
        unresolved=unresolved,
        cost=spent + (previous.cost if previous is not None else Decimal("0")),
    )


def _factor(spent: Decimal, estimated: Decimal) -> Decimal:
    """How much of the static estimate the next rows are priced at, 0 to 1.

    1 until a row is asked, or when the asked rows had no estimate to compare.
    """
    if estimated <= 0:
        return Decimal("1")
    return min(Decimal("1"), SAFETY * spent / estimated)


def _affordable(
    rows: Sequence[int],
    outcomes: Mapping[int, RowOutcome],
    client: Any,
    remaining: Decimal,
    factor: Decimal,
) -> list[int]:
    chosen = []
    spent = Decimal("0")
    for row in rows:
        estimate = row_estimate(outcomes[row], client) * factor
        if spent + estimate > remaining:
            break
        spent += estimate
        chosen.append(row)
    return chosen


def _read(
    result: ExecutionResult,
    chunk: Sequence[int],
    fields: Sequence[str],
    values: dict[str, dict[int, Any]],
    unresolved: dict[int, str],
) -> None:
    """Each chunk row's answers, or the cause it has none, from its pass."""
    frame = result.to_pandas()
    for position, row in enumerate(chunk):
        missing = False
        for name in fields:
            value = _cell(frame[name].iloc[position])
            if value is None:
                missing = True
            else:
                values[name][row] = value
        if missing:
            status = str(frame["status"].iloc[position])
            unresolved[row] = _CAUSES.get(status, NO_ANSWER)
