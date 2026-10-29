"""Delivery: turns a ledger snapshot into what the client receives.

Three outputs, all readings of one :class:`~ondine.harness.ledger.LedgerSnapshot`:

1. **The dataframe.** Every input row, in input order, with the output values
   kept or blanked by status: accepted, repaired, unverified and unchecked
   rows keep their value; rejected and failed rows are blanked. Under a
   checked contract five columns follow: ``status``, ``reason``, ``checks``,
   ``repair_method``, ``delivered_by``.
2. **The quarantine.** One entry per rejected, unverified or failed row: its
   inputs, every attempt's raw answer (within the snapshot's cap), every
   failed check with its rule and reason, the models used and the cost.
3. **The run report.** Totals per status and the invariant that they sum to
   the row count, coverage, the top rejection reasons, the pass rate per
   rule, cost, and warnings.

Why delivery is built this way:

* **It decides nothing.** The ledger owns every status; delivery only reads
  the snapshot. Before the harness, a row's fate was re-derived by a blanking
  pass matching marker strings and by a quality report counting empty cells.
  Both become readings here, so they can no longer disagree with the ledger.
* **The value comes from the judge, not the ledger.** The ledger records
  which attempt a row delivers, not the values: it only needs a verdict's
  pass or fail. Delivery takes the judged answers as a separate input,
  a mapping keyed by ``(row, attempt)`` and built from the
  judge's verdicts with :meth:`JudgedAnswer.from_verdict`. Giving every
  judged attempt, not only the delivered one, lets the quarantine name each
  failed check's rule and the ``reason`` column show the path taken.
* **No contract, no change.** Under the unchecked contract the dataframe is
  the input rows with today's output values, and no column is added. The
  quarantine and the report are new outputs next to it, never changes to it.
* **Plain structures inside, pandas only at the edge.** Rows are tuples in a
  fixed column order, like the dict rows ``ResultContainerImpl`` holds;
  :meth:`Delivery.to_pandas` builds a frame only when asked. That keeps
  delivery deterministic and cheap to compare: the same snapshot and judged
  answers always give an equal :class:`Delivery`.
* **Wiring defects are loud, model behaviour is data.** An input whose row
  count differs from the snapshot's, or a column clash with the harness
  columns, raises :class:`ValueError`: no correct frame exists. A delivered
  attempt without a judged answer is delivered empty and listed in the
  report's violations, so a bookkeeping slip never ships a wrong value and
  never hides.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ondine.harness.ledger import PROVIDER_SUB_REASON, StopReason
from ondine.harness.status import FINAL_STATUSES, RowStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ondine.harness.contract import OutputContract
    from ondine.harness.judge import Verdict
    from ondine.harness.ledger import AttemptRecord, LedgerSnapshot, RowOutcome

#: The columns a checked contract adds, in this order. Delivery owns the set;
#: later features fill values, never add columns (design v1, section 2).
HARNESS_COLUMNS: tuple[str, ...] = (
    "status",
    "reason",
    "checks",
    "repair_method",
    "delivered_by",
)

#: Warn when more than this share of the rows is rejected. A judgement call:
#: early stop fires at 50% of the first rows; a fifth is already a run whose
#: prompt or contract deserves a look before the next one.
DEFAULT_REJECTION_WARNING_THRESHOLD = 0.2

#: How many rejection reasons the report lists.
DEFAULT_TOP_REASONS = 10

#: Statuses whose row keeps its value in the dataframe.
_KEEPS_VALUE = frozenset(
    {RowStatus.ACCEPTED, RowStatus.REPAIRED, RowStatus.UNVERIFIED, RowStatus.UNCHECKED}
)
#: Statuses that get a quarantine entry.
_QUARANTINED = (RowStatus.REJECTED, RowStatus.UNVERIFIED, RowStatus.FAILED)
#: Statuses whose answer failed a check, for the rejection reasons.
_FAILED_A_CHECK = frozenset({RowStatus.REJECTED, RowStatus.UNVERIFIED})


# ---------------------------------------------------------------------------
# Input: the judged answers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckFailure:
    """One failed check: the field, the rule, and the judge's sentence.

    ``field`` and ``rule`` are None only when the judged answer was not given
    and the text comes from the ledger, which keeps reasons as sentences.
    ``attempt`` is set in the quarantine, where checks from every attempt are
    listed together.
    """

    field: str | None
    rule: str | None
    text: str
    attempt: int | None = None


@dataclass(frozen=True)
class JudgedField:
    """One field of a judged answer: what delivery may write in its column.

    Attributes:
        name: The output column.
        outcome: "value", "absent" (a valid empty answer) or "failed".
        value: The deliverable value; for a failed field, the best candidate,
            delivered only when the row is unverified.
        required: Whether a failure of this field fails the row.
        format_fixed: The value was found inside a wordy answer.
        failures: Every failed check of this field.
        fixed_failures: For a format-fixed field, the checks the answer
            failed before the fix: the check that blocked it.
    """

    name: str
    outcome: str
    value: Any = None
    required: bool = True
    format_fixed: bool = False
    failures: tuple[CheckFailure, ...] = ()
    fixed_failures: tuple[CheckFailure, ...] = ()


@dataclass(frozen=True)
class JudgedAnswer:
    """The judge's result for one attempt, in the form delivery reads.

    A copy rather than the judge's own :class:`~ondine.harness.judge.Verdict`
    so delivery depends on the few attributes it needs, and a verdict changed
    after the fact cannot change what was delivered.
    """

    fields: tuple[JudgedField, ...]

    @classmethod
    def from_verdict(cls, verdict: Verdict) -> JudgedAnswer:
        """Copy what delivery needs from ``verdict``."""
        return cls(
            tuple(
                JudgedField(
                    name=str(item.name),
                    outcome=str(item.outcome),
                    value=item.value,
                    required=bool(item.required),
                    format_fixed=bool(item.format_fixed),
                    failures=_failures(item.reasons),
                    fixed_failures=_failures(item.fixed_reasons),
                )
                for item in verdict.fields
            )
        )

    @property
    def passed(self) -> bool:
        """No required field failed."""
        return not any(f.required and f.outcome == "failed" for f in self.fields)

    @property
    def format_fixed(self) -> bool:
        """Any field was delivered through the format fix."""
        return any(f.format_fixed for f in self.fields)

    @property
    def failures(self) -> tuple[CheckFailure, ...]:
        """Every failed check, in field order."""
        return tuple(failure for f in self.fields for failure in f.failures)

    def get(self, name: str) -> JudgedField | None:
        """The field named ``name``, or None when the answer lacks it."""
        for item in self.fields:
            if item.name == name:
                return item
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuarantineEntry:
    """One rejected, unverified or failed row, with what a client needs to fix it.

    Attributes:
        row: Position in the input.
        status: Rejected, unverified or failed.
        reason: The ledger's reason (the check that blocked delivery last, or
            the failure that lost the row).
        sub_reason: The ledger's sub-reason, e.g. a provider cut a repair short.
        inputs: The row's input values.
        attempts: Every answer received. A raw answer is None past the
            snapshot's raw-answer cap (decision D3) or when none came back.
        failed_checks: Every failed check across every attempt, by attempt.
        models: The models that answered, in attempt order, each once.
        cost: What the row cost across all its attempts.
    """

    row: int
    status: RowStatus
    reason: str
    sub_reason: str | None
    inputs: Mapping[str, Any]
    attempts: tuple[AttemptRecord, ...]
    failed_checks: tuple[CheckFailure, ...]
    models: tuple[str, ...]
    cost: Decimal

    def to_record(self) -> dict[str, Any]:
        """The entry as one flat row of the quarantine table."""
        return {
            "row": self.row,
            "status": self.status.value,
            "reason": self.reason,
            "inputs": dict(self.inputs),
            "raw_answers": [a.raw_answer for a in self.attempts],
            "failed_checks": [c.text for c in self.failed_checks],
            "models": list(self.models),
            "cost": self.cost,
        }


@dataclass(frozen=True)
class RejectionReason:
    """A blocking check, counted over the rows it blocked.

    Grouped by field and rule rather than by sentence: sentences quote the
    answer, so every row's would be unique and nothing would rank.
    """

    field: str | None
    rule: str | None
    count: int
    example: str


@dataclass(frozen=True)
class RulePassRate:
    """How often a rule did not fail, over each row's final judged answer.

    "Final" is the answer the row's status rests on: the delivered attempt,
    else the latest judged one. The judge stops a field at its first failed
    rule, so a later rule of that field counts as passed there: this is a
    "did not fail" rate, which is what points at a broken prompt.
    """

    field: str
    rule: str
    passed: int
    judged: int

    @property
    def rate(self) -> float:
        return self.passed / self.judged if self.judged else 1.0


@dataclass(frozen=True)
class RunReport:
    """What happened in the run, in numbers a client can act on.

    Attributes:
        total: Input rows.
        counts: Rows per final status; they sum to ``total``.
        coverage: (accepted + repaired) / total. None without a contract,
            where nothing is accepted or repaired, and for an empty input.
        top_rejection_reasons: Blocking checks of rejected and unverified
            rows, most frequent first.
        rejected_after_provider_failure: Rejected rows whose repair a provider
            failure cut short; counted apart, as they point at the provider.
        rule_pass_rates: One entry per declared rule, plus any rule seen
            failing, in contract order.
        cost_total: Every attempt of every row.
        cost_per_delivered_row: Total over rows that delivered a value; None
            when none did.
        cost_on_rejected: Spent on rows that were rejected.
        warnings: Signs of a broken prompt or contract, or a stopped run.
        violations: Broken bookkeeping, from the ledger and from delivery.
        stop_reason: Why the run stopped early, if it did.
        invariant_holds: The status counts sum to the row count and agree
            with the snapshot's own counts.
        invariant_line: That fact, as a sentence for the report.
    """

    total: int
    counts: Mapping[RowStatus, int]
    coverage: float | None
    top_rejection_reasons: tuple[RejectionReason, ...]
    rejected_after_provider_failure: int
    rule_pass_rates: tuple[RulePassRate, ...]
    cost_total: Decimal
    cost_per_delivered_row: Decimal | None
    cost_on_rejected: Decimal
    warnings: tuple[str, ...]
    violations: tuple[str, ...]
    stop_reason: StopReason | None
    invariant_holds: bool
    invariant_line: str

    def render(self) -> str:
        """The report as plain text, one fact per line."""
        statuses = ", ".join(f"{s.value} {self.counts[s]}" for s in FINAL_STATUSES)
        lines = [f"Rows: {self.total}", f"Status: {statuses}", self.invariant_line]
        if self.coverage is None:
            why = "no rows" if self.total == 0 else "no contract"
            lines.append(f"Coverage: not measured ({why})")
        else:
            lines.append(f"Coverage: {self.coverage:.1%} (accepted + repaired)")
        if self.stop_reason is not None:
            lines.append(f"Run stopped: {self.stop_reason.value}")
        if self.top_rejection_reasons:
            lines.append("Top rejection reasons:")
            for reason in self.top_rejection_reasons:
                lines.append(
                    f"  {reason.count} x {_rule_name(reason.field, reason.rule)} "
                    f"(e.g. {reason.example})"
                )
        if self.rejected_after_provider_failure:
            lines.append(
                "Rejected after a provider failure cut a repair short: "
                f"{self.rejected_after_provider_failure}"
            )
        if self.rule_pass_rates:
            lines.append("Pass rate per rule:")
            for item in self.rule_pass_rates:
                lines.append(
                    f"  {item.field}.{item.rule}: {item.passed}/{item.judged} "
                    f"({item.rate:.1%})"
                )
        per_row = (
            "n/a"
            if self.cost_per_delivered_row is None
            else str(self.cost_per_delivered_row)
        )
        lines.append(
            f"Cost: total {self.cost_total}; per delivered row {per_row}; "
            f"on rejected rows {self.cost_on_rejected}"
        )
        for title, items in (
            ("Warnings", self.warnings),
            ("Violations", self.violations),
        ):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  - {item}" for item in items)
        return "\n".join(lines)


@dataclass(frozen=True)
class Delivery:
    """The three outputs of a run.

    ``rows`` holds one tuple per input row, values in ``columns`` order, so
    the delivery is immutable and two deliveries compare by value. Read them
    as dicts with :meth:`records`, or as a frame with :meth:`to_pandas`.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    quarantine: tuple[QuarantineEntry, ...]
    report: RunReport

    def records(self) -> list[dict[str, Any]]:
        """Fresh dict rows, the shape ``ResultContainerImpl`` holds."""
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def to_pandas(self) -> Any:
        """The client's dataframe."""
        import pandas as pd

        return pd.DataFrame(self.records(), columns=list(self.columns))

    def quarantine_to_pandas(self) -> Any:
        """The quarantine table, one row per entry."""
        import pandas as pd

        return pd.DataFrame(
            [entry.to_record() for entry in self.quarantine],
            columns=list(_QUARANTINE_COLUMNS),
        )


_QUARANTINE_COLUMNS = (
    "row",
    "status",
    "reason",
    "inputs",
    "raw_answers",
    "failed_checks",
    "models",
    "cost",
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def harness_column_conflicts(columns: Iterable[str]) -> list[str]:
    """The harness column names already taken by ``columns``, in their order.

    The builder should call this with the input and output columns of a
    checked contract, so a clash stops the build before money is spent;
    :func:`deliver` refuses one too, as a last guard.
    """
    taken = set(HARNESS_COLUMNS)
    return [column for column in dict.fromkeys(columns) if column in taken]


def deliver(
    snapshot: LedgerSnapshot,
    judged: Mapping[tuple[int, int], JudgedAnswer],
    input_rows: Sequence[Mapping[str, Any]],
    contract: OutputContract,
    *,
    input_columns: Sequence[str] | None = None,
    rejection_warning_threshold: float = DEFAULT_REJECTION_WARNING_THRESHOLD,
    top_reasons: int = DEFAULT_TOP_REASONS,
) -> Delivery:
    """Build the dataframe, the quarantine and the run report.

    Args:
        snapshot: The ledger's frozen snapshot of the run.
        judged: ``(row, attempt)`` -> :class:`JudgedAnswer`, built from the
            judge's verdicts. Must hold every delivered attempt; should hold
            every judged attempt.
        input_rows: The original input rows, in input order.
        contract: The run's contract: output columns, whether it is checked,
            and the partial-rows policy.
        input_columns: The input's column order. Defaults to the keys of the
            input rows, in first-seen order.
        rejection_warning_threshold: Warn when more than this share of the
            rows is rejected.
        top_reasons: How many rejection reasons the report lists.

    Raises:
        ValueError: The input's row count differs from the snapshot's, or a
            checked contract's harness column clashes with an input or
            output column. Both are wiring defects with no correct output.
    """
    total = len(snapshot.rows)
    if len(input_rows) != total:
        raise ValueError(
            f"the ledger snapshot has {total} rows but {len(input_rows)} input "
            f"rows were given; delivery needs exactly one input row per ledger row"
        )
    inputs = list(input_columns) if input_columns is not None else _keys(input_rows)
    outputs = list(contract.output_columns)
    if contract.checked:
        clashes = harness_column_conflicts([*inputs, *outputs])
        if clashes:
            raise ValueError(
                f"column(s) {clashes} clash with the harness columns "
                f"{list(HARNESS_COLUMNS)}; rename them"
            )
    columns = [*inputs, *(c for c in outputs if c not in inputs)]
    if contract.checked:
        columns.extend(HARNESS_COLUMNS)

    by_row = _by_row(judged, total)
    violations = list(snapshot.violations)
    keep_valid = contract.policy.partial_rows == "keep_valid_fields"
    rows: list[tuple[Any, ...]] = []
    quarantine: list[QuarantineEntry] = []
    for outcome, input_row in zip(snapshot.rows, input_rows, strict=True):
        answers = by_row.get(outcome.row, {})
        view = _RowView(outcome, answers)
        values, delivered_by = view.values(outputs, keep_valid, violations)
        record = {column: input_row.get(column) for column in inputs}
        record.update(values)
        if contract.checked:
            record.update(
                status=outcome.status.value,
                reason=view.reason_column(),
                checks=view.checks_column(),
                repair_method=view.repair_method(),
                delivered_by=delivered_by,
            )
        rows.append(tuple(record[column] for column in columns))
        if outcome.status in _QUARANTINED:
            quarantine.append(view.quarantine_entry(input_row))

    report = _report(
        snapshot,
        by_row,
        contract,
        violations,
        rejection_warning_threshold,
        top_reasons,
    )
    return Delivery(tuple(columns), tuple(rows), tuple(quarantine), report)


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RowView:
    """One row's outcome next to its judged answers, read every way delivery needs."""

    outcome: RowOutcome
    answers: Mapping[int, JudgedAnswer]

    @property
    def final_attempt(self) -> int | None:
        """The attempt the status rests on: delivered, else the latest judged."""
        if self.outcome.delivered_attempt is not None:
            return self.outcome.delivered_attempt
        return max(self.answers, default=None)

    @property
    def final_answer(self) -> JudgedAnswer | None:
        attempt = self.final_attempt
        return None if attempt is None else self.answers.get(attempt)

    def model_of(self, attempt: int | None) -> str | None:
        for record in self.outcome.attempts:
            if record.attempt == attempt:
                return record.model
        return None

    def values(
        self, outputs: Sequence[str], keep_valid: bool, violations: list[str]
    ) -> tuple[dict[str, Any], str | None]:
        """The output values to deliver, and the model that produced them."""
        cells: dict[str, Any] = dict.fromkeys(outputs)
        status = self.outcome.status
        if status in _KEEPS_VALUE:
            attempt = self.outcome.delivered_attempt
            answer = None if attempt is None else self.answers.get(attempt)
            if answer is None:
                violations.append(
                    f"row {self.outcome.row}: attempt {attempt} is delivered "
                    f"({status.value}) but no judged answer was given for it; "
                    f"delivered empty"
                )
                return cells, None
            flagged = status is RowStatus.UNVERIFIED
            for name in outputs:
                item = answer.get(name)
                if item is not None and (
                    item.outcome == "value" or (flagged and item.outcome == "failed")
                ):
                    cells[name] = item.value
            return cells, self.outcome.delivered_by
        if status is RowStatus.REJECTED and keep_valid:
            answer = self.final_answer
            if answer is None:
                return cells, None
            kept = False
            for name in outputs:
                item = answer.get(name)
                if item is not None and item.outcome == "value":
                    cells[name] = item.value
                    kept = True
            return cells, self.model_of(self.final_attempt) if kept else None
        return cells, None

    def path(self) -> str:
        """Each attempt and what came of it, in attempt order."""
        attempts = sorted(
            {record.attempt for record in self.outcome.attempts} | set(self.answers)
        )
        steps = []
        for attempt in attempts:
            answer = self.answers.get(attempt)
            if answer is None:
                label = "answered"
            elif not answer.passed:
                label = "failed check"
            elif answer.format_fixed:
                label = "format fix"
            else:
                label = "passed"
            steps.append(f"attempt {attempt} {label}")
        return " -> ".join(steps)

    def reason_column(self) -> str:
        """The ledger's reason (the check that blocked last), then the path.

        For a format fix, the ledger's note names the fix but not what it
        fixed: the judge keeps that apart, in ``fixed_reasons``. It is added
        here, unless the ledger's reason already holds it.
        """
        reason = self.outcome.reason
        answer = self.final_answer
        if self.outcome.status is RowStatus.REPAIRED and answer is not None:
            before = [
                failure.text
                for item in answer.fields
                if item.format_fixed
                for failure in item.fixed_failures
                if failure.text not in reason
            ]
            if before:
                reason = f"{reason} (before the fix: {'; '.join(before)})"
        path = self.path()
        return f"{reason} [path: {path}]" if reason and path else reason

    def checks_column(self) -> str:
        """Per-field results of the answer the status rests on."""
        answer = self.final_answer
        parts = []
        for name, status in self.outcome.field_statuses:
            item = answer.get(name) if answer is not None else None
            text = status.value
            if status is RowStatus.ACCEPTED and item and item.outcome == "absent":
                text = "absent"
            elif status is RowStatus.REPAIRED and item is not None:
                text = f"repaired ({'format' if item.format_fixed else 're-ask'})"
            parts.append(f"{name}: {text}")
        return "; ".join(parts)

    def repair_method(self) -> str:
        """The step that made the delivered answer: none, format or re-ask.

        A repaired row without a format fix was repaired by a later attempt
        (the ledger's only other repair), which F1/F2 call re-ask.
        """
        status = self.outcome.status
        if status not in (RowStatus.REPAIRED, RowStatus.UNVERIFIED):
            return "none"
        answer = self.final_answer
        if answer is not None and answer.format_fixed:
            return "format"
        if status is RowStatus.REPAIRED:
            return "re-ask"
        earlier = [k for k in self.answers if k < (self.final_attempt or 0)]
        return "re-ask" if earlier else "none"

    def quarantine_entry(self, input_row: Mapping[str, Any]) -> QuarantineEntry:
        outcome = self.outcome
        checks = tuple(
            CheckFailure(f.field, f.rule, f.text, attempt)
            for attempt in sorted(self.answers)
            for f in self.answers[attempt].failures
        )
        if not checks:
            checks = tuple(
                CheckFailure(None, None, text) for text in outcome.failed_checks
            )
        return QuarantineEntry(
            row=outcome.row,
            status=outcome.status,
            reason=outcome.reason,
            sub_reason=outcome.sub_reason,
            inputs=MappingProxyType(dict(input_row)),
            attempts=outcome.attempts,
            failed_checks=checks,
            models=tuple(dict.fromkeys(a.model for a in outcome.attempts)),
            cost=outcome.cost,
        )


# ---------------------------------------------------------------------------
# The run report
# ---------------------------------------------------------------------------


def _report(
    snapshot: LedgerSnapshot,
    by_row: Mapping[int, Mapping[int, JudgedAnswer]],
    contract: OutputContract,
    violations: list[str],
    rejection_threshold: float,
    top_reasons: int,
) -> RunReport:
    total = len(snapshot.rows)
    counts = dict.fromkeys(FINAL_STATUSES, 0)
    for outcome in snapshot.rows:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    summed = sum(counts.values())
    agrees = all(snapshot.counts.get(s, 0) == counts[s] for s in counts)
    holds = summed == total and agrees and set(counts) == set(FINAL_STATUSES)
    if holds:
        line = f"Totals: {summed} of {total} rows have exactly one status (holds)"
    else:
        line = (
            f"Totals: INVARIANT BROKEN: statuses sum to {summed} for {total} rows "
            f"(snapshot counts {dict((s.value, n) for s, n in snapshot.counts.items())})"
        )
        violations.append(line)

    views = [_RowView(o, by_row.get(o.row, {})) for o in snapshot.rows]
    delivered = sum(counts[s] for s in _KEEPS_VALUE)
    cost_total = sum((o.cost for o in snapshot.rows), Decimal("0"))
    cost_rejected = sum(
        (o.cost for o in snapshot.rows if o.status is RowStatus.REJECTED),
        Decimal("0"),
    )
    coverage = None
    if contract.checked and total:
        coverage = (counts[RowStatus.ACCEPTED] + counts[RowStatus.REPAIRED]) / total
    rates = _rule_pass_rates(contract, views) if contract.checked else ()

    warnings = [
        f"rule {r.field}.{r.rule} failed on {r.judged - r.passed} of {r.judged} "
        f"judged rows: a rule failing on most rows usually means a broken prompt "
        f"or contract"
        for r in rates
        if 2 * (r.judged - r.passed) > r.judged
    ]
    rejected = counts[RowStatus.REJECTED]
    if contract.checked and total and rejected / total > rejection_threshold:
        warnings.append(
            f"rejected {rejected} of {total} rows ({rejected / total:.1%}), above "
            f"the {rejection_threshold:.0%} warning threshold"
        )
    if snapshot.stop_reason is not None:
        warnings.append(
            f"the run stopped early ({snapshot.stop_reason.value}); rows it never "
            f"reached are failed"
        )

    return RunReport(
        total=total,
        counts=MappingProxyType(counts),
        coverage=coverage,
        top_rejection_reasons=_top_reasons(views, top_reasons),
        rejected_after_provider_failure=sum(
            1
            for o in snapshot.rows
            if o.status is RowStatus.REJECTED and o.sub_reason == PROVIDER_SUB_REASON
        ),
        rule_pass_rates=rates,
        cost_total=cost_total,
        cost_per_delivered_row=cost_total / delivered if delivered else None,
        cost_on_rejected=cost_rejected,
        warnings=tuple(warnings),
        violations=tuple(violations),
        stop_reason=snapshot.stop_reason,
        invariant_holds=holds,
        invariant_line=line,
    )


def _top_reasons(views: Sequence[_RowView], limit: int) -> tuple[RejectionReason, ...]:
    """Blocking checks of rows that failed one, counted once per row."""
    counts: dict[tuple[str | None, str | None], int] = {}
    examples: dict[tuple[str | None, str | None], str] = {}
    for view in views:
        if view.outcome.status not in _FAILED_A_CHECK:
            continue
        answer = view.final_answer
        keys: dict[tuple[str | None, str | None], str]
        if answer is None:
            keys = {(None, None): view.outcome.reason}
        else:
            blocking = [
                failure for f in answer.fields if f.required for failure in f.failures
            ] or list(answer.failures)
            keys = {}
            for failure in blocking:
                keys.setdefault((failure.field, failure.rule), failure.text)
        for key, text in keys.items():
            counts[key] = counts.get(key, 0) + 1
            examples.setdefault(key, text)
    ranked = sorted(counts, key=lambda k: (-counts[k], k[0] or "", k[1] or ""))
    return tuple(
        RejectionReason(key[0], key[1], counts[key], examples[key])
        for key in ranked[: max(limit, 0)]
    )


def _rule_pass_rates(
    contract: OutputContract, views: Sequence[_RowView]
) -> tuple[RulePassRate, ...]:
    """Pass rate per rule over each row's final judged answer."""
    keys = _declared_rules(contract)
    judged = 0
    failed: dict[tuple[str, str], int] = {}
    for view in views:
        answer = view.final_answer
        if answer is None:
            continue
        judged += 1
        for key in dict.fromkeys(
            (f.field, f.rule)
            for f in answer.failures
            if f.field is not None and f.rule is not None
        ):
            failed[key] = failed.get(key, 0) + 1
    seen = sorted(key for key in failed if key not in keys)
    return tuple(
        RulePassRate(name, rule, judged - failed.get((name, rule), 0), judged)
        for name, rule in [*keys, *seen]
    )


def _declared_rules(contract: OutputContract) -> list[tuple[str, str]]:
    """Every (field, rule) the contract declares, in contract order.

    The rule names are the judge's :class:`~ondine.harness.judge.Reason`
    rules. Schema validators are not listed: which fields they touch is only
    known when one fails, so they appear once they do.
    """
    keys: list[tuple[str, str]] = []
    for item in contract.fields:
        if item.required:
            keys.append((item.name, "required"))
        if item.type not in ("text", "any"):
            keys.append((item.name, "type"))
        if item.allowed is not None:
            keys.append((item.name, "allowed_values"))
        for rule in ("minimum", "maximum", "min_length", "max_length", "pattern"):
            if getattr(item, rule) is not None:
                keys.append((item.name, rule))
    for relation in contract.relations:
        keys.append((relation.left, "relation"))
    return list(dict.fromkeys(keys))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_row(
    judged: Mapping[tuple[int, int], JudgedAnswer], total: int
) -> dict[int, dict[int, JudgedAnswer]]:
    """The judged answers grouped by row, attempts in ascending order.

    Sorted so nothing downstream depends on the mapping's insertion order.
    """
    grouped: dict[int, dict[int, JudgedAnswer]] = {}
    for row, attempt in sorted(judged):
        if 0 <= row < total:
            grouped.setdefault(row, {})[attempt] = judged[(row, attempt)]
    return grouped


def _failures(reasons: Iterable[Any]) -> tuple[CheckFailure, ...]:
    """The judge's reasons as check failures."""
    return tuple(CheckFailure(str(r.field), str(r.rule), str(r.text)) for r in reasons)


def _keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every key of ``rows``, in first-seen order."""
    seen: dict[str, None] = {}
    for row in rows:
        seen.update(dict.fromkeys(row))
    return list(seen)


def _rule_name(name: str | None, rule: str | None) -> str:
    if name is None or rule is None:
        return "unknown rule"
    return f"{name}.{rule}"


__all__ = [
    "DEFAULT_REJECTION_WARNING_THRESHOLD",
    "DEFAULT_TOP_REASONS",
    "HARNESS_COLUMNS",
    "CheckFailure",
    "Delivery",
    "JudgedAnswer",
    "JudgedField",
    "QuarantineEntry",
    "RejectionReason",
    "RulePassRate",
    "RunReport",
    "deliver",
    "harness_column_conflicts",
]
