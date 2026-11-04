"""Delivery: turns a ledger snapshot into what the client receives.

Three outputs, all readings of one :class:`~ondine.harness.ledger.LedgerSnapshot`:

1. **The dataframe.** Every input row, in input order. Under a checked
   contract the output values are kept or blanked by status: accepted and
   repaired rows keep them, unverified rows (``deliver_flagged``) keep them
   flagged, rejected and failed rows are blanked; five columns follow:
   ``status``, ``reason``, ``checks``, ``repair_method``, ``delivered_by``.
2. **The quarantine.** One entry per rejected, unverified or failed row: its
   inputs, every attempt's raw answer (within the snapshot's cap), every
   failed check with its rule and reason, the models used and the cost.
3. **The run report.** Totals per status and the invariant that they sum to
   the row count, coverage, the top rejection reasons, the pass rate per
   rule, cost, and warnings. :func:`merge_reports` combines the reports of
   streaming chunks into the whole run's, with exact totals.

Why delivery is built this way:

* **It decides nothing.** The ledger owns every status; delivery only reads
  the snapshot. Before the harness, a row's fate was re-derived by a blanking
  pass matching marker strings and by a quality report counting empty cells.
  Both become readings here, so they can no longer disagree with the ledger.
* **The values come from the parse stage; everything else from the ledger.**
  The parse stage already wrote each row's judged values into the results,
  so delivery takes them as they are (row -> ``{column: value}``) instead of
  re-judging raw answers, and the pipeline keeps nothing new per row. Which
  of those values ship is the ledger's call: the row's status and, under a
  checked contract, each field's status. The checks, the rules that failed
  and the path taken come from the ledger's copy of each verdict
  (:attr:`RowOutcome.verdicts`).
* **No contract, no change.** Under the unchecked contract the dataframe is
  the input rows with the values given, untouched, and no column is added:
  the values are today's output, whatever the status. The quarantine and the
  report are new outputs next to it, never changes to it.
* **Plain structures inside, pandas only at the edge.** Rows are tuples in a
  fixed column order, like the dict rows ``ResultContainerImpl`` holds;
  :meth:`Delivery.to_pandas` builds a frame only when asked. That keeps
  delivery deterministic and cheap to compare: the same snapshot and values
  always give an equal :class:`Delivery`.
* **Wiring defects are loud, model behaviour is data.** An input whose row
  count differs from the snapshot's, or a column clash with the harness
  columns, raises :class:`ValueError`: no correct frame exists. A delivered
  row without values is delivered empty and listed in the report's
  violations, so a bookkeeping slip never ships a wrong value and never
  hides.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ondine.harness.ledger import PROVIDER_SUB_REASON, StopReason
from ondine.harness.status import FINAL_STATUSES, RowStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ondine.harness.contract import OutputContract
    from ondine.harness.ledger import (
        AttemptRecord,
        Judgement,
        LedgerSnapshot,
        RowOutcome,
    )

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

#: Statuses whose row delivers a value.
_KEEPS_VALUE = frozenset(
    {RowStatus.ACCEPTED, RowStatus.REPAIRED, RowStatus.UNVERIFIED, RowStatus.UNCHECKED}
)
#: Field statuses whose value ships, by row status, under a checked contract.
#: A rejected row ships passing fields only under ``keep_valid_fields``.
_PASSING_FIELDS = frozenset({RowStatus.ACCEPTED, RowStatus.REPAIRED})
_SHIPS = {
    RowStatus.ACCEPTED: _PASSING_FIELDS,
    RowStatus.REPAIRED: _PASSING_FIELDS,
    RowStatus.UNVERIFIED: _PASSING_FIELDS | {RowStatus.UNVERIFIED},
}
#: Statuses that get a quarantine entry.
_QUARANTINED = (RowStatus.REJECTED, RowStatus.UNVERIFIED, RowStatus.FAILED)
#: Statuses whose answer failed a check, for the rejection reasons.
_FAILED_A_CHECK = frozenset({RowStatus.REJECTED, RowStatus.UNVERIFIED})


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckFailure:
    """One failed check: the field, the rule, and the judge's sentence.

    ``field`` and ``rule`` are None only when the ledger kept the sentence
    without them. ``attempt`` is set in the quarantine, where checks from
    every attempt are listed together.
    """

    field: str | None
    rule: str | None
    text: str
    attempt: int | None = None


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
        """The entry as one row of the quarantine table.

        Each failed check stays a small dict with its attempt, field and rule
        next to the sentence, so a client can group the table by rule without
        parsing text.
        """
        return {
            "row": self.row,
            "status": self.status.value,
            "reason": self.reason,
            "sub_reason": self.sub_reason,
            "inputs": dict(self.inputs),
            "raw_answers": [a.raw_answer for a in self.attempts],
            "failed_checks": [
                {
                    "attempt": c.attempt,
                    "field": c.field,
                    "rule": c.rule,
                    "reason": c.text,
                }
                for c in self.failed_checks
            ],
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
            rows, most frequent first, at most the limit asked for.
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
        rejection_reasons: Every blocking check, ranked like
            ``top_rejection_reasons`` but uncapped, so chunk reports merge
            into exact totals (:func:`merge_reports`).
        judged_rows: Rows with a final judged answer: the ``judged`` of
            every pass rate.
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
    rejection_reasons: tuple[RejectionReason, ...] = ()
    judged_rows: int = 0

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
    "sub_reason",
    "inputs",
    "raw_answers",
    "failed_checks",
    "models",
    "cost",
)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def column_order(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every key of ``records``, once each, in first-seen order.

    The column names of rows given as mappings, when nothing else states
    them: the rows of a result, or the input rows handed to :func:`deliver`.
    Reads ``records`` once, so any iterable will do.
    """
    seen: dict[str, None] = {}
    for record in records:
        seen.update(dict.fromkeys(record))
    return list(seen)


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
    values: Mapping[int, Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    contract: OutputContract,
    *,
    input_columns: Sequence[str] | None = None,
    rejection_warning_threshold: float = DEFAULT_REJECTION_WARNING_THRESHOLD,
    top_reasons: int = DEFAULT_TOP_REASONS,
) -> Delivery:
    """Build the dataframe, the quarantine and the run report.

    Args:
        snapshot: The ledger's frozen snapshot of the run (or of one
            streaming chunk).
        values: Snapshot row number (``RowOutcome.row``) -> the output values
            the parse stage wrote for that row, ``{output column: value}``.
            Extra keys (``_row_index``) are ignored; a row or column missing
            here delivers None. Under a checked contract, a failed field's
            value reaches the client only on an unverified row
            (``deliver_flagged``): write the judge's candidate
            (``FieldVerdict.value``) there, since ``Verdict.values`` gives
            None for it. Delivery blanks everything else the status forbids,
            so a candidate written for any other row never ships. The values
            must be the delivered attempt's (``RowOutcome.delivered_attempt``).
            In a pipeline run the latest answer is that attempt, because a
            later attempt only comes from the auto-retry pass, which re-runs
            failed rows only (D-D).
        input_rows: The original input rows, in input order, one per
            snapshot row.
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
    inputs = (
        list(input_columns) if input_columns is not None else column_order(input_rows)
    )
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

    violations = list(snapshot.violations)
    stray = _stray_rows(values, total)
    if stray:
        violations.append(stray)
    keep_valid = contract.policy.partial_rows == "keep_valid_fields"
    rows: list[tuple[Any, ...]] = []
    quarantine: list[QuarantineEntry] = []
    for outcome, input_row in zip(snapshot.rows, input_rows, strict=True):
        view = _RowView(outcome)
        given = values.get(outcome.row)
        record = {column: input_row.get(column) for column in inputs}
        if contract.checked:
            cells, delivered_by = view.cells(outputs, given, keep_valid, violations)
            record.update(cells)
            record.update(
                status=outcome.status.value,
                reason=view.reason_column(),
                checks=view.checks_column(),
                repair_method=view.repair_method(),
                delivered_by=delivered_by,
            )
        else:
            record.update(_as_given(outputs, given, outcome, violations))
        rows.append(tuple(record[column] for column in columns))
        if outcome.status in _QUARANTINED:
            quarantine.append(view.quarantine_entry(input_row))

    report = _report(
        snapshot, contract, violations, rejection_warning_threshold, top_reasons
    )
    return Delivery(tuple(columns), tuple(rows), tuple(quarantine), report)


def merge_reports(
    reports: Iterable[RunReport],
    contract: OutputContract,
    *,
    rejection_warning_threshold: float = DEFAULT_REJECTION_WARNING_THRESHOLD,
    top_reasons: int = DEFAULT_TOP_REASONS,
) -> RunReport:
    """The whole run's report from the reports of its parts (streaming chunks).

    Exact, not estimated: counts, costs, rejection reasons and pass rates are
    sums over the parts, and coverage, the cost per delivered row and the
    warnings are recomputed from those sums. For parts that cover disjoint
    rows, given in input order, the result equals the report of one snapshot
    holding all their rows, except that each part's violations keep a
    ``chunk N:`` prefix so a slip can be traced to its chunk.

    Args:
        reports: The parts' reports, in input order.
        contract: The run's contract, for the declared rules' order.
        rejection_warning_threshold: As for :func:`deliver`.
        top_reasons: As for :func:`deliver`.
    """
    parts = list(reports)
    counts = dict.fromkeys(FINAL_STATUSES, 0)
    reasons: dict[tuple[str | None, str | None], int] = {}
    examples: dict[tuple[str | None, str | None], str] = {}
    failed: dict[tuple[str, str], int] = {}
    seen_rules: set[tuple[str, str]] = set()
    judged = 0
    violations: list[str] = []
    for number, part in enumerate(parts):
        for status in FINAL_STATUSES:
            counts[status] += part.counts.get(status, 0)
        for reason in part.rejection_reasons:
            key = (reason.field, reason.rule)
            reasons[key] = reasons.get(key, 0) + reason.count
            examples.setdefault(key, reason.example)
        for rate in part.rule_pass_rates:
            key = (rate.field, rate.rule)
            seen_rules.add(key)
            failed[key] = failed.get(key, 0) + rate.judged - rate.passed
        judged += part.judged_rows
        violations.extend(f"chunk {number}: {v}" for v in part.violations)

    total = sum(part.total for part in parts)
    holds = all(part.invariant_holds for part in parts)
    stops = {part.stop_reason for part in parts if part.stop_reason is not None}
    declared = _declared_rules(contract)
    extra = sorted(key for key in seen_rules if key not in declared)
    rates = (
        tuple(
            RulePassRate(name, rule, judged - failed.get((name, rule), 0), judged)
            for name, rule in [*declared, *extra]
        )
        if contract.checked
        else ()
    )
    return _assemble(
        contract,
        total=total,
        counts=counts,
        holds=holds,
        reasons=_ranked(reasons, examples),
        rates=rates,
        judged=judged,
        rejected_after_provider_failure=sum(
            part.rejected_after_provider_failure for part in parts
        ),
        cost_total=sum((part.cost_total for part in parts), Decimal("0")),
        cost_on_rejected=sum((part.cost_on_rejected for part in parts), Decimal("0")),
        violations=violations,
        stop_reason=next((r for r in StopReason if r in stops), None),
        rejection_threshold=rejection_warning_threshold,
        top_reasons=top_reasons,
        broken_detail="",
    )


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RowView:
    """One row's outcome, read every way delivery needs."""

    outcome: RowOutcome

    @cached_property
    def verdicts(self) -> dict[int, Judgement]:
        return dict(self.outcome.verdicts)

    @property
    def final_attempt(self) -> int | None:
        """The attempt the status rests on: delivered, else the latest judged."""
        if self.outcome.delivered_attempt is not None:
            return self.outcome.delivered_attempt
        return max((a for a, _ in self.outcome.verdicts), default=None)

    @property
    def final_verdict(self) -> Judgement | None:
        attempt = self.final_attempt
        return None if attempt is None else self.verdicts.get(attempt)

    def model_of(self, attempt: int | None) -> str | None:
        for record in self.outcome.attempts:
            if record.attempt == attempt:
                return record.model
        return None

    def cells(
        self,
        outputs: Sequence[str],
        given: Mapping[str, Any] | None,
        keep_valid: bool,
        violations: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        """Under a checked contract: the values that ship, and who made them.

        A field ships when its row status allows it and its own status (from
        the deciding verdict) is one that row status delivers. A field the
        ledger has no status for takes the row's.
        """
        cells: dict[str, Any] = dict.fromkeys(outputs)
        status = self.outcome.status
        ships = _SHIPS.get(status)
        delivered_by = self.outcome.delivered_by
        if ships is None and status is RowStatus.REJECTED and keep_valid:
            ships = _PASSING_FIELDS
            delivered_by = self.model_of(self.final_attempt)
        if ships is None:
            return cells, None
        if given is None:
            if status in _KEEPS_VALUE:
                violations.append(_no_values(self.outcome))
            return cells, None
        fields = dict(self.outcome.field_statuses)
        kept = False
        for name in outputs:
            if fields.get(name, status) in ships and name in given:
                cells[name] = given[name]
                kept = True
        if status is RowStatus.REJECTED and not kept:
            delivered_by = None
        return cells, delivered_by

    def path(self) -> str:
        """Each attempt and what came of it, in attempt order.

        An attempt that failed and was never judged shows what failed on it
        ("failed (provider)"), so a row an auto-retry recovered or rejected
        still shows the attempt the provider lost. A row whose one attempt
        failed has no path: its reason already names that failure.
        """
        verdicts = self.verdicts
        failed = dict(self.outcome.failed_attempts)
        if not verdicts and not self.outcome.attempts and len(failed) == 1:
            return ""
        attempts = sorted(
            {a.attempt for a in self.outcome.attempts} | set(verdicts) | set(failed)
        )
        steps = []
        for attempt in attempts:
            verdict = verdicts.get(attempt)
            if verdict is None:
                what = failed.get(attempt)
                label = "answered" if what is None else f"failed ({what})"
            elif not verdict.passed:
                label = "failed check"
            elif any(f.format_fixed for f in verdict.fields):
                label = "format fix"
            else:
                label = "passed"
            steps.append(f"attempt {attempt} {label}")
        return " -> ".join(steps)

    def reason_column(self) -> str:
        """The ledger's reason (the check that blocked last), then the path.

        The reason is the ledger's, unchanged: it already names what a format
        fix repaired, so delivery adds only the path.
        """
        reason = self.outcome.reason
        path = self.path()
        return f"{reason} [path: {path}]" if reason and path else reason

    def checks_column(self) -> str:
        """Per-field results of the verdict the status rests on."""
        verdict = self.final_verdict
        by_name = {f.name: f for f in verdict.fields} if verdict is not None else {}
        parts = []
        for name, status in self.outcome.field_statuses:
            item = by_name.get(name)
            text = status.value
            if status is RowStatus.ACCEPTED and item is not None and item.absent:
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
        verdict = self.final_verdict
        if verdict is not None and any(f.format_fixed for f in verdict.fields):
            return "format"
        if status is RowStatus.REPAIRED:
            return "re-ask"
        final = self.final_attempt or 0
        earlier = [a for a, _ in self.outcome.verdicts if a < final]
        return "re-ask" if earlier else "none"

    def failures(self) -> list[CheckFailure]:
        """Every failed check of every judged attempt, by attempt then field."""
        return [
            failure
            for attempt, verdict in self.outcome.verdicts
            for failure in _failures_of(verdict, attempt)
        ]

    def quarantine_entry(self, input_row: Mapping[str, Any]) -> QuarantineEntry:
        outcome = self.outcome
        checks = tuple(self.failures())
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


def _as_given(
    outputs: Sequence[str],
    given: Mapping[str, Any] | None,
    outcome: RowOutcome,
    violations: list[str],
) -> dict[str, Any]:
    """Without a contract: the values as given, whatever the status.

    They are today's output: a row the ledger counts as failed may still
    hold what today's run wrote for it (an error policy's default value), and
    today's frame shows it, so delivery does too.
    """
    if given is None:
        if outcome.status in _KEEPS_VALUE:
            violations.append(_no_values(outcome))
        return dict.fromkeys(outputs)
    return {name: given.get(name) for name in outputs}


def _stray_rows(values: Mapping[int, Any], total: int) -> str:
    """A violation naming the keys of ``values`` that are not a row, or "".

    Such values have nowhere to go: they mean the caller's row numbers and
    the snapshot's disagree (a chunk given the whole run's values, say). One
    line however many there are, naming the first few.
    """
    stray = sorted(
        (key for key in values if not (isinstance(key, int) and 0 <= key < total)),
        key=repr,
    )
    if not stray:
        return ""
    shown = ", ".join(repr(key) for key in stray[:_STRAY_SHOWN])
    more = f" and {len(stray) - _STRAY_SHOWN} more" if len(stray) > _STRAY_SHOWN else ""
    noun = "row" if len(stray) == 1 else "rows"
    return (
        f"values for {noun} {shown}{more} are outside the {total} input rows; ignored"
    )


#: How many stray rows a violation names.
_STRAY_SHOWN = 5


def _no_values(outcome: RowOutcome) -> str:
    return (
        f"row {outcome.row}: delivered ({outcome.status.value}, attempt "
        f"{outcome.delivered_attempt}) but no values were given for it; "
        f"delivered empty"
    )


def _failures_of(verdict: Judgement, attempt: int | None) -> list[CheckFailure]:
    """The failed checks of one verdict, in field order.

    A failed field with no reason gets the ledger's own sentence for it, so
    no failed field goes unlisted.
    """
    failures: list[CheckFailure] = []
    for item in verdict.fields:
        if not item.failed:
            continue
        if not item.reasons:
            failures.append(
                CheckFailure(item.name, None, f"{item.name}: failed a check", attempt)
            )
            continue
        count = len(item.reasons)
        rules = (*item.rules[:count], *(None,) * (count - len(item.rules)))
        failures.extend(
            CheckFailure(item.name, rule or None, text, attempt)
            for text, rule in zip(item.reasons, rules, strict=True)
        )
    return failures


# ---------------------------------------------------------------------------
# The run report
# ---------------------------------------------------------------------------


def _report(
    snapshot: LedgerSnapshot,
    contract: OutputContract,
    violations: list[str],
    rejection_threshold: float,
    top_reasons: int,
) -> RunReport:
    total = len(snapshot.rows)
    counts = dict.fromkeys(FINAL_STATUSES, 0)
    for outcome in snapshot.rows:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    agrees = all(snapshot.counts.get(s, 0) == counts[s] for s in counts)
    holds = agrees and set(counts) == set(FINAL_STATUSES)
    detail = (
        f" (snapshot counts {dict((s.value, n) for s, n in snapshot.counts.items())})"
    )

    views = [_RowView(o) for o in snapshot.rows]
    judged = sum(1 for view in views if view.final_verdict is not None)
    return _assemble(
        contract,
        total=total,
        counts=counts,
        holds=holds,
        reasons=_rejection_reasons(views),
        rates=_rule_pass_rates(contract, views, judged) if contract.checked else (),
        judged=judged,
        rejected_after_provider_failure=sum(
            1
            for o in snapshot.rows
            if o.status is RowStatus.REJECTED and o.sub_reason == PROVIDER_SUB_REASON
        ),
        cost_total=sum((o.cost for o in snapshot.rows), Decimal("0")),
        cost_on_rejected=sum(
            (o.cost for o in snapshot.rows if o.status is RowStatus.REJECTED),
            Decimal("0"),
        ),
        violations=violations,
        stop_reason=snapshot.stop_reason,
        rejection_threshold=rejection_threshold,
        top_reasons=top_reasons,
        broken_detail=detail,
    )


def _assemble(
    contract: OutputContract,
    *,
    total: int,
    counts: dict[RowStatus, int],
    holds: bool,
    reasons: tuple[RejectionReason, ...],
    rates: tuple[RulePassRate, ...],
    judged: int,
    rejected_after_provider_failure: int,
    cost_total: Decimal,
    cost_on_rejected: Decimal,
    violations: list[str],
    stop_reason: StopReason | None,
    rejection_threshold: float,
    top_reasons: int,
    broken_detail: str,
) -> RunReport:
    """The report from its totals: what one snapshot and merged parts share.

    Everything derived (the invariant, coverage, cost per delivered row,
    warnings) is computed here and only here, so a merged report and a
    single snapshot's report cannot derive them differently.
    """
    summed = sum(counts.values())
    holds = holds and summed == total
    if holds:
        line = f"Totals: {summed} of {total} rows have exactly one status (holds)"
    else:
        line = (
            f"Totals: INVARIANT BROKEN: statuses sum to {summed} for {total} rows"
            f"{broken_detail}"
        )
        violations.append(line)

    delivered = sum(counts[s] for s in _KEEPS_VALUE)
    coverage = None
    if contract.checked and total:
        coverage = (counts[RowStatus.ACCEPTED] + counts[RowStatus.REPAIRED]) / total

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
    if stop_reason is not None:
        warnings.append(
            f"the run stopped early ({stop_reason.value}); rows it never "
            f"reached are failed"
        )

    return RunReport(
        total=total,
        counts=MappingProxyType(counts),
        coverage=coverage,
        top_rejection_reasons=reasons[: max(top_reasons, 0)],
        rejected_after_provider_failure=rejected_after_provider_failure,
        rule_pass_rates=rates,
        cost_total=cost_total,
        cost_per_delivered_row=cost_total / delivered if delivered else None,
        cost_on_rejected=cost_on_rejected,
        warnings=tuple(warnings),
        violations=tuple(violations),
        stop_reason=stop_reason,
        invariant_holds=holds,
        invariant_line=line,
        rejection_reasons=reasons,
        judged_rows=judged,
    )


def _rejection_reasons(views: Sequence[_RowView]) -> tuple[RejectionReason, ...]:
    """Blocking checks of rows that failed one, counted once per row, ranked."""
    counts: dict[tuple[str | None, str | None], int] = {}
    examples: dict[tuple[str | None, str | None], str] = {}
    for view in views:
        if view.outcome.status not in _FAILED_A_CHECK:
            continue
        verdict = view.final_verdict
        keys: dict[tuple[str | None, str | None], str]
        if verdict is None:
            keys = {(None, None): view.outcome.reason}
        else:
            required = {f.name for f in verdict.fields if f.required}
            failures = _failures_of(verdict, None)
            blocking = [f for f in failures if f.field in required] or failures
            keys = {}
            for failure in blocking:
                keys.setdefault((failure.field, failure.rule), failure.text)
        for key, text in keys.items():
            counts[key] = counts.get(key, 0) + 1
            examples.setdefault(key, text)
    return _ranked(counts, examples)


def _ranked(
    counts: Mapping[tuple[str | None, str | None], int],
    examples: Mapping[tuple[str | None, str | None], str],
) -> tuple[RejectionReason, ...]:
    """Most frequent first; ties by field, then rule."""
    ranked = sorted(counts, key=lambda k: (-counts[k], k[0] or "", k[1] or ""))
    return tuple(
        RejectionReason(key[0], key[1], counts[key], examples[key]) for key in ranked
    )


def _rule_pass_rates(
    contract: OutputContract, views: Sequence[_RowView], judged: int
) -> tuple[RulePassRate, ...]:
    """Pass rate per rule over each row's final judged answer."""
    keys = _declared_rules(contract)
    failed: dict[tuple[str, str], int] = {}
    for view in views:
        verdict = view.final_verdict
        if verdict is None:
            continue
        for key in dict.fromkeys(
            (f.field, f.rule)
            for f in _failures_of(verdict, None)
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
    "QuarantineEntry",
    "RejectionReason",
    "RulePassRate",
    "RunReport",
    "column_order",
    "deliver",
    "harness_column_conflicts",
    "merge_reports",
]
