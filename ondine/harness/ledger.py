"""The row ledger: the single record of what happened to every input row.

Before the harness, a row's fate was known in five places — the execution
context's lost-row set and counters, the result's error list, a post-step that
blanked cells by matching marker strings, the quality report counting empty
cells, and grounding's own columns. Each knew a slice, and they could disagree.
The ledger replaces all of them with one owner: stages *report events* here,
and only the ledger turns those events into a status.

How it works, and why it is built this way:

* **Every row exists from the start, as pending.** The ledger is created with
  the input's row count, so a row nobody reported on cannot go missing: at
  snapshot time it becomes ``failed``, "run stopped before this row". No code
  path can produce a row without a status.
* **Events are stored, statuses are derived.** Async model calls finish in any
  order, and sub-pipelines (streaming chunks, the auto-retry pass) report
  later still. Rather than updating a status as each event lands — which makes
  the result depend on arrival order — the ledger keeps the events and decides
  a row's status from all of them at once, ordered by attempt number. The same
  events in any order give the same statuses.
* **The status rules live here and nowhere else.** Failed vs rejected, the
  worst-required-field rule, the provider sub-reason, which answer an
  unverified row delivers, early stop: changing one of them changes this
  module only.
* **Today's views are readings of the ledger.** The error list,
  ``lost_row_indices`` and ``is_complete`` are computed from the snapshot, and
  with no contract they say what ``ExecutionContext`` and ``ExecutionResult``
  say today (the #262/#312 guard). Rejected rows get their own list. Where the
  snapshot's view is deliberately truer than today's (row order, recoveries,
  rows a stopped run never reached), :meth:`RowLedger.reported_losses` reads
  the same events exactly as ``ExecutionContext`` used to keep them, so the
  migration can change today's outputs one deliberate step at a time.

The ledger reads the judge's own :class:`~ondine.harness.judge.Verdict`, with
no adapter and no second vocabulary: ``Verdict.passed`` for the row, and per
``FieldVerdict`` its ``name``, ``outcome``, ``required``, ``format_fixed`` and
``reasons``. A field passes unless its outcome is ``"failed"``; an ``"absent"``
optional field is a valid answer. The ledger copies only those attributes, so
the judge may add fields to its verdict without touching this module.

The policy has one source too: :meth:`LedgerPolicy.from_contract` reads
whether the run is checked and its run policy from the contract, and adds only
the raw-answer cap (decision D3), which is a storage limit, not a rule.

A broken invariant (an event for a row outside the input, two different
verdicts for one attempt, a verdict that contradicts its own fields) is a
defect upstream, never model behaviour. A ``strict`` ledger raises
:class:`LedgerInvariantError` at once — that is how tests run. Otherwise the
violation is kept and shown in the snapshot, so the run report states it
loudly instead of the run crashing on a bookkeeping slip.
"""

from __future__ import annotations

import operator
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from itertools import islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from ondine.core.models import ErrorInfo
from ondine.harness.contract import RunPolicy, policy_problems
from ondine.harness.status import FINAL_STATUSES, RowStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ondine.harness.contract import OutputContract
    from ondine.harness.judge import Verdict

# ---------------------------------------------------------------------------
# Public vocabulary
# ---------------------------------------------------------------------------


class LedgerInvariantError(RuntimeError):
    """A rule the ledger guarantees was broken by whoever reported to it.

    Raised only by a ``strict`` ledger. Never caused by a model's answer — a
    bad answer is a verdict, and a verdict is data.
    """


class StopReason(str, Enum):
    """Why a run stopped before reaching every row.

    Listed in precedence order: when a run records more than one (an early
    stop, then a crash while winding down), the rows never reached are blamed
    on the first one listed here. Policy decisions come before accidents, and
    the choice cannot depend on which report landed first.
    """

    EARLY_STOP = "early stop"
    BUDGET_CAP = "budget cap"
    CRASH = "crash"


#: ``sub_reason`` of a rejected row whose repair a provider failure cut short.
#: The row stays rejected — its answer did fail a check — but the report counts
#: these separately, because they say something about the provider, not the
#: prompt.
PROVIDER_SUB_REASON = "provider"


#: Raw answers kept for this many quarantined rows by default (decision D3).
DEFAULT_RAW_ANSWER_CAP = 1000


@dataclass(frozen=True)
class LedgerPolicy:
    """What the ledger needs from a contract to decide statuses.

    Build it with :meth:`from_contract`, so that "checked" and the run policy
    have one source — the contract — and cannot disagree with the judge's
    verdicts. The run policy is the contract's own :class:`RunPolicy`, held
    rather than copied field by field, and validated by the same rule.

    Attributes:
        checked: False for the unchecked contract (none declared). Answered
            rows are then ``unchecked``, nothing can be rejected and early
            stop is off, which keeps a run without a contract identical to
            today.
        run: The contract's run policy.
        raw_answer_cap: Raw answers are kept for this many quarantined rows
            (rejected, unverified, failed), the first ones by input order.
            Statuses and counts are exact regardless (decision D3). A storage
            limit, not a rule, so it is not part of the contract.
    """

    checked: bool
    run: RunPolicy
    raw_answer_cap: int = DEFAULT_RAW_ANSWER_CAP

    def __post_init__(self) -> None:
        problems = policy_problems(self.run) if self.checked else []
        if (
            isinstance(self.raw_answer_cap, bool)
            or not isinstance(self.raw_answer_cap, int)
            or self.raw_answer_cap < 0
        ):
            problems.append(
                f"raw_answer_cap {self.raw_answer_cap!r}: use 0 or more rows"
            )
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_contract(
        cls,
        contract: OutputContract,
        *,
        raw_answer_cap: int = DEFAULT_RAW_ANSWER_CAP,
    ) -> LedgerPolicy:
        """The ledger policy of a run under ``contract``."""
        return cls(contract.checked, contract.policy, raw_answer_cap)

    @property
    def deliver_flagged(self) -> bool:
        """The client opted in to delivering failed values as ``unverified``."""
        return self.checked and self.run.on_failure == "deliver_flagged"

    @property
    def early_stop_fraction(self) -> float | None:
        """Stop when *more than* this share of the window is rejected; None: off.

        Always off without a contract: nothing can be rejected there.
        """
        return self.run.early_stop_fraction if self.checked else None

    @property
    def early_stop_window(self) -> int:
        """How many rows, by input order, early stop looks at."""
        return self.run.early_stop_window


@dataclass(frozen=True)
class AttemptRecord:
    """One billed model answer for a row.

    ``raw_answer`` is whatever reference the caller reported — ideally a cache
    key, not the text, since the ledger holds it until the snapshot. In a
    snapshot it is kept only for quarantined rows within the D3 cap, and is
    None everywhere else.
    """

    attempt: int
    model: str
    tokens: int
    cost: Decimal
    raw_answer: Any = None


@dataclass(frozen=True)
class RowOutcome:
    """Everything the ledger concluded about one row.

    Attributes:
        row: Position in the full input.
        status: The row's one status.
        reason: Human-readable cause. Empty for a clean accepted or unchecked
            row; otherwise the check that blocked delivery last, the repair
            taken, or the failure that lost the row.
        sub_reason: :data:`PROVIDER_SUB_REASON` when a rejected (or
            unverified) row's repair was cut short by a provider failure.
        delivered_attempt: The attempt whose value delivery should use, or
            None for a row delivered empty. Attempts from an auto-retry pass
            are numbered after the parent's (see :meth:`RowLedger.merge`).
        delivered_by: The model that produced that attempt, when known.
        field_statuses: Per-field status of the deciding verdict, for the
            checks column. Empty without a contract.
        failed_checks: Every failed check's reason, across every attempt.
        attempts: Every answer received, by attempt number.
    """

    row: int
    status: RowStatus
    reason: str = ""
    sub_reason: str | None = None
    delivered_attempt: int | None = None
    delivered_by: str | None = None
    field_statuses: tuple[tuple[str, RowStatus], ...] = ()
    failed_checks: tuple[str, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()

    @property
    def cost(self) -> Decimal:
        """What this row cost across all its attempts."""
        return sum((a.cost for a in self.attempts), Decimal("0"))


@dataclass(frozen=True)
class LedgerSnapshot:
    """The ledger frozen for delivery: one final status per row.

    No row is pending here — a row nobody decided is ``failed``, "run stopped
    before this row" — so ``counts`` always sums to the row count. Later
    events reach the ledger, never an existing snapshot.

    ``errors``, ``skipped_rows``, ``failed_rows``, ``lost_row_indices`` and
    ``is_complete`` are today's views. Without a contract they match what
    ``ExecutionContext`` / ``ExecutionResult`` report today, with two
    deliberate differences: ``errors`` is ordered by row rather than by
    arrival (arrival order is not reproducible), and ``lost_row_indices`` is
    exact even past the error-recording limit.
    """

    rows: tuple[RowOutcome, ...]
    counts: Mapping[RowStatus, int]
    stop_reason: StopReason | None
    violations: tuple[str, ...]
    errors: tuple[ErrorInfo, ...]
    skipped_rows: int
    failed_rows: int

    @property
    def lost_row_indices(self) -> list[int]:
        """Rows that produced no usable answer (status ``failed``), sorted."""
        return [r.row for r in self.rows if r.status is RowStatus.FAILED]

    @property
    def rejected_row_indices(self) -> list[int]:
        """Rows whose answer failed a check, sorted — kept apart from losses (#312)."""
        return [r.row for r in self.rows if r.status is RowStatus.REJECTED]

    @property
    def is_complete(self) -> bool:
        """True when every row delivered a value that passed, or was never checked.

        Without a contract this is today's meaning — nothing skipped or failed.
        With one, complete means every row accepted or repaired: a rejected or
        unverified row is a hole in the result even though the run finished.
        """
        return all(r.status in _COMPLETE_STATUSES for r in self.rows)


@dataclass(frozen=True)
class ReportedLosses:
    """The losses reported to one ledger, read the way ``ExecutionContext`` kept them.

    Migration step 1 makes the ledger the only record of row fate, but the
    result's error list and loss counters must not change until a later step
    changes them on purpose (the parity net freezes them). The snapshot's own
    view decides differently by design, so this second view reads the same
    events with today's rules:

    * a row is lost the first time a failure is reported for it, and that
      first report is its cause, whatever arrives later;
    * ``errors`` is in report order and bounded; the counts stay exact;
    * only failures reported to this ledger count. A sub-pipeline's failures
      arrive by :meth:`RowLedger.merge` and never reached today's context, and
      a recovery merged later does not take a reported row back;
    * a row nothing reported on (never reached) is not here.

    Attributes:
        errors: One entry per lost row, in report order, at most the limit
            asked for.
        skipped_rows: Rows whose first report was a provider failure.
        failed_rows: Rows whose first report was a stage failure or an
            unparseable answer.
    """

    errors: tuple[ErrorInfo, ...]
    skipped_rows: int
    failed_rows: int


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


class RowLedger:
    """The only owner of every row's fate in one run (or one sub-pipeline).

    Create it with the input's row count and the contract's policy; report
    events as they happen; ask it for a row's outcome, the counts, whether
    early stop is reached, or a frozen snapshot. Thread-safe: stages on
    different threads may report into the same ledger.

    Polling is cheap. Each row's live status is cached and re-derived only
    after a new event for that row, so :meth:`early_stop_reached` and
    :meth:`counts` cost the rows changed since the last call, not the whole
    input, and the model-call stage may ask once per batch.
    """

    #: Error entries kept in the back-compatibility error list. Same bound and
    #: reason as ``ExecutionContext.MAX_RECORDED_ROW_ERRORS``: enough to see a
    #: pattern, never the reason a run runs out of memory. Counts stay exact.
    MAX_RECORDED_ROW_ERRORS: ClassVar[int] = 1000

    def __init__(
        self,
        row_count: int,
        policy: LedgerPolicy,
        *,
        strict: bool = False,
    ) -> None:
        if row_count < 0:
            raise ValueError(f"row_count cannot be negative, got {row_count}")
        self._row_count = row_count
        self._policy = policy
        self._strict = strict
        # Only rows with at least one event have an entry; the rest are pending.
        self._rows: dict[int, _RowEvents] = {}
        self._stops: set[StopReason] = set()
        self._violations: list[str] = []
        # Stage failures carry no attempt; this numbers them in report order,
        # the one tie the pipeline's own order decides (see _lost).
        self._failure_sequence = 0
        # Live status of each row with events, re-derived only when ``_dirty``
        # says an event changed it. A row's live status depends on its own
        # events and the fixed policy alone (the stop reason only matters to a
        # row with no events, at snapshot), so the cache is exact.
        self._live: dict[int, RowStatus] = {}
        self._dirty: set[int] = set()
        self._tally: dict[RowStatus, int] = dict.fromkeys(RowStatus, 0)
        # First direct loss report per row, for reported_losses(). A dict keeps
        # insertion order, which is report order, so the bounded error list is
        # a slice and never a sort; the skip count keeps the counts O(1).
        self._reported: dict[int, _Cause] = {}
        self._reported_skipped = 0
        self._lock = threading.Lock()

    @property
    def row_count(self) -> int:
        """How many input rows this ledger covers."""
        return self._row_count

    @property
    def policy(self) -> LedgerPolicy:
        """The policy this ledger decides statuses under."""
        return self._policy

    def widen(self, row_count: int) -> None:
        """Cover ``row_count`` rows; the rows added start pending.

        For a caller that learns the input's size late: a context driven
        outside a pipeline (stage tests, backends with no loader), or a
        streamed file whose length is known only chunk by chunk. A pipeline
        that knows its input opens the ledger at the exact size instead.
        Never shrinks: an event already reported cannot fall outside.
        """
        with self._lock:
            self._row_count = max(self._row_count, row_count)

    # -- events -------------------------------------------------------------

    def record_answer(
        self,
        row: int,
        attempt: int,
        *,
        model: str,
        tokens: int,
        cost: Decimal,
        raw_answer: Any = None,
    ) -> None:
        """A model answered ``row`` on ``attempt``.

        Reporting the same attempt twice (resume replaying a cached answer)
        keeps one record, so cost is never counted twice. Which one is decided
        by content, never by arrival (see :func:`_answer_rank`).
        """
        with self._lock:
            index = self._checked_row(row)
            if index is None or not self._checked_attempt(attempt):
                return
            record = AttemptRecord(attempt, model, tokens, cost, raw_answer)
            self._add_answer(index, record, merged=False)

    def record_verdict(
        self, row: int, attempt: int, verdict: Verdict, *, stage: str
    ) -> None:
        """The judge's verdict on ``row``'s answer from ``attempt``.

        ``stage`` is the stage that judged the answer (today, the parse
        stage). Without a contract an unparseable answer is a lost row, and
        today's error list names that stage, so the ledger must know it.
        """
        judgement, contradiction = _Judgement.of(verdict, stage)
        with self._lock:
            index = self._checked_row(row)
            if index is None or not self._checked_attempt(attempt):
                return
            if bool(verdict.checked) != self._policy.checked:
                self._violation(
                    f"row {row} attempt {attempt}: a "
                    f"{'checked' if verdict.checked else 'unchecked'} verdict in "
                    f"a{'' if self._policy.checked else 'n un'}checked run; the "
                    f"judge and the ledger were given different contracts"
                )
            if contradiction:
                self._violation(
                    f"row {row} attempt {attempt}: the verdict says passed "
                    f"but a required field failed; treated as not deliverable"
                )
            self._add_verdict(index, attempt, judgement)
            if not self._policy.checked and not judgement.passed:
                # Without a contract an unparseable answer is a lost row, as
                # the parse step has always recorded it.
                self._report_loss(index, judgement.unparseable_cause())

    def record_provider_failure(
        self,
        rows: Iterable[int],
        attempt: int,
        *,
        stage: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        """The provider gave no answer for ``rows`` on ``attempt``.

        Takes several rows because a failed mega-batch loses all of its rows
        at once. ``message`` is kept verbatim: it is the provider's own words,
        and the most useful thing a client can read about a loss.
        """
        with self._lock:
            if not self._checked_attempt(attempt):
                return
            for row in rows:
                checked = self._checked_row(row)
                if checked is not None:
                    failure = _Failure(attempt, True, stage, message, retryable)
                    self._add_failure(checked, failure)
                    self._report_loss(checked, failure.cause)

    def record_stage_failure(
        self, rows: Iterable[int], *, stage: str, message: str
    ) -> None:
        """A stage could not process ``rows`` (prompt formatting, batch splitting).

        A stage failure carries no attempt number; it is placed at the first
        attempt of its pass. When a provider failure on that same attempt is
        also reported for the row, the provider is the cause: the stage merely
        tripped over the hole the provider left, as ``ExecutionContext`` has
        always ruled.
        """
        with self._lock:
            for row in rows:
                checked = self._checked_row(row)
                if checked is not None:
                    failure = _Failure(1, False, stage, message, False)
                    self._add_failure(checked, failure)
                    self._report_loss(checked, failure.cause)

    def record_run_stopped(self, reason: StopReason) -> None:
        """The run stopped. Rows still undecided at snapshot become failed.

        In-flight answers may still be reported afterwards and count as usual;
        only rows with no event at all are blamed on the stop.
        """
        with self._lock:
            self._stops.add(StopReason(reason))

    def merge(
        self,
        child: RowLedger,
        parent_rows: Sequence[int],
        *,
        retry: bool = False,
    ) -> None:
        """Fold a sub-pipeline's ledger into this one.

        Args:
            child: The ledger of a streaming chunk or an auto-retry pass.
            parent_rows: ``parent_rows[i]`` is this ledger's row for the
                child's row ``i``. Must cover every child row, with no parent
                row named twice.
            retry: True when the child re-ran these rows *after* this ledger's
                own attempts (the auto-retry pass). Its pipeline numbers
                attempts from 1 again, so they are renumbered to follow this
                ledger's last attempt for each row; a recovery then is the
                latest attempt and replaces the failure. False for a streaming
                chunk, whose attempts are compared as reported, so a row seen
                by two chunks keeps the higher attempt.

        Two ledgers reporting the same row on the same attempt with different
        content (two answers, or two different verdicts) means two pipelines
        ran one row: a violation. The record kept is chosen by content, so
        the result never depends on which ledger merged first.

        A child's stop reason and its violations carry over to this ledger.
        """
        answers, verdicts, failures, stops, violations = child._export()
        mapping = list(parent_rows)
        with self._lock:
            if len(mapping) != child.row_count:
                self._violation(
                    f"merge: the mapping names {len(mapping)} parent rows "
                    f"for a child of {child.row_count} rows; merge refused"
                )
                return
            maybe_checked = [self._checked_row(row) for row in mapping]
            checked = [row for row in maybe_checked if row is not None]
            if len(checked) != len(maybe_checked):
                return  # each bad row is already a recorded violation
            if len(set(checked)) != len(checked):
                self._violation(
                    "merge: two child rows map to the same parent row; merge refused"
                )
                return
            offsets = {
                child_row: self._last_attempt(parent_row) if retry else 0
                for child_row, parent_row in enumerate(checked)
            }
            for child_row, record in answers:
                attempt = record.attempt + offsets[child_row]
                self._add_answer(
                    checked[child_row], replace(record, attempt=attempt), merged=True
                )
            for child_row, attempt, judgement in verdicts:
                self._add_verdict(
                    checked[child_row], attempt + offsets[child_row], judgement
                )
            # In the child's report order, so stage failures keep their
            # pipeline order after the merge.
            for child_row, failure in sorted(failures, key=lambda f: f[1].sequence):
                attempt = failure.attempt + offsets[child_row]
                self._add_failure(checked[child_row], replace(failure, attempt=attempt))
            self._stops |= stops
            for violation in violations:
                self._violation(f"merged ledger: {violation}")

    # -- queries ------------------------------------------------------------

    def outcome(self, row: int) -> RowOutcome:
        """The live outcome of ``row``. ``pending`` while nothing has decided it."""
        if not 0 <= row < self._row_count:
            raise IndexError(f"row {row} is outside 0..{self._row_count - 1}")
        with self._lock:
            outcome, _ = _decide(
                row, self._rows.get(row), self._policy, self._stop_reason(), False
            )
        return outcome

    def counts(self) -> Mapping[RowStatus, int]:
        """Live counts per status, pending included. They sum to the row count.

        Cost: re-derives only the rows with an event since the last call to
        this or :meth:`early_stop_reached` (amortised, one derivation per
        event), plus a copy of the seven-entry tally. Safe to poll.
        """
        with self._lock:
            self._refresh()
            tally = {
                status: self._tally[status]
                for status in (*FINAL_STATUSES, RowStatus.PENDING)
            }
            tally[RowStatus.PENDING] += self._row_count - len(self._live)
        return MappingProxyType(tally)

    def early_stop_reached(self) -> bool:
        """True when more than the policy's share of the first rows is rejected.

        "First rows" means by input order, and the answer waits until every
        one of them is decided: rows 50+ finishing first must not stand in for
        rows 0-49, and a row still in flight could change the count.

        Cost: the rows changed since the last poll, plus a look-up per row of
        the window; no row is re-derived without a new event.
        """
        fraction = self._policy.early_stop_fraction
        if fraction is None:
            return False
        window = min(self._row_count, self._policy.early_stop_window)
        if window == 0:
            return False
        with self._lock:
            self._refresh()
            rejected = 0
            for row in range(window):
                events = self._rows.get(row)
                if events is None or not events.decided:
                    return False
                rejected += self._live[row] is RowStatus.REJECTED
        return rejected > fraction * window

    def snapshot(self) -> LedgerSnapshot:
        """Freeze the ledger: every row final, counts summing to the row count."""
        with self._lock:
            stop = self._stop_reason()
            rows: list[RowOutcome] = []
            errors: list[ErrorInfo] = []
            tally = dict.fromkeys(FINAL_STATUSES, 0)
            skipped = failed = quarantined = 0
            for row in range(self._row_count):
                outcome, cause = _decide(
                    row, self._rows.get(row), self._policy, stop, True
                )
                keep_raw = False
                if outcome.status in _QUARANTINED_STATUSES:
                    quarantined += 1
                    keep_raw = quarantined <= self._policy.raw_answer_cap
                if not keep_raw and outcome.attempts:
                    outcome = replace(
                        outcome,
                        attempts=tuple(
                            replace(a, raw_answer=None) for a in outcome.attempts
                        ),
                    )
                if cause is not None:
                    if cause.error_type == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                    if len(errors) < self.MAX_RECORDED_ROW_ERRORS:
                        errors.append(cause.error_info(row))
                tally[outcome.status] += 1
                rows.append(outcome)
            if sum(tally.values()) != self._row_count:  # pragma: no cover
                self._violation(
                    f"counts sum to {sum(tally.values())}, not {self._row_count}"
                )
            return LedgerSnapshot(
                rows=tuple(rows),
                counts=MappingProxyType(tally),
                stop_reason=stop,
                violations=tuple(self._violations),
                errors=tuple(errors),
                skipped_rows=skipped,
                failed_rows=failed,
            )

    def reported_losses(self, limit: int | None = None) -> ReportedLosses:
        """The losses reported to this ledger, by today's rules (see the class).

        Args:
            limit: Error entries to build, the first ones by report order;
                ``None`` for :attr:`MAX_RECORDED_ROW_ERRORS`. ``0`` reads the
                counts alone.

        Cost: the entries built, never the input's size, so a context may
        read its counters at every checkpoint.
        """
        if limit is None:
            limit = self.MAX_RECORDED_ROW_ERRORS
        with self._lock:
            errors = tuple(
                cause.error_info(row)
                for row, cause in islice(self._reported.items(), max(limit, 0))
            )
            skipped = self._reported_skipped
            failed = len(self._reported) - skipped
        return ReportedLosses(errors=errors, skipped_rows=skipped, failed_rows=failed)

    # -- internals ----------------------------------------------------------

    def _report_loss(self, row: int, cause: _Cause) -> None:
        """Keep ``cause`` if it is the first loss reported for ``row``."""
        if row in self._reported:
            return
        self._reported[row] = cause
        if cause.error_type == "skipped":
            self._reported_skipped += 1

    def _events(self, row: int) -> _RowEvents:
        """``row``'s events, for a caller about to add one: its status is stale."""
        events = self._rows.get(row)
        if events is None:
            events = self._rows[row] = _RowEvents()
        stale = self._live.pop(row, None)
        if stale is not None:
            self._tally[stale] -= 1
        self._dirty.add(row)
        return events

    def _refresh(self) -> None:
        """Re-derive the live status of every row with a new event."""
        for row in self._dirty:
            outcome, _ = _decide(row, self._rows[row], self._policy, None, False)
            self._live[row] = outcome.status
            self._tally[outcome.status] += 1
        self._dirty.clear()

    def _add_answer(self, row: int, record: AttemptRecord, *, merged: bool) -> None:
        """Store an answer; one per attempt, the same one in any arrival order.

        A repeat inside one ledger is resume replaying a cached answer. A
        repeat with different content arriving by merge means two
        sub-pipelines answered one row on one attempt: a defect upstream.
        """
        answers = self._events(row).answers
        existing = answers.get(record.attempt)
        if existing is None:
            answers[record.attempt] = record
            return
        if merged and _answer_rank(existing) != _answer_rank(record):
            self._violation(
                f"row {row} attempt {record.attempt}: two merged ledgers each "
                f"answered it ({existing.model}, {record.model}); kept one by "
                f"content"
            )
        answers[record.attempt] = max(existing, record, key=_answer_rank)

    def _add_failure(self, row: int, failure: _Failure) -> None:
        self._failure_sequence += 1
        self._events(row).failures.append(
            replace(failure, sequence=self._failure_sequence)
        )

    def _add_verdict(self, row: int, attempt: int, judgement: _Judgement) -> None:
        """Store a verdict; a different one for the same attempt is a conflict.

        The same verdict twice is normal (resume re-judges cached answers, and
        the judge is deterministic). Two different ones mean a defect upstream;
        the safer one is kept — the one that delivers nothing it doubts — so
        the conflict can only ever blank a value, never ship a bad one. Every
        choice here is by content, so arrival order never shows.
        """
        verdicts = self._events(row).verdicts
        existing = verdicts.get(attempt)
        if existing is None:
            verdicts[attempt] = judgement
            return
        if existing == judgement:
            # Equal verdicts may differ only in the stage that reported them.
            verdicts[attempt] = min(existing, judgement, key=lambda j: j.stage)
            return
        self._violation(
            f"row {row} attempt {attempt}: two different verdicts; kept the safer one"
        )
        verdicts[attempt] = min(existing, judgement, key=_Judgement.safety_rank)

    def _last_attempt(self, row: int) -> int:
        events = self._rows.get(row)
        return 0 if events is None else events.last_attempt

    def _stop_reason(self) -> StopReason | None:
        for reason in StopReason:
            if reason in self._stops:
                return reason
        return None

    def _checked_row(self, row: Any) -> int | None:
        try:
            index = operator.index(row)
        except TypeError:
            index = None
        if index is None or not 0 <= index < self._row_count:
            self._violation(
                f"event for row {row!r}, outside the input's rows "
                f"0..{self._row_count - 1}; ignored"
            )
            return None
        return index

    def _checked_attempt(self, attempt: Any) -> bool:
        if isinstance(attempt, int) and attempt >= 1:
            return True
        self._violation(f"attempt numbers start at 1, got {attempt!r}; event ignored")
        return False

    def _violation(self, message: str) -> None:
        if self._strict:
            raise LedgerInvariantError(message)
        self._violations.append(message)

    def _export(self) -> tuple[list, list, list, set[StopReason], list[str]]:
        """A consistent copy of this ledger's events, for a parent's merge."""
        with self._lock:
            answers = [
                (row, record)
                for row, events in self._rows.items()
                for record in events.answers.values()
            ]
            verdicts = [
                (row, attempt, judgement)
                for row, events in self._rows.items()
                for attempt, judgement in events.verdicts.items()
            ]
            failures = [
                (row, failure)
                for row, events in self._rows.items()
                for failure in events.failures
            ]
            return answers, verdicts, failures, set(self._stops), list(self._violations)


# ---------------------------------------------------------------------------
# Internal event records
# ---------------------------------------------------------------------------

_COMPLETE_STATUSES = frozenset(
    {RowStatus.ACCEPTED, RowStatus.REPAIRED, RowStatus.UNCHECKED}
)
_QUARANTINED_STATUSES = frozenset(
    {RowStatus.REJECTED, RowStatus.UNVERIFIED, RowStatus.FAILED}
)
# Worst-required-field order: the row takes the highest rank among its
# required fields.
_FIELD_RANK = {
    RowStatus.ACCEPTED: 0,
    RowStatus.REPAIRED: 1,
    RowStatus.UNVERIFIED: 2,
    RowStatus.REJECTED: 3,
}


@dataclass(frozen=True)
class _FieldJudgement:
    name: str
    required: bool
    #: The judge's outcome was "failed"; a value or a valid absence is not.
    failed: bool
    reasons: tuple[str, ...]
    format_fixed: bool

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return self.reasons or (f"{self.name}: failed a check",)


@dataclass(frozen=True)
class _Judgement:
    """A verdict copied into the ledger's own immutable form.

    Copying, rather than keeping the judge's object, makes the ledger immune
    to a verdict mutated after the fact, and gives two reports of the same
    verdict a well-defined equality. ``stage`` (who reported it) is not part
    of that equality: the same verdict from two reports is one verdict.
    """

    #: As ``Verdict.passed``: no required field failed, so it may be delivered.
    passed: bool
    fields: tuple[_FieldJudgement, ...]
    stage: str = field(default="", compare=False)

    @classmethod
    def of(cls, verdict: Verdict, stage: str) -> tuple[_Judgement, bool]:
        """The ledger's copy of ``verdict``, and whether it contradicted itself.

        A field passes unless its outcome is ``"failed"``: a value and a valid
        absence both pass.
        """
        fields = tuple(
            _FieldJudgement(
                name=str(item.name),
                required=bool(item.required),
                failed=item.outcome == "failed",
                reasons=tuple(str(r) for r in item.reasons),
                format_fixed=bool(item.format_fixed),
            )
            for item in verdict.fields
        )
        required_failed = any(f.required and f.failed for f in fields)
        contradiction = bool(verdict.passed) and required_failed
        judgement = cls(bool(verdict.passed) and not required_failed, fields, stage)
        return judgement, contradiction

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """Every failed check's reason, in field order, each sentence once.

        Under the unchecked contract the judge puts today's single parse
        message on every column; counting it once per column would turn one
        parse failure into several, unlike today's error list.
        """
        return tuple(
            dict.fromkeys(
                reason for f in self.fields if f.failed for reason in f.failure_reasons
            )
        )

    @property
    def failed_check_count(self) -> int:
        return len(self.failure_reasons)

    def unparseable_cause(self) -> _Cause:
        """Why this verdict lost its row, without a contract: it could not be read.

        Reported under the stage that judged the answer, with each distinct
        reason once: today's parse step records one message per row, under
        its own name.
        """
        text = "; ".join(self.failure_reasons) or "the answer could not be parsed"
        return _Cause("parse_failure", self.stage or "judge", text)

    def safety_rank(self) -> tuple[bool, int, str]:
        """Lower is safer: not passed first, then more failed checks.

        The last element only makes the order total, so two different
        verdicts with the same safety are still chosen by content.
        """
        return (self.passed, -self.failed_check_count, repr(self.fields))


# Without a contract, an answer that no verdict or failure contradicts passes:
# the "unchecked" contract accepts whatever today's parsing produces.
_IMPLICIT_PASS = _Judgement(passed=True, fields=())


def _answer_rank(record: AttemptRecord) -> tuple:
    """A total order on two answers reported for one attempt.

    Higher spend first, so a duplicate can never hide money spent; the rest
    only makes the choice independent of which report arrived first.
    """
    return (record.cost, record.tokens, record.model, repr(record.raw_answer))


@dataclass(frozen=True)
class _Failure:
    attempt: int
    provider: bool
    stage: str
    message: str
    retryable: bool
    timestamp: datetime = field(default_factory=datetime.now, compare=False)
    #: Report order within the ledger. Breaks only the tie the attempt number
    #: cannot: two stage failures of one row, which stages report in pipeline
    #: order, so the first is the cause (as ``ExecutionContext`` rules).
    sequence: int = field(default=0, compare=False)

    @property
    def text(self) -> str:
        return f"{self.stage}: {self.message}"

    @property
    def cause(self) -> _Cause:
        """This failure as a lost row's cause, in the terms of today's error list."""
        return _Cause(
            "skipped" if self.provider else "parse_failure",
            self.stage,
            self.message,
            self.retryable,
            self.timestamp,
        )

    @property
    def tie_key(self) -> tuple:
        """Among failures of one attempt and kind, the cause sorts first.

        Stage failures: pipeline order. Provider failures report concurrently
        (two retries, two chunks), so their order means nothing; content
        decides instead.
        """
        if self.provider:
            return (self.stage, self.message, self.retryable)
        return (self.sequence,)


@dataclass
class _RowEvents:
    answers: dict[int, AttemptRecord] = field(default_factory=dict)
    verdicts: dict[int, _Judgement] = field(default_factory=dict)
    failures: list[_Failure] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        """Has a verdict or a failure: final for early stop's purposes."""
        return bool(self.verdicts or self.failures)

    @property
    def last_attempt(self) -> int:
        return max(
            (*self.answers, *self.verdicts, *(f.attempt for f in self.failures)),
            default=0,
        )


@dataclass(frozen=True)
class _Cause:
    """Why a failed row was lost, in the terms of today's error list."""

    error_type: str
    stage: str
    message: str
    retryable: bool = False
    timestamp: datetime | None = None

    def error_info(self, row: int) -> ErrorInfo:
        info = ErrorInfo(
            row_index=row,
            stage=self.stage,
            error_type=self.error_type,
            message=self.message,
            retryable=self.retryable,
        )
        if self.timestamp is not None:
            info.timestamp = self.timestamp
        return info


# ---------------------------------------------------------------------------
# The status rules
# ---------------------------------------------------------------------------


def _decide(
    row: int,
    events: _RowEvents | None,
    policy: LedgerPolicy,
    stop: StopReason | None,
    final: bool,
) -> tuple[RowOutcome, _Cause | None]:
    """One row's outcome from all of its events — the only place statuses are set.

    A pure function of the events, never of their arrival order, which is what
    makes the ledger order-independent. Returns the cause too when the row is
    failed, for today's error list.
    """
    if events is None or not (events.answers or events.verdicts or events.failures):
        if not final:
            return RowOutcome(row, RowStatus.PENDING), None
        why = "run stopped before this row"
        if stop is not None:
            why = f"{why}: {stop.value}"
        return RowOutcome(row, RowStatus.FAILED, why), _Cause(
            "not_processed", "run", why
        )

    attempts = tuple(events.answers[k] for k in sorted(events.answers))
    failed_checks = tuple(
        reason
        for attempt in sorted(events.verdicts)
        for reason in events.verdicts[attempt].failure_reasons
    )
    base = RowOutcome(
        row, RowStatus.FAILED, failed_checks=failed_checks, attempts=attempts
    )

    judgements = dict(events.verdicts)
    if not policy.checked:
        failed_attempts = {f.attempt for f in events.failures}
        for attempt in events.answers:
            if attempt not in judgements and attempt not in failed_attempts:
                judgements[attempt] = _IMPLICIT_PASS

    if judgements:
        latest_attempt = max(judgements)
        latest = judgements[latest_attempt]
        if latest.passed:
            return _delivered(base, events, policy, latest_attempt), None
        if policy.checked:
            return _failed_check(base, events, policy, latest_attempt), None

    return _lost(base, events, policy)


def _delivered(
    base: RowOutcome, events: _RowEvents, policy: LedgerPolicy, attempt: int
) -> RowOutcome:
    """The latest verdict passed: accepted, repaired, or (no contract) unchecked."""
    answer = events.answers.get(attempt)
    delivered = replace(
        base,
        delivered_attempt=attempt,
        delivered_by=answer.model if answer is not None else None,
    )
    if not policy.checked:
        return replace(delivered, status=RowStatus.UNCHECKED)

    statuses, notes = _field_statuses(events, attempt, RowStatus.REJECTED)
    verdict = events.verdicts[attempt]
    required = [
        s for f, (_, s) in zip(verdict.fields, statuses, strict=True) if f.required
    ]
    status = max(required, key=_FIELD_RANK.__getitem__, default=RowStatus.ACCEPTED)
    return replace(
        delivered, status=status, reason="; ".join(notes), field_statuses=statuses
    )


def _failed_check(
    base: RowOutcome, events: _RowEvents, policy: LedgerPolicy, latest_attempt: int
) -> RowOutcome:
    """The latest verdict failed a check: rejected, or unverified if flagged.

    A provider failure on a later attempt than that verdict means a repair
    was under way when the provider failed. The answer still failed a check,
    so the row stays rejected, with the provider as a sub-reason (design v1,
    "failed vs rejected").
    """
    flagged = policy.deliver_flagged
    status = RowStatus.UNVERIFIED if flagged else RowStatus.REJECTED
    deciding = latest_attempt
    if flagged:
        # Deliver the failed answer with the fewest failed checks; a tie goes
        # to the highest attempt. Only failed answers are candidates: an
        # earlier answer that passed has zero failed checks, and delivering it
        # under "unverified" would contradict its own field statuses. (Had it
        # been the latest, the row would be accepted or repaired instead.)
        deciding = min(
            (k for k, judgement in events.verdicts.items() if not judgement.passed),
            key=lambda k: (events.verdicts[k].failed_check_count, -k),
        )
    verdict = events.verdicts[deciding]
    statuses, _ = _field_statuses(events, deciding, status)
    reason = (
        "; ".join(verdict.failure_reasons) or "the answer was judged not deliverable"
    )

    interruption = min(
        (f for f in events.failures if f.provider and f.attempt > latest_attempt),
        key=lambda f: (-f.attempt, f.tie_key),
        default=None,
    )
    sub_reason = None
    if interruption is not None:
        sub_reason = PROVIDER_SUB_REASON
        reason = f"{reason} (repair cut short by provider failure: {interruption.text})"

    answer = events.answers.get(deciding) if flagged else None
    return replace(
        base,
        status=status,
        reason=reason,
        sub_reason=sub_reason,
        delivered_attempt=deciding if flagged else None,
        delivered_by=answer.model if answer is not None else None,
        field_statuses=statuses,
    )


def _lost(
    base: RowOutcome, events: _RowEvents, policy: LedgerPolicy
) -> tuple[RowOutcome, _Cause]:
    """No deliverable answer and no answer that failed a check: failed.

    The cause is the latest failure by attempt. On the same attempt a provider
    failure outranks an unparseable answer, which outranks a stage failure:
    the provider failing is what leaves a later stage nothing to work with.
    Remaining ties break on :attr:`_Failure.tie_key`, never on arrival.

    An unparseable answer (no contract) is reported under the stage that
    judged it, with each distinct reason once: today's parse step records one
    message per row, under its own name.
    """
    # Sort keys: (-attempt, -precedence, tie). Lowest wins; the tie element
    # is only compared between candidates of one precedence, so its shape may
    # differ between precedences.
    candidates: list[tuple[tuple, _Cause, str]] = []
    for failure in events.failures:
        precedence = 2 if failure.provider else 0
        candidates.append(
            (
                (-failure.attempt, -precedence, failure.tie_key),
                failure.cause,
                failure.text,
            )
        )
    if not policy.checked:
        for attempt, judgement in events.verdicts.items():
            if not judgement.passed:
                cause = judgement.unparseable_cause()
                text = cause.message
                candidates.append(((-attempt, -1, (text,)), cause, text))
    if not candidates:
        text = "an answer was received but never judged"
        return replace(base, reason=text), _Cause("not_judged", "judge", text)
    _, cause, text = min(candidates, key=lambda c: c[0])
    return replace(base, reason=text), cause


def _field_statuses(
    events: _RowEvents, attempt: int, failed_status: RowStatus
) -> tuple[tuple[tuple[str, RowStatus], ...], list[str]]:
    """Per-field statuses of the verdict on ``attempt``, and notes for the reason.

    A passing field is repaired when a format fix was applied, or when it
    failed on an earlier attempt and passes now; otherwise accepted. A failing
    field takes ``failed_status``. Notes name each repair and each failed
    optional field, so a delivered row's reason still says what happened.
    """
    verdict = events.verdicts[attempt]
    earlier: dict[str, tuple[str, ...]] = {}
    for k in sorted(events.verdicts):
        if k < attempt:
            for f in events.verdicts[k].fields:
                if f.failed:
                    earlier[f.name] = f.failure_reasons
    statuses: list[tuple[str, RowStatus]] = []
    notes: list[str] = []
    for f in verdict.fields:
        if f.failed:
            statuses.append((f.name, failed_status))
            if not f.required:
                notes.append(f"optional field failed: {'; '.join(f.failure_reasons)}")
        elif f.format_fixed:
            statuses.append((f.name, RowStatus.REPAIRED))
            detail = f" ({'; '.join(f.reasons)})" if f.reasons else ""
            notes.append(f"{f.name}: format fix{detail}")
        elif f.name in earlier:
            statuses.append((f.name, RowStatus.REPAIRED))
            notes.append(f"{'; '.join(earlier[f.name])} -> passed on attempt {attempt}")
        else:
            statuses.append((f.name, RowStatus.ACCEPTED))
    return tuple(statuses), notes
