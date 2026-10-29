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
* **Kept in a few bytes per row.** The ledger is opened on every run, so the
  common row (answered once, judged once or never) is kept in parallel arrays
  rather than as records, and only rows with more to say get a full record.
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
``FieldVerdict`` its ``name``, ``outcome``, ``required``, ``format_fixed``,
``reasons`` and ``fixed_reasons``. A field passes unless its outcome is ``"failed"``; an ``"absent"``
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
from array import array
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
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

    Temporary: a second reading of the events, with today's rules, next to
    the snapshot's own. Migration step 3 deletes it, together with
    :meth:`RowLedger.reported_losses`, when ``result.errors`` and the loss
    counters are read from the snapshot instead.

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
            asked for and at most ``RowLedger.MAX_RECORDED_ROW_ERRORS``.
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

    Cheap to keep and to poll, because it is opened on every run:

    * **Memory.** The common row (answered once, judged once or, without a
      contract, never) is kept in a few bytes of parallel arrays, with the
      values many rows share (model, verdict, failure) stored once. Only a
      row whose events take another shape (several attempts, an answer and a
      failure) gets a full record. See :class:`_CompactRows`.
    * **Polling.** Each row's live status is re-derived when an event for it
      lands, so :meth:`counts` and :meth:`early_stop_reached` read stored
      statuses and the model-call stage may ask once per batch. A compact
      row's status depends only on its shape, so it is derived once per
      shape, not once per row.
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
        self._row_count = 0
        self._policy = policy
        self._strict = strict
        # A row's events live in one of two places: the compact columns (the
        # common shapes), or a full record here once its events need one.
        self._compact = _CompactRows()
        self._full: dict[int, _RowEvents] = {}
        self._stops: set[StopReason] = set()
        self._violations: list[str] = []
        # Numbers each failure in report order: the one tie the pipeline's own
        # order decides (see _lost). One number per report, shared by its rows.
        self._failure_sequence = 0
        # Live status code of every row (_STATUS_BY_CODE), kept current as
        # events land. A row's live status depends on its own events and the
        # fixed policy alone (the stop reason only matters to a row with no
        # events, at snapshot), so it never goes stale.
        self._status = bytearray()
        self._tally: dict[RowStatus, int] = dict.fromkeys(RowStatus, 0)
        # Live status of each compact shape (see _CompactRows.shape).
        self._shape_status: dict[int, RowStatus] = {}
        # For reported_losses(): which rows were reported lost, the first
        # entries in report order, and exact counts.
        self._reported = bytearray()
        self._reported_errors: list[tuple[int, _Cause]] = []
        self._reported_skipped = 0
        self._reported_failed = 0
        # The ledger's copy of a clean unchecked pass, one per reporting stage
        # and column list: nearly every verdict of a run without a contract.
        self._clean_passes: dict[tuple[str, tuple[Any, ...]], _Judgement] = {}
        self._lock = threading.Lock()
        self._widen(row_count)

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
            self._widen(row_count)

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

    def record_batch_answer(
        self,
        rows: Iterable[int],
        attempt: int,
        *,
        model: str,
        tokens: int,
        cost: Decimal,
    ) -> None:
        """One billed model call answered every row of ``rows`` on ``attempt``.

        A mega-batch answers several rows with one call. Its tokens and cost
        are shared among the rows, the first row taking what does not divide
        evenly, so the rows' tokens and costs sum exactly to the call's. The
        shares are rounded down to nine decimal places past the cost's own, so
        they stay small enough to keep compactly; the first row absorbs that
        rounding too.
        """
        batch = list(rows)
        if not batch:
            return
        token_share, extra_tokens = divmod(tokens, len(batch))
        first_cost, cost_share = _split_cost(cost, len(batch))
        with self._lock:
            if not self._checked_attempt(attempt):
                return
            first = AttemptRecord(
                attempt, model, token_share + extra_tokens, first_cost
            )
            other = AttemptRecord(attempt, model, token_share, cost_share)
            for position, row in enumerate(batch):
                index = self._checked_row(row)
                if index is not None:
                    self._add_answer(index, other if position else first, merged=False)

    def record_verdict(
        self, row: int, attempt: int, verdict: Verdict, *, stage: str
    ) -> None:
        """The judge's verdict on ``row``'s answer from ``attempt``.

        ``stage`` is the stage that judged the answer (today, the parse
        stage). Without a contract an unparseable answer is a lost row, and
        today's error list names that stage, so the ledger must know it.
        """
        judgement = self._clean_pass(verdict, stage)
        if judgement is not None:
            # Fast path: the run's common case, with nothing to check.
            with self._lock:
                index = self._checked_row(row)
                if index is not None and self._checked_attempt(attempt):
                    self._add_verdict(index, attempt, judgement)
            return
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
                # the parse step has always recorded it, stamped when it was
                # reported, as the parse step's own failure report was.
                cause = replace(judgement.unparseable_cause(), timestamp=datetime.now())
                self._report_loss(index, cause)

    def _clean_pass(self, verdict: Verdict, stage: str) -> _Judgement | None:
        """The ledger's copy of ``verdict`` if it is a clean unchecked pass.

        Without a contract nearly every verdict is one: every field a value,
        no reason, no format fix. All such verdicts on the same columns from
        the same stage copy to equal judgements, so the copy is made once and
        shared. None for any other verdict, or under a checked policy; those
        take the full path, with its consistency checks.
        """
        if self._policy.checked or verdict.checked:
            return None
        names = []
        for item in verdict.fields:
            if (
                item.outcome != "value"
                or not item.required
                or item.reasons
                or item.format_fixed
                or item.fixed_reasons
            ):
                return None
            names.append(item.name)
        key = (stage, tuple(names))
        judgement = self._clean_passes.get(key)
        if judgement is None:
            judgement = _Judgement.of(verdict, stage)[0]
            self._clean_passes[key] = judgement
        return judgement

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
            failure = self._numbered(_Failure(attempt, True, stage, message, retryable))
            self._add_failures(rows, failure)

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
            failure = self._numbered(_Failure(1, False, stage, message, False))
            self._add_failures(rows, failure)

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
            # Read before any child event lands: they move the last attempt.
            offsets = [self._last_attempt(row) for row in checked] if retry else None

            def offset(child_row: int) -> int:
                return 0 if offsets is None else offsets[child_row]

            for child_row, record in answers:
                attempt = record.attempt + offset(child_row)
                if attempt != record.attempt:
                    record = replace(record, attempt=attempt)
                self._add_answer(checked[child_row], record, merged=True)
            for child_row, attempt, judgement in verdicts:
                self._add_verdict(
                    checked[child_row], attempt + offset(child_row), judgement
                )
            # In the child's report order, so stage failures keep their
            # pipeline order after the merge. One child report stays one
            # report here (per attempt it lands on), shared by its rows.
            renumbered: dict[tuple[int, int], _Failure] = {}
            for child_row, failure in sorted(failures, key=lambda f: f[1].sequence):
                attempt = failure.attempt + offset(child_row)
                key = (failure.sequence, attempt)
                ours = renumbered.get(key)
                if ours is None:
                    ours = self._numbered(replace(failure, attempt=attempt))
                    renumbered[key] = ours
                self._add_failure(checked[child_row], ours)
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
                row, self._events_of(row), self._policy, self._stop_reason(), False
            )
        return outcome

    def counts(self) -> Mapping[RowStatus, int]:
        """Live counts per status, pending included. They sum to the row count.

        Cost: a copy of the seven-entry tally, kept current as events land.
        Safe to poll.
        """
        with self._lock:
            tally = {
                status: self._tally[status]
                for status in (*FINAL_STATUSES, RowStatus.PENDING)
            }
        return MappingProxyType(tally)

    def early_stop_reached(self) -> bool:
        """True when more than the policy's share of the first rows is rejected.

        "First rows" means by input order, and the answer waits until every
        one of them is decided: rows 50+ finishing first must not stand in for
        rows 0-49, and a row still in flight could change the count.

        Cost: a look-up per row of the window; no row is re-derived.
        """
        fraction = self._policy.early_stop_fraction
        if fraction is None:
            return False
        window = min(self._row_count, self._policy.early_stop_window)
        if window == 0:
            return False
        rejected_code = _STATUS_CODES[RowStatus.REJECTED]
        with self._lock:
            rejected = 0
            for row in range(window):
                if not self._decided(row):
                    return False
                rejected += self._status[row] == rejected_code
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
                    row, self._events_of(row), self._policy, stop, True
                )
                keep_raw = False
                if outcome.status in _QUARANTINED_STATUSES:
                    quarantined += 1
                    keep_raw = quarantined <= self._policy.raw_answer_cap
                if not keep_raw and any(
                    a.raw_answer is not None for a in outcome.attempts
                ):
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
                ``None`` for :attr:`MAX_RECORDED_ROW_ERRORS`, which is also
                the most this ledger keeps. ``0`` reads the counts alone.

        Cost: the entries built, never the input's size, so a context may
        read its counters at every checkpoint.
        """
        if limit is None:
            limit = self.MAX_RECORDED_ROW_ERRORS
        with self._lock:
            errors = tuple(
                cause.error_info(row)
                for row, cause in islice(self._reported_errors, max(limit, 0))
            )
            skipped, failed = self._reported_skipped, self._reported_failed
        return ReportedLosses(errors=errors, skipped_rows=skipped, failed_rows=failed)

    # -- internals ----------------------------------------------------------

    def _widen(self, row_count: int) -> None:
        added = row_count - self._row_count
        if added <= 0:
            return
        self._row_count = row_count
        self._compact.widen(row_count)
        self._status.extend(bytes(added))
        self._reported.extend(bytes(added))
        self._tally[RowStatus.PENDING] += added

    def _report_loss(self, row: int, cause: _Cause) -> None:
        """Count ``row`` lost, under ``cause``, if no loss was reported for it yet."""
        if self._reported[row]:
            return
        self._reported[row] = 1
        if cause.error_type == "skipped":
            self._reported_skipped += 1
        else:
            self._reported_failed += 1
        if len(self._reported_errors) < self.MAX_RECORDED_ROW_ERRORS:
            self._reported_errors.append((row, cause))

    def _events_of(self, row: int) -> _RowEvents | None:
        """All of ``row``'s events, or None. Read only: may be a fresh copy."""
        events = self._full.get(row)
        return events if events is not None else self._compact.events(row)

    def _full_record(self, row: int) -> _RowEvents:
        """``row``'s full record, moving its events out of the compact columns."""
        events = self._full.get(row)
        if events is None:
            events = self._compact.take(row) or _RowEvents()
            self._full[row] = events
        return events

    def _restatus(self, row: int) -> None:
        """Re-derive ``row``'s live status after an event; keep the tally exact."""
        events = self._full.get(row)
        if events is not None:
            status = _decide(row, events, self._policy, None, False)[0].status
        else:
            shape = self._compact.shape(row)
            cached = self._shape_status.get(shape)
            if cached is None:
                cached = _decide(
                    row, self._compact.events(row), self._policy, None, False
                )[0].status
                self._shape_status[shape] = cached
            status = cached
        self._tally[_STATUS_BY_CODE[self._status[row]]] -= 1
        self._tally[status] += 1
        self._status[row] = _STATUS_CODES[status]

    def _decided(self, row: int) -> bool:
        """Has a verdict or a failure: final for early stop's purposes."""
        events = self._full.get(row)
        return events.decided if events is not None else self._compact.decided(row)

    def _add_answer(self, row: int, record: AttemptRecord, *, merged: bool) -> None:
        """Store an answer; one per attempt, the same one in any arrival order.

        A repeat inside one ledger is resume replaying a cached answer. A
        repeat with different content arriving by merge means two
        sub-pipelines answered one row on one attempt: a defect upstream.
        """
        if row not in self._full and self._compact.store_answer(row, record):
            self._restatus(row)
            return
        answers = self._full_record(row).answers
        existing = answers.get(record.attempt)
        if existing is None:
            answers[record.attempt] = record
        else:
            if merged and _answer_rank(existing) != _answer_rank(record):
                self._violation(
                    f"row {row} attempt {record.attempt}: two merged ledgers "
                    f"each answered it ({existing.model}, {record.model}); kept "
                    f"one by content"
                )
            answers[record.attempt] = max(existing, record, key=_answer_rank)
        self._restatus(row)

    def _add_failures(self, rows: Iterable[int], failure: _Failure) -> None:
        """Add one reported failure to each of ``rows``, and count them lost."""
        cause = failure.cause
        for row in rows:
            index = self._checked_row(row)
            if index is not None:
                self._add_failure(index, failure)
                self._report_loss(index, cause)

    def _add_failure(self, row: int, failure: _Failure) -> None:
        if row not in self._full and self._compact.store_failure(row, failure):
            self._restatus(row)
            return
        self._full_record(row).failures.append(failure)
        self._restatus(row)

    def _add_verdict(self, row: int, attempt: int, judgement: _Judgement) -> None:
        """Store a verdict; a different one for the same attempt is a conflict.

        The same verdict twice is normal (resume re-judges cached answers, and
        the judge is deterministic). Two different ones mean a defect upstream;
        the safer one is kept — the one that delivers nothing it doubts — so
        the conflict can only ever blank a value, never ship a bad one. Every
        choice here is by content, so arrival order never shows.
        """
        if row not in self._full and self._compact.store_verdict(
            row, attempt, judgement
        ):
            self._restatus(row)
            return
        verdicts = self._full_record(row).verdicts
        existing = verdicts.get(attempt)
        if existing is None:
            verdicts[attempt] = judgement
        elif existing == judgement:
            # Equal verdicts may differ only in the stage that reported them.
            verdicts[attempt] = min(existing, judgement, key=lambda j: j.stage)
        else:
            self._violation(
                f"row {row} attempt {attempt}: two different verdicts; kept the "
                f"safer one"
            )
            verdicts[attempt] = min(existing, judgement, key=_Judgement.safety_rank)
        self._restatus(row)

    def _numbered(self, failure: _Failure) -> _Failure:
        """``failure`` with the next report number (see ``_Failure.sequence``)."""
        self._failure_sequence += 1
        return replace(failure, sequence=self._failure_sequence)

    def _last_attempt(self, row: int) -> int:
        events = self._full.get(row)
        if events is not None:
            return events.last_attempt
        return self._compact.last_attempt(row)

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
        answers: list[tuple[int, AttemptRecord]] = []
        verdicts: list[tuple[int, int, _Judgement]] = []
        failures: list[tuple[int, _Failure]] = []
        with self._lock:
            for row in range(self._row_count):
                if not self._status[row]:
                    continue  # no events
                events = self._events_of(row)
                if events is None:  # pragma: no cover - a status needs events
                    continue
                answers.extend((row, record) for record in events.answers.values())
                verdicts.extend(
                    (row, attempt, judgement)
                    for attempt, judgement in events.verdicts.items()
                )
                failures.extend((row, failure) for failure in events.failures)
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
    #: For a format-fixed field, why the answer failed before the fix.
    fixed_reasons: tuple[str, ...] = ()

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
                fixed_reasons=tuple(str(r) for r in item.fixed_reasons),
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


@dataclass(slots=True)
class _RowEvents:
    """A row's events in full, for a row the compact columns cannot hold."""

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
# Compact row storage
# ---------------------------------------------------------------------------

#: Live status codes, one byte per row. Code 0 is a row with no events.
_STATUS_BY_CODE: tuple[RowStatus, ...] = (
    RowStatus.PENDING,
    *(status for status in RowStatus if status is not RowStatus.PENDING),
)
_STATUS_CODES: dict[RowStatus, int] = {
    status: code for code, status in enumerate(_STATUS_BY_CODE)
}

_INT64 = range(-(2**63), 2**63)
_SMALL = range(1, 2**16)  # what an unsigned 16-bit column holds, 0 aside
#: In ``_CompactRows.cost_exponent``: the cost column holds a number in
#: ``_CompactRows.costs`` instead of a coefficient.
_LISTED_COST = 127
_COST_EXPONENTS = range(-128, _LISTED_COST)
#: A cost coefficient of at most this many digits fits a signed 64-bit column.
_COST_DIGITS = 18
#: Batch cost shares keep this many decimal places past the cost's own.
_SHARE_PLACES = 9


class _Listed:
    """Distinct values stored once, each known by a number from 1.

    A column then holds the number, and many rows share one value: the model
    of a run, the verdict "passed", one provider failure across a batch.
    """

    def __init__(self, limit: int) -> None:
        self._values: list[Any] = []
        self._numbers: dict[Any, int] = {}
        self._limit = limit

    def number(self, value: Any, key: Any) -> int | None:
        """The number of ``value`` (known by ``key``), or None when full."""
        number = self._numbers.get(key)
        if number is None:
            if len(self._values) >= self._limit:
                return None
            self._values.append(value)
            number = self._numbers[key] = len(self._values)
        return number

    def __getitem__(self, number: int) -> Any:
        return self._values[number - 1]


class _CompactRows:
    """The common rows in a few bytes each, as parallel arrays indexed by row.

    Most rows of a run are answered once and judged once or, without a
    contract, not judged at all. Kept as records (a dict of answers, a dict of
    verdicts, a list of failures, a ``Decimal``), each such row cost the
    ledger about 700 bytes; at 5M rows, gigabytes for a run that went well.

    Two shapes are kept here, and only these:

    * **answered:** one answer (model, attempt, tokens, cost, raw answer),
      plus at most one verdict, on that same attempt;
    * **lost:** one failure and nothing else.

    A row whose events leave these shapes (a second answer, a verdict on
    another attempt, a failure next to an answer) is moved out by
    :meth:`take` into a full :class:`_RowEvents`, and never comes back: the
    shapes kept here stay simple enough to read back exactly, and the
    status rules see the same events either way (:meth:`events`).

    Values many rows share (model, verdict, failure) are stored once in a
    :class:`_Listed` and the columns hold their numbers; 0 means "none". A
    cost is kept exactly as a 64-bit coefficient and a one-byte exponent
    when it fits, which a priced call's cost does, and listed otherwise.
    Raw answers are kept only when given, in a dict: the pipeline gives
    none, and a caller that does keeps them all until the snapshot anyway.
    """

    def __init__(self) -> None:
        self.model = array("H")
        self.attempt = array("H")
        self.tokens = array("q")
        self.cost = array("q")
        self.cost_exponent = array("b")
        self.verdict = array("I")
        self.failure = array("I")
        self.raw: dict[int, Any] = {}
        self.models = _Listed(_SMALL.stop - 1)
        self.costs = _Listed(_INT64.stop - 1)
        self.verdicts = _Listed(2**32 - 1)
        self.failures = _Listed(2**32 - 1)

    def widen(self, row_count: int) -> None:
        added = row_count - len(self.model)
        for column in (
            self.model,
            self.attempt,
            self.tokens,
            self.cost,
            self.cost_exponent,
            self.verdict,
            self.failure,
        ):
            column.frombytes(bytes(added * column.itemsize))

    # -- storing: each returns False when the row must take a full record ---

    def store_answer(self, row: int, record: AttemptRecord) -> bool:
        """Keep ``record`` as ``row``'s only event, if it fits."""
        if self.model[row] or self.failure[row]:
            return False
        if type(record.attempt) is not int or record.attempt not in _SMALL:
            return False
        if type(record.tokens) is not int or record.tokens not in _INT64:
            return False
        cost = self._encoded_cost(record.cost)
        if cost is None or not isinstance(record.model, str):
            return False
        model = self.models.number(record.model, record.model)
        if model is None:
            return False
        self.model[row] = model
        self.attempt[row] = record.attempt
        self.tokens[row] = record.tokens
        self.cost[row], self.cost_exponent[row] = cost
        if record.raw_answer is not None:
            self.raw[row] = record.raw_answer
        return True

    def store_verdict(self, row: int, attempt: int, judgement: _Judgement) -> bool:
        """Keep ``judgement`` on ``row``'s answer, if it judges that answer."""
        if not self.model[row] or self.verdict[row] or self.attempt[row] != attempt:
            return False
        number = self.verdicts.number(judgement, (judgement, judgement.stage))
        if number is None:
            return False
        self.verdict[row] = number
        return True

    def store_failure(self, row: int, failure: _Failure) -> bool:
        """Keep ``failure`` as ``row``'s only event, if it has none yet."""
        if self.model[row] or self.failure[row]:
            return False
        number = self.failures.number(failure, failure.sequence)
        if number is None:
            return False
        self.failure[row] = number
        return True

    # -- reading ------------------------------------------------------------

    def shape(self, row: int) -> int:
        """What ``row``'s live status depends on, given the policy.

        One answer with no verdict, one answer with a given verdict (its
        number), or one failure. The attempt, model and cost never change a
        status, so rows of one shape share one status.
        """
        if self.verdict[row]:
            return self.verdict[row]
        return 0 if self.model[row] else -1

    def decided(self, row: int) -> bool:
        """Has a verdict or a failure."""
        return bool(self.verdict[row] or self.failure[row])

    def last_attempt(self, row: int) -> int:
        if self.model[row]:
            return self.attempt[row]
        if self.failure[row]:
            return int(self.failures[self.failure[row]].attempt)
        return 0

    def events(self, row: int) -> _RowEvents | None:
        """``row``'s events as a fresh record, or None when it has none here."""
        if self.failure[row]:
            return _RowEvents(failures=[self.failures[self.failure[row]]])
        if not self.model[row]:
            return None
        attempt = self.attempt[row]
        events = _RowEvents()
        events.answers[attempt] = AttemptRecord(
            attempt,
            self.models[self.model[row]],
            self.tokens[row],
            self._cost(row),
            self.raw.get(row),
        )
        if self.verdict[row]:
            events.verdicts[attempt] = self.verdicts[self.verdict[row]]
        return events

    def take(self, row: int) -> _RowEvents | None:
        """Move ``row``'s events out of the columns, into a full record."""
        events = self.events(row)
        self.model[row] = self.attempt[row] = self.verdict[row] = 0
        self.tokens[row] = self.cost[row] = self.cost_exponent[row] = 0
        self.failure[row] = 0
        self.raw.pop(row, None)
        return events

    # -- costs --------------------------------------------------------------

    def _encoded_cost(self, cost: Any) -> tuple[int, int] | None:
        """``cost`` as (coefficient, exponent), or (number, ``_LISTED_COST``)."""
        if not isinstance(cost, Decimal):
            return None
        sign, digits, exponent = cost.as_tuple()
        if (
            isinstance(exponent, int)
            and exponent in _COST_EXPONENTS
            and len(digits) <= _COST_DIGITS
            and (any(digits) or not sign)  # -0 would read back as 0
        ):
            coefficient = int("".join(map(str, digits)))
            return -coefficient if sign else coefficient, exponent
        number = self.costs.number(cost, str(cost))
        return None if number is None else (number, _LISTED_COST)

    def _cost(self, row: int) -> Decimal:
        exponent = self.cost_exponent[row]
        if exponent == _LISTED_COST:
            return self.costs[self.cost[row]]  # type: ignore[no-any-return]
        # The string constructor is exact and keeps the exponent: "100E-4"
        # reads back as 0.0100, never 0.01.
        return Decimal(f"{self.cost[row]}E{exponent}")


def _split_cost(cost: Decimal, count: int) -> tuple[Decimal, Decimal]:
    """(the first row's share, every other row's share) of one call's cost.

    The shares sum exactly to ``cost``. Each other share is ``cost / count``
    rounded down to :data:`_SHARE_PLACES` decimal places past the cost's own,
    and the first row takes what is left. Unrounded, ``cost / count`` carries
    28 digits, too many to keep compactly.
    """
    if count == 1:
        return cost, cost
    share = cost / count
    exponent = cost.as_tuple().exponent if isinstance(cost, Decimal) else None
    if isinstance(exponent, int):
        try:
            share = share.quantize(
                Decimal(1).scaleb(exponent - _SHARE_PLACES), rounding=ROUND_DOWN
            )
        except InvalidOperation:  # pragma: no cover - a cost of 28+ digits
            pass
    return cost - share * (count - 1), share


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
        # A failure reported on the attempt a passing verdict judged means a
        # later stage withdrew that value (grounding blanked it): the row is
        # lost, exactly as an answer with no verdict and a failure is.
        withdrawn = any(f.attempt == latest_attempt for f in events.failures)
        if latest.passed and not withdrawn:
            return _delivered(base, events, policy, latest_attempt), None
        if policy.checked and not latest.passed:
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
            detail = f" ({'; '.join(f.fixed_reasons)})" if f.fixed_reasons else ""
            notes.append(f"{f.name}: format fix{detail}")
        elif f.name in earlier:
            statuses.append((f.name, RowStatus.REPAIRED))
            notes.append(f"{'; '.join(earlier[f.name])} -> passed on attempt {attempt}")
        else:
            statuses.append((f.name, RowStatus.ACCEPTED))
    return tuple(statuses), notes
