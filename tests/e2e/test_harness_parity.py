"""Parity net: today's observable row outcomes, locked per entry path.

The data-quality harness moves "what happened to each row" out of five
scattered places (the context's lost-row set, its error records and counters,
the result's error list, cell blanking, the quality report) into one row
ledger. The design's first migration step is the dangerous one: the ledger
must reproduce today's output *exactly* when no contract is declared, on every
way a run can be entered — ``execute()``, ``execute_async()``,
``execute_stream()``, the auto-retry pass and resume. Several of those run as
separate sub-pipelines with their own context, which is precisely where a
merge can lose or double a row's fate.

``test_pipeline_conformance`` already proves the happy path and one failure
mode at a time (a single skipped row, a single transient hiccup, a dropped
batch, resume across a gap). What it does not pin down is the *combined*
outcome per entry path — delivered values, the error list, ``lost_row_indices``
and ``is_complete`` together — for a frame where one row fails for good and
another only hiccups. That combination is what a ledger has to fold into one
status per row, so it is what this file freezes.

Every expectation here describes current behaviour, observed rather than
wished for. Where current behaviour looks wrong, the test still asserts it and
its docstring says so: the wiring step must change such a behaviour on
purpose, by editing the test, never by accident.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ondine import PipelineBuilder
from ondine.utils.retry_handler import NetworkError
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    TOKEN_PATTERN,
    LedgerClient,
    ProviderError,
    StructuredAnswerBatch,
    StructuredLedgerClient,
    answer_for,
    conformance_frame,
    token_for,
)

if TYPE_CHECKING:
    import pandas as pd

PROMPT = "Answer for {marker}"

#: The row that never answers. Its provider error is not retryable, so the
#: default skip policy drops it on the first attempt.
PERMANENT = token_for(2)

#: The row whose first call raises a retryable error and whose second succeeds.
TRANSIENT = token_for(4)

#: The columns a declared contract adds (interfaces spec, section 4). With no
#: contract, none of them may appear.
CONTRACT_COLUMNS = ("status", "reason", "checks", "repair_method", "delivered_by")


# ── scripted provider ─────────────────────────────────────────────────────


class ScriptedLedgerClient(LedgerClient):
    """A ledger whose failure script is chosen per row, not per client.

    ``LedgerClient`` applies one failure mode to every scripted token, so a
    frame cannot hold a permanent failure and a transient one at once. The
    harness has to fold exactly that mix into per-row statuses, so this
    subclass scripts each token separately while keeping the parent's ledger,
    both prompt dialects and its thread safety.

    Args:
        permanent: Tokens whose every call raises ``ProviderError``, which the
            retry handler classifies as not retryable.
        transient: Token to the number of leading calls that raise
            ``NetworkError`` (retryable) before the token answers.
        empty: Token to the number of leading calls that answer with an empty
            string instead of raising: the model replied, but said nothing.
    """

    def __init__(
        self,
        *,
        permanent: set[str] | None = None,
        transient: dict[str, int] | None = None,
        empty: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._permanent = set(permanent or ())
        self._transient = dict(transient or {})
        self._empty = dict(empty or {})

    def _answer_token(self, token: str) -> str:
        attempt = self._record(token)
        if token in self._permanent:
            raise ProviderError(failure(token, attempt))
        if attempt <= self._transient.get(token, 0):
            raise NetworkError(hiccup(token, attempt))
        if attempt <= self._empty.get(token, 0):
            return ""
        return answer_for(token)


class ScriptedStructuredClient(StructuredLedgerClient):
    """``StructuredLedgerClient`` with the same per-row failure script.

    A batched structured call carries several markers, so a failure scripted
    for any one of them fails the whole call, as a real provider error would.
    ``attempts`` counts every call per token, including failed ones; the
    parent's ``calls`` counts only the calls that answered.
    """

    def __init__(
        self,
        *,
        permanent: set[str] | None = None,
        transient: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._permanent = set(permanent or ())
        self._transient = dict(transient or {})
        self.attempts: Counter[str] = Counter()

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        tokens = TOKEN_PATTERN.findall(prompt)
        for token in tokens:
            self.attempts[token] += 1
        for token in tokens:
            attempt = self.attempts[token]
            if token in self._permanent:
                raise ProviderError(failure(token, attempt))
            if attempt <= self._transient.get(token, 0):
                raise NetworkError(hiccup(token, attempt))
        return super().structured_invoke(prompt, output_cls, **kwargs)


# ── observing a result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Outcome:
    """Everything a caller can see about row fate, in comparable form.

    One value per concern the ledger will own, so a test compares a whole
    outcome at once and a failure prints every difference together instead of
    stopping at the first assertion.

    Attributes:
        columns: Output frame columns, in order.
        rows: ``(marker, answer)`` per output row, in output order. A missing
            cell (``None`` or ``NaN``) reads as ``None``.
        errors: ``(row_index, stage, error_type, message)`` per error, in the
            order ``result.errors`` lists them.
        lost: ``result.lost_row_indices``.
        complete: ``result.is_complete``.
        skipped: ``result.metrics.skipped_rows``.
        failed: ``result.metrics.failed_rows``.
    """

    columns: list[str]
    rows: list[tuple[str, str | None]]
    errors: list[tuple[int, str, str, str]]
    lost: list[int]
    complete: bool
    skipped: int
    failed: int


def _cell(value: Any) -> Any:
    """A missing cell as ``None``, whichever spelling pandas chose for it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def observe(result: Any, answer_column: str = "answer") -> Outcome:
    """Read a result the way a caller does: through its public surface."""
    frame = result.to_pandas()
    assert list(frame.index) == list(range(len(frame))), "output index not 0..n-1"
    return Outcome(
        columns=list(frame.columns),
        rows=[
            (marker, _cell(answer))
            for marker, answer in zip(
                frame["marker"], frame[answer_column], strict=True
            )
        ],
        errors=[
            (error.row_index, error.stage, error.error_type, error.message)
            for error in result.errors
        ],
        lost=result.lost_row_indices,
        complete=result.is_complete,
        skipped=result.metrics.skipped_rows,
        failed=result.metrics.failed_rows,
    )


def answered(*rows: int) -> list[tuple[str, str]]:
    """``(marker, answer)`` for rows that received their own answer."""
    return [(token_for(row), answer_for(token_for(row))) for row in rows]


def failure(token: str, attempt: int = 1) -> str:
    """The message of a scripted permanent failure."""
    return f"scripted failure for {token} (attempt {attempt})"


def hiccup(token: str, attempt: int = 1) -> str:
    """The message of a scripted transient failure."""
    return f"scripted hiccup for {token} (attempt {attempt})"


def skipped(row: int, message: str) -> tuple[int, str, str, str]:
    """The error entry the skip policy records today for a dropped row."""
    return (row, "LLMInvocation", "skipped", message)


def build(client: LedgerClient, frame: pd.DataFrame, **options: Any):
    """A builder over ``frame`` wired to ``client``; options as in conformance."""
    builder = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
        .with_prompt(PROMPT)
        .with_custom_llm_client(client)
        .with_max_retries(options.get("max_retries", 3))
        .with_batch_size(options.get("batch_size", 1))
    )
    if "concurrency" in options:
        builder = builder.with_concurrency(options["concurrency"])
    if "error_policy" in options:
        builder = builder.with_error_policy(options["error_policy"])
    if "checkpoint_dir" in options:
        builder = builder.with_checkpoint_dir(str(options["checkpoint_dir"]))
        builder = builder.with_checkpoint_interval(1)
    return builder


# ── execute() and execute_async(): one outcome, every whole-frame path ────


def _run_sync(client: LedgerClient):
    return build(client, conformance_frame(6), concurrency=1).build().execute()


def _run_concurrent(client: LedgerClient):
    return build(client, conformance_frame(6), concurrency=4).build().execute()


def _run_async(client: LedgerClient):
    pipeline = (
        build(client, conformance_frame(6)).with_async_execution(max_concurrency=4)
    ).build()
    return asyncio.run(pipeline.execute_async())


#: The one outcome every whole-frame entry path gives today for six rows with
#: row 2 permanently failing and row 4 hiccupping once.
WHOLE_FRAME_OUTCOME = Outcome(
    columns=["marker", "answer"],
    rows=answered(0, 1) + [(PERMANENT, None)] + answered(3, 4, 5),
    errors=[skipped(2, failure(PERMANENT))],
    lost=[2],
    complete=False,
    skipped=1,
    failed=0,
)


@pytest.mark.parametrize(
    "run",
    [_run_sync, _run_concurrent, _run_async],
    ids=["execute", "execute-concurrent", "execute_async"],
)
def test_every_whole_frame_path_reports_the_same_row_fates(run):
    """execute(), execute() with concurrency, and execute_async() agree.

    The ledger will be created once per run, so any path that builds its
    context differently is a place the migration can diverge. Here the
    permanent row is dropped and named once, the transient row is retried
    by the request-level handler and keeps its own answer (so it leaves no
    error behind), and every other row is untouched.

    Cost is the price of the five calls that answered: the failed calls
    (row 2 once, row 4's first try) are not charged.
    """
    client = ScriptedLedgerClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    result = run(client)

    assert observe(result) == WHOLE_FRAME_OUTCOME
    assert result.success is True
    assert client.calls == Counter(
        {token_for(row): 1 for row in range(6)} | {TRANSIENT: 2}
    )
    assert result.costs.total_cost == COST_PER_CALL * 5


def test_an_empty_answer_is_delivered_as_an_empty_string_and_not_reported():
    """Suspected bug, locked as-is: one row answering ``""`` is not a loss.

    With one prompt per row, an empty model answer becomes an empty-string
    cell. No error is recorded, ``lost_row_indices`` is empty and the run
    reports complete. The batch path's empty *body* trips the whole-run guard
    (covered in ``test_pipeline_conformance``); a single empty row passes
    silently. Under the harness this row would be ``failed`` or ``rejected``
    once a contract exists, but with no contract the frame must stay exactly
    this.
    """
    client = ScriptedLedgerClient(empty={TRANSIENT: 1})

    result = _run_sync(client)

    assert observe(result) == Outcome(
        columns=["marker", "answer"],
        rows=answered(0, 1, 2, 3) + [(TRANSIENT, "")] + answered(5),
        errors=[],
        lost=[],
        complete=True,
        skipped=0,
        failed=0,
    )
    assert client.calls[TRANSIENT] == 1


# ── JSON mega-batch ───────────────────────────────────────────────────────


def test_a_mega_batch_drops_and_retries_as_a_unit():
    """batch_size=3: the permanent row takes its batch down; a hiccup re-sends
    its whole batch.

    Rows 0-2 share a batch with the permanent row, so all three are lost and
    each carries the *permanent row's* message, not one of its own: the
    error names the batch's cause, not the row's. Row 4's batch is retried as
    a whole, so row 3 is asked twice (it precedes row 4 in the batch) and
    row 5 once (the first try raised before reaching it). One answered call
    is one charge, whatever the batch size.
    """
    client = ScriptedLedgerClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    result = build(client, conformance_frame(6), batch_size=3).build().execute()

    assert observe(result) == Outcome(
        columns=["marker", "answer"],
        rows=[(token_for(row), None) for row in (0, 1, 2)] + answered(3, 4, 5),
        errors=[skipped(row, failure(PERMANENT)) for row in (0, 1, 2)],
        lost=[0, 1, 2],
        complete=False,
        skipped=3,
        failed=0,
    )
    assert client.calls == Counter(
        {
            token_for(0): 1,
            token_for(1): 1,
            token_for(2): 1,
            token_for(3): 2,
            token_for(4): 2,
            token_for(5): 1,
        }
    )
    assert result.costs.total_cost == COST_PER_CALL * 1


# ── execute_stream(): one sub-pipeline per chunk ──────────────────────────


def test_streaming_reports_each_chunk_with_chunk_local_row_indices():
    """Suspected bug, locked as-is: a chunk's errors use the chunk's numbering.

    Each chunk runs as its own sub-pipeline, and today its result knows only
    its own rows: a loss at input row 5 (the second row of the second chunk
    of four) is reported as row 1. Nothing maps it back to the input frame,
    which is exactly the merge the interfaces spec gives to the ledger.

    Values and order survive chunking: concatenated, the chunks hold every
    row's own answer in input order, with both permanent rows missing and the
    transient row recovered inside its chunk.
    """
    late_permanent = token_for(5)
    client = ScriptedLedgerClient(
        permanent={PERMANENT, late_permanent}, transient={token_for(1): 1}
    )

    chunks = list(
        build(client, conformance_frame(6))
        .with_streaming(chunk_size=4)
        .build()
        .execute_stream()
    )

    assert [observe(chunk) for chunk in chunks] == [
        Outcome(
            columns=["marker", "answer"],
            rows=answered(0, 1) + [(PERMANENT, None)] + answered(3),
            errors=[skipped(2, failure(PERMANENT))],
            lost=[2],
            complete=False,
            skipped=1,
            failed=0,
        ),
        Outcome(
            columns=["marker", "answer"],
            rows=answered(4) + [(late_permanent, None)],
            errors=[skipped(1, failure(late_permanent))],
            lost=[1],
            complete=False,
            skipped=1,
            failed=0,
        ),
    ]
    assert client.calls[token_for(1)] == 2
    assert client.duplicated_tokens == {token_for(1)}


# ── the auto-retry pass: a sub-pipeline over the "failed" rows ────────────


def _auto_retry(builder, attempts: int):
    """Turn on the row-level retry pass. It has no builder method today."""
    pipeline = builder.build()
    pipeline.specifications.processing.auto_retry_failed = True
    pipeline.specifications.processing.max_retry_attempts = attempts
    return pipeline


def test_auto_retry_recovers_empty_rows_but_never_retries_skipped_ones():
    """Suspected bug, locked as-is: the retry pass cannot see skipped rows.

    The pass picks rows whose output cells are empty. It runs *before* the
    step that blanks lost cells, so a row the skip policy dropped still holds
    the ``[SKIPPED]`` marker, which is a non-empty string: the pass skips
    over it. Both the permanent row and a transient row whose request-level
    retries ran out (``max_retries=1``) stay lost, each asked exactly once,
    while the row that answered ``""`` is re-asked and recovered.

    Under the harness, auto-retry re-runs *failed* rows; that will change the
    transient row's fate here, on purpose and by editing this test.
    """
    empty_row = token_for(1)
    client = ScriptedLedgerClient(
        permanent={PERMANENT},
        transient={TRANSIENT: 1},
        empty={empty_row: 1},
    )

    result = _auto_retry(
        build(client, conformance_frame(6), max_retries=1, concurrency=1),
        attempts=2,
    ).execute()

    assert observe(result) == Outcome(
        columns=["marker", "answer"],
        rows=answered(0, 1)
        + [(PERMANENT, None)]
        + answered(3)
        + [(TRANSIENT, None)]
        + answered(5),
        errors=[skipped(2, failure(PERMANENT)), skipped(4, hiccup(TRANSIENT))],
        lost=[2, 4],
        complete=False,
        skipped=2,
        failed=0,
    )
    assert client.calls == Counter(
        {token_for(row): 1 for row in range(6)} | {empty_row: 2}
    )
    # Four answered calls in the first pass (the empty one included), plus the
    # one recovering call: the retry pass's cost is merged into the result.
    assert result.costs.total_cost == COST_PER_CALL * 5


def test_auto_retry_recovers_an_empty_row_inside_a_mega_batch():
    """The retry pass merges its rows back by position, also after batching.

    Row 4 answers ``""`` inside a batch of three; no error is recorded for it
    (an empty item is not a loss today), the retry pass re-asks only that
    row, and the recovered answer lands on row 4. The batch the permanent row
    took down stays lost and is not retried, for the reason given in the
    previous test.
    """
    client = ScriptedLedgerClient(permanent={PERMANENT}, empty={TRANSIENT: 1})

    result = _auto_retry(
        build(client, conformance_frame(6), batch_size=3), attempts=1
    ).execute()

    assert observe(result) == Outcome(
        columns=["marker", "answer"],
        rows=[(token_for(row), None) for row in (0, 1, 2)] + answered(3, 4, 5),
        errors=[skipped(row, failure(PERMANENT)) for row in (0, 1, 2)],
        lost=[0, 1, 2],
        complete=False,
        skipped=3,
        failed=0,
    )
    assert client.calls == Counter(
        {token_for(row): 1 for row in range(6)} | {TRANSIENT: 2}
    )
    assert result.costs.total_cost == COST_PER_CALL * 2


# ── resume ────────────────────────────────────────────────────────────────


def test_resume_under_skip_finishes_the_rows_an_aborted_run_left(tmp_path):
    """A run aborted by the fail policy, resumed under skip.

    The first attempt runs one row at a time under ``fail``, so the permanent
    row stops it at row 2 and rows 3-5 are never asked. The provider's own
    exception reaches the caller unwrapped. The resumed run re-asks nothing
    cached (rows 0 and 1), drops the permanent row under the default skip
    policy, retries the transient row, and returns the whole frame in input
    order with the same row fates as an uninterrupted run. Its counters cover
    the whole frame, not only the rows this attempt asked about.
    """
    frame = conformance_frame(6)
    first_client = ScriptedLedgerClient(permanent={PERMANENT})
    first = build(
        first_client,
        frame,
        checkpoint_dir=tmp_path,
        error_policy="fail",
        concurrency=1,
    ).build()

    with pytest.raises(ProviderError, match=PERMANENT):
        first.execute()
    assert first_client.tokens_seen == {token_for(0), token_for(1), PERMANENT}

    second_client = ScriptedLedgerClient(
        permanent={PERMANENT}, transient={TRANSIENT: 1}
    )
    resumed = (
        build(second_client, frame, checkpoint_dir=tmp_path, concurrency=1)
        .build()
        .execute(resume_from=first.session_id)
    )

    assert observe(resumed) == WHOLE_FRAME_OUTCOME
    assert second_client.calls == Counter(
        {PERMANENT: 1, token_for(3): 1, TRANSIENT: 2, token_for(5): 1}
    )
    assert resumed.metrics.total_rows == 6
    assert resumed.metrics.processed_rows == 6


# ── structured output ─────────────────────────────────────────────────────


def _structured_outcome_rows(lost: set[int]) -> list[tuple[str, str | None]]:
    return [
        (token_for(row), None if row in lost else token_for(row)) for row in range(6)
    ]


@pytest.mark.parametrize(
    ("batch_size", "lost", "attempts"),
    [
        # One call per row: only the permanent row is lost; row 4 retried.
        (1, [2], {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1}),
        # Batches of three: the permanent row's batch is lost whole, and the
        # hiccup re-sends its whole batch, so rows 3-5 are each asked twice.
        (3, [0, 1, 2], {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}),
    ],
    ids=["batch_size=1", "batch_size=3"],
)
def test_structured_output_reports_the_same_row_fates(batch_size, lost, attempts):
    """The structured path loses and recovers rows like the plain path.

    It parses through a different back half (the model returns objects, the
    disaggregator maps them to rows), which the design flags as a place the
    judge must also be wired. Every structured field of a lost row is
    missing, and the error list names every lost row with the permanent
    row's message.
    """
    client = ScriptedStructuredClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    result = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(6),
            input_columns=["marker"],
            output_columns=["token", "category"],
        )
        .with_prompt(PROMPT)
        .with_custom_llm_client(client)
        .with_structured_output(StructuredAnswerBatch)
        .with_batch_size(batch_size)
        .build()
        .execute()
    )

    assert observe(result, answer_column="token") == Outcome(
        columns=["marker", "token", "category"],
        rows=_structured_outcome_rows(set(lost)),
        errors=[skipped(row, failure(PERMANENT)) for row in lost],
        lost=lost,
        complete=False,
        skipped=len(lost),
        failed=0,
    )
    categories = [_cell(value) for value in result.to_pandas()["category"]]
    assert categories == [
        None if row in lost else answer_for(token_for(row)) for row in range(6)
    ]
    assert client.attempts == Counter(
        {token_for(row): count for row, count in attempts.items()}
    )


# ── no contract, no new columns ───────────────────────────────────────────


def test_no_contract_means_no_status_column():
    """Today's output has no ``status`` column, even on a run with losses.

    The interfaces spec promises that a run without a contract delivers
    exactly today's frame. A run with lost rows is the case most tempting to
    annotate, so it is the one checked: the frame holds the input column and
    the declared output column, and none of the five columns a contract adds.
    """
    client = ScriptedLedgerClient(permanent={PERMANENT}, transient={TRANSIENT: 1})

    frame = _run_sync(client).to_pandas()

    assert "status" not in frame.columns
    assert not set(CONTRACT_COLUMNS) & set(frame.columns)
    assert list(frame.columns) == ["marker", "answer"]
