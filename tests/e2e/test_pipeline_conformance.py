"""End-to-end conformance: what the pipeline promises about every row.

These tests run the real public API against `LedgerClient`, the only fake in
play. Nothing inside ondine is patched, so each test exercises the same stages,
executor, checkpointer and cost tracker a production run does.

Every test here targets one specific way a run can *look* successful while
having dropped, duplicated, or misplaced work. See `conformance_harness` for
why that is the failure mode worth this much machinery.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ondine import PipelineBuilder
from tests.e2e.conformance_harness import (
    LedgerClient,
    StructuredAnswerBatch,
    StructuredLedgerClient,
    answer_for,
    conformance_frame,
    expected_answers,
    token_for,
)

PROMPT = "Answer for {marker}"


def build(client: LedgerClient, frame: pd.DataFrame, **options):
    """A pipeline over `frame` wired to `client`, with the given options."""
    builder = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
        .with_prompt(PROMPT)
        .with_custom_llm_client(client)
    )
    if "batch_size" in options:
        builder = builder.with_batch_size(options["batch_size"])
    if "concurrency" in options:
        builder = builder.with_concurrency(options["concurrency"])
    if "max_retries" in options:
        builder = builder.with_max_retries(options["max_retries"])
    if "error_policy" in options:
        builder = builder.with_error_policy(options["error_policy"])
    if "checkpoint_dir" in options:
        builder = builder.with_checkpoint_dir(str(options["checkpoint_dir"]))
    if "checkpoint_interval" in options:
        builder = builder.with_checkpoint_interval(options["checkpoint_interval"])
    if "max_budget" in options:
        builder = builder.with_max_budget(options["max_budget"])
    return builder.build()


@pytest.mark.parametrize(
    ("rows", "batch_size", "concurrency"),
    [
        (12, 1, 1),  # the simplest path — a regression here breaks everything
        (12, 5, 1),  # mega-batching: 12 rows do not divide into 5
        (12, 1, 4),  # concurrency: completion order differs from row order
        (12, 5, 4),  # both at once, where reordering bugs actually live
    ],
)
def test_every_row_receives_its_own_answer(rows, batch_size, concurrency):
    """Row N holds the answer to row N — under batching and concurrency.

    Catches: results reordered by completion time, a partially-parsed batch
    padded with a neighbour's result, or a dropped row silently filled in.
    A row-count check cannot see any of these; only per-row values can.
    """
    client = LedgerClient()
    frame = conformance_frame(rows)

    result = build(
        client, frame, batch_size=batch_size, concurrency=concurrency
    ).execute()

    output = result.to_pandas()
    assert list(output["answer"]) == expected_answers(rows)
    assert client.tokens_seen == {token_for(index) for index in range(rows)}
    assert client.duplicated_tokens == set()


def test_reported_row_counts_match_the_rows_that_were_answered():
    """`processed_rows` counts work done, not rows walked past.

    Catches the #150 class of bug directly: a stage that skips a frame and
    reports the skip as success. The ledger is the independent witness — the
    pipeline cannot claim more processed rows than calls it actually made.
    """
    client = LedgerClient()
    result = build(client, conformance_frame(9), batch_size=4).execute()

    assert result.metrics.total_rows == 9
    assert result.metrics.processed_rows == 9
    assert result.metrics.failed_rows == 0
    assert sum(client.calls.values()) == 9


def test_reported_cost_is_the_sum_of_what_the_provider_charged():
    """Cost is accumulated from responses, not estimated after the fact.

    Catches accounting that silently loses a batch: a run whose answers are
    all correct can still under-report cost if a code path adds rows to the
    output without adding their price, which is how a budget cap gets bypassed.
    """
    from tests.e2e.conformance_harness import COST_PER_CALL

    client = LedgerClient()
    result = build(client, conformance_frame(6)).execute()

    calls = sum(client.calls.values())
    assert calls == 6
    assert result.costs.total_cost == COST_PER_CALL * calls


def test_transient_failure_is_retried_and_the_row_still_gets_its_answer():
    """A row whose call hiccups once ends up correct, not blank.

    Catches a retry path that re-runs the request but writes the *failure*
    into the output, and one that retries the wrong row. Row 3 is the only
    one scripted to fail, so any other row changing value is also caught.
    """
    failing = token_for(3)
    client = LedgerClient(fail_tokens={failing}, fail_times=1, transient=True)

    result = build(client, conformance_frame(6), max_retries=3).execute()

    output = result.to_pandas()
    assert list(output["answer"]) == expected_answers(6)
    assert client.calls[failing] == 2  # one hiccup, then the answer


def test_a_dropped_row_is_named_in_the_result_not_only_counted():
    """A partial loss must be reachable from code, not just from the log.

    The default error policy is SKIP, so one dead row out of five returns a
    frame with five rows, `success=True`, and the right columns. Before this
    was fixed, `result.errors` was empty on every run ever executed — the
    only trace of the loss was the string `[SKIPPED]` sitting in a cell.

    Catches: any future change that counts a skip without saying which row it
    was, which is what makes a 3-in-5,000,000 loss undetectable.
    """
    failing = token_for(2)
    client = LedgerClient(fail_tokens={failing}, fail_times=None)

    result = build(client, conformance_frame(5)).execute()

    assert result.metrics.skipped_rows == 1
    assert [error.row_index for error in result.errors] == [2]
    assert failing in result.errors[0].message
    # The surviving rows are still correct — the skip took one row, not the
    # alignment of everything after it.
    output = result.to_pandas()
    assert list(output["answer"].drop(index=2)) == [
        answer_for(token_for(index)) for index in (0, 1, 3, 4)
    ]


def test_streaming_produces_the_same_answers_as_a_single_pass():
    """Chunking is a memory strategy, not a different result.

    Each chunk is its own sub-pipeline, which is where shared state goes
    wrong: a client rebuilt per chunk, a chunk written to the wrong offset,
    or the last partial chunk dropped. 14 rows over chunks of 5 makes the
    ragged tail part of the test rather than a lucky multiple.
    """
    rows = 14
    frame = conformance_frame(rows)

    direct_client = LedgerClient()
    direct = build(direct_client, frame).execute().to_pandas()

    stream_client = LedgerClient()
    streamed = pd.concat(
        [
            chunk.to_pandas()
            for chunk in (
                PipelineBuilder.create()
                .from_dataframe(
                    frame, input_columns=["marker"], output_columns=["answer"]
                )
                .with_prompt(PROMPT)
                .with_custom_llm_client(stream_client)
                .with_streaming(chunk_size=5)
                .build()
                .execute_stream()
            )
        ],
        ignore_index=True,
    )

    assert list(streamed["answer"]) == list(direct["answer"]) == expected_answers(rows)
    # One client object served every chunk, so a stateful client (a session, a
    # shared rate limiter) is shared rather than rebuilt per chunk (#232).
    assert sum(stream_client.calls.values()) == rows
    assert stream_client.duplicated_tokens == set()


def test_a_client_implementing_only_the_declared_interface_completes_a_run():
    """The abstract methods are the whole contract — nothing undeclared.

    Catches the #235 regression class: the executor reaching for a method the
    base class never declared abstract, so a documented custom client dies
    with AttributeError on the first async call. This subclass deliberately
    implements the abstract methods and *nothing* else.
    """
    from decimal import Decimal

    from ondine.adapters.llm_client import LLMClient
    from ondine.core.models import LLMResponse
    from tests.e2e.conformance_harness import answer_for as _answer
    from tests.e2e.conformance_harness import conformance_spec

    class BareMinimumClient(LLMClient):
        def invoke(self, prompt, **kwargs):
            marker = prompt.rsplit(" ", 1)[-1].strip()
            return LLMResponse(
                text=_answer(marker),
                tokens_in=1,
                tokens_out=1,
                model=self.model,
                cost=Decimal("0"),
                latency_ms=0.0,
            )

        def structured_invoke(self, prompt, output_cls, **kwargs):
            return self.invoke(prompt)

        def estimate_tokens(self, text):
            return 1

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(4), input_columns=["marker"], output_columns=["answer"]
        )
        .with_prompt(PROMPT)
        .with_custom_llm_client(BareMinimumClient(conformance_spec()))
        .build()
    )

    assert list(pipeline.execute().to_pandas()["answer"]) == expected_answers(4)


def test_resume_answers_the_rows_that_were_lost_and_no_others(tmp_path):
    """A resumed run finishes the job without paying for it twice.

    Catches both halves of resume going wrong: re-calling rows already
    answered (the cost bug), and trusting the checkpoint so far that rows
    which never completed are left blank (the correctness bug).
    """
    rows = 8
    frame = conformance_frame(rows)
    dead = token_for(5)

    # First attempt dies for good on row 5, after earlier rows have completed.
    first_client = LedgerClient(fail_tokens={dead}, fail_times=None, transient=True)
    first = build(
        first_client,
        frame,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1,
        error_policy="fail",
        max_retries=1,
    )
    with pytest.raises(Exception):
        first.execute()

    # Second attempt: same checkpoint dir, a provider that no longer fails.
    second_client = LedgerClient()
    resumed = build(
        second_client,
        frame,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1,
        error_policy="fail",
    ).execute(resume_from=first.session_id)

    assert list(resumed.to_pandas()["answer"]) == expected_answers(rows)

    # The row that died is redone, and so is anything that finished *above*
    # it — those completions sat above a gap, so the watermark never covered
    # them. That re-work is bounded by the concurrency window.
    #
    # What must never happen is redoing the settled prefix: on a 5M-row run
    # that is the difference between resuming and starting over.
    settled_prefix = {token_for(index) for index in range(5)}
    redone = settled_prefix & second_client.tokens_seen
    assert not redone, f"resume re-called settled rows: {sorted(redone)}"
    assert dead in second_client.tokens_seen, "the row that failed was never redone"


def test_a_dropped_batch_names_every_row_it_took_down():
    """One failed mega-batch is `batch_size` lost rows, each identified.

    Catches a recorder that logs the batch's first row and forgets the rest —
    which would under-report a loss by up to `batch_size - 1` rows while the
    count stayed right.
    """
    client = LedgerClient(fail_tokens={token_for(2)}, fail_times=None)

    result = build(client, conformance_frame(8), batch_size=4).execute()

    assert result.metrics.skipped_rows == 4
    assert [error.row_index for error in result.errors] == [0, 1, 2, 3]


def test_a_provider_returning_nothing_is_reported_as_failure_not_as_data():
    """An empty response body must not become a value in the output.

    Found while benchmarking against a real provider: a reasoning model with a
    token cap too low to reach its answer returns an empty body. The pipeline
    reported `success=True`, `skipped=0`, `errors=[]`, and wrote the literal
    string "null" into all ten rows — a value that survives `isna()` and
    `== ""`, so no caller-side check finds it.

    The marker itself is deliberate (it signals the auto-retry pass), but
    auto-retry is off by default, so on default settings it leaks into user
    data. What must never be silent is the *count*: the rows are gone, and the
    result has to say so.
    """

    class EmptyProvider(LedgerClient):
        def invoke(self, prompt, **kwargs):
            response = super().invoke(prompt, **kwargs)
            response.text = ""
            return response

    client = EmptyProvider()
    result = build(client, conformance_frame(10), batch_size=5).execute()

    assert result.metrics.failed_rows == 10, (
        "a provider that answered nothing was recorded as having answered"
    )
    assert sorted(error.row_index for error in result.errors) == list(range(10))


# ── structured output ─────────────────────────────────────────────────────


def build_structured(client, frame, **options):
    """A structured-output pipeline over `frame`, wired to `client`."""
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            frame, input_columns=["marker"], output_columns=["token", "category"]
        )
        .with_prompt(PROMPT)
        .with_custom_llm_client(client)
        .with_structured_output(StructuredAnswerBatch)
    )
    if "batch_size" in options:
        builder = builder.with_batch_size(options["batch_size"])
    return builder.build()


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_structured_output_gives_each_row_its_own_object(batch_size):
    """Row N's parsed object is row N's, not a neighbour's.

    The structured path bypasses JsonBatchStrategy entirely — the model
    returns a Pydantic object and the disaggregator maps its items onto rows.
    Nothing else in the suite covers that mapping, and shape assertions cannot
    see it: six rows of well-formed objects look identical whether or not
    they landed on the right rows. The answers carry the marker that asked for
    them, so the mapping is checkable.
    """
    client = StructuredLedgerClient()
    frame = conformance_frame(6)

    output = (
        build_structured(client, frame, batch_size=batch_size).execute().to_pandas()
    )

    assert list(output["token"]) == list(frame["marker"])
    assert list(output["category"]) == expected_answers(6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Structured batches are matched to rows by position with no "
        "verification, so a model that reorders its items silently gives rows "
        "each other's answers: 4 of 6 wrong, success=True, zero failed rows. "
        "The plain JSON batch path recovers from the same reordering by id. "
        "See #255 — remove this marker when the paths agree."
    ),
)
def test_a_reordered_structured_batch_does_not_misassign_answers():
    """Models reorder batch responses. That must not become wrong data.

    This is the one failure mode that no counter can catch after the fact:
    every cell is populated, every count is right, and the values belong to
    the wrong rows.
    """
    client = StructuredLedgerClient(reverse=True)
    frame = conformance_frame(6)

    output = build_structured(client, frame, batch_size=3).execute().to_pandas()

    misassigned = int((output["marker"] != output["token"]).sum())
    assert misassigned == 0, f"{misassigned} rows hold another row's answer"
