"""Early stop (floor 2), end to end through a running pipeline.

When more than ``early_stop_fraction`` of the first ``early_stop_window`` rows
have a first answer the contract rejects, the run asks the model about no
later row and raises ``EarlyStopError``. Every run uses the conformance
provider with answers scripted per row, so each test knows which rows the
model was asked about and what the run must report.

Design: ``.claude/designs/early-stop.md``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from ondine import EarlyStopError, PipelineBuilder
from ondine.harness.ledger import StopReason
from ondine.harness.status import RowStatus
from tests.e2e.conformance_harness import (
    AttemptClient,
    conformance_frame,
    token_for,
)

if TYPE_CHECKING:
    import pandas as pd

LABELS = ["refund", "billing", "shipping"]
#: Names two labels, so the format fix cannot pick one: rejected.
BAD = "It is about a refund, or maybe billing."
FAILED = RowStatus.FAILED
REJECTED = RowStatus.REJECTED
REPAIRED = RowStatus.REPAIRED


def _builder(client, rows: int, *, batch_size: int = 1, concurrency: int = 1, **policy):
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(rows), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(batch_size)
        .with_concurrency(concurrency)
        .with_max_retries(1)
        .with_output_field("intent", allowed=LABELS)
    )
    if policy:
        builder = builder.with_contract_policy(**policy)
    return builder


def _bad_rows(*rows: int) -> dict[int, list[str]]:
    return {row: [BAD] for row in rows}


def _asked(client: AttemptClient) -> set[int]:
    return {int(token.removeprefix("row-")) for token in client.calls}


def _statuses(frame: pd.DataFrame) -> list[RowStatus]:
    return [RowStatus(value) for value in frame["status"]]


# ---------------------------------------------------------------------------
# 1-3. A broken prompt stops after the window; a working one does not
# ---------------------------------------------------------------------------


def test_a_broken_prompt_stops_the_run_after_the_first_50_rows():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 200).build()

    with pytest.raises(EarlyStopError) as stopped:
        pipeline.execute()

    assert _asked(client) == set(range(50))
    snapshot = pipeline.row_ledger
    assert [o.status for o in snapshot.rows[:50]] == [REJECTED] * 50
    assert {o.status for o in snapshot.rows[50:]} == {FAILED}
    assert snapshot.rows[50].reason == "run stopped before this row: early stop"
    report = pipeline.run_report
    assert report.stop_reason is StopReason.EARLY_STOP
    assert sum(report.counts.values()) == 200
    error = stopped.value
    assert (error.rejected, error.window, error.fraction) == (50, 50, 0.5)


@pytest.mark.parametrize(("bad", "stops"), [(25, False), (26, True)])
def test_the_run_stops_only_above_the_fraction(bad, stops):
    client = AttemptClient(_bad_rows(*range(bad)))
    pipeline = _builder(client, 120).build()

    if stops:
        with pytest.raises(EarlyStopError):
            pipeline.execute()
        assert _asked(client) == set(range(50))
    else:
        frame = pipeline.execute().to_pandas()
        assert _asked(client) == set(range(120))
        assert _statuses(frame).count(REJECTED) == bad


def test_rows_failing_after_the_window_never_stop_the_run():
    client = AttemptClient(_bad_rows(*range(50, 200)))

    frame = _builder(client, 200).build().execute().to_pandas()

    assert _statuses(frame).count(REJECTED) == 150
    assert _asked(client) == set(range(200))


# ---------------------------------------------------------------------------
# 4-7. When early stop does not apply
# ---------------------------------------------------------------------------


def test_with_early_stop_off_a_broken_prompt_runs_every_row():
    client = AttemptClient({}, default=BAD)

    frame = (
        _builder(client, 120, early_stop_fraction=None).build().execute().to_pandas()
    )

    assert _statuses(frame) == [REJECTED] * 120


def test_a_run_no_longer_than_the_window_returns_its_rows():
    # Every row is already answered when the window is judged: stopping
    # would save nothing.
    client = AttemptClient({}, default=BAD)

    frame = _builder(client, 50).build().execute().to_pandas()

    assert _statuses(frame) == [REJECTED] * 50


def test_deliver_flagged_never_stops():
    client = AttemptClient({}, default=BAD)

    frame = (
        _builder(client, 120, on_failure="deliver_flagged")
        .build()
        .execute()
        .to_pandas()
    )

    assert _statuses(frame) == [RowStatus.UNVERIFIED] * 120


def test_without_a_contract_a_run_never_stops():
    client = AttemptClient({}, default=BAD)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(120), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .build()
    )

    pipeline.execute()

    assert _asked(client) == set(range(120))


# ---------------------------------------------------------------------------
# 8-10. Re-ask, batch prompts, concurrency
# ---------------------------------------------------------------------------


def test_re_ask_does_not_save_a_broken_prompt_and_is_never_paid_for():
    # The count is of first answers: a prompt that fails half its rows doubles
    # the cost of each of them even when re-ask rescues every one.
    client = AttemptClient({row: [BAD, "billing"] for row in range(200)})
    pipeline = _builder(client, 200, re_ask=1).build()

    with pytest.raises(EarlyStopError):
        pipeline.execute()

    assert sum(client.calls.values()) == 50


def test_a_re_ask_pass_over_many_rejected_rows_is_not_stopped_by_its_own_count():
    # Rows 50-199 fail first and are fixed by re-ask. The re-ask pass asks
    # 150 rows that all failed: its own first 50 are 100% "rejected".
    client = AttemptClient({row: [BAD, "billing"] for row in range(50, 200)})

    frame = _builder(client, 200, re_ask=1).build().execute().to_pandas()

    assert _statuses(frame)[50:] == [REPAIRED] * 150


def test_batch_prompts_are_cut_at_the_window_boundary():
    # A window of 12 with 5 rows per prompt: rows 0-11 are judged, so the
    # head is the prompts holding rows 0-4, 5-9 and 10-14 at most.
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(
        client, 100, batch_size=5, early_stop_window=12, early_stop_fraction=0.5
    ).build()

    with pytest.raises(EarlyStopError) as stopped:
        pipeline.execute()

    assert _asked(client) <= set(range(15))
    assert set(range(12)) <= _asked(client)
    assert stopped.value.window == 12


def test_with_concurrency_no_row_past_the_window_is_asked_before_the_check():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 400, concurrency=16).build()

    with pytest.raises(EarlyStopError):
        pipeline.execute()

    assert _asked(client) == set(range(50))


class YieldingClient(AttemptClient):
    """Gives the event loop a turn before each answer, as a network call does,
    so calls in flight really wait on the concurrency limit."""

    async def ainvoke(self, prompt: str, **kwargs: Any):
        await asyncio.sleep(0.001)
        return self.invoke(prompt, **kwargs)


def test_with_concurrency_a_run_that_passes_the_check_answers_every_row():
    # The window and the rest are asked in two passes of the model stage, each
    # in its own event loop; more rows than slots makes both wait on the
    # concurrency limit.
    client = YieldingClient({})
    frame = _builder(client, 400, concurrency=8).build().execute().to_pandas()

    assert _statuses(frame) == [RowStatus.ACCEPTED] * 400
    assert sum(client.calls.values()) == 400


# ---------------------------------------------------------------------------
# 11-12. Resume; what the error tells the client
# ---------------------------------------------------------------------------


def test_resuming_an_early_stopped_run_finishes_it_without_paying_the_window_again(
    tmp_path,
):
    first = AttemptClient({}, default=BAD)
    pipeline = _builder(first, 120).with_checkpoint_dir(str(tmp_path)).build()
    with pytest.raises(EarlyStopError):
        pipeline.execute()

    again = AttemptClient({}, default=BAD)
    resumed = _builder(again, 120).with_checkpoint_dir(str(tmp_path)).build()
    frame = resumed.execute(resume_from=pipeline.session_id).to_pandas()

    assert _asked(again) == set(range(50, 120))
    assert _statuses(frame) == [REJECTED] * 120


def test_the_error_names_the_numbers_the_reason_and_the_ways_out():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 200).build()

    with pytest.raises(EarlyStopError) as stopped:
        pipeline.execute()

    message = str(stopped.value)
    assert "50 of the first 50 rows" in message
    assert "above 50%" in message
    assert "not one of the 3 allowed values" in message
    assert f"resume_from={pipeline.session_id!r}" in message or (
        str(pipeline.session_id) in message
    )
    assert "early_stop_fraction=None" in message
    assert stopped.value.reasons[0].count == 50


# ---------------------------------------------------------------------------
# 13-15. Streams
# ---------------------------------------------------------------------------


def test_a_stream_stops_before_the_chunk_after_the_window():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 200).with_streaming(chunk_size=20).build()
    chunks = []

    def consume() -> None:
        chunks.extend(pipeline.execute_stream())

    with pytest.raises(EarlyStopError):
        consume()

    # Chunks 0-2 cover rows 0-59; row 49 is in chunk 2.
    assert len(chunks) == 3
    assert _asked(client) == set(range(60))
    report = pipeline.run_report
    assert report.stop_reason is StopReason.EARLY_STOP
    assert report.counts[FAILED] == 140


def test_a_stream_whose_window_ends_in_its_last_chunk_finishes():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 60).with_streaming(chunk_size=30).build()

    chunks = list(pipeline.execute_stream())

    assert sum(len(chunk.to_pandas()) for chunk in chunks) == 60
    assert pipeline.run_report.stop_reason is None


def test_a_pipelined_stream_stops_too():
    client = AttemptClient({}, default=BAD)
    pipeline = _builder(client, 200).build()

    with pytest.raises(EarlyStopError):
        list(pipeline.execute_stream_pipelined(chunk_size=25))

    # Chunks 0-1 hold the window; chunk 2 may already be in flight.
    assert set(range(50)) <= _asked(client) <= set(range(75))
    assert pipeline.run_report.stop_reason is StopReason.EARLY_STOP


def test_stream_rows_re_asked_into_repaired_still_count_as_first_answers():
    # A chunk runs its own re-ask before the stream sees it: the count must
    # still be of first answers, the same as without a stream.
    client = AttemptClient({row: [BAD, "billing"] for row in range(200)})
    pipeline = _builder(client, 200, re_ask=1).with_streaming(chunk_size=50).build()

    with pytest.raises(EarlyStopError):
        list(pipeline.execute_stream())

    assert _asked(client) == set(range(50))
    assert all(client.calls[token_for(row)] == 2 for row in range(50))
