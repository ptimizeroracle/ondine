"""A declared output contract, run end to end (harness step 3).

Interfaces spec, section 6, for a running pipeline. With a checked contract
the client's frame comes from the row ledger's snapshot: every input row, in
input order, its value kept or blanked by its status, and five columns that
say what happened (``status``, ``reason``, ``checks``, ``repair_method``,
``delivered_by``). Rejected rows are listed apart from lost ones, the
quarantine table holds every row that did not pass, and the run report counts
every status.

Every run uses the conformance provider (``LedgerClient``): an answer is a
function of the row that asked, every call is recorded, so a test can say
which rows were asked, how often, and what each must deliver.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Literal

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore
from ondine.core.exceptions import PipelineExecutionError
from ondine.harness.contract import ContractError
from ondine.harness.delivery import HARNESS_COLUMNS, merge_reports
from ondine.harness.status import FINAL_STATUSES, RowStatus
from ondine.utils.budget_controller import BudgetExceededError
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    LedgerClient,
    ProviderError,
    conformance_frame,
    token_for,
)

ACCEPTED = RowStatus.ACCEPTED
REPAIRED = RowStatus.REPAIRED
REJECTED = RowStatus.REJECTED
UNVERIFIED = RowStatus.UNVERIFIED
FAILED = RowStatus.FAILED

LABELS = ["refund", "billing"]

#: One row of each kind: accepted, repaired by the format fix, rejected,
#: lost to the provider, accepted.
ANSWERS = {
    0: "refund",
    1: "I think it is billing.",
    2: "a complaint",
    4: "billing",
}
LOST = 3


class ScriptedClient(LedgerClient):
    """Answers each row with a scripted text; records every call as usual.

    ``answers`` maps a row to its text. A row in ``fail_rows`` raises (a
    provider failure), ``fail_times`` times or always.
    """

    def __init__(
        self,
        answers: dict[int, str],
        *,
        fail_rows: set[int] = frozenset(),
        fail_times: int | None = None,
    ) -> None:
        super().__init__(
            fail_tokens={token_for(row) for row in fail_rows}, fail_times=fail_times
        )
        self._scripted = {token_for(row): text for row, text in answers.items()}

    def _answer_token(self, token: str) -> str:
        default = super()._answer_token(token)
        return self._scripted.get(token, default)


def _builder(client: LedgerClient, rows: int = 5, *, columns=("intent",)):
    return (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(rows),
            input_columns=["marker"],
            output_columns=list(columns),
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_max_retries(1)
    )


def _labelled(client: LedgerClient, rows: int = 5):
    return _builder(client, rows).with_output_field("intent", allowed=LABELS)


def _statuses(frame: pd.DataFrame) -> list[RowStatus]:
    return [RowStatus(value) for value in frame["status"]]


def _cells(frame: pd.DataFrame, column: str) -> list:
    return [None if pd.isna(value) else value for value in frame[column]]


def _assert_totals(report, rows: int) -> None:
    assert report.total == rows
    assert sum(report.counts.values()) == rows
    assert set(report.counts) == set(FINAL_STATUSES)
    assert report.invariant_holds


# ---------------------------------------------------------------------------
# Delivery of each status, with the five columns
# ---------------------------------------------------------------------------


def test_each_row_gets_its_status_its_value_and_the_five_columns():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _labelled(client).build().execute()
    frame = result.to_pandas()

    assert list(frame.columns) == ["marker", "intent", *HARNESS_COLUMNS]
    assert _statuses(frame) == [ACCEPTED, REPAIRED, REJECTED, FAILED, ACCEPTED]
    assert _cells(frame, "intent") == ["refund", "billing", None, None, "billing"]
    assert list(frame["repair_method"]) == ["none", "format", "none", "none", "none"]
    assert _cells(frame, "delivered_by") == [
        "ledger-1",
        "ledger-1",
        None,
        None,
        "ledger-1",
    ]
    reasons = list(frame["reason"])
    assert reasons[0] == ""
    assert "format fix" in reasons[1]
    assert "'a complaint' is not one of the 2 allowed values" in reasons[2]
    assert "scripted failure for row-0003" in reasons[3]
    assert "intent: accepted" in frame["checks"][0]
    assert "intent: repaired (format)" in frame["checks"][1]
    assert "intent: rejected" in frame["checks"][2]


def test_rejected_rows_are_blank_and_listed_apart_from_lost_rows():
    """#312: a rejected row is not a lost row, and the run is not complete."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _labelled(client).build().execute()

    assert result.rejected_row_indices == [2]
    assert result.lost_row_indices == [LOST]
    assert [error.row_index for error in result.errors] == [LOST]
    assert result.is_complete is False
    assert result.metrics.failed_rows + result.metrics.skipped_rows == 1

    quarantine = result.quarantine
    assert list(quarantine["row"]) == [2, LOST]
    assert list(quarantine["status"]) == ["rejected", "failed"]
    assert quarantine["inputs"][0] == {"marker": token_for(2)}

    report = result.report
    _assert_totals(report, 5)
    assert report.counts[ACCEPTED] == 2
    assert report.counts[REPAIRED] == 1
    assert report.counts[REJECTED] == 1
    assert report.counts[FAILED] == 1
    assert report.coverage == pytest.approx(3 / 5)


def test_every_row_passing_makes_the_run_complete():
    client = ScriptedClient({row: LABELS[row % 2] for row in range(4)})

    result = _labelled(client, rows=4).build().execute()

    assert _statuses(result.to_pandas()) == [ACCEPTED] * 4
    assert result.is_complete is True
    assert result.rejected_row_indices == []
    assert result.lost_row_indices == []
    assert result.errors == []
    assert result.quarantine.empty


def test_execute_async_delivers_the_same_frame():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).with_async_execution(max_concurrency=1).build()

    result = asyncio.run(pipeline.execute_async())

    assert _statuses(result.to_pandas()) == [
        ACCEPTED,
        REPAIRED,
        REJECTED,
        FAILED,
        ACCEPTED,
    ]


def test_a_run_where_every_row_failed_still_raises_with_a_report():
    """The produced-nothing guard holds under a contract when no row answered."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    # The batch's one failing row loses the whole call: every row is failed.
    pipeline = _labelled(client).with_batch_size(5).build()

    with pytest.raises(PipelineExecutionError):
        pipeline.execute()

    assert [o.status for o in pipeline.row_ledger.rows] == [FAILED] * 5
    _assert_totals(pipeline.run_report, 5)


def test_a_mega_batch_without_a_failure_judges_every_row():
    client = ScriptedClient({**ANSWERS, LOST: "refund"})
    pipeline = _labelled(client).with_batch_size(2).build()

    frame = pipeline.execute().to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, REJECTED, ACCEPTED, ACCEPTED]
    assert _cells(frame, "intent") == ["refund", "billing", None, "refund", "billing"]


class _Ticket(BaseModel):
    intent: Literal["refund", "billing"]


def test_a_schema_declared_as_the_contract_rejects_what_it_forbids():
    client = ScriptedClient(
        {row: json.dumps({"intent": text}) for row, text in ANSWERS.items()},
        fail_rows={LOST},
    )
    pipeline = (
        _builder(client).with_structured_output(_Ticket, as_contract=True).build()
    )

    frame = pipeline.execute().to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, REJECTED, FAILED, ACCEPTED]
    assert _cells(frame, "intent") == ["refund", "billing", None, None, "billing"]


# ---------------------------------------------------------------------------
# Partial rows: whole_row (default) and keep_valid_fields
# ---------------------------------------------------------------------------

PARTIAL = {
    0: json.dumps({"intent": "refund", "amount": 12}),
    1: json.dumps({"intent": "refund", "amount": -5}),
    2: json.dumps({"intent": "billing", "amount": 3}),
}


def _two_fields(client: LedgerClient):
    return (
        _builder(client, rows=3, columns=("intent", "amount"))
        .with_output_field("intent", allowed=LABELS)
        .with_output_field("amount", type="number", minimum=0)
    )


@pytest.mark.parametrize(
    ("partial_rows", "intent_of_row_1"),
    [("whole_row", None), ("keep_valid_fields", "refund")],
)
def test_one_failed_required_field_rejects_the_row(partial_rows, intent_of_row_1):
    client = ScriptedClient(PARTIAL)
    builder = _two_fields(client)
    if partial_rows != "whole_row":
        builder = builder.with_contract_policy(partial_rows=partial_rows)

    result = builder.build().execute()
    frame = result.to_pandas()

    assert _statuses(frame) == [ACCEPTED, REJECTED, ACCEPTED]
    assert _cells(frame, "intent") == ["refund", intent_of_row_1, "billing"]
    assert _cells(frame, "amount") == [12, None, 3]
    assert result.rejected_row_indices == [1]
    assert "amount" in frame["reason"][1]


# ---------------------------------------------------------------------------
# deliver_flagged: an unverified row keeps its value
# ---------------------------------------------------------------------------


def test_deliver_flagged_keeps_the_failing_value_as_unverified():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = (
        _labelled(client).with_contract_policy(on_failure="deliver_flagged").build()
    )

    result = pipeline.execute()
    frame = result.to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, UNVERIFIED, FAILED, ACCEPTED]
    assert _cells(frame, "intent")[2] == "a complaint"
    assert frame["delivered_by"][2] == "ledger-1"
    assert result.rejected_row_indices == []
    assert result.is_complete is False
    assert list(result.quarantine["status"]) == ["unverified", "failed"]


# ---------------------------------------------------------------------------
# Auto-retry re-runs failed rows only
# ---------------------------------------------------------------------------


def test_auto_retry_reruns_failed_rows_and_never_rejected_ones():
    """Interfaces spec, section 5: the retry pass re-runs failed rows only."""
    client = ScriptedClient({**ANSWERS, LOST: "refund"}, fail_rows={LOST}, fail_times=1)
    pipeline = _labelled(client).build()
    pipeline.specifications.processing.auto_retry_failed = True
    pipeline.specifications.processing.max_retry_attempts = 2

    result = pipeline.execute()
    frame = result.to_pandas()

    # The rejected row was asked once; the lost row twice, then recovered.
    assert client.calls[token_for(2)] == 1
    assert client.calls[token_for(LOST)] == 2
    assert _statuses(frame) == [ACCEPTED, REPAIRED, REJECTED, ACCEPTED, ACCEPTED]
    assert _cells(frame, "intent")[LOST] == "refund"
    assert result.lost_row_indices == []
    assert result.errors == []
    assert result.rejected_row_indices == [2]
    _assert_totals(result.report, 5)


# ---------------------------------------------------------------------------
# A budget stop: rows never reached are failed, totals still add up
# ---------------------------------------------------------------------------


def test_a_budget_stop_at_row_120_of_200_fails_every_row_after_it():
    """Interfaces spec, section 6, test 8, under a checked contract."""
    client = ScriptedClient({row: "refund" for row in range(200)})
    pipeline = (
        _labelled(client, rows=200)
        .with_max_budget(float(COST_PER_CALL) * 119.5)
        .build()
    )

    with pytest.raises(BudgetExceededError):
        pipeline.execute()

    assert len(client.calls) == 120
    snapshot = pipeline.row_ledger
    assert sum(snapshot.counts.values()) == 200
    for outcome in snapshot.rows[120:]:
        assert outcome.status is FAILED
        assert outcome.reason.startswith("run stopped before this row")
    report = pipeline.run_report
    _assert_totals(report, 200)
    assert report.counts[FAILED] == 200


# ---------------------------------------------------------------------------
# Streaming: delivery per chunk, one report for the whole run
# ---------------------------------------------------------------------------


def test_each_chunk_delivers_its_rows_and_the_pipeline_reports_the_run():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).with_streaming(chunk_size=2).build()

    chunks = list(pipeline.execute_stream())

    assert [len(chunk.to_pandas()) for chunk in chunks] == [2, 2, 1]
    statuses = [s for chunk in chunks for s in _statuses(chunk.to_pandas())]
    assert statuses == [ACCEPTED, REPAIRED, REJECTED, FAILED, ACCEPTED]
    assert list(chunks[1].to_pandas().columns) == ["marker", "intent", *HARNESS_COLUMNS]
    assert chunks[1].rejected_row_indices == [0]
    assert chunks[1].lost_row_indices == [1]

    report = pipeline.run_report
    _assert_totals(report, 5)
    assert report.counts[REJECTED] == 1
    assert report.counts[FAILED] == 1
    assert report.counts[REPAIRED] == 1


def test_a_pipelined_stream_delivers_per_chunk_and_reports_the_run():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).build()

    chunks = list(pipeline.execute_stream_pipelined(chunk_size=2))

    statuses = [s for chunk in chunks for s in _statuses(chunk.to_pandas())]
    assert statuses == [ACCEPTED, REPAIRED, REJECTED, FAILED, ACCEPTED]
    _assert_totals(pipeline.run_report, 5)
    assert pipeline.run_report.counts[REJECTED] == 1


def test_a_streamed_file_carries_the_five_columns(tmp_path):
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).build()
    path = tmp_path / "streamed.parquet"

    async def drain() -> None:
        async for _ in pipeline.execute_stream_async(chunk_size=2, output_path=path):
            pass

    asyncio.run(drain())

    written = pd.read_parquet(path)
    assert list(written.columns) == ["marker", "intent", *HARNESS_COLUMNS]
    assert list(written["status"]) == [
        "accepted",
        "repaired",
        "rejected",
        "failed",
        "accepted",
    ]


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def test_the_output_file_carries_the_five_columns(tmp_path):
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    path = tmp_path / "out.csv"

    result = _labelled(client).with_output(str(path)).build().execute()

    written = pd.read_csv(path)
    assert list(written.columns) == ["marker", "intent", *HARNESS_COLUMNS]
    assert list(written["status"]) == [
        "accepted",
        "repaired",
        "rejected",
        "failed",
        "accepted",
    ]
    assert _cells(written, "intent") == ["refund", "billing", None, None, "billing"]
    assert list(result.to_pandas()["status"]) == list(written["status"])


# ---------------------------------------------------------------------------
# Provider batch: not covered yet (#338)
# ---------------------------------------------------------------------------


def test_submit_with_a_contract_raises_before_any_work():
    client = ScriptedClient(ANSWERS)
    pipeline = _labelled(client).with_execution_mode("provider_batch").build()

    with pytest.raises(NotImplementedError, match="provider-batch"):
        pipeline.submit()

    assert client.calls == Counter()


# ---------------------------------------------------------------------------
# No contract: today's frame, plus a quarantine and a report next to it
# ---------------------------------------------------------------------------


def test_without_a_contract_the_frame_is_unchanged_and_a_report_exists():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _builder(client).build().execute()
    frame = result.to_pandas()

    assert list(frame.columns) == ["marker", "intent"]
    assert _cells(frame, "intent") == [
        "refund",
        "I think it is billing.",
        "a complaint",
        None,
        "billing",
    ]
    assert result.rejected_row_indices == []
    assert result.lost_row_indices == [LOST]
    assert list(result.quarantine["row"]) == [LOST]
    assert list(result.quarantine["status"]) == ["failed"]
    _assert_totals(result.report, 5)
    assert result.report.counts[RowStatus.UNCHECKED] == 4
    assert result.report.coverage is None


# ---------------------------------------------------------------------------
# Resume: the contract must not change between attempts (D2)
# ---------------------------------------------------------------------------


def _crashing_run(builder_for, tmp_path):
    """A first attempt that crashes on row 3, leaving a checkpoint."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    first = (
        builder_for(client)
        .with_error_policy("fail")
        .with_checkpoint_dir(str(tmp_path))
        .with_checkpoint_interval(1)
        .build()
    )
    with pytest.raises(ProviderError):
        first.execute()
    return first.session_id


def test_resume_under_the_same_contract_delivers_every_row(tmp_path):
    session = _crashing_run(_labelled, tmp_path)
    client = ScriptedClient({**ANSWERS, LOST: "billing"})

    resumed = (
        _labelled(client)
        .with_checkpoint_dir(str(tmp_path))
        .build()
        .execute(resume_from=session)
    )

    assert set(client.calls) == {token_for(LOST), token_for(4)}
    assert _statuses(resumed.to_pandas()) == [
        ACCEPTED,
        REPAIRED,
        REJECTED,
        ACCEPTED,
        ACCEPTED,
    ]


def test_resume_with_a_changed_contract_is_refused_naming_the_change(tmp_path):
    session = _crashing_run(_labelled, tmp_path)
    client = ScriptedClient(ANSWERS)
    changed = (
        _builder(client)
        .with_output_field("intent", allowed=[*LABELS, "complaint"])
        .with_checkpoint_dir(str(tmp_path))
        .build()
    )

    with pytest.raises(ValueError, match="field 'intent'") as refused:
        changed.execute(resume_from=session)

    assert "contract" in str(refused.value)
    assert client.calls == Counter()


def test_resume_with_a_changed_policy_names_the_policy(tmp_path):
    session = _crashing_run(_labelled, tmp_path)
    client = ScriptedClient(ANSWERS)
    changed = (
        _labelled(client)
        .with_contract_policy(on_failure="deliver_flagged")
        .with_checkpoint_dir(str(tmp_path))
        .build()
    )

    with pytest.raises(ValueError, match="run policy"):
        changed.execute(resume_from=session)
    assert client.calls == Counter()


def test_resuming_a_checkpoint_without_a_contract_under_one_is_refused(tmp_path):
    session = _crashing_run(_builder, tmp_path)
    client = ScriptedClient(ANSWERS)

    with pytest.raises(ValueError, match="cannot prove"):
        _labelled(client).with_checkpoint_dir(str(tmp_path)).build().execute(
            resume_from=session
        )
    assert client.calls == Counter()


def test_dropping_the_contract_on_resume_is_refused(tmp_path):
    session = _crashing_run(_labelled, tmp_path)
    client = ScriptedClient(ANSWERS)

    with pytest.raises(ValueError, match="declares none"):
        _builder(client).with_checkpoint_dir(str(tmp_path)).build().execute(
            resume_from=session
        )
    assert client.calls == Counter()


# ---------------------------------------------------------------------------
# Step-3 review fixes
# ---------------------------------------------------------------------------

#: The reviewer's case: one accepted answer, two answers the contract rejects.
#: Each rejected answer shares words with its row's text, so it grounds.
STORE_TEXTS = [
    "please refund my order",
    "complaint about billing",
    "a complaint about refund",
]
STORE_ANSWERS = {0: "refund", 1: "complaint about my order", 2: "refund or billing"}


def _store_run(store, *, policy="reject", verification=None, texts=STORE_TEXTS):
    frame = pd.DataFrame(
        {"marker": [token_for(row) for row in range(3)], "text": list(texts)}
    )
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            frame, input_columns=["marker", "text"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker} {text}")
        .with_custom_llm_client(ScriptedClient(STORE_ANSWERS))
        .with_batch_size(1)
        .with_concurrency(1)
        .with_output_field("intent", allowed=LABELS)
        .with_contract_policy(on_failure=policy)
        .with_context_store(store)
    )
    if verification is None:
        builder = builder.with_grounding(threshold=0.01, action="flag")
    else:
        builder = verification(builder)
    return builder.build().execute()


def _stored_texts(store) -> list[str]:
    return [stored.record.text for stored in store._records.values()]


def test_a_rejected_answer_never_becomes_stored_evidence():
    """Review finding 1: only an answer that passed the contract is evidence.

    Grounding runs before delivery and stores every answer that grounds as
    "input -> answer", which later runs prime their prompts with. A rejected
    candidate must not get there, nor be scored.
    """
    store = InMemoryContextStore()

    result = _store_run(store)
    frame = result.to_pandas()

    assert _statuses(frame) == [ACCEPTED, REJECTED, REJECTED]
    texts = _stored_texts(store)
    assert any("refund" in text for text in texts)  # the accepted answer
    for rejected in ("complaint about my order", "refund or billing"):
        assert not any(rejected in text for text in texts), texts
    assert _cells(frame, "_grounding_score")[1:] == [None, None]


def test_an_unverified_answer_is_delivered_but_never_stored():
    """Under deliver_flagged the failing value ships, flagged, and stays out."""
    store = InMemoryContextStore()

    result = _store_run(store, policy="deliver_flagged")
    frame = result.to_pandas()

    assert _statuses(frame) == [ACCEPTED, UNVERIFIED, UNVERIFIED]
    assert _cells(frame, "intent")[1:] == [
        "complaint about my order",
        "refund or billing",
    ]
    texts = _stored_texts(store)
    for unverified in ("complaint about my order", "refund or billing"):
        assert not any(unverified in text for text in texts), texts


def test_a_rejected_answer_never_contradicts_an_accepted_one():
    """Contradiction detection compares only answers that passed the contract.

    The three rows share their key, so any two different answers conflict:
    only the accepted one may be compared, and it has no rival.
    """
    result = _store_run(
        InMemoryContextStore(),
        verification=lambda builder: builder.with_contradiction_detection(
            key_columns=["text"], value_columns=["intent"]
        ),
        texts=["same ticket"] * 3,
    )
    frame = result.to_pandas()

    flags = _cells(frame, "_contradiction") if "_contradiction" in frame else [None] * 3
    assert flags == [None, None, None]


def test_a_stream_report_merges_the_chunk_reports_with_no_false_violation():
    """Review finding 4: the whole-run report is the chunks' reports merged.

    Built from per-chunk reports, it needs no row value, so it claims no
    delivered row went without one.
    """
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).with_streaming(chunk_size=2).build()

    chunks = list(pipeline.execute_stream())
    report = pipeline.run_report

    _assert_totals(report, 5)
    assert report.violations == ()
    expected = merge_reports(
        [chunk.report for chunk in chunks], pipeline._resolve_contract()
    )
    assert report.counts == expected.counts
    assert report.rejection_reasons == expected.rejection_reasons
    assert report.cost_total == expected.cost_total


def test_a_pipelined_stream_report_has_no_false_violation():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).build()

    list(pipeline.execute_stream_pipelined(chunk_size=2))

    assert pipeline.run_report.violations == ()
    assert pipeline.run_report.counts[ACCEPTED] == 2


def test_an_unchecked_stream_report_counts_every_row():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _builder(client).with_streaming(chunk_size=2).build()

    list(pipeline.execute_stream())
    report = pipeline.run_report

    _assert_totals(report, 5)
    assert report.violations == ()
    assert report.counts[RowStatus.UNCHECKED] == 4
    assert report.counts[FAILED] == 1


def test_a_stream_stopped_by_its_budget_reports_the_rows_it_never_reached():
    client = ScriptedClient({row: "refund" for row in range(6)})
    pipeline = (
        _labelled(client, rows=6)
        .with_streaming(chunk_size=2)
        .with_max_budget(float(COST_PER_CALL) * 3.5)
        .build()
    )

    chunks = []

    def consume() -> None:
        for chunk in pipeline.execute_stream():
            chunks.append(chunk)

    with pytest.raises(BudgetExceededError):
        consume()

    report = pipeline.run_report
    _assert_totals(report, 6)
    reached = sum(len(chunk.to_pandas()) for chunk in chunks)
    assert report.counts[ACCEPTED] == reached
    assert report.counts[FAILED] == 6 - reached
    assert report.stop_reason is not None
    assert report.violations == ()


def test_under_a_contract_the_use_default_reason_does_not_claim_delivery():
    """Review finding 8: delivery blanks a failed row, default or not."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _labelled(client).with_error_policy("use_default").build().execute()
    frame = result.to_pandas()

    assert _statuses(frame)[LOST] is FAILED
    assert _cells(frame, "intent")[LOST] is None
    assert "delivered its default value" not in frame["reason"][LOST]


def test_without_a_contract_the_use_default_reason_is_unchanged():
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _builder(client).with_error_policy("use_default").build().execute()

    outcome = result.row_ledger.rows[LOST]
    assert "(the error policy delivered its default value)" in outcome.reason


def test_the_quarantine_keeps_the_raw_answer_of_a_rejected_row():
    """A rejected row's raw answer is in the quarantine; a delivered row's is not."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})

    result = _labelled(client).build().execute()
    quarantine = result.quarantine.set_index("row")

    assert list(quarantine.loc[2, "raw_answers"]) == ["a complaint"]
    assert ACCEPTED.value not in set(quarantine["status"])


def test_a_merged_result_keeps_its_snapshot_and_releases_its_live_ledger():
    """Review finding 9: a chunk result holds one ledger after the merge."""
    client = ScriptedClient(ANSWERS, fail_rows={LOST})
    pipeline = _labelled(client).with_streaming(chunk_size=2).build()

    chunks = list(pipeline.execute_stream())

    for chunk in chunks:
        assert chunk._live_ledger is None
        assert chunk.row_ledger is not None


def test_an_unchecked_result_read_once_keeps_only_its_snapshot():
    result = _builder(ScriptedClient(ANSWERS)).build().execute()

    assert result.row_ledger is not None
    assert result._live_ledger is None


def test_the_pipeline_and_its_result_share_one_snapshot():
    """Review finding 10: delivery returns what it computes; execute adopts it."""
    pipeline = _labelled(ScriptedClient(ANSWERS, fail_rows={LOST})).build()

    result = pipeline.execute()

    assert pipeline.row_ledger is result.row_ledger
    assert pipeline.run_report is result.report


# ── a source column named like a harness column ────────────────────────────


def _frame_with_extra(column: str, rows: int = 3) -> pd.DataFrame:
    """A frame carrying a column the prompt never reads, as ticket data does."""
    frame = conformance_frame(rows)
    frame[column] = ["open"] * rows
    return frame


@pytest.mark.parametrize("column", ["status", "reason"])
def test_an_undeclared_source_column_that_clashes_is_refused_before_any_call(column):
    """A source column the prompt never reads still clashes with the harness.

    Most ticket datasets carry a `status` column that is not an input column.
    Delivery would overwrite it, so the run must refuse it, and before the
    first model call, or the client pays for a result it then loses.
    """
    client = LedgerClient()
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            _frame_with_extra(column),
            input_columns=["marker"],
            output_columns=["intent"],
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_output_field("intent", allowed=LABELS)
        .build()
    )

    with pytest.raises(ContractError, match=column):
        pipeline.execute()

    assert client.prompts == []


def test_a_stream_refuses_a_clashing_source_column_before_any_call():
    client = LedgerClient()
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            _frame_with_extra("status", rows=4),
            input_columns=["marker"],
            output_columns=["intent"],
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_output_field("intent", allowed=LABELS)
        .with_streaming(chunk_size=2)
        .build()
    )

    with pytest.raises(ContractError, match="status"):
        list(pipeline.execute_stream())

    assert client.prompts == []


def test_a_clashing_source_column_is_fine_without_a_contract():
    """No contract adds no columns, so nothing clashes: today's behaviour."""
    client = LedgerClient()
    result = (
        PipelineBuilder.create()
        .from_dataframe(
            _frame_with_extra("status"),
            input_columns=["marker"],
            output_columns=["intent"],
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .build()
        .execute()
    )

    assert list(result.to_pandas()["status"]) == ["open"] * 3
