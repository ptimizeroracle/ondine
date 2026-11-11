"""Re-ask (floor 2, F3), end to end through a running pipeline.

A row whose answer failed a check is asked again, once per round, with its
previous answer and the reason it was not accepted. Every run uses the
conformance provider with answers scripted per row and per call, so each
test knows which rows must be asked again, what each call must carry, and
what the client must receive.

Design: ``.claude/designs/re-ask.md``, section 6.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.harness.contract import ContractError
from ondine.harness.status import RowStatus
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    AttemptClient,
    ProviderError,
    conformance_frame,
    token_for,
)

LABELS = ["refund", "billing", "shipping"]
ACCEPTED = RowStatus.ACCEPTED
REPAIRED = RowStatus.REPAIRED
REJECTED = RowStatus.REJECTED
UNVERIFIED = RowStatus.UNVERIFIED

#: Names two labels, so the format fix cannot pick one: rejected.
TWO_LABELS = "It is about a refund, or maybe billing."


def _builder(
    client, rows: int, *, re_ask: int | None = 1, columns=("intent",), **policy
):
    builder = (
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
        .with_output_field("intent", allowed=LABELS)
    )
    if re_ask is not None or policy:
        builder = builder.with_contract_policy(
            **({"re_ask": re_ask} if re_ask is not None else {}), **policy
        )
    return builder


def _statuses(frame: pd.DataFrame) -> list[RowStatus]:
    return [RowStatus(value) for value in frame["status"]]


def _cell(frame: pd.DataFrame, row: int, column: str):
    value = frame[column].iloc[row]
    return None if pd.isna(value) else value


# ---------------------------------------------------------------------------
# 1-4. Which rows are asked again, with what, and what the client gets
# ---------------------------------------------------------------------------


def test_a_rejected_answer_fixed_by_the_re_ask_is_delivered_as_repaired():
    client = AttemptClient({1: [TWO_LABELS, "billing"]})

    frame = _builder(client, 3).build().execute().to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, ACCEPTED]
    assert _cell(frame, 1, "intent") == "billing"
    assert _cell(frame, 1, "repair_method") == "re-ask"


def test_the_re_ask_carries_the_original_prompt_the_answer_and_the_reason():
    client = AttemptClient({0: [TWO_LABELS, "refund"]})
    pipeline = _builder(client, 1).with_system_prompt("You sort bank tickets.").build()

    pipeline.execute()

    first, again = client.seen[token_for(0)]
    assert again[0].startswith(first[0])
    assert "Your previous answer was:\n" + TWO_LABELS in again[0]
    assert "It was not accepted: " in again[0]
    assert "not one of the 3 allowed values" in again[0]
    assert again[1].get("system_message") == first[1].get("system_message")


def test_with_re_ask_off_a_rejected_row_is_asked_once_and_stays_rejected():
    with_off = AttemptClient({1: [TWO_LABELS, "billing"]})
    unset = AttemptClient({1: [TWO_LABELS, "billing"]})

    off = _builder(with_off, 3, re_ask=0).build().execute().to_pandas()
    default = _builder(unset, 3, re_ask=None).build().execute().to_pandas()

    assert with_off.calls_for(1) == unset.calls_for(1) == 1
    assert _statuses(off)[1] is REJECTED
    pd.testing.assert_frame_equal(off, default)


def test_only_rows_that_failed_a_check_are_asked_again():
    client = AttemptClient(
        {1: [TWO_LABELS, "billing"], 2: ["shipping"]}, fail={3: {1, 2}}
    )

    _builder(client, 4).build().execute()

    assert [client.calls_for(row) for row in range(4)] == [1, 2, 1, 1]


# ---------------------------------------------------------------------------
# 5-7. When the re-ask does not fix the row
# ---------------------------------------------------------------------------


def test_a_re_ask_that_fails_again_leaves_the_row_rejected_with_both_attempts():
    client = AttemptClient({0: [TWO_LABELS, "a complaint"]})
    result = _builder(client, 1).build().execute()

    frame = result.to_pandas()
    entry = result.quarantine.iloc[0]

    assert _statuses(frame) == [REJECTED]
    assert "attempt 1 failed check" in _cell(frame, 0, "reason")
    assert "attempt 2 failed check" in _cell(frame, 0, "reason")
    assert entry["raw_answers"] == [TWO_LABELS, "a complaint"]
    assert entry["cost"] == 2 * COST_PER_CALL


def test_a_second_round_asks_only_the_rows_the_first_did_not_fix():
    client = AttemptClient(
        {0: [TWO_LABELS, "billing"], 1: [TWO_LABELS, TWO_LABELS, "shipping"]}
    )

    frame = _builder(client, 2, re_ask=2).build().execute().to_pandas()

    assert [client.calls_for(row) for row in range(2)] == [2, 3]
    assert _statuses(frame) == [REPAIRED, REPAIRED]
    assert _cell(frame, 1, "intent") == "shipping"


def test_a_provider_failure_on_the_re_ask_keeps_the_row_rejected():
    client = AttemptClient({0: [TWO_LABELS]}, fail={0: {2, 3}})
    result = _builder(client, 1).build().execute()

    outcome = result.row_ledger.rows[0]

    assert outcome.status is REJECTED
    assert outcome.sub_reason == "provider"


# ---------------------------------------------------------------------------
# 8-12. The same pass under every run shape
# ---------------------------------------------------------------------------


def test_an_unverified_row_is_asked_again_under_deliver_flagged():
    client = AttemptClient({0: [TWO_LABELS, "refund"], 1: [TWO_LABELS, TWO_LABELS]})

    frame = (
        _builder(client, 2, on_failure="deliver_flagged").build().execute().to_pandas()
    )

    assert _statuses(frame) == [REPAIRED, UNVERIFIED]
    assert _cell(frame, 0, "intent") == "refund"


def test_the_failing_field_of_a_two_field_row_is_fixed_by_the_re_ask():
    client = AttemptClient(
        {
            0: [
                json.dumps({"intent": "refund", "urgency": 9}),
                json.dumps({"intent": "refund", "urgency": 4}),
            ]
        }
    )
    pipeline = (
        _builder(
            client, 1, columns=("intent", "urgency"), partial_rows="keep_valid_fields"
        )
        .with_output_field("urgency", type="integer", minimum=1, maximum=5)
        .build()
    )

    frame = pipeline.execute().to_pandas()

    assert _statuses(frame) == [REPAIRED]
    assert (_cell(frame, 0, "intent"), _cell(frame, 0, "urgency")) == ("refund", 4)


def test_rows_asked_several_per_prompt_are_re_asked_too():
    client = AttemptClient({row: [TWO_LABELS, "billing"] for row in (1, 3)})
    pipeline = _builder(client, 5).with_batch_size(5).build()

    frame = pipeline.execute().to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, ACCEPTED, REPAIRED, ACCEPTED]
    assert [client.calls_for(row) for row in range(5)] == [1, 2, 1, 2, 1]


def test_the_re_ask_spends_only_what_the_budget_has_left():
    rows = 6
    client = AttemptClient({row: [TWO_LABELS, "billing"] for row in range(rows)})
    # Enough for the six first answers and two re-asks.
    budget = float(COST_PER_CALL) * (rows + 2) + float(COST_PER_CALL) / 2
    result = _builder(client, rows).with_max_budget(budget).build().execute()

    statuses = _statuses(result.to_pandas())

    assert float(result.costs.total_cost) <= budget
    assert statuses.count(REPAIRED) == 2
    assert statuses.count(REJECTED) == rows - 2


def test_a_row_recovered_by_auto_retry_with_a_bad_answer_is_then_re_asked():
    # Call 1 is lost; auto-retry's call 2 gets a rejected answer; call 3 re-asks.
    client = AttemptClient({0: ["unused", TWO_LABELS, "shipping"]}, fail={0: {1}})
    pipeline = _builder(client, 1).build()
    pipeline.specifications.processing.auto_retry_failed = True
    pipeline.specifications.processing.max_retry_attempts = 1

    frame = pipeline.execute().to_pandas()

    assert client.calls_for(0) == 3
    assert _statuses(frame) == [REPAIRED]
    assert _cell(frame, 0, "intent") == "shipping"


# ---------------------------------------------------------------------------
# 13-16. Resume, streams, the raw-answer cap, validation
# ---------------------------------------------------------------------------


def test_resuming_with_a_different_re_ask_is_refused(tmp_path):
    first = (
        _builder(AttemptClient({}, fail={3: {1, 2}}), 5)
        .with_error_policy("fail")
        .with_checkpoint_dir(str(tmp_path))
        .with_checkpoint_interval(1)
        .build()
    )
    with pytest.raises(ProviderError):
        first.execute()

    resumed = (
        _builder(AttemptClient({}), 5, re_ask=2)
        .with_checkpoint_dir(str(tmp_path))
        .build()
    )

    with pytest.raises(Exception, match="run policy changed"):
        resumed.execute(resume_from=first.session_id)


def test_each_streamed_chunk_re_asks_its_rows_and_the_report_adds_up():
    client = AttemptClient({row: [TWO_LABELS, "billing"] for row in (1, 4)})
    pipeline = _builder(client, 6).with_streaming(chunk_size=3).build()

    chunks = list(pipeline.execute_stream())

    statuses = [s for chunk in chunks for s in _statuses(chunk.to_pandas())]
    assert statuses.count(REPAIRED) == 2
    assert pipeline.run_report.counts[REPAIRED] == 2


def test_rows_past_the_raw_answer_cap_are_re_asked_with_the_reason_only():
    # The ledger holds raw answers for the first 1,000 quarantined rows.
    rows = 1002
    client = AttemptClient({row: [TWO_LABELS, "billing"] for row in range(rows)})

    # Every first answer fails: early stop would end the run after 50 rows.
    pipeline = _builder(client, rows, early_stop_fraction=None).build()
    frame = pipeline.execute().to_pandas()

    # A reason quotes the answer too, so look for the line that quotes it whole.
    quoted = "Your previous answer was:\n" + TWO_LABELS
    carries_answer = [quoted in client.prompts_for(row)[1] for row in range(rows)]
    assert _statuses(frame) == [REPAIRED] * rows
    assert carries_answer == [True] * 1000 + [False, False]
    reason = "is not one of the 3 allowed values."
    assert all(reason in client.prompts_for(row)[1] for row in range(rows))


@pytest.mark.parametrize("value", [-1, 4, 1.5, "once"])
def test_a_re_ask_outside_zero_to_three_is_refused_at_build(value):
    with pytest.raises(ContractError, match="re_ask"):
        _builder(AttemptClient({}), 1, re_ask=value).build()


def test_a_pipelined_stream_re_asks_its_rows_too():
    client = AttemptClient({row: [TWO_LABELS, "billing"] for row in (1, 4)})
    pipeline = _builder(client, 6).build()

    chunks = list(pipeline.execute_stream_pipelined(chunk_size=3))

    statuses = [s for chunk in chunks for s in _statuses(chunk.to_pandas())]
    assert statuses.count(REPAIRED) == 2
    assert [client.calls_for(row) for row in range(6)] == [1, 2, 1, 1, 2, 1]


def test_a_re_asked_rows_rejected_answer_and_its_correction_are_never_stored():
    from ondine.context.memory_store import InMemoryContextStore

    store = InMemoryContextStore()
    texts = ["please refund my order", "complaint about billing", "refund for billing"]
    frame = pd.DataFrame({"marker": [token_for(r) for r in range(3)], "text": texts})
    client = AttemptClient(
        {
            0: ["refund"],
            1: ["complaint about my order", "billing"],
            2: [TWO_LABELS, "refund"],
        }
    )
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            frame, input_columns=["marker", "text"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker} {text}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_output_field("intent", allowed=LABELS)
        .with_contract_policy(re_ask=1)
        .with_context_store(store)
        .with_grounding(threshold=0.01, action="flag")
        .build()
    )

    result = pipeline.execute()

    stored = [record.record.text for record in store._records.values()]  # noqa: SLF001
    assert _statuses(result.to_pandas()) == [ACCEPTED, REPAIRED, REPAIRED]
    for never in (
        "complaint about my order",
        TWO_LABELS,
        "previous answer",
        "not accepted",
    ):
        assert not any(never in text for text in stored), (never, stored)


class YesNoClient(AttemptClient):
    """Answers a yes/no question as the live model did: "No." first, "No"
    again unless the re-ask names the values it may use."""

    def invoke(self, prompt: str, **kwargs: Any):
        self._named = "consistent, inconsistent" in prompt
        return super().invoke(prompt, **kwargs)

    def _answer_token(self, token: str) -> str:
        call = self._record(token)
        if call == 1:
            return "No. The summary gets two details wrong."
        return "inconsistent" if self._named else "No"


def test_a_re_ask_names_the_allowed_values_so_a_yes_no_answer_can_be_fixed():
    client = YesNoClient({})
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(1), input_columns=["marker"], output_columns=["answer"]
        )
        .with_prompt("Is the summary in {marker} consistent with its document?")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_output_field("answer", allowed=["consistent", "inconsistent"])
        .with_contract_policy(re_ask=1)
        .build()
    )

    frame = pipeline.execute().to_pandas()

    assert client.prompts_for(0)[1].endswith(
        "Reply with only one of these values, and nothing else: consistent, inconsistent."
    )
    assert _statuses(frame) == [REPAIRED]
    assert _cell(frame, 0, "answer") == "inconsistent"
