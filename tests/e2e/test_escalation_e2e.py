"""Escalation (floor 2, F4), end to end through a running pipeline.

A row whose answer still fails a check after every re-ask round is asked
once more, by a stronger model. Two scripted conformance clients play the
run's model ("weak") and the escalation model ("strong"), so each test knows
which rows reach the strong one, what it is sent, and what the client gets.

Design: ``.claude/designs/escalation.md``, section 5.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.specifications import LLMProvider, LLMSpec
from ondine.harness.contract import ContractError
from ondine.harness.status import RowStatus
from tests.e2e.conformance_harness import (
    COST_PER_CALL,
    AttemptClient,
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

#: The strong model's price per call at its spec's rates (8 tokens in, 4 out,
#: 0.01 per 1k each way): more than the weak one's, as escalation costs.
STRONG_COST = Decimal("0.00012")


def _strong(answers: dict[int, list[str]], **options) -> AttemptClient:
    spec = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="strong-1",
        input_cost_per_1k_tokens=Decimal("0.01"),
        output_cost_per_1k_tokens=Decimal("0.01"),
    )
    return AttemptClient(answers, spec=spec, cost_per_call=STRONG_COST, **options)


def _builder(weak, strong, rows: int, *, re_ask: int = 1, **policy):
    return (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(rows), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(weak)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_max_retries(1)
        .with_output_field("intent", allowed=LABELS)
        .with_contract_policy(re_ask=re_ask, **policy)
        .with_escalation(client=strong)
    )


def _statuses(frame: pd.DataFrame) -> list[RowStatus]:
    return [RowStatus(value) for value in frame["status"]]


def _cell(frame: pd.DataFrame, row: int, column: str):
    value = frame[column].iloc[row]
    return None if pd.isna(value) else value


# ---------------------------------------------------------------------------
# Which rows reach the strong model, with what, and what the client gets
# ---------------------------------------------------------------------------


def test_a_row_the_weak_model_cannot_fix_is_fixed_by_the_strong_one():
    weak = AttemptClient({0: [TWO_LABELS, TWO_LABELS]})
    strong = _strong({0: ["billing"]})

    frame = _builder(weak, strong, 1).build().execute().to_pandas()

    assert _statuses(frame) == [REPAIRED]
    assert _cell(frame, 0, "intent") == "billing"
    assert _cell(frame, 0, "repair_method") == "escalate"
    assert _cell(frame, 0, "delivered_by") == "strong-1"
    assert _cell(frame, 0, "checks") == "intent: repaired (escalate)"


def test_only_rows_still_failing_after_re_ask_reach_the_strong_model():
    # Row 0 passes, row 1 is fixed by re-ask, row 2 needs the strong model.
    weak = AttemptClient({1: [TWO_LABELS, "billing"], 2: [TWO_LABELS, TWO_LABELS]})
    strong = _strong({2: ["shipping"]})

    frame = _builder(weak, strong, 3).build().execute().to_pandas()

    assert [weak.calls_for(row) for row in range(3)] == [1, 2, 2]
    assert [strong.calls_for(row) for row in range(3)] == [0, 0, 1]
    assert _statuses(frame) == [ACCEPTED, REPAIRED, REPAIRED]
    assert list(frame["repair_method"]) == ["none", "re-ask", "escalate"]


def test_a_run_whose_rows_all_pass_never_calls_the_strong_model():
    weak = AttemptClient({})
    strong = _strong({})

    result = _builder(weak, strong, 4).build().execute()

    assert strong.tokens_seen == set()
    assert result.costs.total_cost == 4 * COST_PER_CALL


def test_the_strong_model_gets_the_prompt_and_the_latest_failed_answer():
    weak = AttemptClient({0: [TWO_LABELS, "a complaint"]})
    strong = _strong({0: ["refund"]})
    pipeline = _builder(weak, strong, 1).with_system_prompt("You sort tickets.").build()

    pipeline.execute()

    (first, _), *_ = weak.seen[token_for(0)]
    (asked, options), *_ = strong.seen[token_for(0)]
    assert asked.startswith(first)
    assert "Your previous answer was:\na complaint" in asked
    assert "Reply with only one of these values" in asked
    assert options.get("system_message") == "You sort tickets."


def test_with_re_ask_off_the_strong_model_is_asked_straight_after_the_run():
    weak = AttemptClient({0: [TWO_LABELS]})
    strong = _strong({0: ["billing"]})

    frame = _builder(weak, strong, 1, re_ask=0).build().execute().to_pandas()

    assert (weak.calls_for(0), strong.calls_for(0)) == (1, 1)
    assert _cell(frame, 0, "repair_method") == "escalate"


# ---------------------------------------------------------------------------
# When escalation fails too
# ---------------------------------------------------------------------------


def test_a_row_the_strong_model_cannot_fix_stays_rejected_with_every_attempt():
    weak = AttemptClient({0: [TWO_LABELS, TWO_LABELS]})
    strong = _strong({0: ["a complaint"]})
    result = _builder(weak, strong, 1).build().execute()

    frame = result.to_pandas()

    assert _statuses(frame) == [REJECTED]
    assert _cell(frame, 0, "reason").endswith(
        "[path: attempt 1 failed check -> attempt 2 failed check "
        "-> attempt 3 failed check]"
    )
    assert result.quarantine.iloc[0]["cost"] == 2 * COST_PER_CALL + STRONG_COST


def test_a_provider_failure_on_escalation_keeps_the_row_rejected():
    weak = AttemptClient({0: [TWO_LABELS, TWO_LABELS]})
    strong = _strong({0: ["billing"]}, fail={0: {1, 2, 3}})

    outcome = _builder(weak, strong, 1).build().execute().row_ledger.rows[0]

    assert outcome.status is REJECTED
    assert outcome.sub_reason == "provider"


# ---------------------------------------------------------------------------
# Budget, policies, run shapes
# ---------------------------------------------------------------------------


def test_escalation_is_budgeted_at_the_strong_models_price():
    rows = 6
    weak = AttemptClient({row: [TWO_LABELS] for row in range(rows)})
    strong = _strong({row: ["billing"] for row in range(rows)})
    # Six weak answers, then room for two strong calls at their estimate
    # (12 tokens each way at 0.01/1k, with the 1.25 margin) but not three.
    estimate = Decimal("0.01") * 24 / 1000 * Decimal("1.25")
    budget = float(COST_PER_CALL * rows + estimate * Decimal("2.5"))

    result = (
        _builder(weak, strong, rows, re_ask=0).with_max_budget(budget).build().execute()
    )

    statuses = _statuses(result.to_pandas())
    assert statuses[:2] == [REPAIRED, REPAIRED]
    assert statuses[2:] == [REJECTED] * (rows - 2)
    assert float(result.costs.total_cost) <= budget


def test_an_unverified_row_is_escalated_under_deliver_flagged():
    weak = AttemptClient({0: [TWO_LABELS, TWO_LABELS], 1: [TWO_LABELS, TWO_LABELS]})
    strong = _strong({0: ["refund"], 1: [TWO_LABELS]})

    frame = (
        _builder(weak, strong, 2, on_failure="deliver_flagged")
        .build()
        .execute()
        .to_pandas()
    )

    assert _statuses(frame) == [REPAIRED, UNVERIFIED]
    assert _cell(frame, 0, "repair_method") == "escalate"


def test_rows_asked_several_per_prompt_are_escalated_one_by_one_too():
    weak = AttemptClient({row: [TWO_LABELS, TWO_LABELS] for row in (1, 3)})
    strong = _strong({row: ["billing"] for row in (1, 3)})
    pipeline = _builder(weak, strong, 5).with_batch_size(5).build()

    frame = pipeline.execute().to_pandas()

    assert _statuses(frame) == [ACCEPTED, REPAIRED, ACCEPTED, REPAIRED, ACCEPTED]
    assert [strong.calls_for(row) for row in range(5)] == [0, 1, 0, 1, 0]


def test_each_streamed_chunk_escalates_its_rows_and_the_report_adds_up():
    weak = AttemptClient({row: [TWO_LABELS, TWO_LABELS] for row in (1, 4)})
    strong = _strong({row: ["billing"] for row in (1, 4)})
    pipeline = _builder(weak, strong, 6).with_streaming(chunk_size=3).build()

    chunks = list(pipeline.execute_stream())

    statuses = [s for chunk in chunks for s in _statuses(chunk.to_pandas())]
    assert statuses.count(REPAIRED) == 2
    assert pipeline.run_report.counts[REPAIRED] == 2
    assert [strong.calls_for(row) for row in range(6)] == [0, 1, 0, 0, 1, 0]


def test_early_stop_still_judges_first_answers_before_any_escalation():
    rows = 60
    weak = AttemptClient({row: [TWO_LABELS] for row in range(rows)})
    strong = _strong({row: ["billing"] for row in range(rows)})

    from ondine import EarlyStopError

    with pytest.raises(EarlyStopError):
        _builder(weak, strong, rows, re_ask=0).build().execute()
    assert strong.tokens_seen == set()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_escalation_without_a_contract_is_refused_at_build():
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(1), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(AttemptClient({}))
        .with_escalation(client=_strong({}))
    )

    with pytest.raises(ContractError, match="with_escalation"):
        builder.build()


@pytest.mark.parametrize("rounds", [0, 4, True, 1.5])
def test_escalation_rounds_outside_one_to_three_are_refused_at_build(rounds):
    builder = _builder(AttemptClient({}), _strong({}), 1)
    builder.with_escalation(client=_strong({}), rounds=rounds)

    with pytest.raises(ContractError, match="escalation rounds"):
        builder.build()


def test_escalation_needs_exactly_one_of_a_model_and_a_client():
    builder = PipelineBuilder.create()

    with pytest.raises(ValueError, match="model or a client"):
        builder.with_escalation()
    with pytest.raises(ValueError, match="model or a client"):
        builder.with_escalation(model="openai/gpt-6-luna", client=_strong({}))


def test_an_escalation_model_is_configured_like_with_llm():
    pipeline = (
        _builder(AttemptClient({}), _strong({}), 1)
        .with_escalation(model="openai/gpt-6-luna", temperature=0.2)
        .build()
    )

    escalation = pipeline.specifications.escalation
    assert escalation.llm.model == "openai/gpt-6-luna"
    assert escalation.llm.provider is LLMProvider.LITELLM
    assert escalation.llm.temperature == 0.2
    assert escalation.rounds == 1
