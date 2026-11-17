"""The model audit (Ruler phase 3), end to end through a running pipeline.

A finished run's sampled rows are asked again, by a reference model, through
the run's own prompt; its answers are compared with the delivered values
under the run's contract. The claim is agreement with that model, never
accuracy. Two scripted conformance clients play the run's model and the
reference, so each test knows what the reference is sent, what it answers,
and what the audit must say.

Design: ``.claude/designs/certified-columns.md`` section 12.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import pytest

from ondine import Pipeline, PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore
from ondine.core.specifications import LLMProvider, LLMSpec
from ondine.ruler import CertificationError
from ondine.ruler.intervals import hypergeom_bounds
from tests.e2e.conformance_harness import AttemptClient, conformance_frame

LABELS = ["refund", "billing"]

#: The reference's price per call at its spec's rates (8 tokens in, 4 out,
#: 0.01 per 1k each way).
REFERENCE_COST = Decimal("0.00012")
#: The audit's estimate for one row: the run's 12 tokens per attempt, priced
#: both ways at 0.01/1k, for two calls (answer + re-ask), with the 1.25 margin.
ROW_ESTIMATE = Decimal("0.01") * 24 / 1000 * 2 * Decimal("1.25")


def _reference(
    answers: dict[int, list[str]], cost_per_call: Decimal = REFERENCE_COST, **options
) -> AttemptClient:
    spec = LLMSpec(
        provider=LLMProvider.OPENAI,
        model="reference-1",
        input_cost_per_1k_tokens=Decimal("0.01"),
        output_cost_per_1k_tokens=Decimal("0.01"),
    )
    return AttemptClient(
        answers, spec=spec, cost_per_call=cost_per_call, default="refund", **options
    )


def _run(rows: int, answers: dict[int, list[str]] | None = None, **build):
    """A finished run: every row answered "refund" unless ``answers`` says."""
    client = AttemptClient(answers or {}, default="refund")
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(rows), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_max_retries(1)
        .with_output_field("intent", allowed=LABELS)
    )
    if build.get("system"):
        builder = builder.with_system_prompt(build["system"])
    pipeline = builder.build()
    return pipeline, pipeline.execute(), client


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------


def test_an_audit_measures_agreement_with_the_reference_model():
    pipeline, result, _ = _run(10)
    reference = _reference({2: ["billing"], 5: ["billing"]})

    audit = pipeline.audit(result, client=reference, max_cost=1.0)

    intent = audit["intent"]
    assert audit.claim == "audit"
    assert "agreement with reference-1" in audit.claim_text
    assert (intent.estimate.compared, intent.estimate.right) == (10, 8)
    assert [(w.row, w.delivered, w.reference) for w in intent.wrong_rows] == [
        (2, "refund", ("billing",)),
        (5, "refund", ("billing",)),
    ]


def test_an_audit_never_calls_itself_accuracy_and_warns_models_share_mistakes():
    pipeline, result, _ = _run(4)

    text = pipeline.audit(result, client=_reference({}), max_cost=1.0).render()

    assert text.startswith("Audit:")
    assert "not accuracy" in text
    assert "Accuracy:" not in text
    assert "models share mistakes" in text


def test_a_sampled_audit_gives_an_exact_range_on_agreement():
    pipeline, result, _ = _run(300)
    wrong = set(range(0, 300, 10))
    reference = _reference({row: ["billing"] for row in wrong})
    sample = pipeline.draw_sample(result, size=50)

    intent = pipeline.audit(result, client=reference, max_cost=1.0, sample=sample)[
        "intent"
    ]

    expected_right = sum(row not in wrong for row in sample.rows)
    assert intent.estimate.compared == 50
    assert intent.estimate.right == expected_right
    low, high = hypergeom_bounds([expected_right], 50, 300)
    assert (intent.estimate.low, intent.estimate.high) == (
        int(low[0]) / 300,
        int(high[0]) / 300,
    )
    assert reference.tokens_seen == {f"row-{row:04d}" for row in sample.rows}


# ---------------------------------------------------------------------------
# What the reference sees, and what it never touches
# ---------------------------------------------------------------------------


def test_the_reference_is_sent_exactly_what_the_run_model_was_sent():
    pipeline, result, run_client = _run(3, system="You sort tickets.")
    reference = _reference({})

    pipeline.audit(result, client=reference, max_cost=1.0)

    for row in range(3):
        assert reference.prompts_for(row) == run_client.prompts_for(row)
    (_, options), *_ = reference.seen["row-0000"]
    assert options.get("system_message") == "You sort tickets."


def test_the_reference_sees_the_run_preprocessed_input():
    frame = pd.DataFrame({"marker": ["  row-0000   needs    cleaning  "]})
    run_client = AttemptClient({})
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["intent"])
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(run_client)
        .with_output_field("intent", allowed=LABELS)
        .build()
    )
    pipeline.specifications.processing.enable_preprocessing = True
    result = pipeline.execute()
    reference = _reference({})

    pipeline.audit(result, client=reference, max_cost=1.0)

    assert reference.prompts_for(0) == run_client.prompts_for(0)


class CountingStore(InMemoryContextStore):
    """A context store that counts every read and write."""

    def __init__(self) -> None:
        super().__init__()
        self.uses = 0

    def search(self, *args, **kwargs):
        self.uses += 1
        return super().search(*args, **kwargs)

    def store(self, *args, **kwargs):
        self.uses += 1
        return super().store(*args, **kwargs)


def test_the_reference_never_reads_or_writes_the_run_context_store():
    store = CountingStore()
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(1), input_columns=["marker"], output_columns=["intent"]
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(AttemptClient({}))
        .with_output_field("intent", allowed=LABELS)
        .with_context_store(store)
        .with_evidence_priming(top_k=3, min_score=0.0)
        .build()
    )
    result = pipeline.execute()
    uses = store.uses
    assert uses > 0  # the run itself searched the store

    pipeline.audit(result, client=_reference({}), max_cost=1.0)

    assert store.uses == uses


def test_an_audit_leaves_the_run_result_untouched():
    pipeline, result, _ = _run(5, {1: ["It is billing."]})
    frame = result.to_pandas().copy()
    report = repr(pipeline.run_report)
    cost = result.costs.total_cost
    attempts = [len(o.attempts) for o in result.row_ledger.rows]

    pipeline.audit(result, client=_reference({1: ["refund"]}), max_cost=1.0)

    pd.testing.assert_frame_equal(result.to_pandas(), frame)
    assert repr(pipeline.run_report) == report
    assert result.costs.total_cost == cost
    assert [len(o.attempts) for o in result.row_ledger.rows] == attempts


# ---------------------------------------------------------------------------
# When the reference cannot give an answer
# ---------------------------------------------------------------------------


def test_a_reference_answer_that_breaks_the_contract_is_re_asked_once():
    pipeline, result, _ = _run(1)
    reference = _reference({0: ["a complaint", "billing"]})

    intent = pipeline.audit(result, client=reference, max_cost=1.0)["intent"]

    assert reference.calls_for(0) == 2
    assert [w.reference for w in intent.wrong_rows] == [("billing",)]


def test_rows_the_reference_cannot_answer_are_unresolved_with_their_cause():
    pipeline, result, _ = _run(3)
    reference = _reference(
        {1: ["a complaint", "still a complaint"]}, fail={2: {1, 2, 3, 4}}
    )

    audit = pipeline.audit(result, client=reference, max_cost=1.0)

    intent = audit["intent"]
    assert (intent.estimate.right, intent.estimate.unresolved) == (1, 2)
    assert intent.unresolved_causes == {"broke the contract": 1, "provider": 1}
    assert "2 unresolved (1 broke the contract, 1 provider)" in audit.render()


def test_each_field_is_audited_on_its_own_rows_with_its_own_causes():
    answers = {
        0: [json.dumps({"intent": "refund", "urgency": 2})],
        1: [json.dumps({"intent": "billing", "urgency": 9})],  # urgency fails
        2: [json.dumps({"intent": "refund", "urgency": 4})],
    }
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(3),
            input_columns=["marker"],
            output_columns=["intent", "urgency"],
        )
        .with_prompt("Classify {marker}")
        .with_custom_llm_client(AttemptClient(answers))
        .with_batch_size(1)
        .with_output_field("intent", allowed=LABELS)
        .with_output_field("urgency", type="integer", minimum=1, maximum=5)
        .with_contract_policy(partial_rows="keep_valid_fields")
        .build()
    )
    result = pipeline.execute()
    broken = json.dumps({"intent": "refund", "urgency": 9})
    reference = _reference(
        {
            0: [broken, broken],  # intent agrees; urgency never valid
            1: [json.dumps({"intent": "billing", "urgency": 3})],
            2: [json.dumps({"intent": "billing", "urgency": 4})],
        }
    )

    audit = pipeline.audit(result, client=reference, max_cost=1.0)

    intent, urgency = audit["intent"], audit["urgency"]
    assert (intent.estimate.compared, intent.estimate.right) == (3, 2)
    assert intent.unresolved_causes == {}
    assert (urgency.estimate.compared, urgency.estimate.right) == (2, 1)
    assert urgency.unresolved_causes == {"broke the contract": 1}
    assert "Rows where every field agrees" in audit.render()


# ---------------------------------------------------------------------------
# Cost: a hard cap, an estimate before spending, nothing paid twice
# ---------------------------------------------------------------------------


def _passes(monkeypatch) -> list[int]:
    """Rows sent in each reference pass, in order, as the audit makes them."""
    sizes: list[int] = []
    original = Pipeline._reference_pass

    def spy(self, frame, plan):
        sizes.append(len(frame))
        return original(self, frame, plan)

    monkeypatch.setattr(Pipeline, "_reference_pass", spy)
    return sizes


def test_the_cap_is_never_crossed_and_rows_over_it_are_unresolved():
    pipeline, result, _ = _run(10)
    reference = _reference({})
    # Room for two rows at their estimate; real calls cost a fifth of it, so
    # later chunks, priced at what the asked rows cost, fit more.
    cap = float(ROW_ESTIMATE * 2)

    audit = pipeline.audit(result, client=reference, max_cost=cap)

    intent = audit["intent"]
    asked = len(reference.tokens_seen)
    # Chunk 1 at the static estimate (0.0006): 2 rows, spending 0.00024.
    # Ratio 0.00024 / 0.0012 = 0.2, factor min(1, 2 * 0.2) = 0.4, so a row
    # is priced 0.00024. Chunk 2: 0.00096 left, 4 rows (spent 0.00072).
    # Chunk 3: 0.00048 left, 2 rows (spent 0.00096). Chunk 4: 0.00024 left,
    # 1 row (spent 0.00108). 0.00012 left: no row fits. 2+4+2+1 = 9.
    assert asked == 9
    assert audit.reference.cost == asked * REFERENCE_COST
    assert audit.reference.cost <= Decimal(str(cap))
    assert intent.unresolved_causes == {"over the cost cap": 10 - asked}
    assert intent.estimate.right == asked


def test_later_chunks_are_priced_by_what_the_reference_charged(monkeypatch):
    pipeline, result, _ = _run(10)
    cap = float(ROW_ESTIMATE * 2)
    reference = _reference({})
    passes = _passes(monkeypatch)

    pipeline.audit(result, client=reference, max_cost=cap)

    # The static rule asks [2, 1, 1, 1, 1]: 6 rows in 5 passes. Calibrated,
    # the same cap asks 9 rows in 4 passes, and never spends past it.
    assert passes == [2, 4, 2, 1]
    assert sum(passes[:2]) > 2 + 1
    assert len(passes) < 5
    assert sum(passes) * REFERENCE_COST <= Decimal(str(cap))


def test_a_later_chunk_is_never_priced_above_the_static_estimate(monkeypatch):
    pipeline, result, _ = _run(12)
    # A call costs 0.0005, so the asked rows cost 5/6 of their estimate:
    # twice that is above 1, and the factor stays 1. Cap: 6 rows at estimate.
    reference = _reference({}, cost_per_call=Decimal("0.0005"))
    passes = _passes(monkeypatch)
    cap = float(ROW_ESTIMATE * 6)

    pipeline.audit(result, client=reference, max_cost=cap)

    # Chunk 1: 6 rows, spent 0.003. Left 0.0006: the static estimate fits
    # one row. A factor of 2 * 5/6 would price a row 0.001 and fit none.
    assert passes == [6, 1]
    assert sum(passes) * Decimal("0.0005") <= Decimal(str(cap))


def test_a_free_reference_is_asked_every_row_without_dividing_by_zero(monkeypatch):
    pipeline, result, _ = _run(60)
    free = AttemptClient(
        {},
        spec=LLMSpec(provider=LLMProvider.OPENAI, model="free-1"),
        cost_per_call=Decimal("0"),
        default="refund",
    )
    passes = _passes(monkeypatch)

    audit = pipeline.audit(result, client=free, max_cost=0.001)

    # Nothing is priced, so nothing is left over; the second chunk has no
    # static estimate to compare the (zero) spending with.
    assert passes == [50, 10]
    assert len(free.tokens_seen) == 60
    assert audit.reference.cost == 0
    assert audit["intent"].unresolved_causes == {}


def test_a_second_audit_with_the_first_asks_only_the_rows_left_unresolved():
    pipeline, result, _ = _run(10)
    reference = _reference({})
    first = pipeline.audit(result, client=reference, max_cost=float(ROW_ESTIMATE * 2))
    asked_first = set(reference.tokens_seen)

    second = pipeline.audit(result, client=reference, max_cost=1.0, previous=first)

    assert all(reference.calls[token] == 1 for token in reference.tokens_seen)
    assert len(reference.tokens_seen) == 10 > len(asked_first)
    assert second["intent"].estimate.right == 10
    assert second.reference.cost == 10 * REFERENCE_COST


def test_a_previous_audit_by_another_model_is_refused():
    pipeline, result, _ = _run(3)
    first = pipeline.audit(result, client=_reference({}), max_cost=1.0)
    other = AttemptClient(
        {}, spec=LLMSpec(provider=LLMProvider.OPENAI, model="other-2")
    )

    with pytest.raises(CertificationError, match="reference-1"):
        pipeline.audit(result, client=other, max_cost=1.0, previous=first)


def test_a_previous_audit_of_another_run_is_refused():
    pipeline, result, _ = _run(3)
    _, other_result, _ = _run(3, {0: ["billing"]})
    first = pipeline.audit(other_result, client=_reference({}), max_cost=1.0)

    with pytest.raises(CertificationError, match="another run"):
        pipeline.audit(result, client=_reference({}), max_cost=1.0, previous=first)


def test_the_estimate_spends_nothing_and_is_never_below_the_actual_cost():
    pipeline, result, _ = _run(6, system="You sort tickets.")
    reference = _reference({row: ["x", "billing"] for row in range(6)})

    estimate = pipeline.estimate_audit(result, client=reference)
    assert reference.tokens_seen == set()
    audit = pipeline.audit(result, client=reference, max_cost=1.0)

    assert estimate.rows == 6
    assert estimate.cost >= audit.reference.cost
    text = estimate.render()
    assert "reference-1" in text
    assert "marker" in text
    assert "a system prompt" in text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_run_model_cannot_audit_itself():
    pipeline, result, run_client = _run(2)

    with pytest.raises(CertificationError, match="cannot audit itself"):
        pipeline.audit(result, client=run_client, max_cost=1.0)


def test_an_audit_needs_exactly_one_of_a_model_and_a_client():
    pipeline, result, _ = _run(1)

    with pytest.raises(ValueError, match="model or a client"):
        pipeline.audit(result, max_cost=1.0)
    with pytest.raises(ValueError, match="model or a client"):
        pipeline.audit(
            result, model="openai/gpt-6-luna", client=_reference({}), max_cost=1.0
        )


@pytest.mark.parametrize("cap", [0, -1.0])
def test_a_cap_that_pays_for_nothing_is_refused(cap):
    pipeline, result, _ = _run(1)

    with pytest.raises(CertificationError, match="max_cost"):
        pipeline.audit(result, client=_reference({}), max_cost=cap)
