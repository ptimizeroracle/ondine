"""Fixes from the harness step-2 review, each pinned by a test.

* **A declared contract does not run yet** (finding 1). Until delivery is
  wired (harness step 3), a run under a checked contract would blank rejected
  rows and report nothing. Every entry path refuses it before any model call.
* **``with_parser(RegexParser(...))`` runs** (finding 2) and returns the
  values it returns on main.
* **One response-format check** (finding 5): the pipeline validates the format
  with :func:`validate_response_format` and builds no parser to discard.
* **One reading of "the model answered"** (finding 6), public in the model
  stage, used by the parse stage.
* **One lookup of the full input** (finding 7): ``ExecutionContext.full_input``.
* **A passing unchecked verdict is cheap to record** (finding 11).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.harness import ledger as ledger_module
from ondine.harness.contract import build_contract
from ondine.harness.judge import judge
from ondine.harness.ledger import LedgerPolicy, RowLedger
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages import response_parser_stage
from ondine.stages.llm_invocation_stage import model_answered
from ondine.stages.parser_factory import validate_response_format
from ondine.stages.response_parser_stage import RegexParser
from tests.e2e.conformance_harness import (
    LedgerClient,
    answer_for,
    conformance_frame,
    token_for,
)

ROWS = 4
UNENFORCED = (
    "Output contracts are declared but not enforced yet; they take effect "
    "when delivery is wired (harness step 3). Remove the with_output_field / "
    "with_output_relation / with_contract_policy / as_contract=True "
    "declarations to run."
)


def _builder(client: LedgerClient, *, columns: list[str] | None = None):
    return (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(ROWS),
            input_columns=["marker"],
            output_columns=columns or ["answer"],
        )
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_concurrency(1)
    )


def _declared(client: LedgerClient):
    return _builder(client).with_output_field("answer", pattern=r"^ANS-row-\d{4}$")


# ---------------------------------------------------------------------------
# Finding 1: a declared contract refuses to run, on every entry path
# ---------------------------------------------------------------------------


def _execute(pipeline):
    return pipeline.execute()


def _execute_async(pipeline):
    return asyncio.run(pipeline.execute_async())


def _execute_stream(pipeline):
    return list(pipeline.execute_stream())


def _execute_stream_async(pipeline):
    async def drain():
        return [chunk async for chunk in pipeline.execute_stream_async(chunk_size=2)]

    return asyncio.run(drain())


def _execute_stream_pipelined(pipeline):
    return list(pipeline.execute_stream_pipelined(chunk_size=2))


@pytest.mark.parametrize(
    ("entry", "configure"),
    [
        (_execute, lambda b: b),
        (_execute_async, lambda b: b.with_async_execution(max_concurrency=1)),
        (_execute_stream, lambda b: b.with_streaming(chunk_size=2)),
        (_execute_stream_async, lambda b: b),
        (_execute_stream_pipelined, lambda b: b),
    ],
    ids=["execute", "execute_async", "stream", "stream_async", "stream_pipelined"],
)
def test_a_declared_contract_refuses_to_run_before_any_model_call(entry, configure):
    client = LedgerClient()
    pipeline = configure(_declared(client)).build()

    with pytest.raises(NotImplementedError) as raised:
        entry(pipeline)

    assert str(raised.value) == UNENFORCED
    assert client.calls == {}
    assert client.prompts == []


def test_the_declared_contract_stays_the_runs_contract():
    """The seam step 3 builds on: the refusal does not hide the contract."""
    pipeline = _declared(LedgerClient()).build()

    contract = pipeline._resolve_contract()

    assert contract is pipeline._components.output_contract
    assert contract.checked


def test_without_a_contract_every_entry_path_still_runs():
    client = LedgerClient()

    result = _builder(client).build().execute()

    assert list(result.to_pandas()["answer"]) == [
        answer_for(token_for(row)) for row in range(ROWS)
    ]


# ---------------------------------------------------------------------------
# Finding 2: with_parser(RegexParser) runs, with today's values
# ---------------------------------------------------------------------------


def test_with_parser_regex_parser_returns_todays_values():
    client = LedgerClient()
    pipeline = (
        _builder(client, columns=["score"])
        .with_parser(RegexParser({"score": r"ANS-(row-\d{4})"}))
        .build()
    )

    result = pipeline.execute()

    assert list(result.to_pandas()["score"]) == [token_for(row) for row in range(ROWS)]
    assert result.errors == []
    assert result.row_ledger is not None
    assert [o.status for o in result.row_ledger.rows] == [RowStatus.UNCHECKED] * ROWS


# ---------------------------------------------------------------------------
# Finding 5: one response-format check, no parser built to be discarded
# ---------------------------------------------------------------------------


class TestValidateResponseFormat:
    def test_raw_reads_text_for_one_column_and_json_for_several(self):
        assert validate_response_format("raw", ["a"]) == "text"
        assert validate_response_format("RAW", ["a", "b"]) == "json"

    def test_json_and_regex_read_as_named(self):
        assert validate_response_format("json", ["a"]) == "json"
        assert validate_response_format("regex", ["a"], {"a": r"(\w+)"}) == "regex"

    def test_regex_without_patterns_is_refused(self):
        with pytest.raises(ValueError, match="requires regex_patterns"):
            validate_response_format("regex", ["a"])

    def test_regex_missing_a_columns_pattern_is_refused(self):
        with pytest.raises(ValueError, match="Missing regex patterns"):
            validate_response_format("regex", ["a", "b"], {"a": r"(\w+)"})

    def test_an_unknown_format_is_refused(self):
        with pytest.raises(ValueError, match="Invalid response_format"):
            validate_response_format("yaml", ["a"])


def test_the_pipeline_builds_no_parser_to_discard(monkeypatch):
    from ondine.api import pipeline as pipeline_module

    def no_parser(*args, **kwargs):  # pragma: no cover - the failure itself
        raise AssertionError("the pipeline built a parser only for its errors")

    monkeypatch.setattr(
        pipeline_module, "create_response_parser", no_parser, raising=False
    )

    result = _builder(LedgerClient()).build().execute()

    assert result.to_pandas()["answer"].notna().all()


def test_a_regex_format_missing_a_pattern_still_fails_the_run():
    """Today's error, raised by the one format check."""
    client = LedgerClient()
    pipeline = _builder(client, columns=["a", "b"]).build()
    pipeline.specifications.prompt.response_format = "regex"
    pipeline.specifications.prompt.regex_patterns = {"a": r"(\w+)"}

    with pytest.raises(Exception, match="Missing regex patterns"):
        pipeline.execute()


# ---------------------------------------------------------------------------
# Finding 6: one public reading of "the model answered"
# ---------------------------------------------------------------------------


def _response(**metadata) -> LLMResponse:
    return LLMResponse(
        text="x",
        tokens_in=1,
        tokens_out=1,
        model="m",
        cost=Decimal("0"),
        latency_ms=1.0,
        metadata=metadata,
    )


def test_model_answered_reads_the_error_policys_mark():
    assert model_answered(_response())
    assert not model_answered(_response(action="skipped"))
    assert not model_answered(_response(action="substituted"))
    assert model_answered("plain text")


def test_the_parse_stage_imports_no_private_name_of_the_model_stage():
    assert not hasattr(response_parser_stage, "_policy_action")


# ---------------------------------------------------------------------------
# Finding 7: one lookup of the full input
# ---------------------------------------------------------------------------


class TestFullInput:
    def test_preprocessed_data_comes_first(self):
        context = ExecutionContext()
        context.intermediate_data["loaded_data"] = "loaded"
        context.intermediate_data["preprocessed_data"] = "preprocessed"

        assert context.full_input() == "preprocessed"

    def test_else_the_loaded_data(self):
        context = ExecutionContext()
        context.intermediate_data["loaded_data"] = "loaded"

        assert context.full_input() == "loaded"

    def test_an_empty_preprocessed_frame_still_counts(self):
        context = ExecutionContext()
        context.intermediate_data["loaded_data"] = "loaded"
        context.intermediate_data["preprocessed_data"] = []

        assert context.full_input() == []

    def test_else_the_default(self):
        assert ExecutionContext().full_input() is None
        assert ExecutionContext().full_input(default=["w"]) == ["w"]


# ---------------------------------------------------------------------------
# Finding 11: a passing unchecked verdict skips the ledger's copy
# ---------------------------------------------------------------------------


def test_passing_unchecked_verdicts_are_copied_once(monkeypatch):
    contract = build_contract(["a", "b"])
    ledger = RowLedger(200, LedgerPolicy.from_contract(contract))
    copies = 0
    original = ledger_module._Judgement.of.__func__

    def counting(cls, verdict, stage):
        nonlocal copies
        copies += 1
        return original(cls, verdict, stage)

    monkeypatch.setattr(ledger_module._Judgement, "of", classmethod(counting))
    for row in range(200):
        ledger.record_answer(row, 1, model="m", tokens=1, cost=Decimal("0.01"))
        verdict = judge(contract, f'{{"a": "x{row}", "b": "y"}}')
        ledger.record_verdict(row, 1, verdict, stage="ResponseParser")

    assert copies <= 1
    snapshot = ledger.snapshot()
    assert snapshot.counts[RowStatus.UNCHECKED] == 200
    assert snapshot.violations == ()


def test_the_fast_path_keeps_every_outcome_of_the_slow_one():
    """Same rows, same events: passing, unparseable and stage-named verdicts."""
    contract = build_contract(["a"])
    policy = LedgerPolicy.from_contract(contract)
    events = [
        (0, judge(contract, "fine"), "ResponseParser"),
        (1, judge(contract, "also fine"), "OtherStage"),
        (2, judge(contract, "fine"), "OtherStage"),
    ]
    fast = RowLedger(3, policy)
    slow = RowLedger(3, policy)
    for row, verdict, stage in events:
        for ledger in (fast, slow):
            ledger.record_answer(row, 1, model="m", tokens=1, cost=Decimal("0.01"))
        fast.record_verdict(row, 1, verdict, stage=stage)
        judgement, _ = ledger_module._Judgement.of(verdict, stage)
        with slow._lock:
            slow._add_verdict(row, 1, judgement)

    assert fast.snapshot() == slow.snapshot()
    assert fast.reported_losses() == slow.reported_losses()
