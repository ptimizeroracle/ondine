"""The parse stage asks the judge, and reports each verdict to the ledger.

Migration step 2 moves the parse stage's per-row logic into the judge: the
stage reads the answer (or lets its client parser read it), asks
``judge(contract, answer, inputs)`` for a verdict, writes the verdict's
values, and reports the verdict to the row ledger. Under the unchecked
contract, which every run uses until contracts are wired, the output must be
byte-identical to the stage before the move.

The proof is a frozen copy of that stage (``_todays_process`` below, copied
verbatim from ``ResponseParserStage.process`` before the move), run side by
side with the real stage over the same answers: same rows, same error list,
same counters, same ledger statuses.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.core.models import LLMResponse, ResponseBatch, RowMetadata
from ondine.core.specifications import PromptSpec
from ondine.harness.contract import OutputField, Relation, RunPolicy, build_contract
from ondine.harness.judge import ParsedAnswer, judge
from ondine.harness.ledger import RowLedger
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.llm_invocation_stage import report_response
from ondine.stages.parser_factory import create_response_parser
from ondine.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
    ResponseParserStage,
)

# ---------------------------------------------------------------------------
# Today's stage, frozen: the oracle
# ---------------------------------------------------------------------------


def _todays_process(parser, batches, output_cols, context) -> ResultContainerImpl:
    """``ResponseParserStage.process`` as it was before step 2, verbatim."""
    results: dict[int, Any] = {}
    for batch in batches:
        for response, metadata in zip(batch.responses, batch.metadata, strict=False):
            try:
                response_text = (
                    response.text if hasattr(response, "text") else str(response)
                )
                parsed = parser.parse(response_text)
                if parsed is None:
                    results[metadata.row_index] = {
                        "_row_index": metadata.row_index,
                        **{col: None for col in output_cols},
                    }
                    continue
                row_data: dict[str, Any] = {"_row_index": metadata.row_index}
                if len(output_cols) == 1:
                    if isinstance(parsed, dict) and output_cols[0] in parsed:
                        row_data[output_cols[0]] = parsed[output_cols[0]]
                    elif isinstance(parsed, dict) and "output" in parsed:
                        row_data[output_cols[0]] = parsed["output"]
                    elif isinstance(parsed, dict):
                        row_data[output_cols[0]] = next(iter(parsed.values()))
                    else:
                        row_data[output_cols[0]] = parsed
                else:
                    source = parsed if isinstance(parsed, dict) else {}
                    if not any(col in source for col in output_cols):
                        if "items" in source and isinstance(source["items"], list):
                            item = source["items"][0] if source["items"] else {}
                            if isinstance(item, dict):
                                source = item.get("result", item)
                        elif "result" in source and isinstance(source["result"], dict):
                            source = source["result"]
                        if hasattr(source, "model_dump"):
                            source = source.model_dump()
                        elif not isinstance(source, dict):
                            source = {}
                    if not any(col in source for col in output_cols):
                        context.record_failed_rows(
                            [metadata.row_index],
                            stage="ResponseParser",
                            message=(
                                f"response is not a JSON object with any of "
                                f"the output columns {list(output_cols)}: "
                                f"{response_text.strip()[:120]!r}"
                            ),
                        )
                    for col in output_cols:
                        row_data[col] = source.get(col)
                results[metadata.row_index] = row_data
            except Exception as e:
                context.record_failed_rows(
                    [metadata.row_index],
                    stage="ResponseParser",
                    message=f"response could not be parsed: {e}",
                )
                results[metadata.row_index] = {
                    "_row_index": metadata.row_index,
                    **{col: None for col in output_cols},
                }
    sorted_results = [results[idx] for idx in sorted(results.keys())]
    return ResultContainerImpl(
        data=sorted_results, columns=["_row_index"] + list(output_cols)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(text: str, **metadata: Any) -> LLMResponse:
    return LLMResponse(
        text=text,
        tokens_in=1,
        tokens_out=1,
        model="m",
        cost=Decimal("0.001"),
        latency_ms=0.0,
        metadata=metadata,
    )


def _batch(responses: list[Any], custom: list[dict] | None = None) -> ResponseBatch:
    custom = custom or [{} for _ in responses]
    return ResponseBatch(
        responses=responses,
        metadata=[
            RowMetadata(row_index=row, custom=extra) for row, extra in enumerate(custom)
        ],
    )


def _report_answers(context: ExecutionContext, batch: ResponseBatch) -> None:
    """What the model stage reports for each response, before parsing."""
    for response, metadata in zip(batch.responses, batch.metadata, strict=True):
        if isinstance(response, LLMResponse):
            report_response(context, response, metadata, stage="LLMInvocation")


def _shape(errors) -> list[tuple]:
    return [(e.row_index, e.stage, e.error_type, e.message) for e in errors]


def _observed(container, context: ExecutionContext) -> tuple:
    snapshot = context.ledger.snapshot()
    return (
        list(container.columns),
        container.to_list(),
        _shape(context.row_errors),
        context.failed_rows,
        context.skipped_rows,
        [(row.status, row.delivered_attempt) for row in snapshot.rows],
        _shape(snapshot.errors),
    )


def _side_by_side(
    texts: list[str],
    columns: list[str],
    *,
    response_format: str = "raw",
    regex_patterns: dict[str, str] | None = None,
    parser: ResponseParser | None = None,
) -> tuple[tuple, tuple]:
    """Today's stage and the new one over the same answers, as the pipeline wires them."""
    if parser is None:
        spec = PromptSpec(
            template="Classify {text}",
            response_format=response_format,
            regex_patterns=regex_patterns,
        )
        todays_parser = create_response_parser(spec, columns)
    else:
        todays_parser = parser
    batch = _batch([_response(text) for text in texts])

    old_context = ExecutionContext(total_rows=len(texts))
    _report_answers(old_context, batch)
    old = _todays_process(todays_parser, [batch], columns, old_context)

    contract = build_contract(
        columns, response_format=response_format, regex_patterns=regex_patterns
    )
    stage = ResponseParserStage(
        parser=parser, output_columns=columns, contract=contract
    )
    new_context = ExecutionContext(total_rows=len(texts))
    _report_answers(new_context, batch)
    new = stage.process(([batch], columns), new_context)

    return _observed(old, old_context), _observed(new, new_context)


ONE = ["answer"]
TWO = ["intent", "amount"]

# Every answer shape the judge's parity tests know, grouped by output shape.
TEXTS_ONE = [
    "Positive",
    "  assistant: Positive  ",
    "",
    "null",
    '{"answer": "x"}',
    '{"output": "o", "other": 1}',
    '{"label": "first", "score": 2}',
    "{}",
    "42",
    "[1, 2]",
    '"just a string"',
    "plain words",
    "```json\n{bad json\n```",
    '```json\n{"answer": "pos"}\n```',
    '```\n{"answer": "pos"}\n```',
]
TEXTS_TWO = [
    '{"intent": "refund", "amount": 5}',
    '{"intent": "refund"}',
    '{"label": "x"}',
    "the intent is refund",
    "",
    "null",
    "[1, 2]",
    '```json\n{"intent": "a", "amount": 3}\n```',
    '{"items": [{"id": 1, "result": {"intent": "a", "amount": 1}}]}',
    '{"items": []}',
    '{"id": 1, "result": {"intent": "a", "amount": 2}}',
    "```json\n{oops\n```",
    "x" * 300,
]


# ---------------------------------------------------------------------------
# Parity: the default parsers, now read by the judge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("response_format", ["raw", "json"])
def test_one_column_output_is_identical_to_todays_stage(response_format):
    old, new = _side_by_side(TEXTS_ONE, ONE, response_format=response_format)

    assert new == old


@pytest.mark.parametrize("response_format", ["raw", "json"])
def test_several_columns_output_is_identical_to_todays_stage(response_format):
    old, new = _side_by_side(TEXTS_TWO, TWO, response_format=response_format)

    assert new == old
    assert old[2], "the corpus must include answers today's stage records as lost"


@pytest.mark.parametrize("columns", [["score", "why"], ["score"]])
def test_regex_output_is_identical_to_todays_stage(columns):
    patterns = {"score": r"SCORE:\s*(\d+)", "why": r"WHY: .+"}
    patterns = {column: patterns[column] for column in columns}
    texts = ["SCORE: 7\nWHY: good", "SCORE: 7 only", "nothing useful", ""]

    old, new = _side_by_side(
        texts, columns, response_format="regex", regex_patterns=patterns
    )

    assert new == old


# ---------------------------------------------------------------------------
# Parity: a client's own parser, whose output the judge maps
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    intent: str
    amount: int


class _Fixed(ResponseParser):
    """Returns the same object whatever it reads."""

    def __init__(self, result: Any) -> None:
        self.result = result

    def parse(self, response: str) -> Any:
        return self.result


class _Raising(ResponseParser):
    def parse(self, response: str) -> Any:
        raise RuntimeError(f"cannot read {response[:5]!r}")


CLIENT_PARSERS = [
    ("json", JSONParser(strict=False)),
    ("strict json", JSONParser(strict=True)),
    ("raw text", RawTextParser()),
    ("regex", RegexParser({"intent": r"intent=(\w+)", "amount": r"amount=(\d+)"})),
    ("pydantic strict", PydanticParser(_Answer)),
    ("pydantic lenient", PydanticParser(_Answer, strict=False)),
    ("returns a str that looks like json", _Fixed("123")),
    ("returns a str with a role prefix", _Fixed("  assistant: x  ")),
    ("returns a scalar", _Fixed(7)),
    ("returns None", _Fixed(None)),
    ("returns an empty dict", _Fixed({})),
    ("returns a model", _Fixed(_Answer(intent="a", amount=1))),
    ("raises", _Raising()),
]
CLIENT_TEXTS = [
    '{"intent": "refund", "amount": 5}',
    '{"intent": "refund"}',
    "intent=cancel amount=3",
    "not json at all",
    "",
    "```json\n{oops\n```",
]


@pytest.mark.parametrize("columns", [ONE, TWO], ids=["one column", "two columns"])
@pytest.mark.parametrize(
    "parser", [p for _, p in CLIENT_PARSERS], ids=[n for n, _ in CLIENT_PARSERS]
)
def test_a_client_parser_output_is_identical_to_todays_stage(parser, columns):
    """The stage calls the client's parser as before; the judge maps its output.

    The response format is left at "raw", as ``with_structured_output``
    leaves it: the contract's own reading of the text must play no part.
    """
    old, new = _side_by_side(CLIENT_TEXTS, columns, parser=parser)

    assert new == old


def test_a_stage_built_without_a_contract_keeps_todays_defaults():
    """Driven directly, with no contract: today's raw-text default parser."""
    texts = ['{"intent": "a", "amount": 1}', "plain"]
    batch = _batch([_response(text) for text in texts])

    old_context = ExecutionContext(total_rows=2)
    old = _todays_process(RawTextParser(), [batch], TWO, old_context)
    new_context = ExecutionContext(total_rows=2)
    new = ResponseParserStage(output_columns=TWO).process(([batch], TWO), new_context)

    assert new.to_list() == old.to_list()
    assert _shape(new_context.row_errors) == _shape(old_context.row_errors)


def test_a_context_without_a_ledger_still_hears_about_lost_rows():
    """A stand-in context (a custom backend, a test double) keeps today's report."""

    class _Recording:
        def __init__(self) -> None:
            self.reports: list[tuple] = []

        def record_failed_rows(self, row_indices, stage, message):
            self.reports.append((list(row_indices), stage, message))

    context = _Recording()
    batch = _batch([_response("not json"), _response('{"intent": "a"}')])

    ResponseParserStage(output_columns=TWO, contract=build_contract(TWO)).process(
        ([batch], TWO), context
    )

    assert context.reports == [
        (
            [0],
            "ResponseParser",
            "response is not a JSON object with any of the output columns "
            "['intent', 'amount']: 'not json'",
        )
    ]


# ---------------------------------------------------------------------------
# Verdicts reach the ledger
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded(monkeypatch) -> list[tuple]:
    """Every ``record_verdict`` call, as (row, attempt, verdict, stage)."""
    calls: list[tuple] = []
    original = RowLedger.record_verdict

    def spy(self, row, attempt, verdict, *, stage, **kwargs):
        calls.append((row, attempt, verdict, stage))
        original(self, row, attempt, verdict, stage=stage, **kwargs)

    monkeypatch.setattr(RowLedger, "record_verdict", spy)
    return calls


def test_each_answered_row_reports_the_judges_verdict(recorded):
    contract = build_contract(TWO)
    texts = ['{"intent": "a", "amount": 1}', "not json", "null"]
    batch = _batch([_response(text) for text in texts])
    context = ExecutionContext(total_rows=3)

    ResponseParserStage(output_columns=TWO, contract=contract).process(
        ([batch], TWO), context
    )

    assert recorded == [
        (row, 1, judge(contract, text), "ResponseParser")
        for row, text in enumerate(texts)
    ]


def test_an_unreadable_answer_is_a_failed_verdict_under_the_parse_stage():
    """The row's ledger outcome names the judge's reason, not a stage failure."""
    contract = build_contract(TWO)
    batch = _batch([_response("not json")])
    context = ExecutionContext(total_rows=1)
    _report_answers(context, batch)

    ResponseParserStage(output_columns=TWO, contract=contract).process(
        ([batch], TWO), context
    )

    row = context.ledger.snapshot().rows[0]
    message = (
        "response is not a JSON object with any of the output columns "
        "['intent', 'amount']: 'not json'"
    )
    assert row.status is RowStatus.FAILED
    assert row.reason == message
    assert row.failed_checks == (message,)
    assert _shape(context.row_errors) == [
        (0, "ResponseParser", "parse_failure", message)
    ]


def test_a_client_parser_verdict_quotes_the_raw_answer_not_the_parsers_output(
    recorded,
):
    batch = _batch([_response("raw words")])
    context = ExecutionContext(total_rows=1)

    ResponseParserStage(
        parser=_Fixed({}), output_columns=TWO, contract=build_contract(TWO)
    ).process(([batch], TWO), context)

    ((_, _, verdict, _),) = recorded
    assert verdict.reasons[0].text.endswith(": 'raw words'")


# ---------------------------------------------------------------------------
# Answers the model never gave: no verdict, today's values and reports
# ---------------------------------------------------------------------------


def _non_answers() -> tuple[ResponseBatch, list[int]]:
    """A batch holding each kind of response that is not the model's answer.

    Row 0 is a real answer. Row 1 was skipped by the error policy, row 2
    filled with the policy's default (unreadable as JSON), row 3 is the
    disaggregator's placeholder for a row its batch answer lacked.
    """
    batch = _batch(
        [
            _response('{"intent": "a", "amount": 1}'),
            _response("[SKIPPED]", error="down", action="skipped"),
            _response("N/A", error="down", action="substituted"),
            "null",
        ],
        custom=[{}, {}, {}, {"from_batch": True, "parse_error": True}],
    )
    return batch, [1, 2, 3]


def _upstream(context: ExecutionContext, batch: ResponseBatch) -> None:
    _report_answers(context, batch)
    context.record_failed_rows(
        [3], stage="BatchDisaggregator", message="batch response was missing 1 of 2"
    )


def test_responses_the_model_never_gave_are_not_judged_as_answers(recorded):
    batch, not_answers = _non_answers()
    context = ExecutionContext(total_rows=4)
    _upstream(context, batch)

    ResponseParserStage(output_columns=TWO, contract=build_contract(TWO)).process(
        ([batch], TWO), context
    )

    assert [row for row, *_ in recorded] == [0]
    assert not set(not_answers) & {row for row, *_ in recorded}


def test_responses_the_model_never_gave_keep_todays_values_and_reports():
    batch, _ = _non_answers()
    old_context = ExecutionContext(total_rows=4)
    _upstream(old_context, batch)
    old = _todays_process(JSONParser(strict=False), [batch], TWO, old_context)
    new_context = ExecutionContext(total_rows=4)
    _upstream(new_context, batch)

    new = ResponseParserStage(output_columns=TWO, contract=build_contract(TWO)).process(
        ([batch], TWO), new_context
    )

    assert _observed(new, new_context) == _observed(old, old_context)
    statuses = [row.status for row in new_context.ledger.snapshot().rows]
    assert statuses == [
        RowStatus.UNCHECKED,
        RowStatus.FAILED,
        RowStatus.FAILED,
        RowStatus.FAILED,
    ]


# ---------------------------------------------------------------------------
# A checked contract: the seam step 3 builds on
# ---------------------------------------------------------------------------

INTENT = OutputField("intent", allowed=["refund", "cancel"], normalise=["casefold"])


def test_a_checked_contract_writes_deliverable_values_and_reports_verdicts():
    """A rejected candidate is never written: no later stage may read it."""
    contract = build_contract(["intent"], fields=[INTENT])
    batch = _batch([_response("Refund"), _response("maybe")])
    context = ExecutionContext(total_rows=2)
    context.open_ledger(contract)
    _report_answers(context, batch)

    container = ResponseParserStage(
        output_columns=["intent"], contract=contract
    ).process(([batch], ["intent"]), context)

    assert [row["intent"] for row in container.to_list()] == ["refund", None]
    snapshot = context.ledger.snapshot()
    assert [row.status for row in snapshot.rows] == [
        RowStatus.ACCEPTED,
        RowStatus.REJECTED,
    ]
    assert snapshot.violations == ()
    assert context.row_errors == []  # a rejection is not a lost row (#312)


def test_under_deliver_flagged_a_failing_candidate_is_written():
    """The one policy that ships a failing value writes it, flagged by delivery."""
    contract = build_contract(
        ["intent"],
        fields=[INTENT],
        policy=RunPolicy(on_failure="deliver_flagged"),
    )
    batch = _batch([_response("Refund"), _response("maybe")])
    context = ExecutionContext(total_rows=2)
    context.open_ledger(contract)
    _report_answers(context, batch)

    container = ResponseParserStage(
        output_columns=["intent"], contract=contract
    ).process(([batch], ["intent"]), context)

    assert [row["intent"] for row in container.to_list()] == ["refund", "maybe"]
    assert [row.status for row in context.ledger.snapshot().rows] == [
        RowStatus.ACCEPTED,
        RowStatus.UNVERIFIED,
    ]


def test_a_checked_contract_judges_each_row_against_its_own_inputs():
    contract = build_contract(
        ["amount"],
        fields=[OutputField("amount", type="integer")],
        relations=[Relation("amount", "<=", input_column="limit")],
        input_columns=["limit"],
    )
    batch = _batch([_response("5"), _response("5")])
    context = ExecutionContext(total_rows=2)
    context.open_ledger(contract)
    context.intermediate_data["loaded_data"] = [{"limit": 10}, {"limit": 1}]

    ResponseParserStage(output_columns=["amount"], contract=contract).process(
        ([batch], ["amount"]), context
    )

    rows = context.ledger.snapshot().rows
    assert [row.status for row in rows] == [RowStatus.ACCEPTED, RowStatus.REJECTED]
    assert rows[1].reason == "amount: 5 is not at most input column limit (1)"


def test_a_contract_over_other_columns_is_a_wiring_defect():
    stage = ResponseParserStage(output_columns=ONE, contract=build_contract(TWO))

    with pytest.raises(ValueError, match="contract"):
        stage.process(([_batch([_response("x")])], ONE), ExecutionContext())


# ---------------------------------------------------------------------------
# The judge's side of the seam: a client parser's output
# ---------------------------------------------------------------------------


def test_a_parsed_str_is_a_value_never_read_again():
    contract = build_contract(ONE, response_format="json")

    verdict = judge(contract, ParsedAnswer("123", text="anything"))

    assert verdict.values == {"answer": "123"}


def test_a_parser_failure_is_todays_message_under_unchecked():
    verdict = judge(build_contract(TWO), ParsedAnswer(None, "x", error="boom"))

    assert verdict.values == {"intent": None, "amount": None}
    assert [r.text for r in verdict.reasons] == ["response could not be parsed: boom"]


def test_a_parser_failure_is_a_reasoned_missing_field_under_a_checked_contract():
    contract = build_contract(["intent"], fields=[INTENT])

    verdict = judge(contract, ParsedAnswer(None, "x", error="boom"))

    assert not verdict.passed
    assert [(r.rule, r.text) for r in verdict.reasons] == [
        ("required", "intent: missing, the parser could not read the answer: boom")
    ]


def test_a_parsed_str_is_checked_as_the_fields_value_under_a_checked_contract():
    contract = build_contract(["intent"], fields=[INTENT])

    verdict = judge(contract, ParsedAnswer(" Cancel ", text='{"intent": "x"}'))

    assert verdict.values == {"intent": "cancel"}


# ---------------------------------------------------------------------------
# The ledger: an unreadable answer is stamped when it is reported
# ---------------------------------------------------------------------------


def test_a_lost_rows_error_is_stamped_when_reported_not_when_read():
    from datetime import datetime, timedelta

    contract = build_contract(TWO)
    context = ExecutionContext(total_rows=1)
    before = datetime.now()
    context.ledger.record_verdict(
        0, 1, judge(contract, "not json"), stage="ResponseParser"
    )
    after = datetime.now()

    (error,) = context.row_errors

    assert before <= error.timestamp <= after
    assert error.timestamp < after + timedelta(seconds=1)


@pytest.mark.parametrize(
    "contract",
    [build_contract(["intent"]), build_contract(["intent"], fields=[INTENT])],
    ids=["unchecked", "checked"],
)
def test_a_value_a_later_stage_withdrew_is_failed_not_delivered(contract):
    """Grounding blanks a judged row: a stage failure on the verdict's attempt."""
    from ondine.harness.ledger import LedgerPolicy

    book = RowLedger(1, LedgerPolicy.from_contract(contract), strict=True)
    book.record_answer(0, 1, model="m", tokens=1, cost=Decimal("0"))
    book.record_verdict(0, 1, judge(contract, "refund"), stage="ResponseParser")
    book.record_stage_failure([0], stage="ContextVerification", message="ungrounded")

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.FAILED
    assert row.reason == "ContextVerification: ungrounded"
    assert row.delivered_attempt is None


def test_a_failure_on_an_earlier_attempt_does_not_withdraw_a_later_value():
    """The auto-retry pass recovers a row: its first attempt's failure stays behind."""
    from ondine.harness.ledger import LedgerPolicy

    contract = build_contract(["intent"])
    book = RowLedger(1, LedgerPolicy.from_contract(contract), strict=True)
    book.record_stage_failure([0], stage="ResponseParser", message="unreadable")
    book.record_answer(0, 2, model="m", tokens=1, cost=Decimal("0"))
    book.record_verdict(0, 2, judge(contract, "refund"), stage="ResponseParser")

    row = book.snapshot().rows[0]

    assert row.status is RowStatus.UNCHECKED
    assert row.delivered_attempt == 2


# ---------------------------------------------------------------------------
# The pipeline resolves the run's contract in one place
# ---------------------------------------------------------------------------


def _pipeline(output_columns: list[str], **prompt: Any):
    from ondine import PipelineBuilder
    from tests.e2e.conformance_harness import LedgerClient, conformance_frame

    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(5),
            input_columns=["marker"],
            output_columns=output_columns,
        )
        .with_prompt("Answer for {marker}", **prompt)
        .with_custom_llm_client(LedgerClient())
        .with_batch_size(1)
        .with_concurrency(1)
    )
    return builder.build()


def test_without_a_declared_contract_the_run_uses_the_unchecked_one():
    pipeline = _pipeline(["answer"])

    contract = pipeline._resolve_contract()

    assert contract == build_contract(["answer"])
    assert contract.checked is False


def test_a_contract_the_reader_cannot_build_falls_back_to_plain_text():
    pipeline = _pipeline(["a", "b"])
    pipeline.specifications.prompt.response_format = "regex"
    pipeline.specifications.prompt.regex_patterns = {"a": r"(\w+)"}

    contract = pipeline._resolve_contract()

    assert contract == build_contract(["a", "b"])
    assert contract.checked is False


def test_a_declared_contract_is_the_runs_contract_end_to_end():
    """The declared contract reaches the judge, the ledger and delivery."""
    pipeline = _pipeline(["answer"])
    contract = build_contract(
        ["answer"], fields=[OutputField("answer", pattern=r"^ANS-row-000[0-2]$")]
    )
    pipeline._components = replace(pipeline._components, output_contract=contract)

    assert pipeline._resolve_contract() is contract
    result = pipeline.execute()

    snapshot = result.row_ledger
    assert [row.status for row in snapshot.rows] == [RowStatus.ACCEPTED] * 3 + [
        RowStatus.REJECTED
    ] * 2
    assert snapshot.violations == ()
    assert list(result.to_pandas()["answer"].iloc[:3]) == [
        "ANS-row-0000",
        "ANS-row-0001",
        "ANS-row-0002",
    ]
    assert result.to_pandas()["answer"].iloc[3:].isna().all()
    assert list(result.to_pandas()["status"]) == ["accepted"] * 3 + ["rejected"] * 2
