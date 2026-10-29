"""The judge: one raw answer in, one verdict out.

Two families of tests, for the two promises the judge makes.

**Parity.** Under the unchecked contract the judge must produce, for every
row, exactly what today's parse step produces: same values, same failure
messages. These tests run each answer through *both* the real
``ResponseParserStage`` (with the parser the factory picks today) and the
judge, and compare. They are the regression gate for moving the parse step
into the judge: if they pass, "no contract" stays byte-identical.

**Verdicts.** Under a checked contract, a bad answer is data, never an
exception; each failure carries a reason naming the field and the rule; the
format fix fires only when exactly one allowed value is in the answer; and
field-to-field rules never pile a second reason onto a field that already
failed.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
import time
import timeit
from decimal import Decimal
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, field_validator, model_validator
from pydantic import Field as PydanticField

from ondine.core.models import LLMResponse, ResponseBatch, RowMetadata
from ondine.core.specifications import PromptSpec
from ondine.harness.contract import (
    MAX_PATTERN_INPUT,
    OutputField,
    Relation,
    build_contract,
)
from ondine.harness.judge import Verdict, judge
from ondine.stages.parser_factory import create_response_parser
from ondine.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    ResponseParser,
    ResponseParserStage,
)

# ---------------------------------------------------------------------------
# Today's parse path, run for real
# ---------------------------------------------------------------------------


class _RecordingContext:
    """Stands in for the execution context; keeps what the parser records."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def record_failed_rows(self, row_indices, stage, message):
        self.messages.append(message)


def _today(answer, columns, response_format="raw", regex_patterns=None, parser=None):
    """One answer through today's factory-chosen parser and parse stage."""
    if parser is None:
        spec = PromptSpec(
            template="Classify {text}",
            response_format=response_format,
            regex_patterns=regex_patterns,
        )
        parser = create_response_parser(spec, list(columns))
    stage = ResponseParserStage(parser=parser, output_columns=list(columns))
    context = _RecordingContext()
    response = LLMResponse(
        text=answer,
        tokens_in=1,
        tokens_out=1,
        model="test",
        cost=Decimal("0"),
        latency_ms=0.0,
    )
    batch = ResponseBatch(responses=[response], metadata=[RowMetadata(row_index=0)])
    container = stage.process(([batch], list(columns)), context)
    row = dict(container.to_list()[0])
    row.pop("_row_index")
    return row, context.messages


def _judged(answer, columns, response_format="raw", regex_patterns=None):
    contract = build_contract(
        list(columns), response_format=response_format, regex_patterns=regex_patterns
    )
    verdict = judge(contract, answer)
    return verdict.values, [reason.text for reason in verdict.reasons]


ONE = ["answer"]
TWO = ["intent", "amount"]
FENCED = '```json\n{"answer": "pos"}\n```'

PARITY_CASES = [
    # plain text, one column
    ("plain text", ONE, "raw", "Positive"),
    ("role prefix", ONE, "raw", "  assistant: Positive  "),
    ("empty text", ONE, "raw", ""),
    ("text null", ONE, "raw", "null"),
    ("json read as text", ONE, "raw", '{"answer": "x"}'),
    # JSON, one column
    ("json key", ONE, "json", '{"answer": "pos"}'),
    ("json fence", ONE, "json", FENCED),
    ("bare fence", ONE, "json", '```\n{"answer": "pos"}\n```'),
    ("output fallback", ONE, "json", '{"output": "o", "other": 1}'),
    ("dict missing the key", ONE, "json", '{"label": "first", "score": 2}'),
    ("empty object", ONE, "json", "{}"),
    ("empty json answer", ONE, "json", ""),
    ("json null", ONE, "json", "null"),
    ("json number", ONE, "json", "42"),
    ("json list", ONE, "json", "[1, 2]"),
    ("json string", ONE, "json", '"just a string"'),
    ("not json", ONE, "json", "plain words"),
    ("broken json in fence", ONE, "json", "```json\n{bad json\n```"),
    # several columns: raw resolves to JSON
    ("multi json", TWO, "raw", '{"intent": "refund", "amount": 5}'),
    ("multi partial", TWO, "raw", '{"intent": "refund"}'),
    ("multi no column", TWO, "raw", '{"label": "x"}'),
    ("multi not json", TWO, "raw", "the intent is refund"),
    ("multi empty", TWO, "raw", ""),
    ("multi null", TWO, "raw", "null"),
    ("multi list", TWO, "raw", "[1, 2]"),
    (
        "multi fence",
        TWO,
        "json",
        '```json\n{"intent": "a", "amount": 3}\n```',
    ),
    (
        "items wrapper",
        TWO,
        "json",
        '{"items": [{"id": 1, "result": {"intent": "a", "amount": 1}}]}',
    ),
    ("empty items", TWO, "json", '{"items": []}'),
    (
        "result wrapper",
        TWO,
        "json",
        '{"id": 1, "result": {"intent": "a", "amount": 2}}',
    ),
    ("broken multi fence", TWO, "json", "```json\n{oops\n```"),
    ("column named output", ["output", "score"], "raw", "hello"),
]


@pytest.mark.parametrize(
    ("columns", "response_format", "answer"),
    [case[1:] for case in PARITY_CASES],
    ids=[case[0] for case in PARITY_CASES],
)
def test_unchecked_judge_matches_todays_parsing(columns, response_format, answer):
    """Same values and same failure messages as the parse stage, row for row."""
    assert _judged(answer, columns, response_format) == _today(
        answer, columns, response_format
    )


REGEX = {"score": r"SCORE:\s*(\d+)", "why": r"WHY: .+"}


@pytest.mark.parametrize(
    ("columns", "answer"),
    [
        (["score", "why"], "SCORE: 7\nWHY: good"),
        (["score", "why"], "SCORE: 7 only"),
        (["score", "why"], "nothing useful"),
        (["score"], "SCORE: 9"),
        (["score"], "no score"),
    ],
)
def test_unchecked_judge_matches_todays_regex_parsing(columns, answer):
    patterns = {column: REGEX[column] for column in columns}

    assert _judged(answer, columns, "regex", patterns) == _today(
        answer, columns, "regex", patterns
    )


def test_unchecked_judge_keeps_todays_first_value_guess():
    """Kept for parity only: it is a known silent-error source.

    A dict without the expected key delivers whichever value comes first.
    A real contract replaces this with a reasoned failure (see
    ``test_checked_contract_never_guesses_a_missing_key``).
    """
    values, reasons = _judged('{"label": "refund", "score": 2}', ONE, "json")

    assert values == {"answer": "refund"}
    assert reasons == []


class _Answer(BaseModel):
    intent: str
    amount: int


class _CustomParser(ResponseParser):
    def __init__(self, result):
        self.result = result

    def parse(self, response):
        return self.result


@pytest.mark.parametrize(
    ("parser", "columns", "text"),
    [
        (PydanticParser(_Answer), ["intent"], '{"intent": "a", "amount": 1}'),
        (PydanticParser(_Answer), TWO, '{"intent": "a", "amount": 1}'),
        (JSONParser(strict=False), TWO, '{"intent": "a", "amount": 1}'),
        (_CustomParser({"answer": "x"}), ONE, "ignored"),
        (_CustomParser({}), TWO, "ignored"),
        (_CustomParser(7), ONE, "ignored"),
        (_CustomParser(None), TWO, "ignored"),
    ],
    ids=[
        "model one col",
        "model two cols",
        "dict",
        "custom dict",
        "custom empty",
        "custom scalar",
        "custom none",
    ],
)
def test_unchecked_judge_maps_an_already_parsed_answer_like_today(
    parser, columns, text
):
    """A custom parser's output (or a structured result) maps exactly as today.

    Values must match. Failure messages quote the answer, and the judge only
    sees the parsed object, so only the values and the fact of a failure are
    compared here.
    """
    today_values, today_messages = _today(text, columns, parser=parser)
    contract = build_contract(list(columns))

    verdict = judge(contract, parser.parse(text))

    assert verdict.values == today_values
    assert bool(verdict.reasons) == bool(today_messages)


def test_no_answer_under_unchecked_is_an_empty_row_like_a_null_parse():
    """``None`` (no answer) maps the way today's parser output ``None`` does."""
    today_values, today_messages = _today("null", TWO, "json")

    verdict = judge(build_contract(TWO), None)

    assert verdict.values == today_values == {"intent": None, "amount": None}
    assert verdict.reasons == ()
    assert today_messages == []


def test_unchecked_verdict_is_flagged_unchecked_and_fails_only_where_today_records():
    contract = build_contract(TWO)

    good = judge(contract, '{"intent": "a", "amount": 1}')
    bad = judge(contract, "not json")

    assert good.checked is False
    assert good.passed is True
    assert bad.passed is False
    assert bad.failed_fields == ("intent", "amount")


# ---------------------------------------------------------------------------
# Checked contracts: allowed values, aliases, normalisation, format fix
# ---------------------------------------------------------------------------


def _intent(**overrides):
    declaration = {
        "allowed": ("refund", "cancel", "card lost", "card"),
        "aliases": {"money back": "refund"},
    }
    declaration.update(overrides)
    return build_contract(["intent"], fields=[OutputField("intent", **declaration)])


def test_an_allowed_value_is_accepted_as_is():
    verdict = judge(_intent(), "refund")

    assert verdict.passed is True
    assert verdict.values == {"intent": "refund"}
    assert verdict["intent"].format_fixed is False


def test_an_alias_delivers_the_allowed_value_it_names():
    verdict = judge(_intent(), "money back")

    assert verdict.values == {"intent": "refund"}
    assert verdict["intent"].format_fixed is False


def test_declared_cleanups_deliver_the_canonical_spelling():
    contract = _intent(normalise=("strip", "strip_punctuation", "casefold"))

    verdict = judge(contract, '{"intent": " Refund. "}')

    assert verdict.values == {"intent": "refund"}


def test_undeclared_cleanup_is_not_applied():
    verdict = judge(_intent(), "Refund")

    assert verdict.passed is False


def test_wordy_answer_with_one_allowed_value_is_format_fixed():
    verdict = judge(_intent(), "I believe the customer wants a refund here.")

    assert verdict.passed is True
    assert verdict.values == {"intent": "refund"}
    assert verdict["intent"].format_fixed is True
    assert verdict.format_fixed is True


def test_a_format_fix_keeps_the_reason_the_answer_failed_before_it():
    """The repaired note must show the path taken, not only the end result."""
    verdict = judge(_intent(), "I believe the customer wants a refund here.")

    fixed = verdict["intent"]
    assert fixed.reasons == ()
    assert [reason.rule for reason in fixed.fixed_reasons] == ["allowed_values"]
    assert fixed.fixed_reasons[0].text == (
        "intent: 'I believe the customer wants a refund he…' is not one of the "
        "4 allowed values"
    )


def test_a_field_that_needed_no_fix_has_no_fixed_reasons():
    assert judge(_intent(), "refund")["intent"].fixed_reasons == ()
    assert judge(_intent(), "refund or cancel")["intent"].fixed_reasons == ()


def test_wordy_answer_with_two_allowed_values_is_not_fixed():
    verdict = judge(_intent(), "Either a refund or cancel the order.")

    assert verdict.passed is False
    assert verdict["intent"].format_fixed is False
    assert verdict.failed_fields == ("intent",)


def test_format_fix_prefers_the_longer_value_that_contains_a_shorter_one():
    """'card lost' is one value, not 'card' and 'card lost'."""
    verdict = judge(_intent(), "Category: card lost")

    assert verdict.values == {"intent": "card lost"}


def test_format_fix_needs_word_boundaries():
    verdict = judge(_intent(), "refunded")

    assert verdict.passed is False


def test_format_fix_finds_an_alias_too():
    verdict = judge(_intent(), "they want their money back now")

    assert verdict.values == {"intent": "refund"}
    assert verdict["intent"].format_fixed is True


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("rating: 2", 2), ("I rate it 12", None), ("it is -2", None), ("2.5 stars", None)],
)
def test_format_fix_on_numbers_takes_whole_numbers_only(answer, expected):
    contract = build_contract(
        ["rating"], fields=[OutputField("rating", type="integer", allowed=(1, 2, 3))]
    )

    verdict = judge(contract, answer)

    assert verdict.values == {"rating": expected}


def test_reason_names_field_and_rule_in_a_short_sentence():
    allowed = tuple(f"label_{index}" for index in range(77))
    contract = build_contract(
        ["intent"], fields=[OutputField("intent", allowed=allowed)]
    )

    verdict = judge(contract, "The customer wants a refund because the item broke")

    (reason,) = verdict.reasons
    assert reason.field == "intent"
    assert reason.rule == "allowed_values"
    assert reason.text == (
        "intent: 'The customer wants a refund because the…' "
        "is not one of the 77 allowed values"
    )
    assert str(reason) == reason.text


def test_failed_field_keeps_its_candidate_for_flagged_delivery():
    """Under deliver-flagged the ledger delivers the failed value, so keep it."""
    verdict = judge(_intent(), "maybe")

    assert verdict.values == {"intent": None}
    assert verdict["intent"].value == "maybe"
    assert verdict["intent"].outcome == "failed"


# ---------------------------------------------------------------------------
# Checked contracts: presence, types, ranges, length, pattern
# ---------------------------------------------------------------------------


TICKET = build_contract(
    fields=[
        OutputField("intent", allowed=("refund", "cancel")),
        OutputField("amount", type="number", minimum=0, maximum=1000),
        OutputField("note", required=False, absent="n/a"),
    ]
)


def test_checked_contract_never_guesses_a_missing_key():
    """The first-value guess is gone: a missing key is a reasoned failure."""
    contract = build_contract(
        ["answer"],
        fields=[OutputField("answer", allowed=("refund", "cancel"))],
        response_format="json",
    )

    verdict = judge(contract, '{"label": "refund"}')

    assert verdict.values == {"answer": None}
    assert [reason.text for reason in verdict.reasons] == [
        "answer: missing from the answer"
    ]


def test_missing_required_field_fails_and_optional_one_is_absent():
    verdict = judge(TICKET, '{"note": "n/a", "amount": 5}')

    assert verdict["intent"].outcome == "failed"
    assert verdict["intent"].reasons[0].text == "intent: missing from the answer"
    assert verdict["note"].outcome == "absent"
    assert verdict.values["note"] is None
    assert verdict.failed_fields == ("intent",)


def test_optional_field_missing_entirely_is_absent_not_failed():
    verdict = judge(TICKET, '{"intent": "refund", "amount": 5}')

    assert verdict.passed is True
    assert verdict["note"].outcome == "absent"


def test_failing_optional_field_does_not_fail_the_row():
    contract = build_contract(
        fields=[
            OutputField("intent", allowed=("refund",)),
            OutputField("score", type="integer", required=False),
        ]
    )

    verdict = judge(contract, '{"intent": "refund", "score": "high"}')

    assert verdict.passed is True
    assert verdict["score"].outcome == "failed"
    assert [reason.field for reason in verdict.reasons] == ["score"]


def test_empty_required_value_is_named_empty():
    verdict = judge(TICKET, '{"intent": "  ", "amount": 1}')

    assert verdict["intent"].reasons[0].text == "intent: is empty"


def test_no_answer_fails_every_required_field_with_a_reason():
    verdict = judge(TICKET, None)

    assert verdict.failed_fields == ("intent", "amount")
    assert verdict["intent"].reasons[0].text == "intent: no answer was returned"
    assert verdict["note"].outcome == "absent"


def test_answer_that_is_not_an_object_names_that_for_each_field():
    verdict = judge(TICKET, "I could not decide")

    assert verdict["amount"].reasons[0].text == (
        "amount: missing, the answer is not a JSON object"
    )


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        (OutputField("n", type="integer"), '"42"', 42),
        (OutputField("n", type="integer"), "4.0", 4),
        (OutputField("n", type="number"), '"1e3"', 1000.0),
        (OutputField("n", type="number"), '" 2.5 "', 2.5),
        (OutputField("n", type="number"), '"3"', 3),
        (OutputField("n", type="boolean"), '"TRUE"', True),
        (OutputField("n", type="boolean"), "false", False),
        (OutputField("n"), "123", "123"),
    ],
)
def test_values_parse_into_the_declared_type(field, raw, expected):
    contract = build_contract(fields=[field, OutputField("other", required=False)])

    verdict = judge(contract, f'{{"n": {raw}}}')

    assert verdict.passed is True
    assert verdict.values["n"] == expected
    assert type(verdict.values["n"]) is type(expected)


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        (OutputField("n", type="integer"), '"abc"', "n: 'abc' is not an integer"),
        (OutputField("n", type="integer"), "4.5", "n: 4.5 is not an integer"),
        (OutputField("n", type="integer"), "true", "n: True is not an integer"),
        (OutputField("n", type="number"), '"NaN"', "n: 'NaN' is not a number"),
        (OutputField("n", type="boolean"), '"yes"', "n: 'yes' is not true or false"),
        (OutputField("n"), "[1, 2]", "n: [1, 2] is not text"),
        (
            OutputField("n", type="number", maximum=1000),
            "1500",
            "n: 1500 is above the maximum 1000",
        ),
        (
            OutputField("n", type="integer", minimum=0),
            "-3",
            "n: -3 is below the minimum 0",
        ),
        (
            OutputField("n", max_length=5),
            '"toolong"',
            "n: is 7 characters long, longer than the maximum 5",
        ),
        (
            OutputField("n", min_length=3),
            '"ab"',
            "n: is 2 characters long, shorter than the minimum 3",
        ),
        (
            OutputField("n", pattern=r"^\d+$"),
            '"abc"',
            "n: 'abc' does not match the pattern ^\\d+$",
        ),
    ],
)
def test_each_failed_rule_gives_its_reason(field, raw, expected):
    contract = build_contract(fields=[field, OutputField("other", required=False)])

    verdict = judge(contract, f'{{"n": {raw}}}')

    assert [reason.text for reason in verdict.reasons] == [expected]


def test_rules_stop_at_the_first_failure_cheapest_first():
    """A value too long for its length rule is not also run through the pattern."""
    contract = build_contract(
        ["code"], fields=[OutputField("code", max_length=3, pattern=r"^\d+$")]
    )

    verdict = judge(contract, "toolong")

    assert [reason.rule for reason in verdict.reasons] == ["max_length"]


def test_pattern_is_not_run_on_a_value_beyond_the_cap():
    """Bounded time: a huge value fails with a reason instead of being scanned."""
    contract = build_contract(["code"], fields=[OutputField("code", pattern=r"\d$")])
    answer = "x" * (MAX_PATTERN_INPUT + 1)

    started = time.perf_counter()
    verdict = judge(contract, answer)

    assert time.perf_counter() - started < 1.0
    assert verdict.reasons[0].rule == "pattern"
    assert "too long to check against the pattern" in verdict.reasons[0].text


def test_value_at_the_cap_is_still_checked():
    contract = build_contract(["code"], fields=[OutputField("code", pattern=r"\d$")])

    verdict = judge(contract, "x" * (MAX_PATTERN_INPUT - 1) + "7")

    assert verdict.passed is True


def test_checked_contract_reads_json_in_a_code_fence():
    verdict = judge(TICKET, '```json\n{"intent": "cancel", "amount": "12.5"}\n```')

    assert verdict.values == {"intent": "cancel", "amount": 12.5, "note": None}


def test_single_field_json_contract_reads_a_plain_text_answer_as_the_value():
    contract = build_contract(
        ["intent"],
        fields=[OutputField("intent", allowed=("refund", "cancel"))],
        response_format="json",
    )

    assert judge(contract, "cancel").values == {"intent": "cancel"}


def test_undeclared_output_column_passes_through_under_a_checked_contract():
    contract = build_contract(
        ["intent", "comment"], fields=[OutputField("intent", allowed=("refund",))]
    )

    verdict = judge(contract, '{"intent": "refund", "comment": {"free": "form"}}')

    assert verdict.passed is True
    assert verdict.values["comment"] == {"free": "form"}


def test_checked_contract_reads_regex_extraction():
    contract = build_contract(
        fields=[OutputField("score", type="integer", maximum=10)],
        response_format="regex",
        regex_patterns={"score": r"SCORE:\s*(\d+)"},
    )

    assert judge(contract, "SCORE: 7").values == {"score": 7}
    assert judge(contract, "SCORE: 70").reasons[0].rule == "maximum"


# ---------------------------------------------------------------------------
# Field-to-field rules
# ---------------------------------------------------------------------------


SPAN = build_contract(
    fields=[
        OutputField("start", type="integer", minimum=0),
        OutputField("end", type="integer"),
    ],
    relations=[Relation("end", ">=", "start")],
)


def test_relation_that_holds_passes():
    assert judge(SPAN, '{"start": 3, "end": 5}').passed is True


def test_broken_relation_fails_both_fields_with_one_sentence():
    verdict = judge(SPAN, '{"start": 5, "end": 3}')

    assert verdict.failed_fields == ("start", "end")
    assert verdict["end"].reasons[0].text == "end: 3 is not at least start (5)"
    assert verdict["end"].reasons[0].rule == "relation"


def test_relation_is_skipped_when_a_field_already_failed():
    """One bad field, one reason: no cascade onto the field it is compared with."""
    verdict = judge(SPAN, '{"start": -1, "end": -5}')

    assert verdict.failed_fields == ("start",)
    assert [reason.rule for reason in verdict.reasons] == ["minimum"]


def test_relation_is_skipped_when_its_left_field_already_failed():
    verdict = judge(SPAN, '{"start": 3, "end": "soon"}')

    assert verdict.failed_fields == ("end",)
    assert [reason.rule for reason in verdict.reasons] == ["type"]


def _input_span():
    return build_contract(
        fields=[OutputField("end", type="number")],
        relations=[Relation("end", ">=", input_column="start")],
        input_columns=["start"],
    )


def test_relation_with_an_input_column_reads_the_row_inputs():
    contract = _input_span()

    assert judge(contract, "9", {"start": "5"}).passed is True
    failed = judge(contract, "3", {"start": 5})
    assert failed.reasons[0].text == "end: 3 is not at least input column start (5)"


@pytest.mark.parametrize("inputs", [None, {}, {"start": None}, {"start": "soon"}])
def test_relation_with_an_unusable_input_fails_with_a_reason(inputs):
    verdict = judge(_input_span(), "3", inputs)

    assert verdict.passed is False
    assert "input column start" in verdict.reasons[0].text


# ---------------------------------------------------------------------------
# Schema-derived contracts
# ---------------------------------------------------------------------------


class Refund(BaseModel):
    intent: Literal["refund", "cancel"]
    order_id: str | None = None
    amount: Annotated[float, PydanticField(gt=0)]

    @field_validator("order_id")
    @classmethod
    def order_ids_have_a_prefix(cls, value):
        if value is not None and not value.startswith("ORD"):
            raise ValueError("order ids start with ORD")
        return value

    @model_validator(mode="after")
    def refunds_need_an_order(self):
        if self.intent == "refund" and self.order_id is None:
            raise ValueError("a refund needs an order id")
        return self


REFUND = build_contract(schema=Refund)


def test_schema_validator_message_becomes_the_reason():
    verdict = judge(REFUND, '{"intent": "cancel", "order_id": "X1", "amount": 2}')

    assert verdict["order_id"].outcome == "failed"
    assert verdict["order_id"].reasons[0].text == "order_id: order ids start with ORD"
    assert verdict["order_id"].reasons[0].rule == "schema"


def test_schema_model_validator_fails_every_field_it_could_concern():
    verdict = judge(REFUND, '{"intent": "refund", "amount": 2}')

    assert verdict.passed is False
    assert "intent: a refund needs an order id" in [r.text for r in verdict.reasons]


def test_schema_exclusive_bound_is_caught_by_the_models_own_validation():
    verdict = judge(REFUND, '{"intent": "cancel", "amount": 0}')

    assert verdict.failed_fields == ("amount",)
    assert verdict["amount"].reasons[0].text == "amount: Input should be greater than 0"


def test_schema_contract_accepts_a_structured_result_object():
    answer = Refund(intent="refund", order_id="ORD1", amount=3.5)

    verdict = judge(REFUND, answer)

    assert verdict.passed is True
    assert verdict.values == {"intent": "refund", "order_id": "ORD1", "amount": 3.5}


def test_schema_validator_that_crashes_becomes_a_reason_not_a_crash():
    class Fragile(BaseModel):
        word: str

        @field_validator("word")
        @classmethod
        def explode(cls, value):
            raise TypeError("validator bug")

    verdict = judge(build_contract(schema=Fragile), '{"word": "hi"}')

    assert verdict.passed is False
    assert "validator bug" in verdict.reasons[0].text
    assert verdict.reasons[0].rule == "schema"


def test_declarative_failure_is_not_repeated_by_the_schema():
    verdict = judge(REFUND, '{"intent": "maybe", "order_id": "ORD1", "amount": 2}')

    assert [reason.rule for reason in verdict.reasons] == ["allowed_values"]


# ---------------------------------------------------------------------------
# Total, deterministic, bounded
# ---------------------------------------------------------------------------


GARBAGE = [
    None,
    "",
    "   ",
    "{",
    "```json\n{",
    "```",
    "\x00\xff�",
    "[[[[",
    "null",
    '{"intent": {"nested": [1, 2]}, "amount": [3]}',
    '{"items": "not a list"}',
    '{"result": 5}',
    "9" * 5000,
    "x" * 1_000_000,
    {"intent": object()},
    ["refund"],
    42,
    math.nan,
    Refund(intent="cancel", amount=1.0),
    b"refund",
]


@pytest.mark.parametrize("answer", GARBAGE, ids=lambda answer: type(answer).__name__)
@pytest.mark.parametrize(
    "contract",
    [build_contract(TWO), build_contract(ONE), TICKET, SPAN, REFUND, _intent()],
    ids=["unchecked two", "unchecked one", "ticket", "span", "schema", "intent"],
)
def test_judge_returns_a_verdict_for_any_answer(contract, answer):
    """Total: garbage, empty and truncated answers are verdicts, not exceptions."""
    verdict = judge(contract, answer)

    assert isinstance(verdict, Verdict)
    assert tuple(verdict.values) == contract.output_columns


def test_verdict_is_the_same_in_another_process():
    """Deterministic: resume re-judges in a new process and must agree."""
    script = textwrap.dedent(
        """
        from ondine.harness.contract import OutputField, build_contract
        from ondine.harness.judge import judge
        allowed = tuple(f"label {i}" for i in range(40)) + ("refund", "cancel")
        contract = build_contract(
            ["intent", "amount"],
            fields=[OutputField("intent", allowed=allowed),
                    OutputField("amount", type="number", maximum=5)],
        )
        for answer in ['{"intent": "say refund or cancel", "amount": 9}',
                       '{"intent": "it is a refund", "amount": "3"}']:
            print(repr(judge(contract, answer)))
        """
    )
    outputs = {
        subprocess.run(  # noqa: S603 - fixed interpreter and inline script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }

    assert len(outputs) == 1


def _best_judge_time(contract, answer) -> float:
    """The fastest of several runs: the least noisy estimate of the work done."""
    return min(timeit.repeat(lambda: judge(contract, answer), number=1, repeat=5))


def test_format_fix_time_grows_with_the_answer_not_beyond():
    """Eight times the answer costs about eight times the time.

    A ratio, not a wall-clock budget: a slow CI machine slows both runs alike.
    Linear work gives about 8; a scan per word, or per pair of values, would
    give 64 and more.
    """
    allowed = tuple(f"label{index}" for index in range(200))
    contract = build_contract(
        ["intent"], fields=[OutputField("intent", allowed=allowed)]
    )
    small_answer = ("word " * 10_000) + "label7"
    large_answer = ("word " * 80_000) + "label7"

    small = _best_judge_time(contract, small_answer)
    large = _best_judge_time(contract, large_answer)

    assert judge(contract, large_answer).values == {"intent": "label7"}
    assert large / small < 24, (small, large)


# ---------------------------------------------------------------------------
# One wording table for contract problems and judge reasons
# ---------------------------------------------------------------------------


def test_contract_problems_and_judge_reasons_share_one_type_wording(monkeypatch):
    """Rewording a type changes it everywhere, so the two never drift apart."""
    from ondine.harness import _phrases
    from ondine.harness.contract import ContractError

    monkeypatch.setitem(_phrases.TYPE_PHRASES, "integer", "a whole number")

    with pytest.raises(ContractError) as caught:
        build_contract(["n"], fields=[OutputField("n", type="integer", allowed=["x"])])
    verdict = judge(
        build_contract(["n"], fields=[OutputField("n", type="integer")]), "x"
    )

    assert "n: allowed value 'x' is not a whole number" in caught.value.problems
    assert verdict.reasons[0].text == "n: 'x' is not a whole number"


def test_schema_reason_text_drops_only_pydantics_generic_prefix():
    from ondine.harness._phrases import schema_reason_text

    assert schema_reason_text("Value error, needs an order id") == "needs an order id"
    assert schema_reason_text("Assertion failed, bad") == "bad"
    assert schema_reason_text("Input should be 'a'") == "Input should be 'a'"
