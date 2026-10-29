"""The output contract: what a client declares a deliverable value to be.

A contract is built once, before any model call, and every later decision in
the harness reads it. Two properties carry the weight, so most tests here aim
at them:

* **A broken contract never reaches a run.** Every problem is reported at build
  time, all at once, so a client fixes the declaration in one pass instead of
  paying for a run per typo.
* **The fingerprint is stable.** Resume refuses a changed contract by comparing
  fingerprints. A fingerprint that drifts between processes would refuse every
  resume; one that ignores a declaration would let a changed contract through.
"""

from __future__ import annotations

import enum
import os
import subprocess
import sys
import textwrap
import timeit
from dataclasses import FrozenInstanceError
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel
from pydantic import Field as PydanticField

from ondine.harness.contract import (
    ContractError,
    OutputField,
    Relation,
    RunPolicy,
    build_contract,
)


def _problems(**kwargs) -> list[str]:
    """The problems a build reports, or an empty list if it succeeds."""
    try:
        build_contract(**kwargs)
    except ContractError as error:
        return list(error.problems)
    return []


# ---------------------------------------------------------------------------
# The unchecked contract
# ---------------------------------------------------------------------------


def test_nothing_declared_gives_the_unchecked_contract():
    """No declaration is still a contract, so callers handle one kind of object."""
    contract = build_contract(["answer"])

    assert contract.checked is False
    assert contract.output_columns == ("answer",)


def test_any_declaration_makes_the_contract_checked():
    contract = build_contract(["answer"], fields=[OutputField("answer")])

    assert contract.checked is True


def test_output_columns_default_to_the_declared_fields():
    contract = build_contract(fields=[OutputField("a"), OutputField("b")])

    assert contract.output_columns == ("a", "b")


def test_a_contract_needs_at_least_one_output_column():
    assert _problems() == ["the contract names no output column"]


# ---------------------------------------------------------------------------
# Frozen and fingerprinted
# ---------------------------------------------------------------------------


def _intent_contract(allowed=("refund", "cancel"), policy=None):
    return build_contract(
        ["intent"],
        fields=[
            OutputField("intent", allowed=allowed, aliases={"money back": "refund"})
        ],
        policy=policy,
    )


def test_contract_cannot_be_changed_after_build():
    contract = _intent_contract()

    with pytest.raises(FrozenInstanceError):
        contract.checked = False  # type: ignore[misc]


def test_a_declaration_mutated_after_build_does_not_change_the_contract():
    """The contract copies what it is given; a caller's list stays the caller's."""
    allowed = ["refund", "cancel"]
    aliases = {"money back": "refund"}
    field = OutputField("intent", allowed=allowed, aliases=aliases)
    contract = build_contract(["intent"], fields=[field])
    before = contract.fingerprint

    allowed.append("other")
    aliases["x"] = "cancel"

    assert contract.fields[0].allowed == ("refund", "cancel")
    assert "x" not in contract.fields[0].aliases
    assert contract.fingerprint == before


def test_same_declarations_give_the_same_fingerprint():
    assert _intent_contract().fingerprint == _intent_contract().fingerprint


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _intent_contract(allowed=("refund", "cancel", "other")),
        lambda: _intent_contract(policy=RunPolicy(on_failure="deliver_flagged")),
        lambda: _intent_contract(policy=RunPolicy(early_stop_fraction=None)),
        lambda: build_contract(["intent"]),
        lambda: build_contract(
            ["intent"],
            fields=[OutputField("intent", allowed=("refund", "cancel"))],
        ),
    ],
    ids=["allowed values", "on failure", "early stop", "unchecked", "aliases"],
)
def test_any_change_in_declarations_changes_the_fingerprint(changed):
    assert changed().fingerprint != _intent_contract().fingerprint


def test_declaration_order_inside_a_field_does_not_change_the_fingerprint():
    """Aliases and cleanups are sets of choices; listing order means nothing."""
    first = build_contract(
        ["a"],
        fields=[
            OutputField(
                "a",
                allowed=("x", "y"),
                aliases={"ex": "x", "why": "y"},
                normalise=("casefold", "strip"),
            )
        ],
    )
    second = build_contract(
        ["a"],
        fields=[
            OutputField(
                "a",
                allowed=("x", "y"),
                aliases={"why": "y", "ex": "x"},
                normalise=("strip", "casefold"),
            )
        ],
    )

    assert first.fingerprint == second.fingerprint


def test_fingerprint_distinguishes_a_number_from_its_text():
    as_text = build_contract(["n"], fields=[OutputField("n", allowed=("1",))])
    as_number = build_contract(
        ["n"], fields=[OutputField("n", type="integer", allowed=(1,))]
    )

    assert as_text.fingerprint != as_number.fingerprint


def test_fingerprint_is_the_same_in_another_process():
    """Resume runs in a new process; hash randomisation must not move the print."""
    script = textwrap.dedent(
        """
        from ondine.harness.contract import OutputField, build_contract
        contract = build_contract(
            ["intent", "amount"],
            fields=[
                OutputField("intent", allowed=("b", "a", "c"),
                            aliases={"z": "a", "y": "b"}, normalise=("casefold",)),
                OutputField("amount", type="number", minimum=0),
            ],
        )
        print(contract.fingerprint)
        """
    )
    prints = {
        subprocess.run(  # noqa: S603 - fixed interpreter and inline script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }

    assert len(prints) == 1


# ---------------------------------------------------------------------------
# Build-time validation
# ---------------------------------------------------------------------------


def test_build_reports_every_problem_at_once():
    """One build, all problems: a client never fixes typos one paid run at a time."""
    problems = _problems(
        output_columns=["start", "end", "intent"],
        fields=[
            OutputField("start", type="number", minimum=10, maximum=1),
            OutputField("intent", allowed=()),
            OutputField("ghost"),
        ],
        relations=[Relation("end", ">=", "nowhere")],
    )

    assert len(problems) == 4
    joined = "\n".join(problems)
    assert "start: minimum 10 is above the maximum 1" in joined
    assert "intent: the allowed values are empty" in joined
    assert "ghost: not an output column" in joined
    assert "nowhere" in joined


def test_unhashable_allowed_values_and_absent_markers_are_listed_problems():
    """A list cannot be looked up; that is a problem to list, not a crash."""
    problems = _problems(
        output_columns=["a", "b", "c"],
        fields=[
            OutputField("a", type="any", allowed=[[1], "ok"]),
            OutputField("b", type="any", required=False, absent={"none": True}),
            OutputField("c", type="number", minimum=2, maximum=1),
        ],
    )

    assert problems == [
        "a: allowed value [1] is a list, which cannot be looked up; use a single "
        "value such as text, a number or true/false",
        "b: the absent marker {'none': True} is a dict, which cannot be looked "
        "up; use a single value such as text, a number or true/false",
        "c: minimum 2 is above the maximum 1",
    ]


def test_contract_error_message_lists_every_problem():
    with pytest.raises(ContractError) as caught:
        build_contract(
            ["a"],
            fields=[OutputField("a", type="number", minimum=2, maximum=1)],
            policy=RunPolicy(on_failure="shrug"),
        )

    message = str(caught.value)
    assert "2 problems" in message
    for problem in caught.value.problems:
        assert problem in message


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "fields": [
                    OutputField("a", allowed=("Yes", "yes"), normalise=("casefold",))
                ]
            },
            "a: allowed values 'Yes' and 'yes' are the same after normalisation",
        ),
        (
            {
                "fields": [
                    OutputField(
                        "a",
                        allowed=("refund", "cancel"),
                        aliases={"Refund ": "cancel"},
                        normalise=("strip", "casefold"),
                    )
                ]
            },
            "a: alias 'Refund ' maps to 'cancel' but is also the allowed value 'refund'",
        ),
        (
            {
                "fields": [
                    OutputField(
                        "a",
                        allowed=("refund", "cancel"),
                        aliases={"X": "refund", "x": "cancel"},
                        normalise=("casefold",),
                    )
                ]
            },
            "a: alias 'x' maps to two values, 'refund' and 'cancel'",
        ),
        (
            {"fields": [OutputField("a", allowed=("refund",), aliases={"r": "nope"})]},
            "a: alias 'r' maps to 'nope', which is not an allowed value",
        ),
        (
            {
                "fields": [
                    OutputField(
                        "a",
                        required=False,
                        absent="N/A",
                        allowed=("n/a", "x"),
                        normalise=("casefold",),
                    )
                ]
            },
            "a: the absent marker 'N/A' is also an allowed value",
        ),
        (
            {"fields": [OutputField("a", absent="n/a")]},
            "a: a required field cannot have an absent marker",
        ),
        (
            {"fields": [OutputField("a", type="decimal")]},
            "a: unknown type 'decimal'",
        ),
        (
            {"fields": [OutputField("a", minimum=1)]},
            "a: minimum and maximum apply to integer and number fields only",
        ),
        (
            {"fields": [OutputField("a", type="integer", pattern=r"^\d+$")]},
            "a: pattern applies to text fields only",
        ),
        (
            {"fields": [OutputField("a", min_length=5, max_length=2)]},
            "a: min_length 5 is above max_length 2",
        ),
        (
            {"fields": [OutputField("a", max_length=-1)]},
            "a: max_length must be a whole number of at least 0",
        ),
        (
            {"fields": [OutputField("a", normalise=("lowercase",))]},
            "a: unknown normalisation 'lowercase'",
        ),
        (
            {"fields": [OutputField("a", type="integer", normalise=("strip",))]},
            "a: normalisations apply to text fields only",
        ),
        (
            {"fields": [OutputField("a", type="integer", allowed=(1, "two"))]},
            "a: allowed value 'two' is not an integer",
        ),
        (
            {"fields": [OutputField("a", pattern="(unclosed")]},
            "a: pattern '(unclosed' is not a valid regular expression",
        ),
        (
            {"fields": [OutputField("a"), OutputField("a")]},
            "a: declared twice",
        ),
    ],
)
def test_each_impossible_declaration_is_named(kwargs, expected):
    problems = _problems(output_columns=["a"], **kwargs)

    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("relation", "fields", "expected"),
    [
        (
            Relation("end", ">=", "start"),
            [OutputField("end", type="number"), OutputField("start")],
            "relation end >= start compares a number with text",
        ),
        (
            Relation("end", ">", "start"),
            [OutputField("end", type="boolean"), OutputField("start", type="boolean")],
            "relation end > start orders true/false values",
        ),
        (
            Relation("end", "~", "start"),
            [OutputField("end"), OutputField("start")],
            "relation end ~ start: unknown comparison '~'",
        ),
        (
            Relation("end", ">="),
            [OutputField("end"), OutputField("start")],
            "relation on end names neither an output field nor an input column",
        ),
        (
            Relation("end", ">=", "start", input_column="s"),
            [OutputField("end"), OutputField("start")],
            "relation on end names both an output field and an input column",
        ),
        (
            Relation("end", ">=", input_column="missing"),
            [OutputField("end"), OutputField("start")],
            "relation end >= input column missing: 'missing' is not an input column",
        ),
    ],
)
def test_relations_are_checked_at_build_time(relation, fields, expected):
    problems = _problems(
        output_columns=["end", "start"],
        fields=fields,
        relations=[relation],
        input_columns=["start_date"],
    )

    assert any(expected in problem for problem in problems), problems


def test_relation_between_integer_and_number_fields_is_compatible():
    contract = build_contract(
        fields=[
            OutputField("start", type="integer"),
            OutputField("end", type="number"),
        ],
        relations=[Relation("end", ">=", "start")],
    )

    assert contract.relations == (Relation("end", ">=", "start"),)


def test_relation_may_name_an_input_column():
    contract = build_contract(
        fields=[OutputField("end", type="number")],
        relations=[Relation("end", ">=", input_column="start")],
        input_columns=["start"],
    )

    assert contract.checked is True


def test_regex_format_needs_a_pattern_per_output_column():
    problems = _problems(
        output_columns=["a", "b"],
        response_format="regex",
        regex_patterns={"a": r"A: (\w+)"},
    )

    assert problems == ["response_format 'regex' needs a pattern for: b"]


def test_unknown_response_format_is_a_problem():
    assert _problems(output_columns=["a"], response_format="yaml") == [
        "unknown response_format 'yaml'; use 'raw', 'json' or 'regex'"
    ]


# ---------------------------------------------------------------------------
# Pattern safety
# ---------------------------------------------------------------------------
#
# Python's ``re`` backtracks and has no timeout, and a timeout would make a
# verdict depend on machine speed. So the contract proves each pattern safe
# before the run, and these tests pin both sides of that proof: what it must
# refuse (exponential and polynomial patterns) and what it must let through
# (the patterns clients write every day). The timing tests compare growth
# ratios, never absolute seconds, so a slow CI machine cannot flip them.

EXPONENTIAL = [
    r"^(a+)+$",
    r"^(\w+\s?)*$",
    r"(.*)*x",
    r"^(a|a)*$",
    r"^(?:a\w|\wb)+$",
    r"^([a-z]{1,3})*$",
    r"(a*)*b",
    r"^(?:x?y?)*z$",
]

POLYNOMIAL = [
    r"\d*\d*\d*x",
    r".*.*.*.*=",
    r"^\s*\s*\s*\s*$",
    r".*a.*b",
    r"^\d+-?\d+$",
    r"^\w+\d+$",
]


@pytest.mark.parametrize("pattern", EXPONENTIAL)
def test_a_pattern_that_can_backtrack_exponentially_is_rejected(pattern):
    """These repeat in more than one way over the same text: 2^n paths."""
    problems = _problems(
        output_columns=["a"], fields=[OutputField("a", pattern=pattern)]
    )

    assert len(problems) == 1
    assert "is not safe on long answers" in problems[0]
    assert "more than one way" in problems[0]


@pytest.mark.parametrize("pattern", POLYNOMIAL)
def test_a_pattern_that_backtracks_polynomially_is_rejected(pattern):
    """Two repeats that take turns over the same characters: n^k paths.

    ``\\d*\\d*\\d*x`` ran past 20 seconds on a 2,000-character answer, well
    inside the length cap, so the cap alone never bounded these.
    """
    problems = _problems(
        output_columns=["a"], fields=[OutputField("a", pattern=pattern)]
    )

    assert len(problems) == 1
    assert "is not safe on long answers" in problems[0]
    assert "take turns" in problems[0]
    assert "neither" in problems[0]  # tells the client how to rewrite it


def test_a_back_reference_is_rejected():
    problems = _problems(
        output_columns=["a"], fields=[OutputField("a", pattern=r"^(a)\1$")]
    )

    assert len(problems) == 1
    assert "back-references" in problems[0]


def test_a_repeat_inside_a_lookahead_that_runs_at_every_character_is_rejected():
    """``(?=.*x)`` inside a repeat re-scans the rest of the text per character."""
    problems = _problems(
        output_columns=["a"],
        fields=[OutputField("a", pattern=r"^(?:(?=.*x)\w)+$")],
    )

    assert len(problems) == 1
    assert "lookahead" in problems[0]


def test_a_pattern_too_large_to_prove_safe_is_rejected_not_guessed():
    problems = _problems(
        output_columns=["a"], fields=[OutputField("a", pattern=r"^\d{5000}$")]
    )

    assert len(problems) == 1
    assert "too large to prove safe" in problems[0]


# Each accepted pattern with an input built to make it backtrack as much as it
# can: pump the repeats, then fail at the very end.
LINEAR = {
    r"^\d{3}-\d{4}$": lambda n: "1" * n,
    r"^\d{4}-\d{2}-\d{2}$": lambda n: "2026-10-" + "2" * n,
    r"^[A-Z]{2}\d+$": lambda n: "AB" + "1" * n + "x",
    r"^(?:ab|cd)+$": lambda n: "ab" * n + "!",
    r"^(?:a|ab)+$": lambda n: "ab" * n + "a!",
    r"^(\d{3},)*\d{3}$": lambda n: "123," * n + "12",
    r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$": lambda n: "a@" + "b.c" * n + "!",
    r"^(?=\d)\d+$": lambda n: "1" * n + "x",
    r"^[a-z]+(?: [a-z]+)*$": lambda n: "ab " * n + "!",
    r"(\w+,)*\w+": lambda n: "," * n,
    r"^(?:\w+,)*\w+$": lambda n: "ab," * n + "!",
    r"^[^a]*a[^b]*b": lambda n: "a" * n,
    r"^(?=.*\d)(?=.*[a-z]).{8,}$": lambda n: "a1" * n + "\n",
    r"^\s*-?\d+(?:\.\d+)?\s*$": lambda n: " " * n + "1" * n + "x",
    r"^\+?\d{1,3}[ -]?\d{3,14}$": lambda n: "+" + "1" * n,
    r"^[A-Z][a-z]+(?:[- ][A-Z][a-z]+)*$": lambda n: "Ab-" * n + "a",
    r"SCORE:\s*(\d+)": lambda n: "SCORE:" + " " * n,
}


@pytest.mark.parametrize("pattern", list(LINEAR))
def test_a_pattern_with_linear_behaviour_is_accepted(pattern):
    contract = build_contract(["a"], fields=[OutputField("a", pattern=pattern)])

    assert contract.fields[0].pattern == pattern


#: Timing checks cross-check the pattern analysis against Python's regex engine.
#: On shared CI runners (notably Windows and macOS) a linear pattern can time
#: like n**1.7, too close to the quadratic cases to gate a merge on, so they
#: run only on request: ONDINE_TIMING_TESTS=1. The guarantee itself (unsafe
#: patterns are refused at build time) is tested deterministically above.
timing_only = pytest.mark.skipif(
    not os.environ.get("ONDINE_TIMING_TESTS"),
    reason="timing check; set ONDINE_TIMING_TESTS=1 to run",
)


def _best_time(compiled, text: str) -> float:
    """The fastest of several runs: the least noisy estimate of the work done."""
    timings = timeit.repeat(lambda: compiled.search(text), number=3, repeat=5)
    return min(timings)


@timing_only
@pytest.mark.parametrize("pattern", list(LINEAR))
def test_an_accepted_pattern_takes_time_in_proportion_to_its_input(pattern):
    """Eight times the input costs about eight times the time, not 64 or more.

    A ratio, not a wall-clock budget: a slow machine slows both runs alike.
    Linear growth gives about 8; the quadratic patterns the check refuses
    give 64 and more. The bound sits far from both.
    """
    compiled = (
        build_contract(["a"], fields=[OutputField("a", pattern=pattern)])
        .rules_for("a")
        .pattern
    )
    attack = LINEAR[pattern]

    small = _best_time(compiled, attack(4_000))
    large = _best_time(compiled, attack(32_000))

    assert large / max(small, 1e-7) < 24, (small, large)


def test_extraction_patterns_are_checked_under_a_checked_contract():
    """A regex answer format runs its patterns on the whole answer, uncapped."""
    problems = _problems(
        fields=[OutputField("a")],
        response_format="regex",
        regex_patterns={"a": r"(a+)+b"},
    )

    assert len(problems) == 1
    assert problems[0].startswith("a: extraction pattern")
    assert "is not safe on long answers" in problems[0]


def test_extraction_patterns_are_not_checked_under_the_unchecked_contract():
    """Parity: with nothing declared, a regex answer format behaves as today."""
    contract = build_contract(
        ["a"], response_format="regex", regex_patterns={"a": r"(a+)+b"}
    )

    assert contract.checked is False
    assert contract.rules_for("a").extractor.pattern == r"(a+)+b"


# ---------------------------------------------------------------------------
# Run policy
# ---------------------------------------------------------------------------


def test_policy_defaults_are_the_decided_ones():
    policy = build_contract(["a"], fields=[OutputField("a")]).policy

    assert policy.on_failure == "reject"
    assert policy.partial_rows == "whole_row"
    assert policy.early_stop_fraction == 0.5
    assert policy.early_stop_window == 50


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (RunPolicy(on_failure="shrug"), "on_failure 'shrug'"),
        (RunPolicy(partial_rows="some"), "partial_rows 'some'"),
        (RunPolicy(early_stop_fraction=1.0), "early_stop_fraction 1.0"),
        (RunPolicy(early_stop_fraction=-0.1), "early_stop_fraction -0.1"),
        (RunPolicy(early_stop_window=0), "early_stop_window 0"),
    ],
)
def test_impossible_policy_is_named(policy, expected):
    problems = _problems(output_columns=["a"], fields=[OutputField("a")], policy=policy)

    assert any(expected in problem for problem in problems), problems


def test_early_stop_can_be_turned_off():
    contract = build_contract(
        ["a"], fields=[OutputField("a")], policy=RunPolicy(early_stop_fraction=None)
    )

    assert contract.policy.early_stop_fraction is None


# ---------------------------------------------------------------------------
# Schema-derived contracts (opt-in)
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Ticket(BaseModel):
    intent: Literal["refund", "cancel"]
    colour: Colour
    amount: Annotated[float, PydanticField(ge=0, le=1000)]
    quantity: Annotated[int, PydanticField(gt=0, lt=10)]
    code: Annotated[str, PydanticField(pattern=r"^[A-Z]{3}\d{2}$", max_length=5)]
    note: str | None = None


def _field(contract, name):
    return next(field for field in contract.fields if field.name == name)


def test_schema_gives_one_field_per_model_field():
    contract = build_contract(schema=Ticket)

    assert contract.checked is True
    assert contract.output_columns == (
        "intent",
        "colour",
        "amount",
        "quantity",
        "code",
        "note",
    )


def test_schema_literal_and_enum_become_allowed_values():
    contract = build_contract(schema=Ticket)

    assert _field(contract, "intent").allowed == ("refund", "cancel")
    assert _field(contract, "colour").allowed == ("red", "blue")
    assert _field(contract, "colour").type == "text"


def test_schema_bounds_become_minimum_and_maximum():
    """Exclusive integer bounds become the next whole number inward."""
    contract = build_contract(schema=Ticket)

    amount = _field(contract, "amount")
    assert (amount.type, amount.minimum, amount.maximum) == ("number", 0, 1000)
    quantity = _field(contract, "quantity")
    assert (quantity.type, quantity.minimum, quantity.maximum) == ("integer", 1, 9)


def test_schema_pattern_and_length_carry_over():
    code = _field(build_contract(schema=Ticket), "code")

    assert code.pattern == r"^[A-Z]{3}\d{2}$"
    assert code.max_length == 5


def test_schema_optional_field_is_not_required():
    contract = build_contract(schema=Ticket)

    assert _field(contract, "note").required is False
    assert _field(contract, "intent").required is True


def test_schema_with_an_unsafe_pattern_is_refused_at_build():
    class Risky(BaseModel):
        word: Annotated[str, PydanticField(pattern=r"^(\w+\s?)*$")]

    problems = _problems(schema=Risky)

    assert len(problems) == 1
    assert problems[0].startswith("word: pattern")


def test_a_field_declared_both_in_the_schema_and_explicitly_is_a_problem():
    problems = _problems(schema=Ticket, fields=[OutputField("intent")])

    assert problems == ["intent: declared both in the schema and as a field"]


def test_schema_change_changes_the_fingerprint():
    class Narrow(BaseModel):
        intent: Literal["refund", "cancel"]

    class Wider(BaseModel):
        intent: Literal["refund", "cancel", "other"]

    narrow = build_contract(schema=Narrow).fingerprint

    assert narrow == build_contract(schema=Narrow).fingerprint
    assert narrow != build_contract(schema=Wider).fingerprint
