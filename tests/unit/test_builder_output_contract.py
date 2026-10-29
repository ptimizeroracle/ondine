"""Declaring an output contract from the builder.

The builder is where a client says what a deliverable value is. Three
properties carry the weight, so most tests here aim at them:

* **Nothing declared means no contract.** ``components.output_contract`` stays
  ``None``, so a pipeline built without declarations runs exactly as before the
  harness existed. A structured-output schema is not a declaration on its own
  (decision D1): enforcing it is opt-in.
* **A broken contract stops ``build()``**, listing every problem at once,
  before a pipeline exists and so before anything is spent.
* **Calls accumulate per column, never silently overwrite.** Two different
  declarations for one column are a build error; the same declaration twice
  is harmless.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.harness.contract import (
    ContractError,
    OutputField,
    Relation,
    RunPolicy,
    build_contract,
)
from ondine.stages.response_parser_stage import RegexParser


class _Ticket(BaseModel):
    intent: Literal["refund", "billing"]
    amount: float


def _builder(
    outputs: tuple[str, ...] = ("intent", "amount"),
    inputs: tuple[str, ...] = ("text", "limit"),
) -> PipelineBuilder:
    frame = pd.DataFrame({column: ["x"] for column in inputs})
    return (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=list(inputs), output_columns=list(outputs))
        .with_prompt("Classify: {text}")
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",  # pragma: allowlist secret
        )
    )


def _problems(builder: PipelineBuilder) -> list[str]:
    """The problems ``build()`` reports, or an empty list if it succeeds."""
    try:
        builder.build()
    except ContractError as error:
        return list(error.problems)
    return []


# ---------------------------------------------------------------------------
# Nothing declared
# ---------------------------------------------------------------------------


def test_nothing_declared_leaves_no_contract():
    """No declaration, no contract: the run resolves the unchecked one itself."""
    pipeline = _builder().build()

    assert pipeline._components.output_contract is None


def test_structured_output_alone_is_not_a_contract():
    """D1: a schema passed for structured output is never enforced by default.

    Enforcing it automatically would add columns and reject rows for every
    existing structured-output user.
    """
    pipeline = _builder().with_structured_output(_Ticket).build()

    assert pipeline._components.output_contract is None


# ---------------------------------------------------------------------------
# Per-column declarations
# ---------------------------------------------------------------------------


def test_allowed_values_for_one_column_is_one_call():
    """The common case: the allowed labels of one column."""
    pipeline = (
        _builder().with_output_field("intent", allowed=["refund", "billing"]).build()
    )

    contract = pipeline._components.output_contract
    assert contract is not None
    assert contract.checked is True
    assert contract.rules_for("intent").declaration.allowed == ("refund", "billing")


def test_undeclared_output_columns_stay_unconstrained():
    """Declaring one column does not invent rules for the others."""
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .build()
        ._components.output_contract
    )

    assert contract.output_columns == ("intent", "amount")
    amount = contract.rules_for("amount").declaration
    assert amount.type == "any"
    assert amount.required is False


def test_keyword_declaration_carries_every_rule():
    """Every rule of the interface spec reaches the contract unchanged."""
    contract = (
        _builder(outputs=("intent", "amount", "code"))
        .with_output_field(
            "intent",
            allowed=["refund", "billing"],
            aliases={"money back": "refund"},
            normalise=["strip", "casefold"],
        )
        .with_output_field(
            "amount", type="number", required=False, absent="n/a", minimum=0
        )
        .with_output_field("code", pattern=r"^[A-Z]{3}$", min_length=3, max_length=3)
        .build()
        ._components.output_contract
    )

    intent = contract.rules_for("intent")
    assert intent.lookup["money back"] == "refund"
    assert intent.declaration.normalise == ("strip", "casefold")
    amount = contract.rules_for("amount").declaration
    assert (amount.type, amount.required, amount.absent, amount.minimum) == (
        "number",
        False,
        "n/a",
        0,
    )
    code = contract.rules_for("code").declaration
    assert (code.pattern, code.min_length, code.max_length) == (r"^[A-Z]{3}$", 3, 3)


def test_a_prepared_output_field_is_accepted():
    """Declarations built elsewhere (a config loader, a loop) pass as they are."""
    declaration = OutputField("amount", type="number", minimum=0, maximum=100)

    contract = (
        _builder().with_output_field(declaration).build()._components.output_contract
    )

    assert contract.rules_for("amount").declaration == declaration


def test_an_output_field_with_extra_keywords_is_refused_at_the_call():
    """Two sources of rules for one declaration would be ambiguous."""
    with pytest.raises(TypeError, match="OutputField"):
        _builder().with_output_field(OutputField("amount"), minimum=0)


def test_the_builder_contract_equals_the_direct_build():
    """The builder adds no rule of its own: same declarations, same fingerprint."""
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund", "billing"])
        .with_output_relation("amount", "<=", input_column="limit")
        .build()
        ._components.output_contract
    )

    direct = build_contract(
        ["intent", "amount"],
        fields=[OutputField("intent", allowed=["refund", "billing"])],
        relations=[Relation("amount", "<=", input_column="limit")],
        input_columns=["text", "limit"],
        response_format="raw",
    )
    assert contract.fingerprint == direct.fingerprint


def test_declarations_may_come_before_the_data_source():
    """Validation waits for build(), so call order does not matter."""
    pipeline = (
        PipelineBuilder.create()
        .with_output_field("intent", allowed=["refund"])
        .from_dataframe(
            pd.DataFrame({"text": ["x"]}),
            input_columns=["text"],
            output_columns=["intent"],
        )
        .with_prompt("Classify: {text}")
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",  # pragma: allowlist secret
        )
        .build()
    )

    assert pipeline._components.output_contract.output_columns == ("intent",)


# ---------------------------------------------------------------------------
# Calling twice
# ---------------------------------------------------------------------------


def test_declarations_for_different_columns_accumulate():
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_output_field("amount", type="number")
        .build()
        ._components.output_contract
    )

    assert contract.rules_for("intent").declaration.allowed == ("refund",)
    assert contract.rules_for("amount").declaration.type == "number"


def test_the_same_declaration_twice_is_harmless():
    """Re-running a configuration step must not break a build."""
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_output_field("intent", allowed=["refund"])
        .with_output_relation("amount", "<=", input_column="limit")
        .with_output_relation("amount", "<=", input_column="limit")
        .build()
        ._components.output_contract
    )

    assert contract.rules_for("intent").declaration.allowed == ("refund",)
    assert len(contract.relations) == 1


def test_two_different_declarations_for_one_column_fail_the_build():
    """Neither merging nor replacing is safe: one of them was a mistake."""
    builder = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_output_field("intent", allowed=["billing"])
    )

    problems = _problems(builder)

    assert any("intent" in problem and "twice" in problem for problem in problems)


def test_a_field_declared_in_the_enforced_schema_and_by_hand_fails_the_build():
    builder = (
        _builder()
        .with_structured_output(_Ticket, as_contract=True)
        .with_output_field("intent", allowed=["refund"])
    )

    problems = _problems(builder)

    assert any("intent" in problem and "schema" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Field-to-field rules
# ---------------------------------------------------------------------------


def test_relations_between_outputs_and_to_an_input_column():
    contract = (
        _builder(outputs=("start", "end", "amount"))
        .with_output_field("start", type="integer")
        .with_output_field("end", type="integer")
        .with_output_relation("end", ">=", "start")
        .with_output_relation("amount", "<=", input_column="limit")
        .build()
        ._components.output_contract
    )

    assert contract.relations == (
        Relation("end", ">=", "start"),
        Relation("amount", "<=", input_column="limit"),
    )


def test_a_relation_alone_makes_a_checked_contract():
    contract = (
        _builder()
        .with_output_relation("amount", "<=", input_column="limit")
        .build()
        ._components.output_contract
    )

    assert contract.checked is True


def test_a_relation_naming_an_unknown_column_fails_the_build():
    builder = _builder().with_output_relation("amount", "<=", input_column="ceiling")

    problems = _problems(builder)

    assert any("ceiling" in problem for problem in problems)


# ---------------------------------------------------------------------------
# The schema switch
# ---------------------------------------------------------------------------


def test_the_schema_becomes_the_contract_only_when_asked():
    contract = (
        _builder()
        .with_structured_output(_Ticket, as_contract=True)
        .build()
        ._components.output_contract
    )

    assert contract.schema is _Ticket
    assert contract.rules_for("intent").declaration.allowed == ("refund", "billing")
    assert contract.rules_for("amount").declaration.type == "number"


def test_the_schema_contract_combines_with_rules_on_other_columns():
    contract = (
        _builder(outputs=("intent", "amount", "note"))
        .with_structured_output(_Ticket, as_contract=True)
        .with_output_field("note", max_length=200)
        .with_output_relation("amount", "<=", input_column="limit")
        .build()
        ._components.output_contract
    )

    assert contract.rules_for("note").declaration.max_length == 200
    assert contract.schema is _Ticket


class _TicketBatch(BaseModel):
    """The batch wrapper with_batch_size(>1) asks for: one row per item."""

    items: list[_Ticket]


@pytest.mark.parametrize(
    "schema", [_Ticket, _TicketBatch], ids=["row model", "batch wrapper"]
)
def test_the_schema_contract_reads_the_row_model_of_a_batch_wrapper(schema):
    """A batch wrapper's rules are its row model's: ``items`` is not a column."""
    contract = (
        _builder()
        .with_batch_size(3)
        .with_structured_output(schema, as_contract=True)
        .build()
        ._components.output_contract
    )

    assert contract.schema is _Ticket
    assert contract.output_columns == ("intent", "amount")
    assert contract.rules_for("intent").declaration.allowed == ("refund", "billing")
    assert contract.rules_for("amount").declaration.type == "number"


def test_a_model_with_more_than_items_is_not_unwrapped():
    """Only the exact wrapper shape is read as rows; anything else is the row."""

    class _Report(BaseModel):
        items: list[_Ticket]
        summary: str

    problems = _problems(
        _builder(outputs=("items", "summary")).with_structured_output(
            _Report, as_contract=True
        )
    )

    assert not any("not an output column" in problem for problem in problems)


def test_a_later_structured_output_call_without_the_switch_drops_it():
    """with_structured_output replaces its whole configuration, switch included."""
    pipeline = (
        _builder()
        .with_structured_output(_Ticket, as_contract=True)
        .with_structured_output(_Ticket)
        .build()
    )

    assert pipeline._components.output_contract is None


# ---------------------------------------------------------------------------
# Run policy
# ---------------------------------------------------------------------------


def test_the_policy_defaults_to_the_decided_one():
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .build()
        ._components.output_contract
    )

    assert contract.policy == RunPolicy()
    assert (contract.policy.early_stop_fraction, contract.policy.early_stop_window) == (
        0.5,
        50,
    )


def test_the_policy_carries_every_option():
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_contract_policy(
            on_failure="deliver_flagged",
            partial_rows="keep_valid_fields",
            early_stop_fraction=0.2,
            early_stop_window=100,
        )
        .build()
        ._components.output_contract
    )

    assert contract.policy == RunPolicy(
        on_failure="deliver_flagged",
        partial_rows="keep_valid_fields",
        early_stop_fraction=0.2,
        early_stop_window=100,
    )


def test_early_stop_can_be_turned_off():
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_contract_policy(early_stop_fraction=None)
        .build()
        ._components.output_contract
    )

    assert contract.policy.early_stop_fraction is None


def test_a_second_policy_call_replaces_the_first_whole():
    """Like every other builder setter: the last call is the policy."""
    contract = (
        _builder()
        .with_output_field("intent", allowed=["refund"])
        .with_contract_policy(on_failure="deliver_flagged")
        .with_contract_policy(early_stop_fraction=None)
        .build()
        ._components.output_contract
    )

    assert contract.policy == RunPolicy(early_stop_fraction=None)


def test_a_policy_with_nothing_to_check_fails_the_build():
    """A policy alone would be silently inert: no row can fail a check."""
    problems = _problems(_builder().with_contract_policy(on_failure="deliver_flagged"))

    assert any("nothing" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Build-time validation
# ---------------------------------------------------------------------------


def test_a_broken_contract_lists_every_problem_at_once():
    """One pass to fix, not one build per typo."""
    builder = (
        _builder()
        .with_output_field("amount", type="number", minimum=10, maximum=1)
        .with_output_field("missing", allowed=["x"])
        .with_output_relation("amount", "<=", input_column="ceiling")
        .with_contract_policy(on_failure="explode", early_stop_window=0)
    )

    problems = _problems(builder)

    assert len(problems) == 5
    assert any("minimum 10 is above the maximum 1" in p for p in problems)
    assert any("missing" in p for p in problems)
    assert any("ceiling" in p for p in problems)
    assert any("explode" in p for p in problems)
    assert any("early_stop_window" in p for p in problems)


def test_a_broken_contract_stops_before_a_pipeline_exists():
    """Nothing is spent: no pipeline, so no client and no model call."""
    builder = _builder().with_output_field("intent", allowed=[])

    with (
        patch("ondine.api.pipeline_builder.Pipeline") as pipeline_class,
        pytest.raises(ContractError, match="allowed values are empty"),
    ):
        builder.build()

    pipeline_class.assert_not_called()


def test_the_response_format_reaches_the_contract():
    """A regex parser's extraction patterns are read, and checked, as patterns."""
    contract = (
        _builder(outputs=("intent",))
        .with_parser(RegexParser({"intent": r"Intent: (\w+)"}))
        .with_output_field("intent", allowed=["refund"])
        .build()
        ._components.output_contract
    )

    assert contract.parse_mode == "regex"
    assert contract.regex_patterns == (("intent", r"Intent: (\w+)"),)


# ---------------------------------------------------------------------------
# Serializable specifications
# ---------------------------------------------------------------------------


def test_the_contract_lives_on_components_not_in_the_specs():
    """Schema models are not serializable, so the contract stays off the specs."""
    pipeline = (
        _builder()
        .with_structured_output(_Ticket, as_contract=True)
        .with_output_relation("amount", "<=", input_column="limit")
        .with_contract_policy(early_stop_fraction=None)
        .build()
    )

    dumped = pipeline.specifications.model_dump(mode="json")

    assert isinstance(dumped, dict)
    assert "output_contract" not in pipeline.specifications.metadata
    assert pipeline._components.output_contract is not None


# ---------------------------------------------------------------------------
# Not yet enforced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        PipelineBuilder.with_output_field,
        PipelineBuilder.with_output_relation,
        PipelineBuilder.with_contract_policy,
        PipelineBuilder.with_structured_output,
    ],
    ids=lambda method: method.__name__,
)
def test_every_contract_option_says_it_is_not_enforced_yet(method):
    """Until delivery is wired, running with a contract raises; the docs say so."""
    doc = " ".join((method.__doc__ or "").split())

    assert "not yet enforced" in doc.lower()
    assert "NotImplementedError" in doc


def test_the_type_keyword_lists_every_field_type():
    """The docstring names every type a declaration may take, "any" included."""
    from ondine.harness.contract import FIELD_TYPES

    doc = " ".join((PipelineBuilder.with_output_field.__doc__ or "").split())

    for field_type in FIELD_TYPES:
        assert f'"{field_type}"' in doc
