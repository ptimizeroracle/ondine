"""The correction a re-asked row is sent: what it says, and what it never does."""

from __future__ import annotations

from ondine.harness.reask import MAX_ANSWER_CHARS, correction


def test_the_correction_quotes_the_answer_then_every_reason():
    text = correction(
        "refund or billing", ["intent: is not one of the 3 allowed values"]
    )

    assert text == (
        "Your previous answer was:\n"
        "refund or billing\n"
        "It was not accepted: intent: is not one of the 3 allowed values.\n"
        "Answer again. Reply with only the answer, in the format asked for above."
    )


def test_several_reasons_are_joined_in_order_and_each_given_once():
    text = correction("x", ["a: too long", "b: not a number", "a: too long"])

    assert "It was not accepted: a: too long; b: not a number.\n" in text


def test_a_long_answer_is_cut_so_the_re_ask_stays_small():
    text = correction("y" * (MAX_ANSWER_CHARS + 500), ["a: wrong"])

    quoted = text.split("\n")[1]
    assert quoted == "y" * MAX_ANSWER_CHARS + " [… 500 more characters]"


def test_without_the_answer_the_correction_gives_the_reasons_only():
    text = correction(None, ["intent: is empty"])

    assert "Your previous answer was:" not in text
    assert text.startswith("Your previous answer was not accepted: intent: is empty.\n")


def test_an_answer_that_was_already_parsed_is_quoted_as_json():
    text = correction({"urgency": 9}, ["urgency: 9 is above the maximum 5"])

    assert '{"urgency": 9}' in text


def test_a_reason_already_ending_in_a_full_stop_gets_no_second_one():
    text = correction("x", ["intent: is empty."])

    assert "is empty.\n" in text
    assert "is empty..\n" not in text


# ---------------------------------------------------------------------------
# The instruction: what would be accepted, from the contract
# ---------------------------------------------------------------------------

from ondine.harness.contract import OutputField, build_contract  # noqa: E402
from ondine.harness.reask import instruction  # noqa: E402


def test_one_labelled_column_is_told_the_allowed_values_by_name():
    # Live finding: asked "is it consistent?", the model answered "No", and a
    # re-ask that did not name the values got "No" again.
    contract = build_contract(
        fields=[OutputField("answer", allowed=["consistent", "inconsistent"])]
    )

    text = instruction(contract, ["answer"])

    assert (
        text
        == "Reply with only one of these values, and nothing else: consistent, inconsistent."
    )


def test_a_long_list_of_values_is_referred_to_not_repeated():
    labels = [f"label_{i}" for i in range(150)]
    contract = build_contract(fields=[OutputField("intent", allowed=labels)])

    text = instruction(contract, ["intent"])

    assert text == "Reply with only one of the 150 allowed values, and nothing else."
    assert "label_0" not in text


def test_a_column_with_no_allowed_values_is_asked_for_the_answer_alone():
    contract = build_contract(
        fields=[OutputField("score", type="integer", minimum=1, maximum=5)]
    )

    assert (
        instruction(contract, ["score"])
        == "Reply with only the answer, and nothing else."
    )


def test_several_columns_keep_the_format_asked_for_and_name_each_failing_columns_values():
    contract = build_contract(
        fields=[
            OutputField("intent", allowed=["refund", "billing"]),
            OutputField("urgency", type="integer", minimum=1, maximum=5),
        ]
    )

    text = instruction(contract, ["intent", "urgency"])

    assert text == (
        "Reply with only the answer, in the format asked for above. "
        "intent must be one of: refund, billing."
    )


def test_the_instruction_replaces_the_generic_closing_line():
    text = correction(
        "No", ["answer: 'No' is not one of the 2 allowed values"], instruction="Say X."
    )

    assert text.endswith(
        "It was not accepted: answer: 'No' is not one of the 2 allowed values.\nSay X."
    )
