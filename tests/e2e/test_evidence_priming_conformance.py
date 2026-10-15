"""What one run may teach the next through the context store (#308).

Evidence priming copies earlier answers into later prompts. Only an answer
that passed a check the user configured may travel that way: an unchecked
answer copied forward makes the next run repeat the first run's mistakes.

Each test runs a first pipeline into a shared store, then a second pipeline
with evidence priming over a similar input, and reads the prompt the second
run actually sent.
"""

from __future__ import annotations

import pandas as pd

from ondine import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore
from tests.e2e.conformance_harness import TOKEN_PATTERN, LedgerClient

FIRST_INPUT = "row-0000 amber basalt cobalt"
#: Shares "amber basalt cobalt" with FIRST_INPUT, so a search finds its record.
SIMILAR_INPUT = "row-0001 amber basalt cobalt"
#: The first run's answer. It shares "amber" with FIRST_INPUT, so it grounds
#: at a low threshold and fails at a high one.
FIRST_ANSWER = "ANS-row-0000 amber"


class AmberClient(LedgerClient):
    """Answers every row with its marker plus the word "amber"."""

    def _answer_token(self, token: str) -> str:
        return f"{super()._answer_token(token)} amber"


class PrimedPromptRecorder(LedgerClient):
    """Answers the row a primed prompt is about, and records the prompt.

    A primed prompt also carries the earlier rows' markers in its evidence
    block, which LedgerClient rejects as a mixed prompt. The row's own prompt
    comes first, so its marker is the first one.
    """

    def _respond(self, prompt: str) -> str:
        return self._answer_token(TOKEN_PATTERN.findall(prompt)[0])


def builder_over(text: str, client: LedgerClient, store: InMemoryContextStore):
    frame = pd.DataFrame({"text": [text]})
    return (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["text"], output_columns=["answer"])
        .with_prompt("Answer for {text}")
        .with_custom_llm_client(client)
        .with_context_store(store)
    )


def primed_prompt_after(first_run) -> str:
    """The prompt a primed second run sends, after `first_run` fills the store."""
    store = InMemoryContextStore()
    first_run(builder_over(FIRST_INPUT, AmberClient(), store)).build().execute()

    recorder = PrimedPromptRecorder()
    (
        builder_over(SIMILAR_INPUT, recorder, store)
        .with_evidence_priming(top_k=3, min_score=0.0)
        .build()
        .execute()
    )
    return recorder.prompts[0]


def test_an_answer_no_check_looked_at_is_never_primed():
    """Confidence scoring and contradiction detection check nothing per answer.

    Before #308 every answer was stored as "input → answer" whatever ran, so
    the next run's prompt received it as a "previously validated" answer.
    """
    prompt = primed_prompt_after(
        lambda builder: builder.with_confidence_scoring().with_contradiction_detection()
    )

    assert FIRST_ANSWER not in prompt


def test_an_answer_that_failed_grounding_is_never_primed():
    prompt = primed_prompt_after(
        lambda builder: builder.with_grounding(threshold=0.99, action="flag")
    )

    assert FIRST_ANSWER not in prompt


def test_an_answer_that_passed_grounding_is_primed_with_its_input():
    """The record must say what was asked, or the model cannot judge its fit."""
    prompt = primed_prompt_after(
        lambda builder: builder.with_grounding(threshold=0.1, action="flag")
    )

    assert f"{FIRST_INPUT} → {FIRST_ANSWER}" in prompt


def test_primed_answers_are_not_presented_as_validated():
    """A grounding pass means "similar to its source text", not "correct"."""
    prompt = primed_prompt_after(
        lambda builder: builder.with_grounding(threshold=0.1, action="flag")
    )

    assert FIRST_ANSWER in prompt
    assert "validated" not in prompt.lower()
