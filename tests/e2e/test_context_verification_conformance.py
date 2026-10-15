"""Conformance of the context-store checks that run after the LLM.

These run the real pipeline against `LedgerClient`, whose answer for each row
is distinct and known in advance. That makes the checks' verdicts exact: a
row flagged when it should not be is a value mismatch, not a judgement call.

The live tests in tests/integration/test_context_store_e2e.py only assert
that the columns exist, so a check that flags every row passes them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore
from tests.e2e.conformance_harness import (
    LedgerClient,
    token_for,
)


def run_with_contradiction_detection(product_ids: list[str]) -> pd.DataFrame:
    """Run one row per product id; LedgerClient gives every row its own answer."""
    frame = pd.DataFrame(
        {
            "product_id": product_ids,
            "marker": [token_for(index) for index in range(len(product_ids))],
        }
    )
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            frame,
            input_columns=["product_id", "marker"],
            output_columns=["answer"],
        )
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(LedgerClient())
        .with_context_store(InMemoryContextStore())
        .with_contradiction_detection(key_columns=["product_id"])
        .build()
    )
    return pipeline.execute().to_pandas()


def flagged_rows(output: pd.DataFrame) -> list[int]:
    return output.index[output["_contradiction"].eq(True)].tolist()


def test_rows_with_different_keys_never_contradict():
    """Different products may get different answers; that is not a conflict.

    The key columns are input columns. Reading them from the parsed result
    rows, which hold only outputs, gave every row the same empty key, so any
    two different answers were flagged (#311).
    """
    output = run_with_contradiction_detection(["p1", "p2", "p3"])

    assert flagged_rows(output) == []


def test_only_the_row_sharing_a_key_is_flagged():
    """Rows 0 and 2 are the same product with different answers; row 1 is not."""
    output = run_with_contradiction_detection(["p1", "p2", "p1"])

    assert flagged_rows(output) == [2]


#: One word per row. A row's input and its answer both carry it, so an answer
#: grounds against its own row and against no other.
WORDS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "heath"]


class EchoWordClient(LedgerClient):
    """Answers each row with that row's word."""

    def _answer_token(self, token: str) -> str:
        return f"{super()._answer_token(token)} {WORDS[int(token[-4:])]}"


def word_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"marker": [f"{token_for(i)} {word}" for i, word in enumerate(WORDS)]}
    )


def grounded_pipeline(client: LedgerClient, frame: pd.DataFrame, checkpoint_dir):
    return (
        PipelineBuilder.create()
        .from_dataframe(frame, input_columns=["marker"], output_columns=["answer"])
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(client)
        .with_context_store(InMemoryContextStore())
        .with_grounding(threshold=0.0, action="flag")
        .with_checkpoint_dir(str(checkpoint_dir))
        .with_checkpoint_interval(1)
        .with_error_policy("fail")
        .with_max_retries(1)
        .with_batch_size(1)
        .with_concurrency(4)
        .build()
    )


def test_a_resumed_run_grounds_each_answer_against_its_own_row(tmp_path):
    """Resume must not pair a row's answer with another row's input.

    On resume the pipeline only re-sends unfinished rows, but the results also
    hold every row restored from the cache. Matching results to inputs by
    position then grounds row 6's answer against row 5's text. A run that
    never stopped is the oracle: every row's score must match it.
    """
    frame = word_frame()
    fresh = grounded_pipeline(EchoWordClient(), frame, tmp_path / "fresh").execute()
    expected_scores = fresh.to_pandas()["_grounding_score"].tolist()
    # The oracle must tell rows apart, or a misaligned run would match it.
    assert all(score > 0 for score in expected_scores)

    dead = token_for(5)
    first = grounded_pipeline(
        EchoWordClient(fail_tokens={dead}, fail_times=None, transient=True),
        frame,
        tmp_path / "resumed",
    )
    with pytest.raises(Exception):
        first.execute()
    resumed = grounded_pipeline(EchoWordClient(), frame, tmp_path / "resumed").execute(
        resume_from=first.session_id
    )

    assert resumed.to_pandas()["_grounding_score"].tolist() == pytest.approx(
        expected_scores
    )
