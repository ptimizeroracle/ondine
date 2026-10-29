"""Resume re-judges the answers it restores, and records their verdicts.

Resume does not ask the model again for a row the previous attempt answered:
it restores that answer from the response cache. Migration step 2 runs every
answer through the judge, the restored ones included. The judge is
deterministic, so a restored answer gets the verdict it got the first time,
and the resumed run's ledger holds a verdict for every answered row. That is
what lets resume work with no stored verdicts (interfaces spec, section 5).
"""

from __future__ import annotations

import json

import pytest

from ondine import PipelineBuilder
from ondine.harness.contract import build_contract
from ondine.harness.judge import judge
from ondine.harness.ledger import RowLedger
from ondine.harness.status import RowStatus
from tests.e2e.conformance_harness import (
    LedgerClient,
    ProviderError,
    answer_for,
    conformance_frame,
    token_for,
)

COLUMNS = ["answer", "note"]
UNREADABLE = token_for(1)
CRASH = token_for(3)


class _JsonClient(LedgerClient):
    """Answers each row as a JSON object, except one row it answers in prose."""

    def __init__(self, *, crash: set[str] | None = None) -> None:
        super().__init__(fail_tokens=crash)

    def _answer_token(self, token: str) -> str:
        answer = super()._answer_token(token)
        if token == UNREADABLE:
            return f"I think it is {answer}"
        return json.dumps({"answer": answer, "note": "ok"})


def _text_for(row: int) -> str:
    token = token_for(row)
    if token == UNREADABLE:
        return f"I think it is {answer_for(token)}"
    return json.dumps({"answer": answer_for(token), "note": "ok"})


def _builder(client: LedgerClient, checkpoint_dir, **options):
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            conformance_frame(5), input_columns=["marker"], output_columns=COLUMNS
        )
        .with_prompt("Answer for {marker}")
        .with_custom_llm_client(client)
        .with_max_retries(1)
        .with_batch_size(1)
        .with_concurrency(1)
        .with_checkpoint_dir(str(checkpoint_dir))
        .with_checkpoint_interval(1)
    )
    if "error_policy" in options:
        builder = builder.with_error_policy(options["error_policy"])
    return builder


@pytest.fixture
def verdicts(monkeypatch) -> list[tuple]:
    """Every verdict reported to any ledger, as (row, attempt, verdict, stage)."""
    calls: list[tuple] = []
    original = RowLedger.record_verdict

    def spy(self, row, attempt, verdict, *, stage):
        calls.append((row, attempt, verdict, stage))
        original(self, row, attempt, verdict, stage=stage)

    monkeypatch.setattr(RowLedger, "record_verdict", spy)
    return calls


def test_resume_rejudges_restored_answers_and_records_their_verdicts(
    tmp_path, verdicts
):
    first = _builder(_JsonClient(crash={CRASH}), tmp_path, error_policy="fail").build()
    with pytest.raises(ProviderError):
        first.execute()
    # The crash came before the parse stage: nothing was judged yet.
    assert verdicts == []

    second_client = _JsonClient()
    resumed = (
        _builder(second_client, tmp_path).build().execute(resume_from=first.session_id)
    )

    # Rows 0-2 were restored from the cache, never asked again.
    assert set(second_client.calls) == {token_for(3), token_for(4)}
    contract = build_contract(COLUMNS)
    by_row = {
        row: (attempt, verdict, stage) for row, attempt, verdict, stage in verdicts
    }
    assert len(verdicts) == 5
    assert by_row == {
        row: (1, judge(contract, _text_for(row)), "ResponseParser") for row in range(5)
    }

    snapshot = resumed.row_ledger
    assert [row.status for row in snapshot.rows] == [
        RowStatus.UNCHECKED,
        RowStatus.FAILED,
        RowStatus.UNCHECKED,
        RowStatus.UNCHECKED,
        RowStatus.UNCHECKED,
    ]
    message = (
        "response is not a JSON object with any of the output columns "
        f"['answer', 'note']: {_text_for(1)!r}"
    )
    assert snapshot.rows[1].failed_checks == (message,)
    # The restored row's unreadable answer is in the error list, as the parse
    # stage has always reported it.
    assert [
        (e.row_index, e.stage, e.error_type, e.message) for e in resumed.errors
    ] == [(1, "ResponseParser", "parse_failure", message)]
    frame = resumed.to_pandas()
    assert list(frame["answer"].fillna("<lost>")) == [
        answer_for(token_for(0)),
        "<lost>",
        answer_for(token_for(2)),
        answer_for(token_for(3)),
        answer_for(token_for(4)),
    ]
