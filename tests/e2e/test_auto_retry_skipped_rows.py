"""End-to-end: auto-retry re-runs the rows a provider error skipped.

Without a contract the retry pass used to pick only rows whose output cells
were blank. It runs before the step that blanks lost cells, so a row the skip
policy dropped still held the ``[SKIPPED]`` marker, a non-empty string, and
was never asked again. The run's ledger already knew the row failed; the pass
now asks it, as it does under a checked contract.
"""

from __future__ import annotations

from tests.e2e.conformance_harness import (
    LedgerClient,
    conformance_frame,
    expected_answers,
    token_for,
)
from tests.e2e.test_pipeline_conformance import build


def _auto_retry(pipeline, attempts: int = 2):
    pipeline.specifications.processing.auto_retry_failed = True
    pipeline.specifications.processing.max_retry_attempts = attempts
    return pipeline


def test_auto_retry_recovers_a_row_a_provider_error_skipped():
    """Row 3 fails once, is skipped, and the retry pass recovers it."""
    failing = token_for(3)
    client = LedgerClient(fail_tokens={failing}, fail_times=1)

    result = _auto_retry(build(client, conformance_frame(6))).execute()

    assert list(result.to_pandas()["answer"]) == expected_answers(6)
    assert client.calls[failing] == 2
    assert {token for token, n in client.calls.items() if n > 1} == {failing}
    # The recovered row is no longer reported lost.
    assert result.errors == []
    assert result.is_complete
    assert result.metrics.skipped_rows == 0


def test_auto_retry_asks_a_dead_row_once_per_attempt_then_reports_it_lost():
    """A row that always fails is asked once per pass, then stays lost."""
    dead = token_for(2)
    client = LedgerClient(fail_tokens={dead}, fail_times=None)

    result = _auto_retry(build(client, conformance_frame(5)), attempts=2).execute()

    answers = list(result.to_pandas()["answer"])
    expected = expected_answers(5)
    assert answers[:2] + answers[3:] == expected[:2] + expected[3:]
    assert answers[2] is None or answers[2] != answers[2]  # None or NaN
    assert client.calls[dead] == 3  # first pass + two retry passes
    assert [error.row_index for error in result.errors] == [2]
    assert not result.is_complete
