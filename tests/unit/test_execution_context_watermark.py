"""The resume watermark must never run ahead of a gap.

``last_processed_row`` is the promise resume relies on: every row up to it has
been processed, so none of them need doing again. When completions arrive out
of order — which is what concurrency is for — a watermark taken from whichever
row finished last breaks that promise, and the rows in the gap are dropped from
the resumed run without a word.
"""

from ondine.orchestration.execution_context import ExecutionContext


def test_a_fresh_context_has_completed_no_rows():
    """Zero would be indistinguishable from "row 0 is done".

    Catches a resume that skips row 0 of a run which crashed before finishing
    anything at all.
    """
    assert ExecutionContext().last_processed_row == -1


def test_the_watermark_stops_below_a_row_that_never_finished():
    """Rows 0, 1, 3, 4 done and row 2 still in flight means the watermark is 1."""
    context = ExecutionContext()

    for row in (0, 1, 3, 4):
        context.complete_rows(row)

    assert context.last_processed_row == 1


def test_closing_the_gap_absorbs_everything_waiting_above_it():
    """The rows finished early are not re-done once the gap closes."""
    context = ExecutionContext()
    for row in (0, 1, 3, 4):
        context.complete_rows(row)

    context.complete_rows(2)

    assert context.last_processed_row == 4
    assert context.pending_completions == set()


def test_a_completed_batch_advances_the_watermark_across_all_its_rows():
    """A mega-batch reports its first row and its size, not one row."""
    context = ExecutionContext()

    context.complete_rows(0, count=4)

    assert context.last_processed_row == 3


def test_reporting_the_same_row_twice_does_not_move_the_watermark_twice():
    """A retry re-reporting a row must not push the watermark past a gap."""
    context = ExecutionContext()
    context.complete_rows(0)

    context.complete_rows(0)

    assert context.last_processed_row == 0
