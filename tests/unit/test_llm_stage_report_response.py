"""What one model-call response tells the row ledger, decided in one place.

``report_response`` (in the LLM stage module) is the only code that knows how
to read a response for the ledger: whether the error policy skipped it or
substituted its default, which input rows a batch prompt answered, and how
the call's cost belongs to them. The live stage, resume and the batch
splitter all go through it. These tests pin each of those rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import (
    SKIPPED_OUTPUT_MARKER,
    LLMResponse,
    PromptBatch,
    ResponseBatch,
    RowMetadata,
)
from ondine.core.specifications import ErrorPolicy, LLMSpec
from ondine.harness.status import RowStatus
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.stages.llm_invocation_stage import (
    LLMInvocationStage,
    report_response,
    skipped_by_error_policy,
)

STAGE = "LLMInvocation"


def _response(text: str = "ok", **metadata: Any) -> LLMResponse:
    return LLMResponse(
        text=text,
        tokens_in=6,
        tokens_out=3,
        model="m",
        cost=Decimal("0.0003"),
        latency_ms=1.0,
        metadata=dict(metadata),
    )


def _batch(*rows: int) -> RowMetadata:
    """A mega-batch prompt's metadata, as the aggregator builds it."""
    return RowMetadata(
        row_index=rows[0],
        custom={
            "is_batch": True,
            "batch_size": len(rows),
            "batch_metadata": {"row_ids": list(rows), "original_count": len(rows)},
        },
    )


class _Fails(LLMClient):
    """A provider that never answers."""

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    def estimate_tokens(self, text: str) -> int:
        return 1


def _context(rows: int = 8) -> ExecutionContext:
    context = ExecutionContext(total_rows=rows)
    context.intermediate_data["token_tracking"] = {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    return context


def _shape(context: ExecutionContext) -> list[tuple[int, str, str]]:
    return [(e.row_index, e.stage, e.error_type) for e in context.row_errors]


def _statuses(context: ExecutionContext) -> list[RowStatus]:
    return [row.status for row in context.ledger.snapshot().rows]


# ── the rows a batch covers ───────────────────────────────────────────────


def test_a_skipped_batch_blames_the_rows_it_held_not_a_range():
    """Rows 3 and 7 were batched together (a resume gap); rows 4-6 were not."""
    context = _context()

    report_response(
        context,
        _response(SKIPPED_OUTPUT_MARKER, action="skipped", error="down"),
        _batch(3, 7),
        stage=STAGE,
    )

    assert _shape(context) == [(3, STAGE, "skipped"), (7, STAGE, "skipped")]
    assert context.skipped_rows == 2


def test_the_stage_reports_a_skipped_batch_at_its_own_rows():
    """The live path: the stage's error policy skips a batch of rows 1 and 5."""

    stage = LLMInvocationStage(
        _Fails(LLMSpec(provider="openai", model="m")),
        error_policy=ErrorPolicy.SKIP,
        max_retries=1,
        concurrency=1,
    )
    context = _context()

    stage.process([PromptBatch(prompts=["p"], metadata=[_batch(1, 5)])], context)

    assert _shape(context) == [(1, STAGE, "skipped"), (5, STAGE, "skipped")]


def test_an_answer_is_recorded_at_every_row_of_its_batch():
    context = _context()

    report_response(context, _response(), _batch(3, 7), stage=STAGE)

    rows = context.ledger.snapshot().rows
    assert [len(rows[r].attempts) for r in (3, 4, 7)] == [1, 0, 1]
    assert rows[3].cost + rows[7].cost == Decimal("0.0003")
    assert rows[3].attempts[0].tokens + rows[7].attempts[0].tokens == 9
    assert context.row_errors == []


# ── a substituted default ─────────────────────────────────────────────────


def test_a_substituted_default_is_a_provider_failure_that_today_never_listed():
    context = _context(2)

    report_response(
        context,
        _response("", action="substituted", error="provider down"),
        RowMetadata(row_index=1),
        stage=STAGE,
    )

    assert _statuses(context) == [RowStatus.FAILED, RowStatus.FAILED]
    outcome = context.ledger.snapshot().rows[1]
    assert outcome.reason.startswith(f"{STAGE}: provider down")
    assert "default" in outcome.reason
    assert outcome.attempts == ()
    # Today's views never named a substituted row.
    assert context.row_errors == []
    assert (context.skipped_rows, context.failed_rows) == (0, 0)


def test_the_stage_marks_its_use_default_answer_as_substituted():
    stage = LLMInvocationStage(
        _Fails(LLMSpec(provider="openai", model="m")),
        error_policy=ErrorPolicy.USE_DEFAULT,
        max_retries=1,
        concurrency=1,
    )
    context = _context(1)

    [batch] = stage.process(
        [PromptBatch(prompts=["p"], metadata=[RowMetadata(row_index=0)])], context
    )

    [response] = batch.responses
    assert response.text == ""  # today's delivered value
    assert not skipped_by_error_policy(response)
    assert _statuses(context) == [RowStatus.FAILED]
    assert context.row_errors == []


# ── the splitter does not report a skip twice ─────────────────────────────


@pytest.mark.parametrize(
    ("metadata", "listed"),
    [
        ({"action": "skipped", "error": "down"}, []),
        # A substituted default reached the splitter as an empty batch, which
        # it has always listed as unparseable: today's list is kept.
        ({"action": "substituted", "error": "down"}, [0, 1]),
    ],
    ids=["skipped", "substituted"],
)
def test_the_splitter_lists_only_what_the_stage_did_not(metadata, listed):
    context = _context(2)
    response = _response(SKIPPED_OUTPUT_MARKER, **metadata)
    report_response(context, response, _batch(0, 1), stage=STAGE)

    BatchDisaggregatorStage().process(
        [ResponseBatch(responses=[response], metadata=[_batch(0, 1)])], context
    )

    skipped_rows = [0, 1] if metadata["action"] == "skipped" else []
    assert [e.row_index for e in context.row_errors] == skipped_rows + listed
    assert _statuses(context) == [RowStatus.FAILED, RowStatus.FAILED]
