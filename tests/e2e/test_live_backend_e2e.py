"""E2E: LiveBackend against the REAL DeepSeek API.

Gated on ``DEEPSEEK_API_KEY`` — skipped in CI unless the secret is
configured, so offline runs stay clean (same pattern as
``test_builder_adaptive_real_api.py``).

What this catches that the unit tests don't:
* ``tests/unit/test_live_backend.py`` mocks both ``create_llm_client``
  AND ``LLMInvocationStage`` at the architectural boundary, so it never
  makes an HTTP call. A wiring regression that only surfaces when the
  real LiteLLM completion path runs through the live engine — e.g. an
  instructor-mode override that breaks a real model, a response-shape
  the flattener mishandles, the ``live-*`` cache losing results across
  submit/collect — would pass the unit suite and fail here.

Contract exercised (end to end):
    LiveBackend.submit()  → real engine → ``live-*`` job id
    LiveBackend.poll()    → BatchProgress(is_terminal=True)
    LiveBackend.collect() → 10 non-empty LLMResponse, one per row
    total cost < $0.01
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — skipping real-API E2E test",
)


def test_live_backend_e2e_deepseek_classification():
    """Drive the full submit → poll → collect lifecycle against DeepSeek.

    Regression: if the LiveBackend's degenerate lifecycle breaks under a
    real provider — submit no longer blocks to completion, poll reports
    non-terminal, collect yields fewer responses than rows, or the
    flattened responses are empty strings — this test fails. The unit
    suite cannot catch any of those because it intercepts the engine.
    """
    from ondine.core.models import PromptBatch, RowMetadata
    from ondine.core.specifications import (
        DatasetSpec,
        DataSourceType,
        LLMSpec,
        PipelineSpecifications,
        ProcessingSpec,
        PromptSpec,
    )
    from ondine.orchestration.backends.base import BatchProgress
    from ondine.orchestration.backends.live import LiveBackend
    from ondine.orchestration.execution_context import ExecutionContext

    api_key = os.environ["DEEPSEEK_API_KEY"]

    # deepseek-v4-flash is a reasoning model: it spends tokens on internal
    # reasoning before emitting visible text, so max_tokens must leave
    # room for both. 64 is comfortably above the ~26 tokens observed per
    # one-word classification while keeping cost negligible.
    llm_spec = LLMSpec(
        model="deepseek/deepseek-v4-flash",
        api_key=api_key,  # pragma: allowlist secret
        temperature=0.0,
        max_tokens=64,
    )

    specs = PipelineSpecifications(
        dataset=DatasetSpec(
            source_type=DataSourceType.DATAFRAME,
            input_columns=["text"],
            output_columns=["category"],
        ),
        prompt=PromptSpec(
            template=(
                "Classify the following text into exactly one category, "
                "replying with a single lowercase word: "
                "fruit, vegetable, grain, dairy, meat, or other.\n\n"
                "Text: {text}"
            ),
        ),
        llm=llm_spec,
        processing=ProcessingSpec(
            batch_size=10,
            concurrency=5,
            max_retries=2,
        ),
    )
    context = ExecutionContext()

    backend = LiveBackend(llm_spec=llm_spec, specs=specs, context=context)

    # A 10-row "DataFrame" represented as one PromptBatch — the shape the
    # pipeline's front half (format → aggregate) hands to the middle.
    texts = [
        "apple",
        "carrot",
        "bread",
        "milk",
        "chicken",
        "banana",
        "spinach",
        "rice",
        "cheese",
        "beef",
    ]
    prompts = [specs.prompt.template.format(text=t) for t in texts]
    batch = PromptBatch(
        prompts=prompts,
        metadata=[RowMetadata(row_index=i) for i in range(len(texts))],
        batch_id=0,
    )

    # ── submit ──────────────────────────────────────────────────────
    job_id = backend.submit([batch])
    assert job_id.startswith("live-"), (
        f"submit() must return a 'live-'-prefixed job id, got {job_id!r}"
    )

    # ── poll ────────────────────────────────────────────────────────
    progress = backend.poll(job_id)
    assert isinstance(progress, BatchProgress)
    assert progress.is_terminal is True, (
        "poll() must report terminal for a live job (work is done at submit)"
    )
    assert progress.completed == len(texts), (
        f"poll completed={progress.completed}, expected {len(texts)}"
    )
    assert progress.failed == 0

    # ── collect ─────────────────────────────────────────────────────
    from ondine.core.models import LLMResponse

    collected = list(backend.collect(job_id))
    assert len(collected) == len(texts), (
        f"collect() yielded {len(collected)} responses, expected {len(texts)}"
    )
    for r in collected:
        assert isinstance(r, LLMResponse)
        assert r.text, f"collected response must be non-empty, got {r.text!r}"
        assert r.text.strip(), (
            f"collected response text is only whitespace, got {r.text!r}"
        )

    # ── cost guard ──────────────────────────────────────────────────
    total_cost = sum((r.cost for r in collected), Decimal("0"))
    assert total_cost < Decimal("0.01"), (
        f"total cost {total_cost} must stay under $0.01 for 10 rows"
    )
