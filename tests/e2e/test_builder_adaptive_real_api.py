"""Real-API smoke test for ``PipelineBuilder.with_adaptive_concurrency``.

Gated by ``OPENAI_API_KEY`` — skipped in CI unless the secret is
configured, so local and offline runs stay clean. Purpose: prove
that toggling the builder method actually runs the pipeline
successfully against a real provider. It does NOT try to induce
429s — the existing ``test_a2_real_http_server.py`` covers 429
behaviour under a local fake server.

What this catches that the unit tests don't:
* Wiring regressions that only surface when LiteLLM is actually
  invoked (e.g. if ``adaptive_concurrency=True`` causes the real
  async path to dead-lock, the unit test's monkey-patched init
  would not notice).
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-API smoke test",
)


def test_adaptive_concurrency_real_openai_smoke():
    """Run a two-row pipeline against real OpenAI with adaptive on.

    Passes if the pipeline completes successfully. The assertion is
    intentionally weak — the provider won't 429 us on two calls, so
    the only invariant we can check is that enabling adaptive does
    not break the happy path.
    """
    from ondine.api.pipeline_builder import PipelineBuilder

    df = pd.DataFrame({"text": ["apple", "banana"]})

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["category"])
        .with_prompt("Reply with exactly one word: the category of '{text}'.")
        .with_llm(model="openai/gpt-4o-mini")
        .with_concurrency(2)
        .with_adaptive_concurrency()
        .with_progress_mode("none")
        .build()
    )

    result = pipeline.execute()

    assert result.success, f"pipeline failed: {result}"
    assert len(result.data) == 2
