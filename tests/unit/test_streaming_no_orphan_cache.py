"""Streaming must not write a response cache nothing can read (issue #150).

The durable cache is keyed ``(session_id, row_index)`` and is the source of
truth for resume. Streaming builds a fresh ``Pipeline`` per chunk, and each new
``ExecutionContext`` takes a new ``uuid4`` session — so every row a chunk wrote
was unreadable by the next run, which generated new ids again.

Measured before this fix, on a 6-row stream over 3 chunks, run twice: 6 LLM
calls each time, and ``responses.db`` grew to 12 rows across 6 dead sessions.
The writes were not merely wasted — up to ``max_pending_chunks`` chunks wrote
that one SQLite file concurrently, which is the contention #147 papered over
with ``busy_timeout``.

These tests pin both halves: streaming writes nothing, and the non-streaming
resume path still works. Only asserting the first would be satisfied by
breaking resume altogether.
"""

import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
import pytest

from ondine.adapters.llm_client import LLMClient
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class _Client(LLMClient):
    """Answers every prompt, and fails after *fail_after* of them."""

    def __init__(self, spec: LLMSpec, fail_after: int | None = None):
        super().__init__(spec)
        self.prompts: list[str] = []
        self.fail_after = fail_after

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        if self.fail_after is not None and len(self.prompts) > self.fail_after:
            raise RuntimeError("simulated provider failure")
        return LLMResponse(
            text="answer",
            tokens_in=1,
            tokens_out=1,
            model="stub",
            cost=Decimal("0"),
            latency_ms=1.0,
            metadata={},
        )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.invoke(prompt, **kwargs)

    def structured_invoke(self, prompt, output_cls, **kwargs):
        return self.invoke(prompt, **kwargs)

    async def structured_invoke_async(self, prompt, output_cls, **kwargs):
        return self.invoke(prompt, **kwargs)

    def estimate_tokens(self, text: str) -> int:
        return 1

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def _pipeline(client, checkpoint_dir, rows=6):
    return (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"t": [f"r{i}" for i in range(rows)]}),
            input_columns=["t"],
            output_columns=["o"],
        )
        .with_prompt("Echo: {t}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_checkpoint_dir(str(checkpoint_dir))
        .with_checkpoint_cleanup(False)
        .build()
    )


def _cache_rows(checkpoint_dir) -> tuple[int, int]:
    """(rows, distinct sessions) in responses.db, or (0, 0) if absent."""
    db = Path(checkpoint_dir) / "responses.db"
    if not db.exists():
        return (0, 0)
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM responses"
    ).fetchone()[0]
    conn.close()
    return (rows, sessions)


@pytest.mark.asyncio
async def test_streaming_writes_no_response_cache(tmp_path):
    """A streamed run must leave no cache behind — nothing could read it."""
    client = _Client(LLMSpec(provider="openai", model="stub"))
    ckpt = tmp_path / "ckpt"

    chunks = [
        c async for c in _pipeline(client, ckpt).execute_stream_async(chunk_size=2)
    ]

    assert len(chunks) == 3
    assert len(client.prompts) == 6
    assert _cache_rows(ckpt) == (0, 0), (
        "streaming wrote cache rows keyed to per-chunk sessions no run can read"
    )


def test_non_streaming_still_writes_its_cache(tmp_path):
    """The durable cache must survive for the path that actually resumes."""
    client = _Client(LLMSpec(provider="openai", model="stub"))
    ckpt = tmp_path / "ckpt"

    _pipeline(client, ckpt).execute()

    rows, sessions = _cache_rows(ckpt)
    assert rows == 6
    assert sessions == 1, "one run must write exactly one session"


def test_resume_reuses_cached_rows(tmp_path):
    """Resume must still skip rows already answered.

    Asserting only that streaming writes nothing would be satisfied by
    disabling the cache everywhere, which would silently make every resume
    re-pay for work already done.
    """
    ckpt = tmp_path / "ckpt"
    client = _Client(LLMSpec(provider="openai", model="stub"), fail_after=3)

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"t": [f"r{i}" for i in range(6)]}),
            input_columns=["t"],
            output_columns=["o"],
        )
        .with_prompt("Echo: {t}")
        .with_custom_llm_client(client)
        .with_batch_size(1)
        .with_error_policy("fail")
        .with_checkpoint_dir(str(ckpt))
        .with_checkpoint_cleanup(False)
        .build()
    )
    with pytest.raises(Exception):
        pipeline.execute()

    conn = sqlite3.connect(Path(ckpt) / "responses.db")
    session = conn.execute("SELECT DISTINCT session_id FROM responses").fetchone()[0]
    cached = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    conn.close()
    assert cached == 3

    client.fail_after = None
    before = len(client.prompts)
    _pipeline(client, ckpt).execute(resume_from=UUID(session))

    assert len(client.prompts) - before == 6 - cached, (
        "resume re-called rows that were already cached"
    )
