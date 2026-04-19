"""Integration tests for checkpoint and resume behavior."""

import gzip
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse


def _extract_row_number(prompt: str) -> int:
    for i in range(10):
        if f"Item {i}" in prompt:
            return i
    raise AssertionError(f"Could not extract row from prompt: {prompt}")


@pytest.mark.integration
@patch("ondine.adapters.provider_registry.ProviderRegistry.get")
def test_checkpoint_resume_restores_partial_progress_without_reprocessing(mock_get):
    """
    Regression this catches:
    a resumed run must not redo already completed rows after a checkpointed crash.
    """
    df = pd.DataFrame({"text": [f"Item {i}" for i in range(5)]})
    all_calls: list[int] = []
    crash_once = {"triggered": False}

    async def mock_invoke(prompt, **kwargs):
        row_num = _extract_row_number(prompt)
        all_calls.append(row_num)
        if row_num == 2 and not crash_once["triggered"]:
            crash_once["triggered"] = True
            raise RuntimeError("simulated crash")

        return LLMResponse(
            text=f"done_{row_num}",
            tokens_in=10,
            tokens_out=5,
            model="test-model",
            cost=Decimal("0.10"),
            latency_ms=10.0,
        )

    mock_client = Mock()
    mock_client.ainvoke = AsyncMock(side_effect=mock_invoke)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.router = None
    mock_client.model = "test-model"
    mock_client.spec = Mock(
        model="test-model",
        input_cost_per_1k_tokens=Decimal("0.10"),
        output_cost_per_1k_tokens=Decimal("0.10"),
    )
    mock_client_class = Mock(return_value=mock_client)
    mock_get.return_value = mock_client_class

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_concurrency(1)
            .with_error_policy("fail")
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        with pytest.raises(RuntimeError, match="simulated crash"):
            pipeline.execute()

        checkpoint_files = list(checkpoint_dir.glob("*.json*"))
        assert len(checkpoint_files) == 1, "Checkpoint should be written after crash"

        cp = checkpoint_files[0]
        if cp.suffix == ".gz":
            with gzip.open(cp, "rb") as f:
                checkpoint_payload = json.loads(f.read().decode("utf-8"))
        else:
            checkpoint_payload = json.loads(cp.read_text())
        checkpoint_data = checkpoint_payload["data"]
        assert checkpoint_data["last_processed_row"] == 1

        # Completed LLM responses live in the durable SQLite response
        # cache (``responses.db``) next to the checkpoint, not inline in
        # the JSON blob. This split is the whole point of A4: the
        # checkpoint stays small, responses stream row-by-row into a
        # crash-atomic backing store.
        from uuid import UUID as _UUID

        from ondine.adapters.response_cache import SqliteResponseCache

        cache = SqliteResponseCache(checkpoint_dir / "responses.db")
        try:
            rows = list(cache.iter_completed(_UUID(checkpoint_data["session_id"])))
            assert len(rows) == 2
            assert [r[0] for r in rows] == [0, 1]
        finally:
            cache.close()

        resumed = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_concurrency(1)
            .with_error_policy("fail")
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        stem = checkpoint_files[0].name.split("checkpoint_", 1)[1].split(".")[0]
        result = resumed.execute(resume_from=UUID(stem))

        output = result.to_pandas()
        assert result.success
        assert output["result"].tolist() == [
            "done_0",
            "done_1",
            "done_2",
            "done_3",
            "done_4",
        ]

        assert all_calls == [0, 1, 2, 2, 3, 4]
        assert all_calls.count(0) == 1
        assert all_calls.count(1) == 1
        assert all_calls.count(2) == 2


@pytest.mark.integration
@patch("ondine.adapters.provider_registry.ProviderRegistry.get")
def test_checkpoint_contains_completed_response_records(mock_get):
    """
    Regression this catches:
    checkpoints must persist enough completed response data to rebuild final output.
    """
    df = pd.DataFrame({"text": [f"Item {i}" for i in range(3)]})

    async def mock_invoke(prompt, **kwargs):
        row_num = _extract_row_number(prompt)
        if row_num == 1:
            raise RuntimeError("checkpoint me")

        return LLMResponse(
            text=f"value_{row_num}",
            tokens_in=8,
            tokens_out=4,
            model="test-model",
            cost=Decimal("0.05"),
            latency_ms=5.0,
        )

    mock_client = Mock()
    mock_client.ainvoke = AsyncMock(side_effect=mock_invoke)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.router = None
    mock_client.model = "test-model"
    mock_client.spec = Mock(model="test-model")
    mock_get.return_value = Mock(return_value=mock_client)

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_concurrency(1)
            .with_error_policy("fail")
            .with_checkpoint_dir(str(checkpoint_dir))
            .build()
        )

        with pytest.raises(RuntimeError, match="checkpoint me"):
            pipeline.execute()

        checkpoint_file = next(checkpoint_dir.glob("*.json*"))
        if checkpoint_file.suffix == ".gz":
            with gzip.open(checkpoint_file, "rb") as f:
                checkpoint_data = json.loads(f.read().decode("utf-8"))["data"]
        else:
            checkpoint_data = json.loads(checkpoint_file.read_text())["data"]

        # Verify completed-response persistence via the durable cache
        # rather than the JSON blob. The checkpoint JSON only carries
        # counters (last_processed_row, accumulated_cost, ...) now.
        from uuid import UUID as _UUID

        from ondine.adapters.response_cache import SqliteResponseCache

        cache = SqliteResponseCache(checkpoint_dir / "responses.db")
        try:
            rows = list(cache.iter_completed(_UUID(checkpoint_data["session_id"])))
            assert len(rows) == 1
            row_index, response, _ = rows[0]
            assert row_index == 0
            assert response.text == "value_0"
        finally:
            cache.close()
