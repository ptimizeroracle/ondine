"""Integration tests for executors with real data flow."""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Set GROQ_API_KEY to run executor integration tests",
)
class TestExecutorIntegration:
    """Integration tests for different executors."""

    def test_sync_executor_full_pipeline(self):
        """Test sync executor with full pipeline execution."""
        df = pd.DataFrame(
            {
                "question": ["What is 1+1?", "What is 2+2?"],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["question"],
                output_columns=["answer"],
            )
            .with_prompt("{question}")
            .with_llm(
                model="groq/openai/gpt-oss-20b",
                temperature=0.0,
            )
            .build()
        )

        result = pipeline.execute()
        df = result.to_pandas()

        assert result.success is True
        assert len(df) == 2
        assert "answer" in df.columns

    @pytest.mark.asyncio
    async def test_async_executor_full_pipeline(self):
        """Test async executor with full pipeline execution."""
        df = pd.DataFrame(
            {
                "question": ["What is async?"],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["question"],
                output_columns=["answer"],
            )
            .with_prompt("Answer briefly: {question}")
            .with_llm(
                model="groq/openai/gpt-oss-20b",
                temperature=0.0,
            )
            .with_async_execution(max_concurrency=2)
            .build()
        )

        result = await pipeline.execute_async()
        df = result.to_pandas()

        assert result.success is True
        assert len(df) == 1

    def test_streaming_executor_full_pipeline(self):
        """Test streaming executor with full pipeline execution."""
        df = pd.DataFrame(
            {
                "number": [str(i) for i in range(10)],
            }
        )

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["number"],
                output_columns=["doubled"],
            )
            .with_prompt("What is {number} * 2?")
            .with_llm(
                model="groq/openai/gpt-oss-20b",
                temperature=0.0,
            )
            .with_streaming(chunk_size=3)
            .build()
        )

        chunks = []
        for chunk_result in pipeline.execute_stream():
            chunks.append(chunk_result)
            assert chunk_result.success is True

        # Should have multiple chunks
        assert len(chunks) > 1
