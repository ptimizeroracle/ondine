"""Integration tests for preprocessing, retry, and quality validation."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from ondine import PipelineBuilder
from ondine.core.models import LLMResponse


def _build_mock_client(side_effect):
    mock_client = Mock()
    mock_client.ainvoke = AsyncMock(side_effect=side_effect)
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.router = None
    mock_client.model = "test-model"
    mock_client.spec = Mock(model="test-model")
    return mock_client


class TestPreprocessingAndRetryE2E:
    """Behavioral integration tests for preprocessing and retry."""

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_preprocessing_cleans_input_before_llm_call(self, mock_get):
        """
        Regression this catches:
        preprocessing must clean noisy input before prompt rendering reaches the LLM.
        """
        prompts = []

        async def mock_invoke(prompt, **kwargs):
            prompts.append(prompt)
            return LLMResponse(
                text="clean",
                tokens_in=10,
                tokens_out=5,
                model="test-model",
                cost=Decimal("0.01"),
                latency_ms=5.0,
            )

        mock_get.return_value = Mock(return_value=_build_mock_client(mock_invoke))

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"text": ["PRODUCT®  ITEM™", "PREMIUM\n\nQUALITY"]}),
                input_columns=["text"],
                output_columns=["cleaned"],
            )
            .with_prompt("Clean: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_processing_batch_size(1)
            .with_concurrency(1)
            .build()
        )
        pipeline.specifications.processing.enable_preprocessing = True
        pipeline.specifications.processing.preprocessing_max_length = 100

        result = pipeline.execute()

        assert result.success
        assert prompts == ["Clean: PRODUCT ITEM", "Clean: PREMIUM QUALITY"]
        assert all("®" not in prompt and "™" not in prompt for prompt in prompts)
        assert all("\n" not in prompt and "  " not in prompt for prompt in prompts)

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_preprocessing_and_auto_retry_restore_quality(self, mock_get):
        """
        Regression this catches:
        preprocessing and auto-retry together should recover rows that initially
        return empty output.
        """
        call_counts = {"NOISY PRODUCT": 0, "CLEAN ITEM": 0}

        async def mock_invoke(prompt, **kwargs):
            text = prompt.split(": ", 1)[1]
            call_counts[text] = call_counts.get(text, 0) + 1

            if text == "NOISY PRODUCT" and call_counts[text] == 1:
                response_text = ""
            else:
                response_text = f"cleaned::{text}"

            return LLMResponse(
                text=response_text,
                tokens_in=10,
                tokens_out=5,
                model="test-model",
                cost=Decimal("0.01"),
                latency_ms=5.0,
            )

        mock_get.return_value = Mock(return_value=_build_mock_client(mock_invoke))

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"text": ["NOISY®  PRODUCT™", "CLEAN ITEM"]}),
                input_columns=["text"],
                output_columns=["cleaned"],
            )
            .with_prompt("Clean: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_processing_batch_size(1)
            .with_concurrency(1)
            .build()
        )
        pipeline.specifications.processing.enable_preprocessing = True
        pipeline.specifications.processing.preprocessing_max_length = 100
        pipeline.specifications.processing.auto_retry_failed = True
        pipeline.specifications.processing.max_retry_attempts = 1

        result = pipeline.execute()
        output = result.to_pandas()
        quality = result.validate_output_quality(["cleaned"])

        assert result.success
        assert output["cleaned"].tolist() == [
            "cleaned::NOISY PRODUCT",
            "cleaned::CLEAN ITEM",
        ]
        assert call_counts["NOISY PRODUCT"] == 2
        assert call_counts["CLEAN ITEM"] == 1
        assert quality.success_rate == 100.0

    @patch("ondine.adapters.provider_registry.ProviderRegistry.get")
    def test_quality_report_generated_from_real_execution(self, mock_get):
        """
        Regression this catches:
        post-execution quality validation must detect persistent empty outputs.
        """

        async def mock_invoke(prompt, **kwargs):
            text = prompt.split(": ", 1)[1]
            response_text = "" if text == "bad" else f"ok::{text}"
            return LLMResponse(
                text=response_text,
                tokens_in=10,
                tokens_out=5,
                model="test-model",
                cost=Decimal("0.01"),
                latency_ms=5.0,
            )

        mock_get.return_value = Mock(return_value=_build_mock_client(mock_invoke))

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                pd.DataFrame({"text": ["good", "bad", "good-2"]}),
                input_columns=["text"],
                output_columns=["cleaned"],
            )
            .with_prompt("Clean: {text}")
            .with_llm(provider="groq", model="test-model", temperature=0.0)
            .with_processing_batch_size(1)
            .with_concurrency(1)
            .build()
        )

        result = pipeline.execute()
        quality = result.validate_output_quality(["cleaned"])

        assert quality.total_rows == 3
        assert quality.empty_outputs == 1
        assert quality.valid_outputs == 2
        assert quality.success_rate == pytest.approx(66.6666666, rel=1e-4)
        assert quality.is_acceptable is False
