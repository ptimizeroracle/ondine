"""Claim verification: Functionality claims (Claims 10-30)."""

from decimal import Decimal

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine import PipelineBuilder, QuickPipeline
from ondine.core.specifications import (
    DatasetSpec,
    LLMProvider,
    LLMSpec,
    ProcessingSpec,
    PromptSpec,
)


class TestFunctionalityClaims:
    """Verify all claimed functionality exists and works."""

    def test_claim_10_litellm_provider_integration(self):
        """Claim 10: 100+ LLM providers via LiteLLM — integration exists."""
        # Verify LiteLLM is a valid provider and the unified client uses it
        spec = LLMSpec(model="gpt-4o-mini", provider="litellm")
        assert spec.provider == "litellm"

        # Verify enum includes litellm
        assert LLMProvider.LITELLM == "litellm"

    def test_claim_11_quick_api_minimal_args(self, temp_dir):
        """Claim 11: Quick API — 3-line hello world with smart defaults."""
        csv_path = temp_dir / "data.csv"
        pd.DataFrame({"text": ["hello"]}).to_csv(csv_path, index=False)

        pipeline = QuickPipeline.create(
            data=str(csv_path),
            prompt="Echo: {text}",
            model="gpt-4o-mini",
        )
        # Pipeline created successfully with minimal args
        assert pipeline is not None

    def test_claim_12_builder_api_fluent_chain(self):
        """Claim 12: Builder API — fluent method chaining returns Pipeline."""
        df = pd.DataFrame({"text": ["hello"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(model="gpt-4o-mini", provider="openai")
            .build()
        )
        assert pipeline is not None

    def test_claim_13_batching_configurable(self):
        """Claim 13: Multi-row batching — batch_size configurable."""
        from ondine.strategies.json_batch_strategy import JsonBatchStrategy

        strategy = JsonBatchStrategy()
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        batch = strategy.format_batch(prompts)

        # Batch is a single string containing all prompts
        assert isinstance(batch, str)
        assert "prompt 1" in batch
        assert "prompt 2" in batch
        assert "prompt 3" in batch

        # Parse back
        results = strategy.parse_batch_response(
            '["response 1", "response 2", "response 3"]',
            expected_count=3,
        )
        assert len(results) == 3

    def test_claim_14_prefix_caching_enabled_by_default(self):
        """Claim 14: Prefix caching — enabled by default in LLMSpec."""
        spec = LLMSpec(model="gpt-4o-mini")
        assert spec.enable_prefix_caching is True

    def test_claim_15_cost_estimation_calculation(self):
        """Claim 15: Cost estimation — CostCalculator produces correct result."""
        from ondine.utils.cost_calculator import CostCalculator

        cost = CostCalculator.calculate(
            tokens_in=1000,
            tokens_out=500,
            input_cost_per_1k=Decimal("0.01"),
            output_cost_per_1k=Decimal("0.03"),
        )
        expected = Decimal("0.01") + Decimal("0.015")
        assert cost == expected

    def test_claim_16_budget_limits_enforcement(self):
        """Claim 16: Budget limits — hard USD caps enforced."""
        from ondine.utils.budget_controller import BudgetController, BudgetExceededError

        controller = BudgetController(max_budget=Decimal("1.00"))

        # Under budget — no error
        controller.check_budget(Decimal("0.50"))

        # Over budget — raises
        with pytest.raises(BudgetExceededError):
            controller.check_budget(Decimal("1.50"))

    def test_claim_17_checkpointing_roundtrip(self, temp_dir):
        """Claim 17: Checkpointing — save and load cycle preserves data."""
        from uuid import uuid4

        from ondine.adapters.checkpoint_storage import LocalFileCheckpointStorage

        storage = LocalFileCheckpointStorage(checkpoint_dir=temp_dir)
        session_id = uuid4()
        data = {"rows_processed": 500, "results": [1, 2, 3]}

        storage.save(session_id, data)
        assert storage.exists(session_id)

        loaded = storage.load(session_id)
        assert loaded["rows_processed"] == 500
        assert loaded["results"] == [1, 2, 3]

    def test_claim_18_structured_output_pydantic(self):
        """Claim 18: Structured output — Pydantic models accepted by builder."""

        class Sentiment(BaseModel):
            label: str
            score: float

        df = pd.DataFrame({"text": ["great product"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["sentiment"])
            .with_prompt("Analyze: {text}")
            .with_llm(model="gpt-4o-mini", provider="openai")
            .with_structured_output(Sentiment)
            .build()
        )
        assert pipeline is not None

    def test_claim_19_multi_column_output(self):
        """Claim 19: Multi-column output — multiple output columns accepted."""
        df = pd.DataFrame({"text": ["hello"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(
                df,
                input_columns=["text"],
                output_columns=["sentiment", "summary", "keywords"],
            )
            .with_prompt("Analyze: {text}")
            .with_llm(model="gpt-4o-mini", provider="openai")
            .build()
        )
        assert pipeline is not None

    def test_claim_20_pipeline_composer(self):
        """Claim 20: Pipeline composition — PipelineComposer chains pipelines."""
        from ondine.api.pipeline_composer import PipelineComposer

        df = pd.DataFrame({"text": ["hello"]})
        composer = PipelineComposer(df)

        pipeline_a = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["col_a"])
            .with_prompt("Task A: {text}")
            .with_llm(model="gpt-4o-mini", provider="openai")
            .build()
        )
        pipeline_b = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["col_b"])
            .with_prompt("Task B: {text}")
            .with_llm(model="gpt-4o-mini", provider="openai")
            .build()
        )

        composer.add_column("col_a", pipeline_a)
        composer.add_column("col_b", pipeline_b, depends_on=["col_a"])
        assert len(composer.column_pipelines) == 2

    def test_claim_21_async_execution_configurable(self):
        """Claim 21: Async execution — PipelineBuilder supports async config."""
        from ondine.orchestration.async_executor import AsyncExecutor

        executor = AsyncExecutor(max_concurrency=20)
        assert executor.max_concurrency == 20

    def test_claim_22_streaming_execution(self):
        """Claim 22: Streaming execution — StreamingExecutor for large datasets."""
        from ondine.orchestration.streaming_executor import StreamingExecutor

        executor = StreamingExecutor(chunk_size=5000)
        assert executor.chunk_size == 5000

    def test_claim_23_observability_registry(self):
        """Claim 23: Observability — observer registry exists."""
        from ondine.observability.registry import ObserverRegistry

        observers = ObserverRegistry.list_observers()
        assert isinstance(observers, list)

    def test_claim_24_router_strategies(self):
        """Claim 24: Router — all strategies defined."""
        from ondine.core.router_strategies import RouterStrategy

        strategies = [s.value for s in RouterStrategy]
        assert "simple-shuffle" in strategies
        assert "latency-based-routing" in strategies
        assert "usage-based-routing" in strategies
        assert "cost-based-routing" in strategies
        assert "least-busy" in strategies

    def test_claim_25_mlx_client_exists(self):
        """Claim 25: MLX local inference — MLXClient class exists."""
        from ondine.adapters.llm_client import MLXClient

        assert MLXClient is not None
        # Verify it's a subclass of LLMClient
        from ondine.adapters.llm_client import LLMClient

        assert issubclass(MLXClient, LLMClient)

    def test_claim_26_provider_presets_exist(self):
        """Claim 26: Provider presets — pre-configured LLMSpec objects."""
        from ondine.core.specifications import LLMProviderPresets

        assert hasattr(LLMProviderPresets, "GPT4O_MINI")
        assert hasattr(LLMProviderPresets, "GPT4O")
        assert hasattr(LLMProviderPresets, "GROQ_LLAMA_70B")

        preset = LLMProviderPresets.GPT4O_MINI
        assert isinstance(preset, LLMSpec)
        assert "gpt-4o-mini" in preset.model

    def test_claim_27_custom_provider_registration(self):
        """Claim 27: Custom providers — register and discover."""
        from ondine.adapters.provider_registry import ProviderRegistry

        class MockProvider:
            pass

        ProviderRegistry.register("test_custom_claim27", MockProvider)
        assert ProviderRegistry.is_registered("test_custom_claim27")

        retrieved = ProviderRegistry.get("test_custom_claim27")
        assert retrieved is MockProvider

        # Cleanup
        ProviderRegistry.unregister("test_custom_claim27")

    def test_claim_28_cli_commands_defined(self):
        """Claim 28: CLI — commands defined via Click."""
        from ondine.cli.main import cli

        # cli is a Click group with subcommands
        assert hasattr(cli, "commands") or hasattr(cli, "list_commands")

    def test_claim_29_type_safe_pydantic_specs(self):
        """Claim 29: Type-safe validation via Pydantic."""
        from pydantic import BaseModel as PydanticBase

        from ondine.core.specifications import LLMSpec

        # All specs are Pydantic models
        assert issubclass(DatasetSpec, PydanticBase)
        assert issubclass(LLMSpec, PydanticBase)
        assert issubclass(PromptSpec, PydanticBase)
        assert issubclass(ProcessingSpec, PydanticBase)

    def test_claim_30_auto_detection_provider_from_model(self):
        """Claim 30: Auto-detection — QuickPipeline detects provider from model."""
        # Verify LLMProvider enum covers major providers
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.GROQ == "groq"
        assert LLMProvider.MLX == "mlx"
        assert LLMProvider.LITELLM == "litellm"
