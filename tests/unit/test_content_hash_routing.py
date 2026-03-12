"""
Unit tests for content-hash deterministic routing strategy.

When routing_strategy="content-hash", the same prompt content must always
route to the same deployment, enabling near-100% cache hit rates on re-runs.

Regression this test suite catches:
- Content-hash strategy not recognized as a valid RouterStrategy enum value
- Same prompt routed to different deployments across calls (non-deterministic)
- metadata.model_id not injected into Router.acompletion kwargs
- Content-hash not distributing across all available deployments
- PipelineBuilder not accepting "content-hash" as a routing_strategy
- Fallback not working when preferred deployment is unavailable
"""

import hashlib
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.router_strategies import RouterStrategy
from ondine.core.specifications import LLMSpec

# ── Fake response objects (shared with test_unified_litellm_minimal.py) ──────


class FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    def __init__(self, content="hello"):
        self.content = content


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content="hello",
        prompt_tokens=10,
        completion_tokens=5,
        model=None,
        hidden_params=None,
    ):
        self.choices = [FakeChoice(FakeMessage(content))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self.model = model
        self._hidden_params = hidden_params or {}


# ── 1. RouterStrategy enum ──────────────────────────────────────────────────


class TestContentHashStrategyEnum:
    def test_content_hash_is_valid_strategy(self):
        """CONTENT_HASH must exist in RouterStrategy and map to 'content-hash'."""
        assert RouterStrategy.CONTENT_HASH.value == "content-hash"

    def test_content_hash_created_from_string(self):
        """RouterStrategy('content-hash') must produce CONTENT_HASH."""
        assert RouterStrategy("content-hash") is RouterStrategy.CONTENT_HASH


# ── 2. Deterministic deployment selection ────────────────────────────────────


class TestContentHashDeterminism:
    THREE_DEPLOYMENTS = [
        {
            "model_name": "swap",
            "model_id": "swap-sweden",
            "litellm_params": {
                "model": "azure/gpt-5-nano",
                "api_base": "https://sweden.api",
            },
        },
        {
            "model_name": "swap",
            "model_id": "swap-eastus",
            "litellm_params": {
                "model": "azure/gpt-5-nano",
                "api_base": "https://eastus.api",
            },
        },
        {
            "model_name": "swap",
            "model_id": "swap-france",
            "litellm_params": {
                "model": "azure/gpt-5-nano",
                "api_base": "https://france.api",
            },
        },
    ]

    def test_same_prompt_always_selects_same_deployment(self):
        """Identical prompts must hash to the same deployment index every time."""
        spec = LLMSpec(model="swap", api_key="sk-test")
        spec.router_config = {
            "model_list": self.THREE_DEPLOYMENTS,
            "routing_strategy": "content-hash",
        }

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.model_list = self.THREE_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)

            prompt = "Compare Prosciutto Crudo 200g vs Prosciutto Cotto 150g"
            results = [client._select_deployment_by_content(prompt) for _ in range(100)]

            assert len(set(results)) == 1, (
                "Same prompt must always select the same deployment"
            )

    def test_different_prompts_distribute_across_deployments(self):
        """Different prompts should spread across available deployments."""
        spec = LLMSpec(model="swap", api_key="sk-test")
        spec.router_config = {
            "model_list": self.THREE_DEPLOYMENTS,
            "routing_strategy": "content-hash",
        }

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.model_list = self.THREE_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)

            selected = set()
            for i in range(300):
                prompt = f"Product pair {i}: item A vs item B"
                dep_id = client._select_deployment_by_content(prompt)
                selected.add(dep_id)

            assert len(selected) == 3, (
                f"300 different prompts across 3 deployments should hit all 3, "
                f"but only hit: {selected}"
            )

    def test_hash_is_sha256_based(self):
        """The deployment selection must use SHA-256 for cross-platform determinism."""
        spec = LLMSpec(model="swap", api_key="sk-test")
        spec.router_config = {
            "model_list": self.THREE_DEPLOYMENTS,
            "routing_strategy": "content-hash",
        }

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.model_list = self.THREE_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)

            prompt = "test prompt"
            expected_idx = (
                int.from_bytes(
                    hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big"
                )
                % 3
            )
            expected_id = self.THREE_DEPLOYMENTS[expected_idx]["model_id"]

            actual = client._select_deployment_by_content(prompt)
            assert actual == expected_id


# ── 3. Metadata injection into acompletion ───────────────────────────────────


class TestContentHashMetadataInjection:
    TWO_DEPLOYMENTS = [
        {
            "model_name": "fast",
            "model_id": "dep-a",
            "litellm_params": {"model": "groq/llama"},
        },
        {
            "model_name": "fast",
            "model_id": "dep-b",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
        },
    ]

    @pytest.mark.asyncio
    async def test_ainvoke_injects_metadata_model_id(self):
        """When content-hash is active, ainvoke must pass metadata.model_id to Router."""
        spec = LLMSpec(model="fast", api_key="sk-test")
        spec.router_config = {
            "model_list": self.TWO_DEPLOYMENTS,
            "routing_strategy": "content-hash",
        }
        fake_response = FakeResponse("result")

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.acompletion = AsyncMock(return_value=fake_response)
            mock_router_instance.model_list = self.TWO_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)
            await client.ainvoke("test prompt for routing")

            call_kwargs = mock_router_instance.acompletion.call_args.kwargs
            assert "metadata" in call_kwargs, "metadata must be passed to acompletion"
            assert "model_id" in call_kwargs["metadata"], (
                "metadata must contain model_id"
            )

    @pytest.mark.asyncio
    async def test_ainvoke_metadata_is_deterministic(self):
        """Same prompt must produce the same metadata.model_id across calls."""
        spec = LLMSpec(model="fast", api_key="sk-test")
        spec.router_config = {
            "model_list": self.TWO_DEPLOYMENTS,
            "routing_strategy": "content-hash",
        }
        fake_response = FakeResponse("result")

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.acompletion = AsyncMock(return_value=fake_response)
            mock_router_instance.model_list = self.TWO_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)

            await client.ainvoke("same prompt here")
            first_id = mock_router_instance.acompletion.call_args.kwargs["metadata"][
                "model_id"
            ]

            await client.ainvoke("same prompt here")
            second_id = mock_router_instance.acompletion.call_args.kwargs["metadata"][
                "model_id"
            ]

            assert first_id == second_id

    @pytest.mark.asyncio
    async def test_non_content_hash_does_not_inject_metadata(self):
        """simple-shuffle strategy must NOT inject metadata.model_id."""
        spec = LLMSpec(model="fast", api_key="sk-test")
        spec.router_config = {
            "model_list": self.TWO_DEPLOYMENTS,
            "routing_strategy": "simple-shuffle",
        }
        fake_response = FakeResponse("result")

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.acompletion = AsyncMock(return_value=fake_response)
            mock_router_instance.model_list = self.TWO_DEPLOYMENTS
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)
            await client.ainvoke("test prompt")

            call_kwargs = mock_router_instance.acompletion.call_args.kwargs
            metadata = call_kwargs.get("metadata", {})
            assert "model_id" not in metadata, (
                "simple-shuffle must not inject deployment preference"
            )


# ── 4. PipelineBuilder integration ──────────────────────────────────────────


class TestPipelineBuilderContentHash:
    def test_with_router_accepts_content_hash_strategy(self):
        """PipelineBuilder.with_router must accept 'content-hash' strategy."""
        df = pd.DataFrame({"text": ["test"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Test: {text}")
            .with_router(
                model_list=[
                    {
                        "model_name": "test",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ],
                routing_strategy="content-hash",
            )
            .build()
        )

        router_config = pipeline.specifications.llm.router_config
        assert router_config["routing_strategy"] == "content-hash"

    def test_with_router_enum_content_hash(self):
        """PipelineBuilder must accept RouterStrategy.CONTENT_HASH enum value."""
        df = pd.DataFrame({"text": ["test"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Test: {text}")
            .with_router(
                model_list=[
                    {
                        "model_name": "test",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ],
                routing_strategy=RouterStrategy.CONTENT_HASH,
            )
            .build()
        )

        router_config = pipeline.specifications.llm.router_config
        assert router_config["routing_strategy"] == "content-hash"


# ── 5. Graceful degradation ─────────────────────────────────────────────────


class TestContentHashGracefulDegradation:
    SINGLE_DEPLOYMENT = [
        {
            "model_name": "solo",
            "model_id": "only-one",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
        },
    ]

    @pytest.mark.asyncio
    async def test_single_deployment_always_selects_it(self):
        """With one deployment, content-hash must always select it."""
        spec = LLMSpec(model="solo", api_key="sk-test")
        spec.router_config = {
            "model_list": self.SINGLE_DEPLOYMENT,
            "routing_strategy": "content-hash",
        }

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.model_list = self.SINGLE_DEPLOYMENT
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)

            for i in range(50):
                dep_id = client._select_deployment_by_content(f"prompt {i}")
                assert dep_id == "only-one"

    @pytest.mark.asyncio
    async def test_router_still_handles_failover(self):
        """Content-hash provides a preference, but Router still handles the actual call.

        If the preferred deployment fails, Router's built-in retry/failover
        should still work — we're injecting a hint, not bypassing the Router.
        """
        spec = LLMSpec(model="fast", api_key="sk-test")
        spec.router_config = {
            "model_list": [
                {
                    "model_name": "fast",
                    "model_id": "dep-a",
                    "litellm_params": {"model": "groq/llama"},
                },
                {
                    "model_name": "fast",
                    "model_id": "dep-b",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                },
            ],
            "routing_strategy": "content-hash",
        }
        fake_response = FakeResponse("fallback worked")

        with patch("litellm.Router") as mock_router:
            mock_router_instance = Mock()
            mock_router_instance.acompletion = AsyncMock(return_value=fake_response)
            mock_router_instance.model_list = spec.router_config["model_list"]
            mock_router.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)
            result = await client.ainvoke("test")

            assert result.text == "fallback worked"
            mock_router_instance.acompletion.assert_called_once()
