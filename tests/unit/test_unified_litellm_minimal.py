"""
Unit tests for the minimal UnifiedLiteLLMClient wrapper.

Tests the core wrapper logic without making real API calls.
All LiteLLM calls are mocked to test behavior in isolation.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class FakeUsage:
    """Fake LiteLLM usage object."""

    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    """Fake LiteLLM message object."""

    def __init__(self, content="hello"):
        self.content = content


class FakeChoice:
    """Fake LiteLLM choice object."""

    def __init__(self, message):
        self.message = message


class FakeResponse:
    """Fake LiteLLM response object."""

    def __init__(self, content="hello", prompt_tokens=10, completion_tokens=5):
        self.choices = [FakeChoice(FakeMessage(content))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class TestUnifiedLiteLLMClient:
    """Test the minimal LiteLLM wrapper."""

    @pytest.mark.asyncio
    async def test_ainvoke_direct_call(self):
        """Test ainvoke makes correct call to litellm.acompletion."""
        # Setup
        spec = LLMSpec(
            provider="litellm",  # Generic provider - model string determines actual provider
            model="openai/gpt-4o-mini",
            api_key="sk-test",
            temperature=0.7,
            max_tokens=100,
        )

        fake_response = FakeResponse("test response", 15, 8)

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_completion:
            mock_completion.return_value = fake_response

            client = UnifiedLiteLLMClient(spec)
            result = await client.ainvoke("test prompt")

            # Verify call was made correctly
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args.kwargs

            assert call_kwargs["model"] == "openai/gpt-4o-mini"
            assert call_kwargs["api_key"] == "sk-test"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["messages"] == [
                {"role": "user", "content": "test prompt"}
            ]

            # Verify response
            assert result.text == "test response"
            assert result.tokens_in == 15
            assert result.tokens_out == 8
            assert result.model == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_ainvoke_with_system_message(self):
        """Test ainvoke includes system message in messages array."""
        spec = LLMSpec(model="groq/llama-3.3-70b-versatile", api_key="gsk-test")
        fake_response = FakeResponse()

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_completion:
            mock_completion.return_value = fake_response

            client = UnifiedLiteLLMClient(spec)
            await client.ainvoke("user prompt", system_message="system instructions")

            call_kwargs = mock_completion.call_args.kwargs
            assert call_kwargs["messages"] == [
                {"role": "system", "content": "system instructions"},
                {"role": "user", "content": "user prompt"},
            ]

    @pytest.mark.asyncio
    async def test_ainvoke_with_extra_params(self):
        """Test extra_params are passed through to LiteLLM."""
        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")
        # Set extra_params after creation (Pydantic model)
        spec.extra_params = {
            "top_p": 0.9,
            "stream": True,
            "caching": True,
            "max_retries": 3,
        }
        fake_response = FakeResponse()

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_completion:
            mock_completion.return_value = fake_response

            client = UnifiedLiteLLMClient(spec)
            await client.ainvoke("test")

            call_kwargs = mock_completion.call_args.kwargs
            # Verify ALL extra params were passed through
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["stream"] is True
            assert call_kwargs["caching"] is True
            assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_ainvoke_with_router(self):
        """Test ainvoke uses Router when configured."""
        router_config = {
            "model_list": [
                {"model_name": "fast-model", "litellm_params": {"model": "groq/llama"}}
            ]
        }
        spec = LLMSpec(
            model="fast-model",  # Will be overridden by router
            api_key="sk-test",
        )
        spec.router_config = router_config
        fake_response = FakeResponse()

        # Patch Router from litellm, not from unified_litellm_client
        with patch("litellm.Router") as MockRouter:
            mock_router_instance = Mock()
            mock_router_instance.acompletion = AsyncMock(return_value=fake_response)
            MockRouter.return_value = mock_router_instance

            client = UnifiedLiteLLMClient(spec)
            await client.ainvoke("test")

            # Verify Router was used, not direct litellm
            mock_router_instance.acompletion.assert_called_once()
            assert client.router is not None

    def test_invoke_script_mode(self):
        """Test invoke() works in script mode (no running loop)."""
        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")
        fake_response = FakeResponse("script response")

        # Mock get_running_loop to raise RuntimeError (no loop)
        with patch(
            "ondine.adapters.unified_litellm_client.asyncio.get_running_loop",
            side_effect=RuntimeError("no running loop"),
        ):
            with patch(
                "ondine.adapters.unified_litellm_client.asyncio.run"
            ) as mock_run:
                mock_run.return_value = LLMResponse(
                    text="script response",
                    tokens_in=10,
                    tokens_out=5,
                    model="openai/gpt-4o-mini",
                    cost=Decimal("0.01"),
                    latency_ms=100,
                )

                client = UnifiedLiteLLMClient(spec)
                result = client.invoke("test")

                # Verify asyncio.run was called (script mode)
                mock_run.assert_called_once()
                assert result.text == "script response"

    def test_invoke_jupyter_mode(self):
        """Test invoke() works in Jupyter mode (running loop exists)."""
        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")

        # Mock running loop (Jupyter scenario)
        mock_loop = Mock()
        expected_response = LLMResponse(
            text="jupyter response",
            tokens_in=10,
            tokens_out=5,
            model="openai/gpt-4o-mini",
            cost=Decimal("0.01"),
            latency_ms=100,
        )

        with patch(
            "ondine.adapters.unified_litellm_client.asyncio.get_running_loop",
            return_value=mock_loop,
        ):
            with patch(
                "ondine.adapters.unified_litellm_client.asyncio.run_coroutine_threadsafe"
            ) as mock_threadsafe:
                # Mock the future result
                mock_future = Mock()
                mock_future.result.return_value = expected_response
                mock_threadsafe.return_value = mock_future

                client = UnifiedLiteLLMClient(spec)
                result = client.invoke("test")

                # Verify run_coroutine_threadsafe was used (Jupyter mode)
                mock_threadsafe.assert_called_once()
                assert mock_threadsafe.call_args.args[1] == mock_loop  # Passed the loop
                assert result.text == "jupyter response"

    @pytest.mark.asyncio
    async def test_structured_invoke_async(self):
        """Test structured output via Instructor."""

        class TestModel(BaseModel):
            result: str

        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")

        # Mock Instructor client
        mock_instructor_client = Mock()
        mock_completions = Mock()
        mock_result = TestModel(result="structured output")
        mock_completions.create = AsyncMock(return_value=mock_result)
        mock_instructor_client.chat = Mock()
        mock_instructor_client.chat.completions = mock_completions

        with patch(
            "ondine.adapters.unified_litellm_client.instructor.from_litellm",
            return_value=mock_instructor_client,
        ):
            client = UnifiedLiteLLMClient(spec)
            result = await client.structured_invoke_async("test", TestModel)

            # Verify Instructor was called correctly
            mock_completions.create.assert_called_once()
            call_kwargs = mock_completions.create.call_args.kwargs
            assert call_kwargs["model"] == "openai/gpt-4o-mini"
            assert call_kwargs["response_model"] == TestModel
            assert call_kwargs["api_key"] == "sk-test"

            # Verify response is serialized JSON
            assert '"result"' in result.text
            assert "structured output" in result.text

    def test_calc_cost_with_litellm(self):
        """Test cost calculation uses LiteLLM's pricing DB."""
        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.completion_cost",
            return_value=0.1234,
        ):
            client = UnifiedLiteLLMClient(spec)
            cost = client._calc_cost(100, 50)

            assert cost == Decimal("0.1234")

    def test_calc_cost_fallback(self):
        """Test cost calculation falls back to spec values when LiteLLM fails."""
        spec = LLMSpec(
            model="custom/model",
            api_key="sk-test",
            input_cost_per_1k_tokens=Decimal("0.01"),
            output_cost_per_1k_tokens=Decimal("0.02"),
        )

        # Mock LiteLLM to raise exception
        with patch(
            "ondine.adapters.unified_litellm_client.litellm.completion_cost",
            side_effect=Exception("Model not in DB"),
        ):
            client = UnifiedLiteLLMClient(spec)
            cost = client._calc_cost(1000, 500)

            # Manual calculation: (1000/1000 * 0.01) + (500/1000 * 0.02) = 0.02
            expected = Decimal("0.01") + Decimal("0.01")
            assert cost == expected

    def test_estimate_tokens(self):
        """Test token estimation."""
        spec = LLMSpec(model="openai/gpt-4o-mini", api_key="sk-test")

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.encode",
            return_value=[1, 2, 3, 4, 5],
        ):  # 5 tokens
            client = UnifiedLiteLLMClient(spec)
            tokens = client.estimate_tokens("test text")

            assert tokens == 5

    def test_estimate_tokens_fallback(self):
        """Test token estimation falls back to word count when encode fails."""
        spec = LLMSpec(model="custom/model", api_key="sk-test")

        with patch(
            "ondine.adapters.unified_litellm_client.litellm.encode",
            side_effect=Exception("Encoding failed"),
        ):
            client = UnifiedLiteLLMClient(spec)
            tokens = client.estimate_tokens("one two three four five")  # 5 words

            # Fallback: words * 1.3 = 5 * 1.3 = 6.5 → 6
            assert tokens == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
