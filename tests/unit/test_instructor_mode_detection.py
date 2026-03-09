"""
Unit tests for Instructor mode detection and structured-output safety.

Regression these tests catch:
- Models that support response_schema (Azure, OpenAI) were assigned Mode.TOOLS,
  which triggers Instructor's parse_tools() assertion when the model returns
  parallel tool calls. JSON_SCHEMA mode avoids tool calling entirely.
- Models where provider registry says tools=False (Groq) were still getting
  Mode.TOOLS because LiteLLM's model info lookup fires first and reports
  supports_function_calling=True.
- When TOOLS mode is unavoidable, parallel_tool_calls must be disabled to
  prevent Instructor's single-tool-call assertion from firing.
"""

import instructor
import pytest
from pydantic import BaseModel, Field

from ondine.adapters.instructor_mode import detect_instructor_mode
from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient
from ondine.core.specifications import LLMSpec


class TestModeDetectionPrefersJsonSchema:
    """Models supporting response_schema should get JSON_SCHEMA, not TOOLS."""

    @pytest.mark.parametrize(
        "model",
        [
            "azure/gpt-5-nano",
            "azure/gpt-4o-mini",
            "openai/gpt-4o-mini",
            "gpt-4o-mini",
        ],
    )
    def test_models_with_response_schema_get_json_schema_mode(self, model):
        mode = detect_instructor_mode(model=model)
        assert mode == instructor.Mode.JSON_SCHEMA, (
            f"{model} supports response_schema but got {mode}; "
            f"should be JSON_SCHEMA to avoid parallel tool call failures"
        )

    def test_router_azure_model_gets_json_schema_mode(self):
        mode = detect_instructor_mode(
            model="swap-reviewer",
            router_model_list=[
                {
                    "model_name": "swap-reviewer",
                    "litellm_params": {
                        "model": "azure/gpt-5-nano",
                        "temperature": 1.0,
                    },
                }
            ],
        )
        assert mode == instructor.Mode.JSON_SCHEMA, (
            "Router with azure/gpt-5-nano supports response_schema "
            f"but got {mode}; should be JSON_SCHEMA"
        )


class TestModeDetectionAnthropicUsesOwnModes:
    """Anthropic rejects JSON_SCHEMA — must use ANTHROPIC_TOOLS or ANTHROPIC_JSON."""

    def test_anthropic_gets_provider_specific_mode(self):
        mode = detect_instructor_mode(model="anthropic/claude-3-haiku-20240307")
        assert mode not in (instructor.Mode.JSON_SCHEMA, instructor.Mode.TOOLS), (
            f"Anthropic got {mode}; Instructor rejects JSON_SCHEMA and generic TOOLS "
            f"for Anthropic — must use ANTHROPIC_TOOLS or ANTHROPIC_JSON"
        )


class TestModeDetectionRespectsProviderRegistry:
    """Provider registry overrides should not be bypassed by LiteLLM model info."""

    def test_groq_gets_json_mode_not_tools(self):
        mode = detect_instructor_mode(model="groq/llama-3.3-70b-versatile")
        assert mode != instructor.Mode.TOOLS, (
            f"Groq got {mode} but provider registry says tools=False; "
            f"should get JSON mode to avoid XML generation issues"
        )


class TestModeDetectionUserOverride:
    """User overrides must always win."""

    def test_user_override_json_schema_honored(self):
        mode = detect_instructor_mode(
            model="azure/gpt-5-nano",
            user_override="json_schema",
        )
        assert mode == instructor.Mode.JSON_SCHEMA

    def test_user_override_json_honored(self):
        mode = detect_instructor_mode(
            model="azure/gpt-5-nano",
            user_override="json",
        )
        assert mode == instructor.Mode.JSON

    def test_user_override_tools_honored(self):
        mode = detect_instructor_mode(
            model="azure/gpt-5-nano",
            user_override="tools",
        )
        assert mode == instructor.Mode.TOOLS


class TestModeDetectionFallbacks:
    """When model info lookup fails, fall back gracefully."""

    def test_unknown_model_gets_json_mode(self):
        mode = detect_instructor_mode(model="unknown-provider/unknown-model")
        assert mode == instructor.Mode.JSON, (
            f"Unknown model got {mode}; should default to JSON (safest)"
        )

    def test_reasoning_model_gets_json_mode(self):
        mode = detect_instructor_mode(
            model="azure/gpt-5-nano",
            extra_params={"reasoning_effort": "low"},
        )
        assert mode != instructor.Mode.TOOLS, (
            f"Reasoning model got {mode}; reasoning models don't support TOOLS"
        )


class _DummySchema(BaseModel):
    score: int = Field(ge=0, le=5)
    reason: str


class TestParallelToolCallsDisabled:
    """When TOOLS mode is used, parallel_tool_calls must be False."""

    def test_tools_mode_passes_parallel_tool_calls_false(self):
        """
        Regression this catches:
        Instructor's parse_tools() asserts len(tool_calls)==1. If the model
        returns parallel tool calls the assertion fires. Setting
        parallel_tool_calls=False in the API request prevents this.
        """
        spec = LLMSpec(provider="openai", model="gpt-4o-mini", api_key="test-key")
        spec.instructor_mode = "tools"
        client = UnifiedLiteLLMClient(spec)

        assert client.instructor_client.mode in (
            instructor.Mode.TOOLS,
            instructor.Mode.TOOLS_STRICT,
        ), f"Expected TOOLS mode but got {client.instructor_client.mode}"

        captured_kwargs = {}

        async def spy_create_with_completion(**kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("spy: stop here")

        client.instructor_client.chat.completions.create_with_completion = (
            spy_create_with_completion
        )

        import asyncio

        with pytest.raises(RuntimeError, match="spy: stop here"):
            asyncio.run(client.structured_invoke_async("test prompt", _DummySchema))

        assert "parallel_tool_calls" in captured_kwargs, (
            "structured_invoke_async did not pass parallel_tool_calls to the API call"
        )
        assert captured_kwargs["parallel_tool_calls"] is False, (
            f"parallel_tool_calls should be False, got {captured_kwargs['parallel_tool_calls']}"
        )
