"""Unit tests for LLM provider presets and with_llm_spec() method."""

from decimal import Decimal

import pandas as pd
import pytest

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.specifications import LLMProvider, LLMProviderPresets, LLMSpec


class TestLLMProviderPresets:
    """Test LLM provider presets configuration."""

    def test_gpt4o_mini_preset_exists(self):
        """Test GPT-4o-mini preset is configured correctly."""
        spec = LLMProviderPresets.GPT4O_MINI

        assert spec.provider == LLMProvider.OPENAI
        assert spec.model == "gpt-4o-mini"
        assert spec.input_cost_per_1k_tokens == Decimal("0.00015")
        assert spec.output_cost_per_1k_tokens == Decimal("0.0006")
        assert spec.temperature == 0.0

    def test_gpt4o_preset_exists(self):
        """Test GPT-4o preset is configured correctly."""
        spec = LLMProviderPresets.GPT4O

        assert spec.provider == LLMProvider.OPENAI
        assert spec.model == "gpt-4o"
        assert spec.input_cost_per_1k_tokens == Decimal("0.0025")
        assert spec.output_cost_per_1k_tokens == Decimal("0.01")

    def test_together_ai_llama_70b_preset_exists(self):
        """Test Together.AI Llama 70B preset is configured correctly."""
        spec = LLMProviderPresets.TOGETHER_AI_LLAMA_70B

        assert spec.provider == LLMProvider.OPENAI_COMPATIBLE
        assert spec.provider_name == "Together.AI"
        assert spec.model == "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        assert spec.base_url == "https://api.together.xyz/v1"
        assert spec.input_cost_per_1k_tokens == Decimal("0.0006")
        assert spec.output_cost_per_1k_tokens == Decimal("0.0006")

    def test_together_ai_llama_8b_preset_exists(self):
        """Test Together.AI Llama 8B preset is configured correctly."""
        spec = LLMProviderPresets.TOGETHER_AI_LLAMA_8B

        assert spec.provider == LLMProvider.OPENAI_COMPATIBLE
        assert spec.provider_name == "Together.AI"
        assert spec.model == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        assert spec.input_cost_per_1k_tokens == Decimal("0.0001")

    def test_ollama_llama_70b_preset_free_pricing(self):
        """Test Ollama Llama 70B preset has zero cost."""
        spec = LLMProviderPresets.OLLAMA_LLAMA_70B

        assert spec.provider == LLMProvider.OPENAI_COMPATIBLE
        assert spec.provider_name == "Ollama-Local"
        assert spec.model == "llama3.1:70b"
        assert spec.base_url == "http://localhost:11434/v1"
        assert spec.input_cost_per_1k_tokens == Decimal("0.0")
        assert spec.output_cost_per_1k_tokens == Decimal("0.0")

    def test_ollama_llama_8b_preset_free_pricing(self):
        """Test Ollama Llama 8B preset has zero cost."""
        spec = LLMProviderPresets.OLLAMA_LLAMA_8B

        assert spec.input_cost_per_1k_tokens == Decimal("0.0")
        assert spec.output_cost_per_1k_tokens == Decimal("0.0")
        assert spec.model == "llama3.1:8b"

    def test_groq_preset_exists(self):
        """Test Groq preset is configured correctly."""
        spec = LLMProviderPresets.GROQ_LLAMA_70B

        assert spec.provider == LLMProvider.GROQ
        assert spec.model == "llama-3.1-70b-versatile"
        assert spec.input_cost_per_1k_tokens == Decimal("0.00059")
        assert spec.output_cost_per_1k_tokens == Decimal("0.00079")

    def test_claude_sonnet_4_preset_exists(self):
        """Test Claude Sonnet 4 preset is configured correctly."""
        spec = LLMProviderPresets.CLAUDE_SONNET_4

        assert spec.provider == LLMProvider.ANTHROPIC
        assert spec.model == "claude-sonnet-4-20250514"
        assert spec.max_tokens == 8192
        assert spec.input_cost_per_1k_tokens == Decimal("0.003")
        assert spec.output_cost_per_1k_tokens == Decimal("0.015")

    def test_presets_have_no_api_keys(self):
        """Test all presets have None for api_key (security requirement)."""
        presets = [
            LLMProviderPresets.GPT4O_MINI,
            LLMProviderPresets.GPT4O,
            LLMProviderPresets.TOGETHER_AI_LLAMA_70B,
            LLMProviderPresets.TOGETHER_AI_LLAMA_8B,
            LLMProviderPresets.OLLAMA_LLAMA_70B,
            LLMProviderPresets.OLLAMA_LLAMA_8B,
            LLMProviderPresets.GROQ_LLAMA_70B,
            LLMProviderPresets.CLAUDE_SONNET_4,
        ]

        for preset in presets:
            assert preset.api_key is None, (
                f"{preset.model} has hardcoded API key - security risk!"
            )

    def test_presets_are_valid_llmspec_instances(self):
        """Test all presets are valid LLMSpec instances."""
        preset_attrs = [
            attr
            for attr in dir(LLMProviderPresets)
            if not attr.startswith("_") and attr.isupper()
        ]

        for preset_name in preset_attrs:
            preset = getattr(LLMProviderPresets, preset_name)
            assert isinstance(preset, LLMSpec), (
                f"{preset_name} is not an LLMSpec instance"
            )

            # Validate using Pydantic
            preset.model_validate(preset.model_dump())

    def test_create_custom_openai_compatible(self):
        """Test factory method for custom OpenAI-compatible providers."""
        spec = LLMProviderPresets.create_custom_openai_compatible(
            provider_name="MyProvider",
            model="custom-model",
            base_url="http://localhost:8080/v1",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            temperature=0.5,
        )

        assert spec.provider == LLMProvider.OPENAI_COMPATIBLE
        assert spec.provider_name == "MyProvider"
        assert spec.model == "custom-model"
        assert spec.base_url == "http://localhost:8080/v1"
        assert spec.input_cost_per_1k_tokens == Decimal("0.001")
        assert spec.output_cost_per_1k_tokens == Decimal("0.002")
        assert spec.temperature == 0.5

    def test_create_custom_openai_compatible_defaults(self):
        """Test factory method with default cost values."""
        spec = LLMProviderPresets.create_custom_openai_compatible(
            provider_name="LocalServer",
            model="test-model",
            base_url="http://localhost:5000/v1",
        )

        assert spec.input_cost_per_1k_tokens == Decimal("0.0")
        assert spec.output_cost_per_1k_tokens == Decimal("0.0")

    def test_preset_override_with_model_copy(self):
        """Test overriding preset fields using Pydantic model_copy."""
        original = LLMProviderPresets.TOGETHER_AI_LLAMA_70B
        modified = original.model_copy(
            update={"api_key": "test-key-123", "temperature": 0.9, "max_tokens": 500}
        )

        # Modified has new values
        assert modified.api_key == "test-key-123"
        assert modified.temperature == 0.9
        assert modified.max_tokens == 500

        # Modified retains original values for unchanged fields
        assert modified.model == original.model
        assert modified.base_url == original.base_url
        assert modified.provider_name == original.provider_name

        # Original is unchanged (immutability)
        assert original.api_key is None
        assert original.temperature == 0.0
        assert original.max_tokens is None

    def test_preset_override_preserves_type_safety(self):
        """Test that overriding presets maintains Pydantic validation."""
        from pydantic import ValidationError

        original = LLMProviderPresets.GPT4O_MINI

        # Valid override should work
        valid = original.model_copy(update={"temperature": 0.5})
        assert valid.temperature == 0.5

        # Invalid override should fail validation when creating new spec (temperature > 2.0)
        with pytest.raises(ValidationError):
            LLMSpec(
                provider=LLMProvider.OPENAI,
                model="test",
                temperature=3.0,  # Invalid: max is 2.0
            )


class TestPipelineBuilderWithLLMSpec:
    """Test PipelineBuilder.with_llm_spec() method."""

    def test_with_llm_spec_accepts_preset(self):
        """Test with_llm_spec accepts preset LLMSpec."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm_spec(LLMProviderPresets.GPT4O_MINI)
        )

        assert builder._llm_spec is not None
        assert builder._llm_spec.model == "gpt-4o-mini"
        assert builder._llm_spec.provider == LLMProvider.OPENAI

    def test_with_llm_spec_accepts_custom_spec(self):
        """Test with_llm_spec accepts custom LLMSpec."""
        df = pd.DataFrame({"text": ["test"]})

        custom_spec = LLMSpec(
            provider=LLMProvider.OPENAI, model="custom-model", temperature=0.8
        )

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm_spec(custom_spec)
        )

        assert builder._llm_spec == custom_spec
        assert builder._llm_spec.temperature == 0.8

    def test_with_llm_spec_accepts_modified_preset(self):
        """Test with_llm_spec accepts modified preset."""
        df = pd.DataFrame({"text": ["test"]})

        modified = LLMProviderPresets.TOGETHER_AI_LLAMA_70B.model_copy(
            update={"temperature": 0.7}
        )

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm_spec(modified)
        )

        assert builder._llm_spec.temperature == 0.7
        assert builder._llm_spec.provider_name == "Together.AI"

    def test_with_llm_spec_rejects_non_spec(self):
        """Test with_llm_spec raises TypeError for non-LLMSpec."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
        )

        with pytest.raises(TypeError, match="Expected LLMSpec"):
            builder.with_llm_spec("not-a-spec")

    def test_with_llm_spec_rejects_dict(self):
        """Test with_llm_spec raises TypeError for dict."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
        )

        with pytest.raises(TypeError, match="Expected LLMSpec"):
            builder.with_llm_spec({"provider": "openai", "model": "gpt-4o-mini"})

    def test_with_llm_still_works(self):
        """Test existing with_llm() method still works (backward compatibility)."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.5)
        )

        assert builder._llm_spec is not None
        assert builder._llm_spec.model == "gpt-4o-mini"
        assert builder._llm_spec.temperature == 0.5

    def test_with_llm_spec_chainable(self):
        """Test with_llm_spec returns self for chaining."""
        pd.DataFrame({"text": ["test"]})

        builder = PipelineBuilder.create()
        result = builder.with_llm_spec(LLMProviderPresets.GPT4O_MINI)

        assert result is builder

    def test_with_llm_spec_can_be_overridden(self):
        """Test with_llm_spec can be called multiple times (last wins)."""
        df = pd.DataFrame({"text": ["test"]})

        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm_spec(LLMProviderPresets.GPT4O_MINI)
            .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
        )

        # Last call wins
        assert builder._llm_spec.model == "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        assert builder._llm_spec.provider_name == "Together.AI"

    def test_with_llm_and_with_llm_spec_can_be_mixed(self):
        """Test with_llm() and with_llm_spec() can be used interchangeably."""
        df = pd.DataFrame({"text": ["test"]})

        # Start with with_llm()
        builder = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Process: {text}")
            .with_llm(provider="openai", model="gpt-4o-mini")
        )

        assert builder._llm_spec.model == "gpt-4o-mini"

        # Override with with_llm_spec()
        builder.with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)

        assert builder._llm_spec.model == "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


class TestPresetsUsability:
    """Test preset usability and convenience features."""

    def test_presets_are_discoverable_via_dir(self):
        """Test presets can be discovered via dir()."""
        preset_names = [
            attr for attr in dir(LLMProviderPresets) if not attr.startswith("_")
        ]

        # Should have at least 8 presets + factory method
        assert "GPT4O_MINI" in preset_names
        assert "GPT4O" in preset_names
        assert "TOGETHER_AI_LLAMA_70B" in preset_names
        assert "TOGETHER_AI_LLAMA_8B" in preset_names
        assert "OLLAMA_LLAMA_70B" in preset_names
        assert "OLLAMA_LLAMA_8B" in preset_names
        assert "GROQ_LLAMA_70B" in preset_names
        assert "CLAUDE_SONNET_4" in preset_names
        assert "create_custom_openai_compatible" in preset_names

    def test_preset_comparison_across_providers(self):
        """Test comparing pricing across different providers."""
        openai_cost = LLMProviderPresets.GPT4O_MINI.input_cost_per_1k_tokens
        together_cost = (
            LLMProviderPresets.TOGETHER_AI_LLAMA_70B.input_cost_per_1k_tokens
        )
        ollama_cost = LLMProviderPresets.OLLAMA_LLAMA_70B.input_cost_per_1k_tokens

        # Ollama should be free
        assert ollama_cost == Decimal("0.0")

        # Together.AI should be cheaper than OpenAI GPT-4o (though not GPT-4o-mini)
        assert together_cost < LLMProviderPresets.GPT4O.input_cost_per_1k_tokens

        # All costs should be positive or zero
        assert openai_cost >= 0
        assert together_cost >= 0

    def test_presets_with_different_temperatures(self):
        """Test creating multiple variations with different temperatures."""
        base = LLMProviderPresets.GPT4O_MINI

        creative = base.model_copy(update={"temperature": 0.9})
        balanced = base.model_copy(update={"temperature": 0.5})
        deterministic = base.model_copy(update={"temperature": 0.0})

        assert creative.temperature == 0.9
        assert balanced.temperature == 0.5
        assert deterministic.temperature == 0.0

        # All should use same model
        assert creative.model == balanced.model == deterministic.model
