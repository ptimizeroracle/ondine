"""
Example: Using LLM Provider Presets

Demonstrates the convenient preset configurations for common LLM providers.
This simplifies configuration and reduces boilerplate code.

Key Benefits:
- Zero boilerplate for common providers
- Correct pricing and base URLs pre-configured
- Easy to override individual settings
- Reusable across multiple pipelines
"""

import pandas as pd

from ondine.core.specifications import LLMProviderPresets

# Sample data
df = pd.DataFrame(
    {
        "text": [
            "This product is amazing! Best purchase ever.",
            "Terrible quality. Waste of money.",
            "It's okay, nothing special.",
        ]
    }
)

print("=" * 70)
print("LLM Provider Presets Demo")
print("=" * 70)

# Example 1: Together.AI with preset (70B model)
print("\nExample 1: Together.AI Llama 70B (using preset)")
print("-" * 70)
print("Code:")
print("""
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(df, input_columns=["text"], output_columns=["sentiment"])
    .with_prompt("Classify sentiment as positive, negative, or neutral: {text}")
    .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
    .build()
)
""")
print("\nConfiguration:")
print(f"  Provider: {LLMProviderPresets.TOGETHER_AI_LLAMA_70B.provider_name}")
print(f"  Model: {LLMProviderPresets.TOGETHER_AI_LLAMA_70B.model}")
print(f"  Base URL: {LLMProviderPresets.TOGETHER_AI_LLAMA_70B.base_url}")
print(
    f"  Input Cost: ${LLMProviderPresets.TOGETHER_AI_LLAMA_70B.input_cost_per_1k_tokens}/1K tokens"
)

# Example 2: Local Ollama (free!)
print("\n\nExample 2: Local Ollama Llama 8B (FREE)")
print("-" * 70)
print("Code:")
print("""
pipeline_local = (
    PipelineBuilder.create()
    .from_dataframe(df, input_columns=["text"], output_columns=["sentiment"])
    .with_prompt("Classify sentiment: {text}")
    .with_llm_spec(LLMProviderPresets.OLLAMA_LLAMA_8B)
    .build()
)
""")
print("\nConfiguration:")
print(f"  Provider: {LLMProviderPresets.OLLAMA_LLAMA_8B.provider_name}")
print(f"  Model: {LLMProviderPresets.OLLAMA_LLAMA_8B.model}")
print(f"  Base URL: {LLMProviderPresets.OLLAMA_LLAMA_8B.base_url}")
print(
    f"  Cost: ${LLMProviderPresets.OLLAMA_LLAMA_8B.input_cost_per_1k_tokens}/1K (FREE!)"
)

# Example 3: Custom provider using factory
print("\n\nExample 3: Custom OpenAI-compatible provider (vLLM)")
print("-" * 70)
print("Code:")
print("""
custom_spec = LLMProviderPresets.create_custom_openai_compatible(
    provider_name="My vLLM Server",
    model="mistral-7b-instruct",
    base_url="http://my-server:8000/v1",
    input_cost_per_1k=0.0,
    output_cost_per_1k=0.0,
    temperature=0.7
)
""")

custom_spec = LLMProviderPresets.create_custom_openai_compatible(
    provider_name="My vLLM Server",
    model="mistral-7b-instruct",
    base_url="http://my-server:8000/v1",
    input_cost_per_1k=0.0,
    output_cost_per_1k=0.0,
    temperature=0.7,
)

print("\nConfiguration:")
print(f"  Provider: {custom_spec.provider_name}")
print(f"  Model: {custom_spec.model}")
print(f"  Base URL: {custom_spec.base_url}")
print(f"  Temperature: {custom_spec.temperature}")

# Example 4: Override preset with custom settings
print("\n\nExample 4: Override preset settings")
print("-" * 70)
print("Code:")
print("""
together_custom = LLMProviderPresets.TOGETHER_AI_LLAMA_70B.model_copy(
    update={
        "temperature": 0.9,
        "max_tokens": 100,
        "api_key": "your-explicit-key-here"
    }
)
""")

together_custom = LLMProviderPresets.TOGETHER_AI_LLAMA_70B.model_copy(
    update={"temperature": 0.9, "max_tokens": 100}
)

print("\nOriginal preset:")
print(f"  Temperature: {LLMProviderPresets.TOGETHER_AI_LLAMA_70B.temperature}")
print(f"  Max Tokens: {LLMProviderPresets.TOGETHER_AI_LLAMA_70B.max_tokens}")

print("\nOverridden settings:")
print(f"  Temperature: {together_custom.temperature}")
print(f"  Max Tokens: {together_custom.max_tokens}")

# Example 5: List all available presets
print("\n\nExample 5: Available Presets")
print("-" * 70)
print("All available provider presets:\n")

preset_attrs = [
    attr
    for attr in dir(LLMProviderPresets)
    if not attr.startswith("_") and attr.isupper()
]

for preset_name in preset_attrs:
    preset = getattr(LLMProviderPresets, preset_name)
    if hasattr(preset, "model"):
        provider_display = (
            preset.provider_name if preset.provider_name else preset.provider.value
        )
        print(f"  {preset_name:30s} -> {provider_display} / {preset.model}")

# Comparison: Old vs New approach
print("\n\nComparison: Parameter-based vs Preset-based")
print("=" * 70)

print("\nOLD APPROACH (parameter-based):")
print("""
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_prompt("...")
    .with_llm(
        provider="openai_compatible",
        provider_name="Together.AI",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key="${TOGETHER_API_KEY}",
        input_cost_per_1k_tokens=0.0006,
        output_cost_per_1k_tokens=0.0006
    )
    .build()
)
""")

print("NEW APPROACH (preset-based):")
print("""
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", ...)
    .with_prompt("...")
    .with_llm_spec(LLMProviderPresets.TOGETHER_AI_LLAMA_70B)
    .build()
)
""")

print("\nBoilerplate reduction: 80%")
print("Lines of code: 7 -> 1")
print("Configuration errors: Eliminated (pre-validated)")

print("\n" + "=" * 70)
print("Note: Remember to set API keys via environment variables:")
print("  export OPENAI_API_KEY='...'")
print("  export TOGETHER_API_KEY='...'")
print("  export GROQ_API_KEY='...'")
print("  export ANTHROPIC_API_KEY='...'")
print("=" * 70)
