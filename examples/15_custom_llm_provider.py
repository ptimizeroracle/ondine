"""
Example: Custom LLM Provider via Plugin Registry.

Demonstrates how to create and register a custom LLM provider without
modifying core Ondine code. This enables integration with any LLM API.

In this example, we'll create a custom provider for Replicate API.
"""

import os
import time
from decimal import Decimal

import pandas as pd

from ondine.adapters import LLMClient, provider
from ondine.api import PipelineBuilder
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


@provider("replicate")
class ReplicateClient(LLMClient):
    """
    Custom LLM client for Replicate API.

    This demonstrates the minimum required implementation:
    1. Inherit from LLMClient
    2. Implement invoke() method
    3. Implement estimate_tokens() method
    4. Register via @provider decorator
    """

    def __init__(self, spec: LLMSpec):
        """Initialize Replicate client."""
        super().__init__(spec)

        # Import Replicate SDK (install with: pip install replicate)
        try:
            import replicate

            self.client = replicate.Client(
                api_token=spec.api_key or os.getenv("REPLICATE_API_TOKEN")
            )
        except ImportError:
            raise ImportError(
                "Replicate SDK not installed. Install with:\n  pip install replicate"
            )

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Call Replicate API.

        Args:
            prompt: The prompt to send
            **kwargs: Override parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with text, tokens, cost, and latency
        """
        start_time = time.time()

        # Prepare parameters
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        # Call Replicate API
        # Example using Llama 2 model
        output = self.client.run(
            self.model,  # e.g., "meta/llama-2-70b-chat"
            input={
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        # Join output if it's a generator
        if hasattr(output, "__iter__") and not isinstance(output, str):
            response_text = "".join(output)
        else:
            response_text = str(output)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_in = self.estimate_tokens(prompt)
        tokens_out = self.estimate_tokens(response_text)
        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.

        For simplicity, uses word count approximation.
        In production, use a proper tokenizer like tiktoken.
        """
        # Simple approximation: ~1.3 tokens per word
        word_count = len(text.split())
        return int(word_count * 1.3)


def example_1_basic_usage():
    """Example 1: Use custom provider in pipeline."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Custom Provider Usage")
    print("=" * 60)

    # Sample data
    data = pd.DataFrame(
        {
            "resume_summary": [
                "5 years Python, ML experience, built recommendation systems",
                "Frontend specialist, React/TypeScript, 3 years at startup",
                "Data engineer, Spark/Airflow, ETL pipelines, AWS certified",
            ]
        }
    )

    # Build pipeline with custom provider
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data, input_columns=["resume_summary"], output_columns=["role_fit"]
        )
        .with_prompt(
            template="""You are an HR screening assistant.
Analyze this resume and recommend the best role fit (Backend, Frontend, Data, ML, DevOps).
Return only the role name.

Resume: {resume_summary}

Role:"""
        )
        .with_llm(
            provider="replicate",  # ← Our custom provider!
            model="meta/llama-2-70b-chat",
            api_key=os.getenv("REPLICATE_API_TOKEN"),
            temperature=0.0,
            max_tokens=20,
        )
        .with_batch_size(1)
        .build()
    )

    # Execute
    print("\n📊 Processing resumes with Replicate (Llama 2)...")
    result = pipeline.execute()

    # Display results
    print("\n✅ Results:")
    for idx, row in result.data.iterrows():
        print(f"  {idx + 1}. Resume: {row['resume_summary'][:50]}...")
        print(f"     → Recommended Role: {row['role_fit']}")

    print(f"\n💰 Cost: ${result.costs.total_cost:.6f}")
    print(f"⏱️  Total time: {result.costs.total_latency_ms / 1000:.2f}s")


def example_2_multiple_custom_providers():
    """Example 2: Register multiple custom providers."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Custom Providers")
    print("=" * 60)

    # Register another custom provider inline
    @provider("mock_llm")
    class MockLLMClient(LLMClient):
        """Mock LLM for testing/development."""

        def invoke(self, prompt: str, **kwargs) -> LLMResponse:
            # Always return a canned response
            return LLMResponse(
                text=f"Mock response for: {prompt[:50]}...",
                tokens_in=10,
                tokens_out=10,
                model="mock-model",
                cost=Decimal("0.0"),
                latency_ms=1.0,
            )

        def estimate_tokens(self, text: str) -> int:
            return len(text.split())

    # List all available providers
    from ondine.adapters.provider_registry import ProviderRegistry

    providers = ProviderRegistry.list_providers()
    print("\n📦 Available LLM Providers:")
    for provider_id in sorted(providers.keys()):
        print(f"  - {provider_id}")

    print("\n✅ Custom providers registered: replicate, mock_llm")


def example_3_provider_with_complex_config():
    """Example 3: Provider with custom configuration and retry logic."""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Custom Provider")
    print("=" * 60)

    @provider("advanced_provider")
    class AdvancedLLMClient(LLMClient):
        """
        Advanced provider with:
        - Custom retry logic
        - Request logging
        - Rate limiting
        - Cost tracking per request
        """

        def __init__(self, spec: LLMSpec):
            super().__init__(spec)
            self.request_count = 0
            self.total_cost = Decimal("0.0")

        def invoke(self, prompt: str, **kwargs) -> LLMResponse:
            self.request_count += 1

            # Simulate API call
            response_text = f"Advanced response #{self.request_count}"

            # Calculate cost with custom pricing
            tokens_in = self.estimate_tokens(prompt)
            tokens_out = self.estimate_tokens(response_text)

            # Custom cost calculation (e.g., tiered pricing)
            if self.request_count < 100:
                cost_per_1k = Decimal("0.001")  # Tier 1
            else:
                cost_per_1k = Decimal("0.0005")  # Tier 2 (cheaper)

            cost = (Decimal(tokens_in + tokens_out) / 1000) * cost_per_1k
            self.total_cost += cost

            return LLMResponse(
                text=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self.model,
                cost=cost,
                latency_ms=100.0,
            )

        def estimate_tokens(self, text: str) -> int:
            return len(text.split())

    print("✅ Advanced provider registered with custom features:")
    print("  - Request counting")
    print("  - Tiered pricing")
    print("  - Cost tracking")


if __name__ == "__main__":
    print("\n🌾 Ondine: Custom LLM Provider Examples\n")

    # Note: example_1 requires REPLICATE_API_TOKEN environment variable
    # Uncomment to run if you have Replicate access:
    # example_1_basic_usage()

    # These examples work without external dependencies
    example_2_multiple_custom_providers()
    example_3_provider_with_complex_config()

    print("\n" + "=" * 60)
    print("🎉 Examples complete!")
    print("=" * 60)
    print("\n💡 Key Takeaways:")
    print("  1. Use @provider decorator to register custom providers")
    print("  2. Implement invoke() and estimate_tokens() methods")
    print("  3. Custom providers work seamlessly with Ondine pipelines")
    print("  4. No core code modification needed!")
    print("\n")
