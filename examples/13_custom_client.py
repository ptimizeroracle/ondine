"""
Custom LLM Client Example - Advanced usage

This example demonstrates how to create a fully custom LLM client
by extending the LLMClient base class. This gives you maximum control
for exotic APIs or custom logic.

Use cases:
- Custom authentication schemes
- Non-OpenAI-compatible APIs
- Custom retry logic
- Custom token counting
- Caching layer
"""

from decimal import Decimal
from typing import Any

import pandas as pd

from ondine import PipelineBuilder
from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class MyCustomLLMClient(LLMClient):
    """
    Example custom LLM client.

    In a real scenario, this could interface with:
    - Custom API (non-OpenAI format)
    - Local model with custom loading
    - Caching layer
    - Custom authentication
    """

    def __init__(self, spec: LLMSpec):
        """Initialize custom client."""
        super().__init__(spec)

        # Custom initialization
        print(f"🚀 Initializing {spec.provider_name or 'Custom'} client")
        print(f"   Model: {spec.model}")
        print(f"   Base URL: {spec.base_url}")

        # In a real implementation, you'd initialize your custom API client here
        # Example: self.client = MyCustomAPI(base_url=spec.base_url, ...)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Invoke custom LLM.

        In a real implementation, you'd call your custom API here.
        """
        import time

        start_time = time.time()

        # EXAMPLE: Replace this with your actual API call
        # response = self.client.generate(prompt)

        # For demo purposes, we'll use a simple mock response
        response_text = f"[Custom LLM Response to: {prompt[:50]}...]"

        latency_ms = (time.time() - start_time) * 1000

        # Custom token counting (replace with your logic)
        tokens_in = len(prompt.split())  # Simple word count
        tokens_out = len(response_text.split())

        # Calculate cost
        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMResponse(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=f"{self.spec.provider_name or 'Custom'}/{self.model}",
            cost=cost,
            latency_ms=latency_ms,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Replace with your custom tokenization logic.
        """
        # Simple word count for demo
        return len(text.split())


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame(
        {
            "text": [
                "This is a test input",
                "Another example to process",
                "Third item in the dataset",
            ]
        }
    )

    # Create custom client spec
    custom_spec = LLMSpec(
        provider="openai",  # Dummy provider (won't be used)
        model="my-custom-model-v1",
        base_url="https://my-custom-api.example.com/v1",
        provider_name="MyCustomLLM",
        temperature=0.7,
        max_tokens=100,
        input_cost_per_1k_tokens=Decimal("0.001"),
        output_cost_per_1k_tokens=Decimal("0.002"),
    )

    # Instantiate custom client
    custom_client = MyCustomLLMClient(custom_spec)

    # Build pipeline with custom client
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["text"],
            output_columns=["result"],
        )
        .with_prompt("Process this: {text}")
        .with_custom_llm_client(custom_client)  # Inject custom client
        .with_batch_size(10)
        .build()
    )

    print("\n" + "=" * 60)
    print("🔧 CUSTOM LLM CLIENT DEMO")
    print("=" * 60 + "\n")

    # Execute pipeline
    print("▶ Executing pipeline with custom client...\n")
    result = pipeline.execute()

    # Display results
    print("✅ Results:")
    print(result.data)
    print("\n📊 Stats:")
    print(f"   Processed: {result.metrics.processed_rows} rows")
    print(f"   Total cost: ${result.costs.total_cost}")
    print(f"   Total tokens: {result.costs.total_tokens:,}")
    print(f"   Duration: {result.metrics.total_duration_seconds:.2f}s")

    print("\n" + "=" * 60)
    print("💡 TIP: Replace MyCustomLLMClient.invoke() with your actual API!")
    print("=" * 60 + "\n")
