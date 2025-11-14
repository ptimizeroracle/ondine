"""
Real-world demo of the Plugin Architecture feature.

This example demonstrates:
1. Using a custom LLM provider via the Plugin Registry
2. Injecting a custom pipeline stage for data enrichment
3. Real execution with Groq API
"""

import os
from decimal import Decimal

import pandas as pd

from ondine import PipelineBuilder
from ondine.adapters import ProviderRegistry, provider
from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages import PipelineStage, stage

# ============================================================================
# EXAMPLE 1: Custom LLM Provider (for non-OpenAI-compatible APIs)
# ============================================================================


@provider("custom_groq")
class CustomGroqProvider(LLMClient):
    """
    Example of a custom provider wrapper.

    In practice, you might use this for:
    - Custom API format (non-OpenAI-compatible)
    - Adding custom authentication logic
    - Implementing rate limiting specific to your needs
    - Custom retry logic
    """

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """Delegate to standard Groq client."""
        from llama_index.llms.groq import Groq

        llm = Groq(
            model=self.spec.model,
            api_key=self.spec.api_key,
            temperature=self.spec.temperature,
        )

        response = llm.complete(prompt)

        # Custom cost calculation (example)
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(str(response)) * 1.3
        cost = (
            input_tokens / 1000 * self.spec.input_cost_per_1k_tokens
            + output_tokens / 1000 * self.spec.output_cost_per_1k_tokens
        )

        return LLMResponse(
            text=str(response),
            tokens_in=int(input_tokens),
            tokens_out=int(output_tokens),
            model=self.spec.model,
            cost=Decimal(str(cost)),
            latency_ms=0.0,
        )


# ============================================================================
# EXAMPLE 2: Custom Pipeline Stage (for RAG, moderation, etc.)
# ============================================================================


@stage("sentiment_enrichment")
class SentimentEnrichmentStage(PipelineStage):
    """
    Custom stage that adds sentiment analysis before LLM processing.

    In practice, you might use custom stages for:
    - RAG retrieval (fetch context from vector DB)
    - Content moderation (filter inappropriate inputs)
    - Data augmentation (add metadata, translations, etc.)
    - Fact-checking (verify claims before processing)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "SentimentEnrichmentStage"

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Add sentiment flags to each row."""
        print(f"\n🔍 Running {self.name}...")

        if context.data is None:
            return context

        # Simple sentiment heuristic (in production, use a real model)
        def get_sentiment(text):
            text_lower = str(text).lower()
            positive_words = ["great", "excellent", "amazing", "good", "love"]
            negative_words = ["bad", "terrible", "awful", "hate", "poor"]

            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return "positive"
            if neg_count > pos_count:
                return "negative"
            return "neutral"

        # Add sentiment column
        if "review" in context.data.columns:
            context.data["_sentiment"] = context.data["review"].apply(get_sentiment)
            print(f"✅ Added sentiment analysis for {len(context.data)} rows")

        return context

    def validate_input(self, context: ExecutionContext) -> bool:
        """Validate that input data exists."""
        return context.data is not None

    def estimate_cost(self, context: ExecutionContext) -> Decimal:
        """No cost for local processing."""
        return Decimal("0.0")


# ============================================================================
# DEMO: Using the Plugin Architecture
# ============================================================================


def demo_custom_provider():
    """Demo 1: Using a custom LLM provider."""
    print("\n" + "=" * 80)
    print("DEMO 1: Custom LLM Provider (Registered)")
    print("=" * 80)

    print("\n✅ Custom provider 'custom_groq' is now registered!")
    print("   You can use it with: .with_llm(provider='custom_groq', ...")
    print(
        "\n📝 Note: The custom provider is defined in this file using @provider decorator"
    )
    print("   See lines 28-66 for the implementation.")
    print("\n💡 Use case: Non-OpenAI-compatible APIs (Cohere, AI21, custom formats)")


def demo_custom_stage():
    """Demo 2: Injecting a custom pipeline stage."""
    print("\n" + "=" * 80)
    print("DEMO 2: Custom Pipeline Stage (Sentiment Enrichment)")
    print("=" * 80)

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY not set. Skipping custom stage demo.")
        return

    # Create sample data with reviews
    df = pd.DataFrame(
        {
            "review": [
                "This product is amazing and I love it!",
                "Terrible quality, complete waste of money.",
                "It works fine, nothing special.",
            ],
        }
    )

    # Build pipeline with custom stage
    from ondine.stages import RawTextParser

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["review"], output_columns=["response"])
        .with_stage(
            "sentiment_enrichment",  # ← Our custom stage!
            position="before_prompt",
            threshold=0.5,
        )
        .with_prompt(
            template="Based on this review: {review}\n\nGenerate a customer service response in 20 words."
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY"),
            input_cost_per_1k_tokens="0.00059",
            output_cost_per_1k_tokens="0.00079",
        )
        .with_parser(RawTextParser())
        .with_batch_size(3)
        .build()
    )

    # Execute
    result = pipeline.execute()

    print("\n📊 Results:")
    # Check if sentiment column was added
    if "_sentiment" in result.data.columns:
        print(result.data[["review", "_sentiment", "response"]])
        print("\n✅ Custom stage successfully added '_sentiment' column!")
    else:
        print(result.data[["review", "response"]])
        print(
            "\n⚠️  Note: Custom stage execution is registered but not yet integrated into the pipeline."
        )
        print("   This feature is under development.")

    # Calculate total cost from metadata
    total_cost = sum(
        item.get("cost", Decimal("0"))
        for item in result.metadata.get("stages", {}).values()
        if isinstance(item, dict)
    )
    print(f"\n💰 Total cost: ${total_cost:.6f}")


def demo_list_plugins():
    """Demo 3: Discovering available plugins."""
    print("\n" + "=" * 80)
    print("DEMO 3: Plugin Discovery")
    print("=" * 80)

    from ondine.stages import StageRegistry

    print("\n🔌 Available LLM Providers:")
    providers = ProviderRegistry.list_providers()
    for provider_id in sorted(providers):
        print(f"  - {provider_id}")

    print("\n🧩 Available Custom Stages:")
    stages = StageRegistry.list_stages()
    if stages:
        for stage_name in sorted(stages):
            print(f"  - {stage_name}")
    else:
        print("  (No custom stages registered yet)")


if __name__ == "__main__":
    # List all available plugins
    demo_list_plugins()

    # Demo custom provider
    demo_custom_provider()

    # Demo custom stage
    demo_custom_stage()

    print("\n" + "=" * 80)
    print("✅ Plugin Architecture Demo Complete!")
    print("=" * 80)
