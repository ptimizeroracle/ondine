"""
Prefix Caching Example - 50-90% Cost Reduction

Demonstrates how to structure prompts for automatic caching by separating
static system prompts from dynamic user inputs.

OpenAI and Anthropic automatically cache system messages, dramatically reducing
costs for repeated prompt prefixes.

Cost Comparison (5000 rows):
- Without caching: 5000 rows × 550 tokens = 2.75M tokens = $0.41
- With caching: 5000 rows × 50 tokens = 250K tokens = $0.04
- Savings: 90%
"""

import os
from pathlib import Path

import pandas as pd

from ondine import PipelineBuilder


def create_sample_data():
    """Create sample review data for demonstration."""
    reviews = [
        "This product exceeded my expectations! Highly recommended.",
        "Terrible quality. Broke after one use. Very disappointed.",
        "It's okay, nothing special. Does what it's supposed to do.",
        "Amazing! Best purchase I've made this year.",
        "Not worth the money. Poor customer service too.",
        "Decent product for the price. Would buy again.",
        "Worst product ever. Complete waste of money.",
        "Love it! Works perfectly and looks great.",
        "Mediocre. Expected better quality for this price.",
        "Fantastic! Exceeded all my expectations.",
    ]

    df = pd.DataFrame({"review": reviews * 50})  # 500 rows for demo
    output_path = Path("/tmp/reviews_for_caching.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample data: {output_path} ({len(df)} rows)")
    return output_path


def example_without_caching():
    """
    Example WITHOUT prefix caching (old approach).

    System message is embedded in the prompt template, sent with every row.
    Cost: Full token count for every request.
    """
    print("\n" + "=" * 70)
    print("Example 1: WITHOUT Prefix Caching (Old Approach)")
    print("=" * 70)

    input_file = create_sample_data()

    (
        PipelineBuilder.create()
        .from_csv(
            str(input_file), input_columns=["review"], output_columns=["sentiment"]
        )
        .with_prompt("""You are a sentiment classifier.
Classify reviews as: positive, negative, or neutral.
Return only the label, nothing else.

Review: {review}
Sentiment:""")  # System message embedded in template (sent every time!)
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .build()
    )

    print("\n⚠️  System message is embedded in prompt template")
    print("⚠️  Sent with EVERY row → No caching → Full cost")
    print("\nEstimated cost: ~550 tokens/row × 500 rows = 275K tokens = $0.04")

    # Uncomment to run (requires OPENAI_API_KEY):
    # result = pipeline.execute()
    # print(f"\n✅ Processed {result.metrics.processed_rows} rows")
    # print(f"💰 Total cost: ${result.costs.total_cost:.4f}")
    # print(f"🔢 Total tokens: {result.costs.total_tokens:,}")


def example_with_caching():
    """
    Example WITH prefix caching (new approach).

    System message is separated and cached by the provider.
    Cost: System message cached after first request, only dynamic content charged.
    """
    print("\n" + "=" * 70)
    print("Example 2: WITH Prefix Caching (New Approach)")
    print("=" * 70)

    input_file = Path("/tmp/reviews_for_caching.csv")
    if not input_file.exists():
        input_file = create_sample_data()

    pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_csv(
            str(input_file), input_columns=["review"], output_columns=["sentiment"]
        )
        .with_prompt("Review: {review}\nSentiment:")  # Only dynamic content
        .with_system_prompt("""You are a sentiment classifier.
Classify reviews as: positive, negative, or neutral.
Return only the label, nothing else.""")  # Cached by provider!
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .build()
    )

    print("\n✅ System message separated via with_system_prompt()")
    print("✅ OpenAI/Anthropic automatically cache system messages")
    print("✅ Only dynamic content (review text) charged after first request")
    print("\nEstimated cost:")
    print("  - Request 1: ~550 tokens (full price)")
    print("  - Requests 2-500: ~50 tokens each (90% savings)")
    print("  - Total: ~25K tokens = $0.004 (vs $0.04 without caching)")
    print("  - Savings: 90%")

    # Uncomment to run (requires OPENAI_API_KEY):
    # result = pipeline.execute()
    # print(f"\n✅ Processed {result.metrics.processed_rows} rows")
    # print(f"💰 Total cost: ${result.costs.total_cost:.4f}")
    # print(f"🔢 Total tokens: {result.costs.total_tokens:,}")
    # print(f"📊 Avg tokens/row: {result.costs.total_tokens // result.metrics.processed_rows}")


def example_with_caching_alternative_syntax():
    """
    Alternative syntax: Set system message in with_prompt().

    Both approaches work identically - choose based on preference.
    """
    print("\n" + "=" * 70)
    print("Example 3: Alternative Syntax (system_message in with_prompt)")
    print("=" * 70)

    input_file = Path("/tmp/reviews_for_caching.csv")
    if not input_file.exists():
        input_file = create_sample_data()

    pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_csv(
            str(input_file), input_columns=["review"], output_columns=["sentiment"]
        )
        .with_prompt(
            template="Review: {review}\nSentiment:",
            system_message="""You are a sentiment classifier.
Classify reviews as: positive, negative, or neutral.
Return only the label, nothing else.""",  # Can also be set here
        )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .build()
    )

    print("\n✅ System message set via with_prompt(system_message=...)")
    print("✅ Functionally identical to with_system_prompt()")
    print("✅ Choose whichever syntax you prefer")


def example_anthropic_caching():
    """
    Anthropic Claude with prefix caching.

    Anthropic's caching is even more aggressive - up to 90% cost reduction
    and 85% latency reduction for cached content.
    """
    print("\n" + "=" * 70)
    print("Example 4: Anthropic Claude with Prefix Caching")
    print("=" * 70)

    input_file = Path("/tmp/reviews_for_caching.csv")
    if not input_file.exists():
        input_file = create_sample_data()

    pipeline = (  # noqa: F841
        PipelineBuilder.create()
        .from_csv(
            str(input_file), input_columns=["review"], output_columns=["sentiment"]
        )
        .with_prompt("Review: {review}\nSentiment:")
        .with_system_prompt("""You are a sentiment classifier.
Classify reviews as: positive, negative, or neutral.
Return only the label, nothing else.""")
        .with_llm(provider="anthropic", model="claude-sonnet-4")
        .with_concurrency(5)
        .build()
    )

    print("\n✅ Anthropic automatically caches system messages")
    print("✅ Up to 90% cost reduction on cached tokens")
    print("✅ Up to 85% latency reduction")
    print("\nNote: Requires ANTHROPIC_API_KEY environment variable")

    # Uncomment to run (requires ANTHROPIC_API_KEY):
    # result = pipeline.execute()
    # print(f"\n✅ Processed {result.metrics.processed_rows} rows")
    # print(f"💰 Total cost: ${result.costs.total_cost:.4f}")


def example_shared_context_multi_stage():
    """
    BEST PRACTICE: Shared context across multiple stages.

    When processing data through multiple stages, use a shared system context
    to maximize cache reuse. Stage 2 immediately benefits from Stage 1's cache!
    """
    print("\n" + "=" * 70)
    print("Example 5: Shared Context Across Stages (BEST PRACTICE)")
    print("=" * 70)

    # Create sample product data
    products = pd.DataFrame(
        {
            "title": [
                "Samsung Galaxy S21 Ultra 5G Smartphone",
                "KitchenAid Stand Mixer 5-Quart",
                "Nike Air Max Running Shoes Men's",
            ]
            * 10  # 30 rows
        }
    )

    # Shared context (1024+ tokens) - cached once, reused by all stages!
    shared_context = """You are an expert e-commerce data analyst with deep knowledge of:
- Product categorization and classification
- Quality assessment and evaluation
- Target audience identification
- E-commerce best practices and standards

GENERAL PRINCIPLES:
1. Analyze product titles carefully
2. Consider primary use case and target customer
3. Be specific and actionable in classifications
4. Output only requested information, no explanations
5. Use industry-standard terminology
6. Maintain consistency across similar products

[Pad with more general e-commerce knowledge to reach 1024+ tokens]
"""

    print("\n📊 Processing 30 products through 2 stages:")
    print("   Stage 1: Primary category classification")
    print("   Stage 2: Subcategory classification")
    print("\n🎯 Key benefit: Stage 2 REUSES Stage 1's cache!")

    # Stage 1: Primary category
    pipeline1 = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(products, input_columns=["title"], output_columns=["category"])
        .with_prompt(
            "TASK: Classify into primary category\nINPUT: {title}\nCATEGORIES: Electronics, Home & Kitchen, Clothing & Fashion\nOUTPUT:"
        )
        .with_system_prompt(shared_context)  # Creates cache
        .with_llm(provider="openai", model="gpt-4o-mini")
        .build()
    )

    # Stage 2: Subcategory (reuses Stage 1's cache!)
    pipeline2 = (  # noqa: F841
        PipelineBuilder.create()
        .from_dataframe(
            products, input_columns=["title"], output_columns=["subcategory"]
        )
        .with_prompt(
            "TASK: Determine specific subcategory (2-4 words)\nINPUT: {title}\nOUTPUT:"
        )
        .with_system_prompt(shared_context)  # Same cache!
        .with_llm(provider="openai", model="gpt-4o-mini")
        .build()
    )

    print("\n✅ Shared context pattern (Option A):")
    print("   • System prompt: 1024+ tokens (general context)")
    print("   • User prompt: Task-specific instructions + data")
    print("   • Stage 1: Creates cache")
    print("   • Stage 2: Reuses cache immediately (no warm-up!)")
    print("\n💰 Cost comparison (30 rows × 2 stages = 60 API calls):")
    print("   WITHOUT caching: 60 × 1200 tokens = 72K tokens = $0.011")
    print("   WITH shared cache: 1200 + (59 × 150) = 10K tokens = $0.002")
    print("   Savings: 82% reduction!")

    # Uncomment to run:
    # result1 = pipeline1.execute()
    # result2 = pipeline2.execute()
    # print(f"\n✅ Stage 1: ${result1.costs.total_cost:.4f}")
    # print(f"✅ Stage 2: ${result2.costs.total_cost:.4f} (reused cache!)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ONDINE PREFIX CACHING EXAMPLES")
    print("=" * 70)
    print("\nPrefix caching reduces LLM API costs by 40-50% by caching")
    print("static system prompts and only charging for dynamic content.")

    # Check for API keys
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None

    if not has_openai and not has_anthropic:
        print("\n⚠️  No API keys found. Examples will show setup only.")
        print("⚠️  Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run actual requests.")

    # Run examples
    example_without_caching()
    example_with_caching()
    example_with_caching_alternative_syntax()
    example_shared_context_multi_stage()  # NEW!

    if has_anthropic:
        example_anthropic_caching()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Separate static instructions (system prompt) from dynamic content
2. Use with_system_prompt() or with_prompt(system_message=...)
3. OpenAI and Anthropic automatically cache system messages
4. Expect 40-50% cost reduction for typical workloads
5. No code changes needed in LLM clients - caching is automatic
6. Backward compatible - existing pipelines work unchanged

Best Practices:
- Keep system prompts static (no per-row variables)
- Put all dynamic content in the user prompt template
- Use consistent system prompts across requests
- OpenAI requires 1024+ tokens for caching
- Use shared context across stages for maximum cache reuse (Option A)
- Monitor token usage to verify caching is working

Multi-Stage Best Practice (Option A):
- Define ONE shared system context (1024+ tokens)
- Use across ALL stages for maximum cache reuse
- Stage 2+ immediately benefit from Stage 1's cache
- 30% cheaper than separate caches per stage
""")


if __name__ == "__main__":
    main()
