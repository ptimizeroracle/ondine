"""
Example: Response Deduplication for Cost Savings

This example demonstrates how to use Ondine's response deduplication feature
to avoid redundant LLM API calls and save costs.

Deduplication caches LLM responses based on the complete request content
(prompt, system message, model, temperature, etc.). When identical requests
are made, the cached response is returned instead of calling the LLM API again.

This is especially useful when:
- Processing datasets with duplicate rows
- Running experiments with the same data multiple times
- Developing and testing pipelines
"""

from ondine import PipelineBuilder


def main():
    """Demonstrate response deduplication with cost savings."""

    # Sample data with duplicates (common in real-world datasets)
    data = [
        {"product_id": "P001", "review": "Great product, highly recommend!"},
        {"product_id": "P002", "review": "Excellent quality and fast shipping."},
        {
            "product_id": "P001",
            "review": "Great product, highly recommend!",
        },  # Duplicate
        {"product_id": "P003", "review": "Good value for the price."},
        {
            "product_id": "P002",
            "review": "Excellent quality and fast shipping.",
        },  # Duplicate
        {
            "product_id": "P001",
            "review": "Great product, highly recommend!",
        },  # Duplicate
    ]

    print("Dataset has 6 rows, but 3 are duplicates")
    print("Without deduplication: 6 API calls")
    print("With deduplication: 3 API calls (50% savings)\n")

    # Create temporary CSV for demo
    import csv
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "review"])
        writer.writeheader()
        writer.writerows(data)
        csv_path = f.name

    # Example 1: Without deduplication (baseline)
    print("=" * 60)
    print("Example 1: Without Deduplication (Baseline)")
    print("=" * 60)

    pipeline_no_dedup = (
        PipelineBuilder.create()
        .from_csv(csv_path, input_columns=["review"])
        .with_prompt(
            "Classify this product review as: positive, negative, or neutral\n\nReview: {review}\nSentiment:"
        )
        .with_system_prompt(
            "You are a sentiment classifier. Respond with only the sentiment label."
        )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .build()
    )

    result_no_dedup = pipeline_no_dedup.execute()
    print(f"Rows processed: {result_no_dedup.metrics.total_rows}")
    print(f"API calls made: {result_no_dedup.metrics.total_rows}")
    print(f"Cost: ${result_no_dedup.costs.total_cost}")
    print()

    # Example 2: With in-memory deduplication
    print("=" * 60)
    print("Example 2: With In-Memory Deduplication")
    print("=" * 60)

    pipeline_dedup_memory = (
        PipelineBuilder.create()
        .from_csv(csv_path, input_columns=["review"])
        .with_prompt(
            "Classify this product review as: positive, negative, or neutral\n\nReview: {review}\nSentiment:"
        )
        .with_system_prompt(
            "You are a sentiment classifier. Respond with only the sentiment label."
        )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .with_deduplication()  # Enable in-memory deduplication
        .build()
    )

    result_dedup_memory = pipeline_dedup_memory.execute()
    print(f"Rows processed: {result_dedup_memory.metrics.total_rows}")
    print("API calls made: ~3 (duplicates served from cache)")
    print(f"Cost: ${result_dedup_memory.costs.total_cost}")
    print("Savings: ~50% compared to no deduplication!")
    print()

    # Example 3: With persistent cache (great for development)
    print("=" * 60)
    print("Example 3: With Persistent Cache")
    print("=" * 60)
    print("Cache persists across runs - perfect for iterative development!\n")

    # First run - populates cache
    print("First run (populates cache):")
    pipeline_dedup_persist = (
        PipelineBuilder.create()
        .from_csv(csv_path, input_columns=["review"])
        .with_prompt(
            "Classify this product review as: positive, negative, or neutral\n\nReview: {review}\nSentiment:"
        )
        .with_system_prompt(
            "You are a sentiment classifier. Respond with only the sentiment label."
        )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .with_deduplication("ondine_demo_cache.db")  # Persistent cache
        .build()
    )

    result_run1 = pipeline_dedup_persist.execute()
    print(f"Cost: ${result_run1.costs.total_cost}")
    print()

    # Second run - uses cache (much faster and cheaper)
    print("Second run (uses cache):")
    pipeline_dedup_persist2 = (
        PipelineBuilder.create()
        .from_csv(csv_path, input_columns=["review"])
        .with_prompt(
            "Classify this product review as: positive, negative, or neutral\n\nReview: {review}\nSentiment:"
        )
        .with_system_prompt(
            "You are a sentiment classifier. Respond with only the sentiment label."
        )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_concurrency(5)
        .with_deduplication("ondine_demo_cache.db")  # Same cache file
        .build()
    )

    result_run2 = pipeline_dedup_persist2.execute()
    print(f"Cost: ${result_run2.costs.total_cost}")
    print("Time: Much faster (no API calls for cached responses)")
    print()

    # Clean up
    import os

    os.unlink(csv_path)
    if os.path.exists("ondine_demo_cache.db"):
        os.unlink("ondine_demo_cache.db")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ Deduplication reduces API calls for duplicate content")
    print("✓ In-memory cache: Fast, but lost on process exit")
    print("✓ Persistent cache: Survives restarts, great for development")
    print("✓ Cache key includes: prompt, system_message, model, temperature, etc.")
    print("✓ Thread-safe: Can be used with concurrent execution")
    print()
    print("When to use:")
    print("- Datasets with duplicate rows (common in logs, reviews, support tickets)")
    print("- Iterative development and testing")
    print("- Re-running pipelines on similar data")
    print("- Any scenario with redundant API calls")


if __name__ == "__main__":
    main()
