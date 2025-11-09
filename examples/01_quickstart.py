"""
Quickstart Example - Two Ways to Build Pipelines.

This example shows both the Quick API (simplified, auto-detects settings)
and the Builder API (full control, explicit configuration).
"""

import pandas as pd

from ondine import PipelineBuilder, QuickPipeline

# Create sample data
data = pd.DataFrame(
    {
        "product_name": [
            "Apple iPhone 13 Pro Max 256GB",
            "Samsung Galaxy S22 Ultra 512GB",
            'MacBook Pro 16" M2 32GB RAM',
        ],
        "description": [
            "Latest iPhone with advanced camera system",
            "Premium Android flagship with S Pen",
            "Powerful laptop for professionals",
        ],
    }
)

print("=" * 80)
print("OPTION 1: Quick API (Recommended for getting started)")
print("=" * 80)

# Quick API: Minimal configuration, smart defaults
pipeline_quick = QuickPipeline.create(
    data=data,
    prompt="""Categorize this product into ONE of these categories:
    - Electronics > Smartphones
    - Electronics > Laptops
    - Electronics > Accessories

    Product: {product_name}
    Description: {description}

    Category:""",
    model="gpt-4o-mini",
)

# Note what was auto-detected:
print("✓ Auto-detected input columns: product_name, description (from {placeholders})")
print("✓ Auto-detected provider: openai (from model name)")
print("✓ Auto-set batch_size: 10 (smart default for small datasets)")
print("✓ Auto-set concurrency: 5 (safe default for OpenAI)")
print("✓ Auto-enabled retries: 3 attempts\n")

# Alternatively, if you need more control:
print("=" * 80)
print("OPTION 2: Builder API (Full control)")
print("=" * 80)

# Builder API: Explicit configuration
pipeline_builder = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_name", "description"],
        output_columns=["category"],
    )
    .with_prompt(
        """Categorize this product into ONE of these categories:
        - Electronics > Smartphones
        - Electronics > Laptops
        - Electronics > Accessories

        Product: {product_name}
        Description: {description}

        Category:"""
    )
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    .with_batch_size(10)
    .with_concurrency(3)
    .build()
)

print("✓ Explicit input/output columns")
print("✓ Explicit provider and model")
print("✓ Explicit batch size and concurrency")
print("✓ Full control over every parameter\n")

# Use the Quick API pipeline for this demo
pipeline = pipeline_quick

# Estimate cost before running
print("Estimating cost...")
estimate = pipeline.estimate_cost()
print(f"Estimated cost: ${estimate.total_cost:.4f}")
print(f"Estimated tokens: {estimate.total_tokens}")

# Execute pipeline
print("\nProcessing data...")
result = pipeline.execute()

# Display results
print("\nResults:")
print(result.data[["product_name", "category"]])

# Show metrics
print("\nMetrics:")
print(f"  Processed rows: {result.metrics.processed_rows}")
print(f"  Duration: {result.metrics.total_duration_seconds:.2f}s")
print(f"  Total cost: ${result.costs.total_cost:.4f}")
print(
    f"  Cost per row: ${float(result.costs.total_cost) / result.metrics.total_rows:.6f}"
)
