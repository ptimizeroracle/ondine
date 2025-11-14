"""
Cost Control Example - Budget limits and tracking.

This example shows how to control costs and monitor spending
during pipeline execution.
"""

import pandas as pd

from ondine import PipelineBuilder

# Create larger dataset
data = pd.DataFrame(
    {
        "text": [
            f"This is sample text number {i} that needs processing." for i in range(100)
        ]
    }
)

# Build pipeline with cost controls
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["text"],
        output_columns=["summary"],
    )
    .with_prompt("Summarize in 5 words: {text}")
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    # Cost control settings
    .with_max_budget(1.0)  # Maximum $1.00
    .with_batch_size(20)  # Process in batches
    .with_concurrency(5)  # 5 concurrent requests
    .with_rate_limit(60)  # 60 requests per minute
    .with_checkpoint_interval(25)  # Checkpoint every 25 rows
    .build()
)

# Get cost estimate
print("Cost Estimation:")
estimate = pipeline.estimate_cost()
print(f"  Estimated total: ${estimate.total_cost:.4f}")
print(f"  Estimated tokens: {estimate.total_tokens:,}")
print(f"  Cost per row: ${float(estimate.total_cost) / estimate.rows:.6f}")

# Check if within budget
if estimate.total_cost > 1.0:
    print("\nWARNING: Estimated cost exceeds budget!")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != "y":
        print("Aborted.")
        exit()

# Execute with cost tracking
print("\nExecuting pipeline...")
result = pipeline.execute()

# Display final costs
print("\nFinal Costs:")
print(f"  Actual total: ${result.costs.total_cost:.4f}")
print(f"  Total tokens: {result.costs.total_tokens:,}")
print(f"  Input tokens: {result.costs.input_tokens:,}")
print(f"  Output tokens: {result.costs.output_tokens:,}")
print(
    f"  Cost per row: ${float(result.costs.total_cost) / result.metrics.total_rows:.6f}"
)

# Compare estimate vs actual
variance = float(result.costs.total_cost - estimate.total_cost)
variance_pct = (variance / float(estimate.total_cost)) * 100
print(f"\nEstimate Variance: ${variance:+.4f} ({variance_pct:+.1f}%)")
