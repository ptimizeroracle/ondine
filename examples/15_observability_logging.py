"""
Example: Simple observability with LoggingObserver.

Demonstrates how to add logging observability to a pipeline
for debugging and monitoring.
"""

import pandas as pd

from ondine import PipelineBuilder

# Sample data
data = pd.DataFrame(
    {
        "product": [
            "Apple MacBook Pro 16-inch",
            "Sony WH-1000XM4 Headphones",
            "Samsung Galaxy S23",
        ]
    }
)

# Build pipeline with logging observer
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data, input_columns=["product"], output_columns=["brand", "category"]
    )
    .with_prompt(
        """
        Extract the brand and category from this product name.
        Return JSON: {"brand": "...", "category": "..."}

        Product: {product}
        """
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    # Add logging observer for observability
    .with_observer(
        "logging",
        config={
            "log_level": "INFO",
            "include_prompts": True,  # Show prompt previews
            "prompt_preview_length": 200,
        },
    )
    .build()
)

# Execute
print("Executing pipeline with logging observability...")
result = pipeline.execute()

print("\nResults:")
print(result.data)

print(f"\nTotal cost: ${result.costs.total_cost:.4f}")
print(f"Total tokens: {result.costs.total_tokens}")
