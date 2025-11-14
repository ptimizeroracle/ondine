"""
Structured Output Example - JSON parsing.

This example demonstrates extracting structured data from LLM
responses using JSON output.
"""

import pandas as pd

from ondine import PipelineBuilder

# Sample product descriptions
data = pd.DataFrame(
    {
        "description": [
            "Apple iPhone 13 Pro 256GB Graphite - Brand new, unlocked, with warranty",
            "Used Samsung Galaxy S21 128GB in good condition, minor scratches",
            "MacBook Air M1 8GB 256GB SSD - Like new, barely used, original box",
        ]
    }
)

# Build pipeline with JSON output
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["description"],
        output_columns=["brand", "model", "storage", "condition"],
    )
    .with_prompt(
        """Extract product information from this description.
        Return ONLY a JSON object with these fields:
        - brand: manufacturer name
        - model: product model
        - storage: storage capacity (e.g., "256GB")
        - condition: "new", "used", or "refurbished"

        Description: {description}

        JSON:"""
    )
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    .build()
)

# Execute
print("Extracting structured data...")
result = pipeline.execute()

# Display results
print("\nExtracted Information:")
print(result.data)

# Export to CSV
output_file = "/tmp/extracted_products.csv"
result.data.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
