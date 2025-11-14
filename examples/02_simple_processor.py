"""
Simple Processor Example - Minimal configuration.

This example shows the DatasetProcessor convenience wrapper
for simple use cases.
"""

# Create sample CSV (in real use, this would be an existing file)
import pandas as pd

from ondine import DatasetProcessor

sample_data = pd.DataFrame(
    {
        "customer_review": [
            "This product is absolutely amazing! Best purchase ever.",
            "Terrible quality. Broke after one week. Very disappointed.",
            "It's okay. Nothing special but does the job.",
            "Exceeded my expectations! Highly recommend to everyone.",
            "Not worth the price. There are better alternatives.",
        ]
    }
)
sample_data.to_csv("/tmp/reviews.csv", index=False)

# Process with minimal configuration
processor = DatasetProcessor(
    data="/tmp/reviews.csv",
    input_column="customer_review",
    output_column="sentiment",
    prompt="""Analyze the sentiment of this review and respond with ONLY one word:
    - Positive
    - Negative
    - Neutral

    Review: {customer_review}

    Sentiment:""",
    llm_config={
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
)

# Run on sample first
print("Testing on 3 rows...")
sample_result = processor.run_sample(n=3)
print(sample_result)

# Estimate full cost
print(f"\nFull dataset estimated cost: ${processor.estimate_cost():.4f}")

# Run on full dataset
print("\nProcessing full dataset...")
result = processor.run()
print(result)

# Analyze results
print("\nSentiment Distribution:")
print(result["sentiment"].value_counts())
