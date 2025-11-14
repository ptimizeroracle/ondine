"""
Groq Example - Fast inference with Groq's API.

This example demonstrates using Groq as the LLM provider.
Groq offers very fast inference speeds.
"""

# Load var env for Groq API key
import os

import pandas as pd
from dotenv import load_dotenv

from ondine import PipelineBuilder

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Create sample data
data = pd.DataFrame(
    {
        "question": [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "What year did World War II end?",
        ]
    }
)

# Build pipeline with Groq
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["question"],
        output_columns=["answer"],
    )
    .with_prompt("Answer this question concisely: {question}\n\nAnswer:")
    .with_llm(
        provider="groq",
        model="gpt-oss-20b",  # Current Groq model
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    .with_batch_size(5)
    .with_concurrency(3)
    .build()
)

# Estimate cost
print("Estimating cost...")
estimate = pipeline.estimate_cost()
print(f"Estimated cost: ${estimate.total_cost:.4f}")
print(f"Estimated tokens: {estimate.total_tokens:,}")

# Execute pipeline
print("\nProcessing with Groq (fast inference)...")
result = pipeline.execute()

# Display results
print("\nResults:")
for idx, row in result.data.iterrows():
    print(f"Q: {data.loc[idx, 'question']}")
    print(f"A: {row['answer']}\n")

# Show metrics
print("Metrics:")
print(f"  Processed: {result.metrics.processed_rows} rows")
print(f"  Duration: {result.metrics.total_duration_seconds:.2f}s")
print(f"  Throughput: {result.metrics.rows_per_second:.2f} rows/sec")
print(f"  Total cost: ${result.costs.total_cost:.4f}")
print(
    f"  Cost per row: ${float(result.costs.total_cost) / result.metrics.total_rows:.6f}"
)
