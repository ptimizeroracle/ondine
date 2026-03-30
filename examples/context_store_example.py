"""
Context Store Example - Anti-hallucination verification pipeline.

This example shows how to build a pipeline with the Context Store features:
grounding verification, contradiction detection, and confidence scoring.
These features help detect when LLM responses are unsupported by the input
text, contradict each other across rows, or have low confidence.
"""

import pandas as pd

from ondine import PipelineBuilder

# ---------------------------------------------------------------------------
# 1. Sample data: product classification with intentional duplicates
#    to demonstrate contradiction detection.
# ---------------------------------------------------------------------------

data = pd.DataFrame(
    {
        "product_id": [
            "SKU-001",
            "SKU-002",
            "SKU-003",
            "SKU-001",  # duplicate — should get same category as first SKU-001
            "SKU-004",
        ],
        "product_name": [
            "Organic Fuji Apples 3lb Bag",
            "Kirkland Signature Almond Butter 27oz",
            "Blue Diamond Roasted Almonds 16oz",
            "Organic Fuji Apples 3lb Bag",
            "Dawn Ultra Dish Soap 28oz",
        ],
    }
)

print("Input data:")
print(data.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 2. Build pipeline with full anti-hallucination stack.
# ---------------------------------------------------------------------------

pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_id", "product_name"],
        output_columns=["category"],
    )
    .with_prompt(
        "Classify this grocery product into exactly ONE category.\n"
        "Choose from: Produce, Pantry, Snacks, Household.\n"
        "Reply with ONLY the category name.\n\n"
        "Product: {product_name}\n\n"
        "Category:"
    )
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    # --- Context Store: evidence-backed verification ---
    .with_context_store()  # auto-selects Rust or in-memory backend
    .with_grounding(threshold=0.3, action="flag")
    .with_contradiction_detection(
        key_columns=["product_id"],
        tolerance=0.5,
    )
    .with_confidence_scoring(scoring_mode="sigmoid")
    .build()
)

# ---------------------------------------------------------------------------
# 3. Cost estimation (runs before any LLM call).
# ---------------------------------------------------------------------------

print("Estimating cost...")
estimate = pipeline.estimate_cost()
print(f"  Estimated total: ${estimate.total_cost:.4f}")
print(f"  Estimated tokens: {estimate.total_tokens:,}")
print()

# ---------------------------------------------------------------------------
# 4. Execute and inspect results.
# ---------------------------------------------------------------------------

print("Processing data...")
result = pipeline.execute()

print("\nResults:")
# Show the original columns plus the quality columns added by the
# context-store features.
display_cols = [
    "product_id",
    "product_name",
    "category",
]

# Grounding, contradiction, and confidence columns are added automatically
# when the corresponding features are enabled.
quality_cols = [
    col
    for col in [
        "grounding_score",
        "grounding_flag",
        "contradiction_flag",
        "confidence_score",
    ]
    if col in result.data.columns
]

print(result.data[display_cols + quality_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Interpret the quality columns.
# ---------------------------------------------------------------------------

print("\n--- Quality Column Guide ---")
print("grounding_score    : 0-1 similarity between LLM output and source text")
print("grounding_flag     : True when grounding_score < threshold (0.3)")
print("contradiction_flag : True when rows with the same product_id got")
print("                     different categories")
print("confidence_score   : 0-1 composite score (sigmoid mode uses grounding")
print("                     score only; linear mode blends grounding + support)")

# ---------------------------------------------------------------------------
# 6. Metrics summary.
# ---------------------------------------------------------------------------

print(f"\nProcessed rows: {result.metrics.processed_rows}")
print(f"Duration: {result.metrics.total_duration_seconds:.2f}s")
print(f"Total cost: ${result.costs.total_cost:.4f}")
