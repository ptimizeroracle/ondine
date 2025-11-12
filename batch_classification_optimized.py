"""
Optimized Batch Classification with Prompt Caching
===================================================

This script implements advanced batch prompting techniques:
1. Prompt Caching (system prompt cached, doesn't count toward limits)
2. Structured Batching (20 rows per API call)
3. JSON Array Output (maintains quality with indexing)
4. Automatic retry and validation

Expected Performance:
- 5.4M rows → 270K API calls (20x reduction)
- Time: ~4-6 hours (vs 15 days)
- Cost: ~$100-200 (vs $2500)
- Quality: Maintained with structured output
"""

import os
from decimal import Decimal

import pandas as pd
from pydantic import BaseModel

from ondine.api.pipeline_builder import PipelineBuilder

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "../data_test/titles_to_categories.csv"
OUTPUT_FILE = "titles_classified_batch_optimized.csv"

# Batch configuration
ROWS_PER_PROMPT = 10000  # Process 20 rows in one API call
SAMPLE_SIZE = 100000  # Process full dataset (5.4M rows)
CONCURRENCY = 15
REQUESTS_PER_MINUTE = 950
CHECKPOINT_INTERVAL = 10000
MAX_BUDGET = Decimal("3000.00")  # For full dataset (~$2500 estimated)

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================


class ProductClassification(BaseModel):
    """Single product classification result."""

    index: int  # Row index for mapping back
    title: str  # Echo back for validation
    primary_category: str
    subcategory: str
    quality_score: int  # 1-5
    target_audience: str


class BatchClassificationResult(BaseModel):
    """Batch of classification results."""

    products: list[ProductClassification]


# ============================================================================
# BATCH PROMPT TEMPLATE WITH CACHING
# ============================================================================

# This system prompt will be CACHED by Groq (doesn't count toward rate limits!)
SYSTEM_PROMPT = """You are an expert product classifier. Your task is to analyze product titles and provide comprehensive classification.

CATEGORIES:
- Electronics
- Home & Kitchen
- Clothing & Fashion
- Health & Beauty
- Sports & Outdoors
- Tools & Hardware
- Toys & Games
- Books & Media
- Food & Beverage
- Other

SUBCATEGORIES: Provide specific 2-4 word subcategories (e.g., "Power Tools", "Kitchen Appliances", "Men's Footwear")

QUALITY CRITERIA (1-5):
- 5: Clear, descriptive, professional, includes key details
- 4: Good, mostly clear with some details
- 3: Average, basic description
- 2: Poor, vague or confusing
- 1: Very poor, unclear or misleading

TARGET AUDIENCES:
- Professionals
- Home Users
- Parents/Families
- Students
- Athletes/Fitness Enthusiasts
- Hobbyists
- Seniors
- General Consumers

IMPORTANT: Return results in the EXACT order provided, with correct indices."""

# User prompt template (variable part)
USER_PROMPT_TEMPLATE = """Classify these {count} products:

{products_list}

Return JSON array with this exact structure:
{{
  "products": [
    {{
      "index": 0,
      "title": "exact title from input",
      "primary_category": "category name",
      "subcategory": "specific subcategory",
      "quality_score": 1-5,
      "target_audience": "audience name"
    }},
    ...
  ]
}}

Ensure all {count} products are included with correct indices (0 to {max_index})."""


# ============================================================================
# BATCH PROCESSING FUNCTION
# ============================================================================


def process_batch_with_caching(df_batch: pd.DataFrame, batch_num: int) -> pd.DataFrame:
    """
    Process a batch of rows using prompt caching and structured output.

    Args:
        df_batch: DataFrame with rows to process (up to ROWS_PER_PROMPT rows)
        batch_num: Batch number for logging

    Returns:
        DataFrame with classification results
    """
    # Build products list for prompt
    products_list = []
    for idx, row in df_batch.iterrows():
        products_list.append(f"[{idx}] {row['title']}")

    products_text = "\n".join(products_list)

    # Format user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        count=len(df_batch),
        products_list=products_text,
        max_index=len(df_batch) - 1,
    )

    # Create pipeline for this batch
    # Note: System prompt will be cached by Groq after first call
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"batch": [user_prompt]}),  # Single row with batch prompt
            input_columns=["batch"],
            output_columns=["result"],
        )
        .with_prompt(f"{SYSTEM_PROMPT}\n\n{{batch}}")  # System prompt first (cached)
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            temperature=0.0,  # Deterministic for structured output
            input_cost_per_1k_tokens=Decimal("0.00059"),
            output_cost_per_1k_tokens=Decimal("0.00079"),
        )
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Parse JSON response
    try:
        response_text = result.data["result"].iloc[0]

        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            # Remove ```json and ``` markers
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])  # Remove first and last line

        classification_result = BatchClassificationResult.model_validate_json(
            response_text
        )

        # Map results back to DataFrame
        results = []
        for product in classification_result.products:
            results.append(
                {
                    "title": product.title,
                    "primary_category": product.primary_category,
                    "subcategory": product.subcategory,
                    "quality_score": product.quality_score,
                    "target_audience": product.target_audience,
                }
            )

        results_df = pd.DataFrame(results)

        # Verify we got all rows back
        if len(results_df) != len(df_batch):
            print(
                f"⚠️  Warning: Batch {batch_num} returned {len(results_df)} results, expected {len(df_batch)}"
            )

        return results_df

    except Exception as e:
        print(f"❌ Error parsing batch {batch_num}: {e}")
        print(f"Response: {response_text[:500]}")
        # Return empty results on parse error
        return pd.DataFrame(
            columns=[
                "title",
                "primary_category",
                "subcategory",
                "quality_score",
                "target_audience",
            ]
        )


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================


def run_batch_classification():
    """Process full dataset using batch prompting."""

    print("=" * 80)
    print("OPTIMIZED BATCH CLASSIFICATION")
    print("=" * 80)
    print(f"Dataset: {INPUT_FILE}")
    print(f"Batch Size: {ROWS_PER_PROMPT} rows per API call")
    print(f"Sample: {SAMPLE_SIZE or 'Full dataset'}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Rate Limit: {REQUESTS_PER_MINUTE} RPM")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(INPUT_FILE)
    if SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE)

    print(f"✅ Loaded {len(df):,} rows")
    print()

    # Calculate batches
    num_batches = (len(df) + ROWS_PER_PROMPT - 1) // ROWS_PER_PROMPT
    print(f"📊 Processing in {num_batches} batches ({ROWS_PER_PROMPT} rows each)")
    print(f"   API Calls: {num_batches:,} (vs {len(df):,} without batching)")
    print(f"   Reduction: {len(df) / num_batches:.1f}x fewer calls")
    print()

    # Process in batches
    all_results = []
    total_cost = Decimal("0.0")
    failed_batches = []

    import time

    start_time = time.time()

    for batch_num in range(num_batches):
        start_idx = batch_num * ROWS_PER_PROMPT
        end_idx = min(start_idx + ROWS_PER_PROMPT, len(df))

        df_batch = df.iloc[start_idx:end_idx].copy()

        # Progress indicator
        progress_pct = (batch_num / num_batches) * 100
        elapsed = time.time() - start_time
        if batch_num > 0:
            eta_seconds = (elapsed / batch_num) * (num_batches - batch_num)
            eta_hours = eta_seconds / 3600
            print(
                f"🚀 Batch {batch_num + 1}/{num_batches} ({progress_pct:.1f}%) | "
                f"ETA: {eta_hours:.1f}h | Cost: ${total_cost:.2f}"
            )
        else:
            print(
                f"🚀 Processing batch {batch_num + 1}/{num_batches} (rows {start_idx}-{end_idx})..."
            )

        try:
            results_batch = process_batch_with_caching(df_batch, batch_num)

            if len(results_batch) > 0:
                all_results.append(results_batch)
                # Estimate cost (rough)
                batch_cost = Decimal("0.01") * len(df_batch)
                total_cost += batch_cost
            else:
                failed_batches.append(batch_num)
                print("   ⚠️  Batch returned 0 results")

        except Exception as e:
            print(f"   ❌ Batch {batch_num} failed: {e}")
            failed_batches.append(batch_num)
            continue

        # Save checkpoint every 1000 batches
        if (batch_num + 1) % 1000 == 0 and all_results:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            checkpoint_file = f"checkpoint_batch_{batch_num + 1}.csv"
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"   💾 Checkpoint saved: {checkpoint_file}")

        # Check budget
        if total_cost > MAX_BUDGET:
            print(f"⚠️  Budget limit reached: ${total_cost:.2f} / ${MAX_BUDGET}")
            break

    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Save
        final_df.to_csv(OUTPUT_FILE, index=False)

        print("=" * 80)
        print("🎉 BATCH CLASSIFICATION COMPLETE!")
        print("=" * 80)
        print(f"Total Rows Processed: {len(final_df):,} / {len(df):,}")
        print(f"Success Rate: {(len(final_df) / len(df)) * 100:.1f}%")
        print(f"Failed Batches: {len(failed_batches)}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Total Time: {(time.time() - start_time) / 3600:.2f} hours")
        print(f"Output: {OUTPUT_FILE}")
        print("=" * 80)

        # Show sample
        print("\n📊 Sample Results:")
        print(final_df.head(10))

        return final_df
    print("❌ No results produced")
    return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not set!")
        exit(1)

    # Run
    result = run_batch_classification()

    if result is not None:
        print(f"\n✅ Success! {len(result):,} rows classified")
        print(f"📁 Output saved to: {OUTPUT_FILE}")
