"""
Multi-Stage Classification Pipeline with Ondine + Groq
=======================================================

This example demonstrates:
1. Multi-stage classification (4 stages, 4 new columns)
2. Scalability features (async execution, streaming, rate limiting)
3. Cost control and budget enforcement
4. Checkpoint/resume for large datasets (5.4M rows)
5. Groq provider for fast, cost-effective inference

Dataset: titles_to_categories.csv (5.4M product titles)

Classification Stages:
1. Primary Category (broad classification)
2. Subcategory (detailed classification)
3. Quality Score (product title quality: 1-5)
4. Target Audience (who is this product for?)

Author: Multi-Agent Framework
Date: 2025-11-09
"""

import os
from decimal import Decimal

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.specifications import LLMProviderPresets

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
INPUT_FILE = "titles_to_categories.csv"
OUTPUT_FILE = "titles_classified_multi_stage.csv"

# Processing configuration
SAMPLE_SIZE = 100  # Start with 100 rows for testing (change to None for full dataset)
CONCURRENCY = 10  # Parallel LLM calls (Groq can handle high concurrency)
BATCH_SIZE = 1000  # Checkpoint every 1000 rows

# Budget control
MAX_BUDGET = Decimal("5.00")  # $5 maximum budget

# Rate limiting (Groq free tier: ~30 requests/minute, paid: much higher)
# Adjust based on your Groq plan
REQUESTS_PER_MINUTE = 25  # Conservative for free tier

# ============================================================================
# STAGE 1: PRIMARY CATEGORY CLASSIFICATION
# ============================================================================


def create_primary_category_pipeline():
    """
    Stage 1: Classify into broad primary categories

    Input: title
    Output: primary_category
    """
    import pandas as pd

    # Load and sample data
    df = pd.read_csv(INPUT_FILE)
    if SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE)

    return (
        PipelineBuilder.create()
        .from_dataframe(
            df, input_columns=["title"], output_columns=["primary_category"]
        )
        .with_prompt("""
Classify this product title into ONE primary category.

Product Title: {title}

Choose from these categories:
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

Respond with ONLY the category name, nothing else.
""")
        .with_llm_spec(LLMProviderPresets.GROQ_LLAMA_70B)
        .with_concurrency(CONCURRENCY)
        .with_rate_limit(REQUESTS_PER_MINUTE)
        .with_checkpoint_interval(BATCH_SIZE)
        .with_max_budget(float(MAX_BUDGET))
        .build()
    )


# ============================================================================
# STAGE 2: SUBCATEGORY CLASSIFICATION
# ============================================================================


def create_subcategory_pipeline():
    """
    Stage 2: Classify into detailed subcategories

    Input: title, primary_category
    Output: subcategory
    """
    import pandas as pd

    # Load data from stage 1
    df = pd.read_csv("titles_classified_stage1.csv")

    return (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["title", "primary_category"],
            output_columns=["subcategory"],
        )
        .with_prompt("""
Given this product title and its primary category, classify it into a specific subcategory.

Product Title: {title}
Primary Category: {primary_category}

Provide a specific subcategory (2-4 words) that describes this product more precisely.

Examples:
- "Power Tools" for drills
- "Kitchen Appliances" for blenders
- "Men's Footwear" for shoes

Respond with ONLY the subcategory name, nothing else.
""")
        .with_llm_spec(LLMProviderPresets.GROQ_LLAMA_70B)
        .with_concurrency(CONCURRENCY)
        .with_rate_limit(REQUESTS_PER_MINUTE)
        .with_checkpoint_interval(BATCH_SIZE)
        .with_max_budget(float(MAX_BUDGET))
        .build()
    )


# ============================================================================
# STAGE 3: QUALITY SCORE
# ============================================================================


def create_quality_score_pipeline():
    """
    Stage 3: Assess product title quality

    Input: title
    Output: quality_score (1-5)
    """
    import pandas as pd

    # Load data from stage 2
    df = pd.read_csv("titles_classified_stage2.csv")

    return (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["title"], output_columns=["quality_score"])
        .with_prompt("""
Rate the quality of this product title on a scale of 1-5.

Product Title: {title}

Rating Criteria:
- 5: Clear, descriptive, professional, includes key details
- 4: Good, mostly clear with some details
- 3: Average, basic description
- 2: Poor, vague or confusing
- 1: Very poor, unclear or misleading

Respond with ONLY a number (1-5), nothing else.
""")
        .with_llm_spec(LLMProviderPresets.GROQ_LLAMA_70B)
        .with_concurrency(CONCURRENCY)
        .with_rate_limit(REQUESTS_PER_MINUTE)
        .with_checkpoint_interval(BATCH_SIZE)
        .with_max_budget(float(MAX_BUDGET))
        .build()
    )


# ============================================================================
# STAGE 4: TARGET AUDIENCE
# ============================================================================


def create_target_audience_pipeline():
    """
    Stage 4: Identify target audience

    Input: title, primary_category
    Output: target_audience
    """
    import pandas as pd

    # Load data from stage 3
    df = pd.read_csv("titles_classified_stage3.csv")

    return (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["title", "primary_category"],
            output_columns=["target_audience"],
        )
        .with_prompt("""
Identify the primary target audience for this product.

Product Title: {title}
Category: {primary_category}

Choose ONE target audience:
- Professionals
- Home Users
- Parents/Families
- Students
- Athletes/Fitness Enthusiasts
- Hobbyists
- Seniors
- General Consumers

Respond with ONLY the audience name, nothing else.
""")
        .with_llm_spec(LLMProviderPresets.GROQ_LLAMA_70B)
        .with_concurrency(CONCURRENCY)
        .with_rate_limit(REQUESTS_PER_MINUTE)
        .with_checkpoint_interval(BATCH_SIZE)
        .with_max_budget(float(MAX_BUDGET))
        .build()
    )


# ============================================================================
# ORCHESTRATION: RUN ALL STAGES
# ============================================================================


def run_multi_stage_classification():
    """
    Execute all 4 classification stages sequentially.

    Each stage:
    1. Reads output from previous stage (or original CSV for stage 1)
    2. Adds a new classification column
    3. Saves intermediate results
    4. Tracks cost and progress
    """

    print("=" * 80)
    print("MULTI-STAGE CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {INPUT_FILE}")
    print(f"Sample Size: {SAMPLE_SIZE or 'Full dataset (5.4M rows)'}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Rate Limit: {REQUESTS_PER_MINUTE} requests/minute")
    print(f"Budget: ${MAX_BUDGET}")
    print("=" * 80)
    print()

    total_cost = Decimal("0.0")

    # -------------------------------------------------------------------------
    # STAGE 1: Primary Category
    # -------------------------------------------------------------------------
    print("🚀 STAGE 1: Primary Category Classification")
    print("-" * 80)

    stage1 = create_primary_category_pipeline()
    result1 = stage1.execute()

    # Save intermediate results
    result1.data.to_csv("titles_classified_stage1.csv", index=False)

    print("✅ Stage 1 Complete")
    print(f"   Rows Processed: {result1.rows_processed}")
    print(f"   Cost: ${result1.cost.total_cost}")
    print(f"   Time: {result1.execution_time:.2f}s")
    print("   Output: titles_classified_stage1.csv")
    print()

    total_cost += result1.cost.total_cost

    # -------------------------------------------------------------------------
    # STAGE 2: Subcategory
    # -------------------------------------------------------------------------
    print("🚀 STAGE 2: Subcategory Classification")
    print("-" * 80)

    stage2 = create_subcategory_pipeline()
    result2 = stage2.execute()

    # Save intermediate results
    result2.data.to_csv("titles_classified_stage2.csv", index=False)

    print("✅ Stage 2 Complete")
    print(f"   Rows Processed: {result2.rows_processed}")
    print(f"   Cost: ${result2.cost.total_cost}")
    print(f"   Time: {result2.execution_time:.2f}s")
    print("   Output: titles_classified_stage2.csv")
    print()

    total_cost += result2.cost.total_cost

    # -------------------------------------------------------------------------
    # STAGE 3: Quality Score
    # -------------------------------------------------------------------------
    print("🚀 STAGE 3: Quality Score Assessment")
    print("-" * 80)

    stage3 = create_quality_score_pipeline()
    result3 = stage3.execute()

    # Save intermediate results
    result3.data.to_csv("titles_classified_stage3.csv", index=False)

    print("✅ Stage 3 Complete")
    print(f"   Rows Processed: {result3.rows_processed}")
    print(f"   Cost: ${result3.cost.total_cost}")
    print(f"   Time: {result3.execution_time:.2f}s")
    print("   Output: titles_classified_stage3.csv")
    print()

    total_cost += result3.cost.total_cost

    # -------------------------------------------------------------------------
    # STAGE 4: Target Audience
    # -------------------------------------------------------------------------
    print("🚀 STAGE 4: Target Audience Identification")
    print("-" * 80)

    stage4 = create_target_audience_pipeline()
    result4 = stage4.execute()

    # Save final results
    result4.data.to_csv(OUTPUT_FILE, index=False)

    print("✅ Stage 4 Complete")
    print(f"   Rows Processed: {result4.rows_processed}")
    print(f"   Cost: ${result4.cost.total_cost}")
    print(f"   Time: {result4.execution_time:.2f}s")
    print(f"   Output: {OUTPUT_FILE}")
    print()

    total_cost += result4.cost.total_cost

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("🎉 ALL STAGES COMPLETE!")
    print("=" * 80)
    print(f"Total Rows Processed: {result4.rows_processed}")
    print(f"Total Cost: ${total_cost}")
    print(
        f"Total Time: {sum([r.execution_time for r in [result1, result2, result3, result4]]):.2f}s"
    )
    print(f"Final Output: {OUTPUT_FILE}")
    print()
    print("New Columns Added:")
    print("  1. primary_category")
    print("  2. subcategory")
    print("  3. quality_score")
    print("  4. target_audience")
    print("=" * 80)

    # Show sample results
    print("\n📊 Sample Results (first 5 rows):")
    print("-" * 80)
    print(result4.data.head())
    print()

    return result4


# ============================================================================
# ALTERNATIVE: PIPELINE COMPOSITION (All stages in one pipeline)
# ============================================================================


def create_composed_pipeline():
    """
    Alternative approach: Use PipelineComposer to chain all stages.

    This is more elegant but requires all stages to be defined upfront.
    The sequential approach above is more flexible for debugging.
    """
    from ondine.api.pipeline_composer import PipelineComposer

    # Create individual stage pipelines
    stage1 = create_primary_category_pipeline()
    stage2 = create_subcategory_pipeline()
    stage3 = create_quality_score_pipeline()
    stage4 = create_target_audience_pipeline()

    # Compose them
    return (
        PipelineComposer()
        .add_pipeline("primary_category", stage1)
        .add_pipeline("subcategory", stage2)
        .add_pipeline("quality_score", stage3)
        .add_pipeline("target_audience", stage4)
        .build()
    )


# ============================================================================
# COST ESTIMATION (Run before full processing)
# ============================================================================


def estimate_cost_for_full_dataset():
    """
    Estimate cost for processing the full 5.4M row dataset.

    Uses Groq's pricing:
    - Input: $0.00059 per 1K tokens
    - Output: $0.00079 per 1K tokens
    """

    print("=" * 80)
    print("COST ESTIMATION FOR FULL DATASET")
    print("=" * 80)

    # Assumptions
    total_rows = 5_389_902
    avg_tokens_per_prompt = 150  # Estimated
    avg_tokens_per_response = 20  # Short responses
    stages = 4

    # Groq pricing
    input_cost_per_1k = Decimal("0.00059")
    output_cost_per_1k = Decimal("0.00079")

    # Calculate
    total_input_tokens = total_rows * avg_tokens_per_prompt * stages
    total_output_tokens = total_rows * avg_tokens_per_response * stages

    input_cost = (Decimal(total_input_tokens) / 1000) * input_cost_per_1k
    output_cost = (Decimal(total_output_tokens) / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print(f"Total Rows: {total_rows:,}")
    print(f"Stages: {stages}")
    print(f"Total LLM Calls: {total_rows * stages:,}")
    print()
    print(f"Estimated Input Tokens: {total_input_tokens:,}")
    print(f"Estimated Output Tokens: {total_output_tokens:,}")
    print()
    print(f"Estimated Input Cost: ${input_cost:.2f}")
    print(f"Estimated Output Cost: ${output_cost:.2f}")
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print()
    print("⚠️  This is an estimate. Actual cost may vary.")
    print("=" * 80)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY environment variable not set!")
        print()
        print("Please set your Groq API key:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        print()
        print("Get a free API key at: https://console.groq.com/")
        exit(1)

    # Show cost estimation first
    estimate_cost_for_full_dataset()

    # Run the multi-stage classification
    result = run_multi_stage_classification()

    print("\n✅ Pipeline execution complete!")
    print(f"📁 Final output saved to: {OUTPUT_FILE}")
