"""
MLX Integration Example - Qwen3 on Apple Silicon

This example demonstrates using MLX (Apple's ML framework) for fast,
local LLM inference on M-series chips with the Qwen3 model.

Requirements:
- macOS with Apple Silicon (M1/M2/M3/M4)
- pip install ondine[mlx]
- HuggingFace account (for model downloads)

Benefits:
- 100% Free (no API costs)
- Privacy (data never leaves your machine)
- Fast inference on Apple Silicon
- No internet needed after model download

Setup:
1. Install MLX extra:
   pip install ondine[mlx]

2. Set HuggingFace token (for first-time model download):
   export HUGGING_FACE_HUB_TOKEN="your_token_here"

3. Run this example!
"""

import os

# Check if on Mac
import platform

import pandas as pd

from ondine import PipelineBuilder

if platform.system() != "Darwin":
    print("❌ This example requires macOS with Apple Silicon")
    print("   For Linux/Windows, use examples/05_groq_example.py instead")
    exit(1)

# Check for HuggingFace token
if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    print("⚠️  HuggingFace token not found!")
    print("   Set with: export HUGGING_FACE_HUB_TOKEN='your_token'")
    print("   Get token at: https://huggingface.co/settings/tokens")
    exit(1)

# Create sample data: HR resume screening
data = pd.DataFrame(
    {
        "resume_summary": [
            "5 years Python engineer, ML/AI experience, Stanford CS degree, worked at Google",
            "Recent bootcamp grad, 2 projects in React, seeking junior frontend role",
            "Senior DevOps with 8 years AWS/K8s, led team of 6, startup experience",
            "Marketing manager, 10 years B2B SaaS, grew ARR from $1M to $20M",
            "Data scientist PhD, published 12 papers, expert in NLP and transformers",
        ]
    }
)

print("\n" + "=" * 60)
print("🍎 MLX + QWEN3 LOCAL INFERENCE EXAMPLE")
print("=" * 60)

# Build pipeline with MLX
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["resume_summary"],
        output_columns=["role_fit"],
    )
    .with_prompt(
        """You are an HR screening assistant. Analyze this resume and recommend the best role fit.

Roles: Software Engineer, Data Scientist, DevOps Engineer, Marketing Manager, Product Manager

Resume: {resume_summary}

Recommended Role:"""
    )
    .with_llm(
        provider="mlx",
        model="mlx-community/Qwen3-1.7B-4bit",  # Fast, small model
        max_tokens=20,
        input_cost_per_1k_tokens=0.0,  # Free!
        output_cost_per_1k_tokens=0.0,
    )
    .with_batch_size(5)
    .with_concurrency(1)  # MLX works best with concurrency=1
    .build()
)

print("\n📊 Pipeline Configuration:")
print("   Provider: MLX (Apple Silicon)")
print("   Model: mlx-community/Qwen3-1.7B-4bit")
print(f"   Rows: {len(data)}")
print("   Cost: $0.00 (local model)")

# First run will download model (~1-2GB)
print("\n⏳ Processing (first run downloads model)...")
print("   This may take 1-2 minutes for initial download")
print("   Subsequent runs will be instant!\n")

result = pipeline.execute()

# Display results
print("\n" + "=" * 60)
print("✅ RESULTS")
print("=" * 60)

for idx, row in result.data.iterrows():
    print(f"\n{idx + 1}. Resume: {row['resume_summary'][:60]}...")
    print(f"   Recommended Role: {row['role_fit']}")

# Show performance metrics
print("\n" + "=" * 60)
print("📈 PERFORMANCE METRICS")
print("=" * 60)
print(f"Rows processed: {result.metrics.processed_rows}")
print(f"Total duration: {result.metrics.total_duration_seconds:.2f}s")
print(f"Throughput: {result.metrics.rows_per_second:.2f} rows/sec")
print(
    f"Avg time per row: {result.metrics.total_duration_seconds / result.metrics.total_rows:.2f}s"
)
print(f"\n💰 Cost: ${result.costs.total_cost} (FREE!)")
print(f"🎯 Total tokens: {result.costs.total_tokens:,}")

print("\n" + "=" * 60)
print("💡 TIPS")
print("=" * 60)
print("1. Model is cached - subsequent runs are instant!")
print("2. Try different Qwen3 models:")
print("   - mlx-community/Qwen3-1.7B-4bit (fast, small)")
print("   - mlx-community/Qwen3-7B-4bit (slower, better quality)")
print("3. Adjust max_tokens for longer/shorter responses")
print("4. Use concurrency=1 for MLX (optimal for M-series)")
print("=" * 60 + "\n")
