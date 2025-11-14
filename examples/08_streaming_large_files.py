"""
Streaming Execution Example - Memory-efficient processing.

This example demonstrates streaming execution for processing
large datasets that don't fit in memory.
"""

import pandas as pd

from ondine import PipelineBuilder


# For demonstration, create a sample file
# In real use, this would be a large CSV (100K+ rows)
def create_large_sample():
    """Create sample large dataset."""
    print("Creating sample dataset...")

    # Create 1000 rows (in real use, this would be 100K+)
    data = pd.DataFrame(
        {
            "text": [f"Sample text {i} for processing" for i in range(1000)],
        }
    )

    data.to_csv("/tmp/large_dataset.csv", index=False)
    print("✅ Created /tmp/large_dataset.csv with 1,000 rows")


def streaming_example():
    """Process large file with streaming."""

    print("\nBuilding streaming pipeline...")

    # Build pipeline with streaming execution
    pipeline = (
        PipelineBuilder.create()
        .from_csv(
            "/tmp/large_dataset.csv",
            input_columns=["text"],
            output_columns=["summary"],
        )
        .with_prompt("Summarize in 3 words: {text}")
        .with_llm(
            provider="groq",
            model="openai/gpt-oss-120b",
            temperature=0.0,
        )
        .with_streaming(chunk_size=100)  # Process 100 rows at a time
        .build()
    )

    print("Processing in streaming mode (100 rows/chunk)...")
    print("Memory usage will remain constant!\n")

    # Process stream
    total_rows = 0
    for chunk_idx, chunk_result in enumerate(pipeline.execute_stream()):
        rows_in_chunk = len(chunk_result)
        total_rows += rows_in_chunk

        print(
            f"✅ Chunk {chunk_idx + 1}: Processed {rows_in_chunk} rows "
            f"(total: {total_rows})"
        )

        # In real use, you could write each chunk immediately
        # chunk_result.to_csv(f"output_chunk_{chunk_idx}.csv")

    print(f"\n✅ Streaming complete! Processed {total_rows} rows")
    print("💡 Memory usage stayed constant throughout processing")


def comparison_example():
    """Show memory usage comparison."""

    print("\n" + "=" * 60)
    print("Memory Usage Comparison:")
    print("=" * 60)

    print("\n❌ Traditional approach (loads all in memory):")
    print("   df = pd.read_csv('large_file.csv')  # 1GB+ in memory")
    print("   result = process_all(df)           # Another 1GB+")
    print("   Total memory: 2GB+")

    print("\n✅ Streaming approach (constant memory):")
    print("   for chunk in pipeline.execute_stream(chunk_size=1000):")
    print("       process(chunk)  # Only 1000 rows in memory")
    print("   Total memory: ~10MB (constant)")

    print("\n💡 Benefits:")
    print("   - Process unlimited dataset sizes")
    print("   - Constant memory footprint")
    print("   - Can run on smaller machines")
    print("   - Early results available")


if __name__ == "__main__":
    # Create sample data
    create_large_sample()

    # Note: Uncomment to run (requires API key)
    # streaming_example()

    # Show comparison
    comparison_example()

    print("\n📝 To run streaming:")
    print("   export GROQ_API_KEY='your-key'")
    print("   python examples/08_streaming_large_files.py")
