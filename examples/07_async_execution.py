"""
Async Execution Example - Non-blocking processing.

This example demonstrates async/await support for integration
with FastAPI, aiohttp, and other async frameworks.
"""

import asyncio

import pandas as pd

from ondine import PipelineBuilder

# Sample data
data = pd.DataFrame(
    {
        "question": [
            "What is Python?",
            "What is async/await?",
            "What is FastAPI?",
        ]
    }
)


async def main():
    """Async main function."""

    print("Building async pipeline...")

    # Build pipeline with async execution
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["question"],
            output_columns=["answer"],
        )
        .with_prompt("Answer concisely: {question}")
        .with_llm(
            provider="groq",
            model="openai/gpt-oss-120b",
            temperature=0.0,
        )
        .with_async_execution(max_concurrency=3)  # Async executor
        .build()
    )

    print("Executing asynchronously...")

    # Execute with async/await (non-blocking)
    result = await pipeline.execute_async()

    print("\n✅ Async execution complete!")
    print(f"Rows: {result.metrics.total_rows}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Cost: ${result.costs.total_cost}")

    print("\nResults:")
    for idx, row in result.data.iterrows():
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}\n")


# FastAPI Integration Example
async def fastapi_example():
    """
    Example of using async pipeline in FastAPI.

    Usage:
        from fastapi import FastAPI
        from ondine import PipelineBuilder

        app = FastAPI()

        @app.post("/process")
        async def process_data(data: dict):
            df = pd.DataFrame([data])

            pipeline = (
                PipelineBuilder.create()
                .from_dataframe(df, ...)
                .with_llm(...)
                .with_async_execution()  # Non-blocking!
                .build()
            )

            result = await pipeline.execute_async()
            return result.data.to_dict()
    """
    print("\n📝 FastAPI integration example above")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())

    # Show FastAPI example
    asyncio.run(fastapi_example())
