"""Sentiment analysis on a CSV, with cost tracking — a complete, runnable example.

This is the "hello world" of a real Ondine job: take a column of free text,
ask an LLM to label each row, and see exactly what it cost. It uses
``QuickPipeline``, which infers the input column from the prompt's ``{text}``
placeholder and picks sensible batching/concurrency defaults for you.

Run it:

    export OPENAI_API_KEY="..."        # or any provider's key
    uv run python examples/sentiment_analysis_with_costs.py

No key handy? The script still prints the *estimated* cost (a local token
count — no API call, no spend) and then stops before executing.
"""

import os

from ondine import QuickPipeline

# A tiny inline dataset so the script is self-contained. In practice you would
# point QuickPipeline at a CSV/Excel/Parquet path instead — the rest is identical.
REVIEWS = [
    "The product arrived a day early and works exactly as described.",
    "Terrible experience — the item broke within an hour and support ignored me.",
    "It was fine. Nothing special, but it does the job.",
    "Absolutely love it, best purchase I've made all year!",
    "Shipping took three weeks longer than promised, very frustrating.",
]


def main() -> None:
    import pandas as pd

    frame = pd.DataFrame({"text": REVIEWS})

    # One call builds the whole pipeline. The input column ("text") is read
    # straight out of the prompt's {text} placeholder; "sentiment" is where the
    # answer lands. Swap the model/provider for any Ondine supports.
    pipeline = QuickPipeline.create(
        data=frame,
        prompt=(
            "Classify the sentiment of this review as exactly one word — "
            "positive, negative, or neutral:\n\n{text}"
        ),
        output_columns="sentiment",
        model="gpt-4o-mini",
        max_tokens=5,  # a one-word answer never needs more
    )

    # Estimate first. This counts tokens locally — no API call, no charge — so
    # you always know the bill before committing to it.
    estimate = pipeline.estimate_cost()
    print("Estimated before running:")
    print(f"  rows:   {estimate.rows}")
    print(f"  tokens: {estimate.total_tokens:,}")
    print(f"  cost:   ${estimate.total_cost:.6f}\n")

    # Executing spends real money, so require a key rather than failing mid-run.
    if not any(key.endswith("_API_KEY") and os.environ[key] for key in os.environ):
        print("No *_API_KEY set — set one (e.g. OPENAI_API_KEY) to run for real.")
        return

    result = pipeline.execute()

    # result.to_pandas() is your input frame plus the new "sentiment" column.
    print("Results:")
    print(result.to_pandas()[["text", "sentiment"]].to_string(index=False))

    # And here is what it actually cost — measured, not estimated.
    print("\nActual cost:")
    print(f"  rows processed: {result.metrics.processed_rows}")
    print(f"  tokens:         {result.costs.total_tokens:,}")
    print(f"  total:          ${result.costs.total_cost:.6f}")
    if result.costs.rows:
        print(f"  per row:        ${result.costs.total_cost / result.costs.rows:.6f}")


if __name__ == "__main__":
    main()
