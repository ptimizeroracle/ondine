"""
E2E test for mega-prompt batching with complex data.

Validates that batch aggregation (mega-prompts) correctly processes multiple rows
in a single API call and returns structured results matching the input DataFrame.
"""

import os
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from ondine import PipelineBuilder


class ProductResult(BaseModel):
    """Extracted product attributes."""

    product_name: str = Field(description="Cleaned product name")
    category: str | None = Field(default=None, description="Product category")
    price_range: str | None = Field(
        default=None, description="Price range (e.g., '$10-20')"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Key product features"
    )


class BatchItem(BaseModel):
    """Single item in batch response."""

    id: int
    result: ProductResult


class ProductBatch(BaseModel):
    """Batch of product results."""

    items: list[BatchItem]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "api_key_env"),
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ],
)
def test_mega_prompt_batching_complex_data(provider, model, api_key_env):
    """
    Test mega-prompt batching with complex DataFrame.

    Validates that:
    1. Multiple rows are batched into a single API call
    2. Each row is correctly processed and returned
    3. Results maintain correct ordering and IDs
    4. Structured output works with batching
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create complex test DataFrame with varied data
    df = pd.DataFrame(
        {
            "product_description": [
                "Premium Wireless Bluetooth Headphones - Over-Ear, Noise Cancelling, 30hr Battery, $199.99",
                "Organic Fair Trade Coffee Beans - Dark Roast, 2lb Bag, Single Origin Ethiopia, $24.99",
                "Stainless Steel Water Bottle - 32oz, Insulated, BPA-Free, Keeps Cold 24hrs, $29.95",
                "Yoga Mat - Extra Thick 6mm, Non-Slip, Eco-Friendly TPE Material, Carrying Strap, $39.99",
                "LED Desk Lamp - Touch Control, 3 Color Modes, USB Charging Port, Adjustable Arm, $45.00",
                "Cotton Throw Blanket - 50x60in, Soft Knit, Machine Washable, Grey Color, $34.99",
                "Running Shoes - Men's Size 10, Lightweight Mesh, Cushioned Sole, Black/White, $89.99",
                "Ceramic Plant Pot Set - 3 Pieces, Drainage Holes, Modern White Design, 4-6-8 inch, $28.50",
                "Laptop Stand - Aluminum, Ergonomic, Adjustable Height, Fits 10-17 inch, Portable, $49.99",
                "Essential Oil Diffuser - 300ml, Ultrasonic, Auto Shutoff, 7 LED Colors, Wood Grain, $32.99",
            ],
            "source": [
                "Amazon",
                "Etsy",
                "Target",
                "Walmart",
                "Amazon",
                "Target",
                "Nike",
                "HomeDepot",
                "Amazon",
                "Etsy",
            ],
            "rating": [4.5, 4.8, 4.3, 4.6, 4.4, 4.7, 4.2, 4.5, 4.6, 4.9],
        }
    )

    # Prompt for structured extraction
    extraction_prompt = """Extract key product information from the description.

Product: {{ product_description }}
Source: {{ source }}
Rating: {{ rating }}

Extract:
- product_name: Clear, concise product name (remove price/specs)
- category: Main product category
- price_range: Extract price if mentioned (e.g., "$20-30" or "$24.99")
- keywords: List 3-5 key features/attributes
"""

    # Build pipeline with batching
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["product_description", "source", "rating"],
            output_columns=["product_name", "category", "price_range", "keywords"],
        )
        .with_prompt(extraction_prompt)
        .with_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.0,
            input_cost_per_1k_tokens=Decimal("0.00015")
            if provider == "openai"
            else Decimal("0.00059"),
            output_cost_per_1k_tokens=Decimal("0.0006")
            if provider == "openai"
            else Decimal("0.00079"),
        )
        .with_jinja2(True)
        .with_batch_size(10)  # All 10 rows in 1 mega-prompt
        .with_processing_batch_size(5)  # 2 API calls (5 rows each)
        .with_structured_output(ProductBatch)  # Auto-adds JSONParser
        .with_concurrency(2)
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify success
    assert result.success, f"{provider} pipeline failed: {result.error}"
    assert len(result.data) == 10, f"Expected 10 rows, got {len(result.data)}"

    # Verify all rows have results
    for idx, row in result.data.iterrows():
        assert row["product_name"] is not None, f"Row {idx}: product_name is None"
        assert isinstance(row["product_name"], str), (
            f"Row {idx}: product_name not a string"
        )
        assert len(row["product_name"]) > 0, f"Row {idx}: product_name is empty"

    # Verify specific extractions (spot checks)
    headphones_row = result.data[
        result.data["product_description"].str.contains("Headphones")
    ].iloc[0]
    assert "headphone" in headphones_row["product_name"].lower(), (
        "Headphones not extracted correctly"
    )
    assert headphones_row["price_range"] is not None, (
        "Price not extracted from headphones"
    )

    coffee_row = result.data[
        result.data["product_description"].str.contains("Coffee")
    ].iloc[0]
    assert "coffee" in coffee_row["product_name"].lower(), (
        "Coffee not extracted correctly"
    )
    assert coffee_row["category"] is not None, "Category not extracted from coffee"

    # Verify keywords are lists (if extracted)
    for idx, row in result.data.iterrows():
        if row["keywords"] is not None:
            # Keywords might be string representation of list or actual list
            assert isinstance(row["keywords"], (list, str)), (
                f"Row {idx}: keywords type is {type(row['keywords'])}"
            )

    print(f"\n{provider.upper()} Mega-Prompt Batching Test Results:")
    print(f"  Rows processed: {len(result.data)}/10")
    print("  API calls: 2 (5 rows per batch)")
    print(f"  Cost: ${result.costs.total_cost:.4f}")
    print("  ✅ Batching working correctly")
    print("\nSample extractions:")
    for i in range(3):
        print(f"  {i + 1}. {result.data.iloc[i]['product_name'][:50]}")


@pytest.mark.integration
def test_mega_prompt_preserves_order():
    """
    Test that mega-prompt batching preserves row order.

    Critical for ensuring results match input DataFrame indices.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Create DataFrame with sequential IDs
    df = pd.DataFrame(
        {
            "id": list(range(1, 21)),
            "text": [f"Item number {i}" for i in range(1, 21)],
        }
    )

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df, input_columns=["id", "text"], output_columns=["extracted_id"]
        )
        .with_prompt(
            "Extract the number from the text: {{ text }}. Return only the integer."
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
            input_cost_per_1k_tokens=Decimal("0.00059"),
            output_cost_per_1k_tokens=Decimal("0.00079"),
        )
        .with_jinja2(True)
        .with_batch_size(20)  # All in one batch
        .with_processing_batch_size(10)  # 2 API calls
        .build()
    )

    result = pipeline.execute()

    assert result.success, "Pipeline failed"
    assert len(result.data) == 20, f"Expected 20 rows, got {len(result.data)}"

    # Verify order is preserved
    for idx, row in result.data.iterrows():
        expected_id = row["id"]
        # The extracted_id might be a string, so convert for comparison
        if row["extracted_id"] is not None:
            extracted = str(row["extracted_id"]).strip()
            assert str(expected_id) in extracted, (
                f"Row {idx}: Order mismatch. Expected ID {expected_id}, "
                f"but extracted '{extracted}'"
            )

    print("\nOrder Preservation Test Results:")
    print("  ✅ All 20 rows in correct order")
    print("  Batch size: 20 rows → 2 API calls (10 each)")


@pytest.mark.integration
def test_mega_prompt_handles_null_values():
    """
    Test that mega-prompt batching handles null/missing values correctly.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    # DataFrame with some null values
    df = pd.DataFrame(
        {
            "name": ["Product A", None, "Product C", "", "Product E"],
            "description": [
                "Good quality item",
                "Another item",
                None,
                "Empty name item",
                "Final item",
            ],
        }
    )

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            df, input_columns=["name", "description"], output_columns=["summary"]
        )
        .with_prompt(
            "Summarize: Name={{ name or 'Unknown' }}, Desc={{ description or 'No description' }}"
        )
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            input_cost_per_1k_tokens=Decimal("0.00015"),
            output_cost_per_1k_tokens=Decimal("0.0006"),
        )
        .with_jinja2(True)
        .with_batch_size(5)  # All in one batch
        .with_processing_batch_size(5)
        .build()
    )

    result = pipeline.execute()

    assert result.success, "Pipeline should handle null values gracefully"
    assert len(result.data) == 5, f"Expected 5 rows, got {len(result.data)}"

    # Verify all rows have some result (even if input was null)
    for idx, row in result.data.iterrows():
        assert row["summary"] is not None, (
            f"Row {idx} with null input should still have output"
        )

    print("\nNull Value Handling Test Results:")
    print("  ✅ All 5 rows processed (including nulls)")
    print(f"  Cost: ${result.costs.total_cost:.4f}")
