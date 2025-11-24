import os

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from ondine import PipelineBuilder
from ondine.stages.response_parser_stage import JSONParser


# Define models (same as user script)
class BaconResult(BaseModel):
    cleaned_description: str = Field(description="Description")
    pack_size: float | None = Field(default=None)
    explanation: str | None = Field(default=None)


class BatchItem(BaseModel):
    id: int
    result: BaconResult


class BaconBatch(BaseModel):
    items: list[BatchItem]


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider,model,api_key_env,base_url",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY", None),
        ("groq", "llama-3.3-70b-versatile", "GROQ_API_KEY", None),
        ("anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY", None),
    ],
)
def test_structured_output_e2e(provider, model, api_key_env, base_url):
    """
    E2E test for structured output across multiple providers.

    Tests that LlamaIndex's structured_predict works correctly with:
    - OpenAI: Native tool calling via structured_predict (baseline)
    - Groq: LLMTextCompletionProgram workaround (due to XML bug)
    - Anthropic: Native tool calling via structured_predict (Claude 3 Haiku)

    Requires respective API keys in environment.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} not set")

    # Create dummy data
    df = pd.DataFrame(
        {
            "DISTRB_ITM_DESC": ["BACON 15 LB", "BACON SLICED"],
            "DISTRB_PACK_SZ": ["15 LB", "15 LB"],
        }
    )

    # Build pipeline (provider-agnostic)
    builder = (
        PipelineBuilder.create()
        .from_dataframe(
            df,
            input_columns=["DISTRB_ITM_DESC", "DISTRB_PACK_SZ"],
            output_columns=["cleaned_description", "pack_size", "explanation"],
        )
        .with_prompt("""Extract bacon product attributes from the description and package size.

Description: {{DISTRB_ITM_DESC}}
Package Size: {{DISTRB_PACK_SZ}}

Extract:
- cleaned_description: A clear product description
- pack_size: The numeric package size (e.g., "15 LB" → 15.0)
- explanation: Brief reasoning for your extraction

CRITICAL: Parse the package size number from DISTRB_PACK_SZ field.""")
        .with_parser(JSONParser())
        .with_jinja2(True)
        .with_batch_size(2)
        .with_processing_batch_size(2)
        .with_rate_limit(60)
        .with_structured_output(BaconBatch)
    )

    # Configure LLM based on provider
    llm_kwargs = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "temperature": 0.0,
    }

    if base_url:
        llm_kwargs["base_url"] = base_url

    builder = builder.with_llm(**llm_kwargs)

    pipeline = builder.build()

    # Execute
    result = pipeline.execute()

    # Verify
    assert result.success, f"{provider} pipeline failed"
    assert len(result.data) == 2, f"{provider} returned wrong number of rows"
    assert "cleaned_description" in result.data.columns
    assert result.data["pack_size"].notnull().all(), (
        f"{provider} returned null pack_size values"
    )

    # Print results for manual inspection
    print(f"\n{provider.upper()} E2E Results:")
    print(result.data[["cleaned_description", "pack_size"]])
    print(f"Cost: ${result.costs.total_cost:.4f}")
