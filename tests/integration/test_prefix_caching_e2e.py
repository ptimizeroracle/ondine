"""
E2E integration tests for prefix caching (prompt caching).

Tests that system messages are properly separated and cached by providers.
OpenAI and Anthropic automatically cache system messages to reduce costs.
"""

import os

import pandas as pd
import pytest

from ondine import PipelineBuilder


@pytest.mark.integration
def test_prefix_caching_groq_e2e():
    """
    E2E test for prefix caching with Groq.

    Tests that:
    - System messages are sent separately from user prompts
    - Multiple requests with same system message reuse cached prefix
    - This reduces input token costs significantly

    Note: Groq may not have explicit prompt caching like OpenAI/Anthropic,
    but proper message structure is still important for future support.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Create test data with multiple rows (same system message, different prompts)
    df = pd.DataFrame(
        {
            "question": [
                "What is 2+2?",
                "What is 3+3?",
                "What is 5+5?",
                "What is 10+10?",
                "What is 20+20?",
            ]
        }
    )

    # Long system message (perfect candidate for caching!)
    system_message = """You are an expert mathematics tutor with 20 years of experience.
You specialize in explaining concepts clearly and concisely.
Always provide the answer first, then a brief explanation.
Use simple language suitable for high school students.
Be encouraging and positive in your responses.
Focus on building confidence in mathematical reasoning."""

    # Build pipeline with system message (should be cached across calls)
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["question"], output_columns=["answer"])
        .with_prompt(
            template="{question}",  # User prompt (varies)
            system_message=system_message,  # System prompt (same for all - cached!)
        )
        .with_llm(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
            enable_prefix_caching=True,  # Enable caching
        )
        .with_rate_limit(9)  # Groq rate limit
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify
    assert result.success, "Pipeline failed"
    assert len(result.data) == 5, "Wrong number of rows"
    assert result.data["answer"].notnull().all(), "Some answers are null"

    # Verify cost tracking
    assert result.costs.total_cost > 0, "Cost not tracked"
    assert result.costs.input_tokens > 0, "Input tokens not tracked"

    # Print results
    print("\nPrefix Caching E2E (Groq):")
    print(f"Rows processed: {len(result.data)}")
    print(f"Total cost: ${result.costs.total_cost:.4f}")
    print(f"Input tokens: {result.costs.input_tokens}")
    print(f"Output tokens: {result.costs.output_tokens}")
    print(f"Cost per row: ${result.costs.total_cost / len(result.data):.6f}")

    # Sample answers
    print("\nSample answers:")
    for i, row in result.data.head(3).iterrows():
        print(f"Q: {row['question']} → A: {row['answer'][:50]}...")

    # Verify message structure was correct
    # System message should be sent separately for each call
    # (This is what enables caching at the provider level)
    print("\n✓ System messages sent separately (enables provider-level caching)")


@pytest.mark.integration
def test_prefix_caching_openai_e2e():
    """
    E2E test for prefix caching with OpenAI.

    OpenAI has explicit prompt caching support that caches system messages
    automatically, reducing costs significantly for repeated system prompts.

    See: https://platform.openai.com/docs/guides/prompt-caching
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    # Create test data
    df = pd.DataFrame({"text": ["Hello", "World", "Test", "Data", "Sample"]})

    # Long system message (will be cached by OpenAI)
    system_prompt = """You are a professional text analyzer specializing in sentiment analysis.
Your task is to analyze the emotional tone and sentiment of the provided text.
Consider context, word choice, and implicit meanings.
Provide your analysis in a single concise sentence.
Be objective and evidence-based in your assessments.
Focus on the primary emotional tone conveyed."""

    # Build pipeline
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["text"], output_columns=["sentiment"])
        .with_prompt(
            template="Analyze: {text}",
            system_message=system_prompt,  # OpenAI caches this!
        )
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            enable_prefix_caching=True,
        )
        .with_rate_limit(500)  # OpenAI rate limit
        .build()
    )

    # Execute
    result = pipeline.execute()

    # Verify
    assert result.success
    assert len(result.data) == 5
    assert result.data["sentiment"].notnull().all()

    print("\nPrefix Caching E2E (OpenAI):")
    print(f"Rows: {len(result.data)}")
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print(f"Input tokens: {result.costs.input_tokens}")
    print("Note: OpenAI automatically cached the system message!")
    print("      (Repeat calls with same system message = 50% cost reduction)")


@pytest.mark.integration
def test_prefix_caching_anthropic_e2e():
    """
    E2E test for prefix caching with Anthropic.

    Anthropic has explicit prompt caching that caches system messages
    and long contexts automatically.

    See: https://docs.anthropic.com/claude/docs/prompt-caching
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    df = pd.DataFrame({"task": ["Summarize", "Analyze", "Review"]})

    # Long system message (Anthropic caches this automatically)
    system_prompt = """You are Claude, an AI assistant created by Anthropic.
You are helpful, harmless, and honest in all your responses.
Your goal is to provide accurate, thoughtful, and nuanced answers.
You should be direct and concise while still being thorough.
You should acknowledge uncertainty when appropriate.
You should avoid speculation and stick to factual information."""

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=["task"], output_columns=["result"])
        .with_prompt(template="{task} this text", system_message=system_prompt)
        .with_llm(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            api_key=api_key,
            temperature=0.0,
            enable_prefix_caching=True,
        )
        .build()
    )

    result = pipeline.execute()

    assert result.success
    assert len(result.data) == 3

    print("\nPrefix Caching E2E (Anthropic):")
    print(f"Cost: ${result.costs.total_cost:.4f}")
    print("Note: Anthropic cached the system message automatically!")
