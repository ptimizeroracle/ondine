"""Claim verification: Use case claims (Claims 64-72).

These tests verify the pipeline can be CONFIGURED for each claimed use case.
They use mock LLM clients — we're testing data flow, not LLM quality.
"""

import pandas as pd

from ondine import PipelineBuilder


def _build_pipeline(df, input_cols, output_cols, prompt):
    """Helper to build a pipeline without executing."""
    return (
        PipelineBuilder.create()
        .from_dataframe(df, input_columns=input_cols, output_columns=output_cols)
        .with_prompt(prompt)
        .with_llm(model="gpt-4o-mini", provider="openai")
        .build()
    )


class TestUseCaseClaims:
    """Verify pipeline supports all claimed use cases."""

    def test_claim_64_data_cleaning(self):
        """Claim 64: Data cleaning and normalization."""
        df = pd.DataFrame({"raw_address": ["123 Main St.", "456 Oak Ave"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["raw_address"],
            output_cols=["clean_address"],
            prompt="Normalize this address to standard format: {raw_address}",
        )
        assert pipeline is not None

    def test_claim_65_sentiment_analysis(self):
        """Claim 65: Sentiment analysis at scale."""
        df = pd.DataFrame({"review": ["Great product!", "Terrible service"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["review"],
            output_cols=["sentiment"],
            prompt="Classify the sentiment of this review as positive, negative, or neutral: {review}",
        )
        assert pipeline is not None

    def test_claim_66_information_extraction(self):
        """Claim 66: Information extraction."""
        df = pd.DataFrame({"text": ["John Smith, CEO of Acme Corp, announced..."]})
        pipeline = _build_pipeline(
            df,
            input_cols=["text"],
            output_cols=["entities"],
            prompt="Extract all named entities (person, org, location) from: {text}",
        )
        assert pipeline is not None

    def test_claim_67_auto_categorization(self):
        """Claim 67: Auto-categorization (products, documents, emails)."""
        df = pd.DataFrame(
            {
                "product_name": ["iPhone 15", "Running Shoes"],
                "description": ["Latest smartphone", "Athletic footwear"],
            }
        )
        pipeline = _build_pipeline(
            df,
            input_cols=["product_name", "description"],
            output_cols=["category"],
            prompt="Categorize this product: {product_name} - {description}",
        )
        assert pipeline is not None

    def test_claim_68_content_generation(self):
        """Claim 68: Content generation (descriptions, summaries, titles)."""
        df = pd.DataFrame({"product": ["Wireless headphones with noise canceling"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["product"],
            output_cols=["title", "description"],
            prompt="Generate a marketing title and description for: {product}",
        )
        assert pipeline is not None

    def test_claim_69_translation(self):
        """Claim 69: Translation to multiple languages."""
        df = pd.DataFrame({"text": ["Hello, how are you?"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["text"],
            output_cols=["translation"],
            prompt="Translate to French: {text}",
        )
        assert pipeline is not None

    def test_claim_70_data_enrichment(self):
        """Claim 70: Data enrichment with LLM-generated insights."""
        df = pd.DataFrame({"company": ["Apple Inc.", "Tesla"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["company"],
            output_cols=["industry", "headquarters"],
            prompt="Provide the industry and headquarters for: {company}",
        )
        assert pipeline is not None

    def test_claim_71_product_matching(self):
        """Claim 71: Product matching and scoring."""
        df = pd.DataFrame(
            {
                "product_a": ["iPhone 15 Pro"],
                "product_b": ["Samsung Galaxy S24"],
            }
        )
        pipeline = _build_pipeline(
            df,
            input_cols=["product_a", "product_b"],
            output_cols=["similarity_score"],
            prompt="Rate similarity 0-100 between: {product_a} vs {product_b}",
        )
        assert pipeline is not None

    def test_claim_72_content_moderation(self):
        """Claim 72: Content moderation at scale."""
        df = pd.DataFrame({"comment": ["Great article!", "This is spam!!!"]})
        pipeline = _build_pipeline(
            df,
            input_cols=["comment"],
            output_cols=["is_safe", "reason"],
            prompt="Moderate this comment. Is it safe? Why? Comment: {comment}",
        )
        assert pipeline is not None
