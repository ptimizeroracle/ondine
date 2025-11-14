"""
Example: Custom Pipeline Stages via Stage Registry.

Demonstrates how to create and inject custom processing stages into
Ondine pipelines. This enables:
- RAG (Retrieval-Augmented Generation)
- Content moderation
- Data enrichment
- Multi-step processing
- Any custom logic between standard stages
"""

import time

import pandas as pd

from ondine.api import PipelineBuilder
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages import JSONParser, PipelineStage, stage


@stage("data_enrichment")
class DataEnrichmentStage(PipelineStage):
    """
    Custom stage: Enrich data with external information.

    Demonstrates how to add custom columns before prompt formatting.
    """

    def __init__(self, source: str = "api"):
        super().__init__(name="data_enrichment")
        self.source = source

    def execute(self, context: ExecutionContext):
        """
        Add enrichment data to each row.

        Args:
            context: Execution context with data

        Returns:
            Modified context with enriched data
        """
        print(f"  🔍 Enriching data from {self.source}...")

        enriched_rows = []
        for _, row in context.data.iterrows():
            enriched_row = row.to_dict()

            # Simulate fetching external data
            enriched_row["enriched_context"] = (
                f"[External data for: {row.get('name', 'N/A')}]"
            )
            enriched_row["metadata"] = {"source": self.source, "timestamp": time.time()}

            enriched_rows.append(enriched_row)

        context.data = pd.DataFrame(enriched_rows)
        return context

    def process(self, data):
        """Required by PipelineStage protocol."""
        return data

    def validate_input(self, data):
        """Required by PipelineStage protocol."""
        return True

    def estimate_cost(self, data):
        """Required by PipelineStage protocol."""
        return 0.0  # No LLM cost for enrichment


@stage("content_filter")
class ContentFilterStage(PipelineStage):
    """
    Custom stage: Filter or sanitize content before LLM processing.

    Demonstrates pre-processing to block inappropriate content.
    """

    def __init__(self, block_patterns: list[str] | None = None):
        super().__init__(name="content_filter")
        self.block_patterns = block_patterns or ["spam", "offensive", "inappropriate"]

    def execute(self, context: ExecutionContext):
        """
        Filter content based on patterns.

        Marks rows as blocked if they contain forbidden patterns.
        """
        print(f"  🛡️  Filtering content (blocking: {', '.join(self.block_patterns)})...")

        filtered_rows = []
        blocked_count = 0

        for _, row in context.data.iterrows():
            row_dict = row.to_dict()

            # Check all text columns for blocked patterns
            is_blocked = False
            for col, value in row_dict.items():
                if isinstance(value, str):
                    for pattern in self.block_patterns:
                        if pattern.lower() in value.lower():
                            is_blocked = True
                            break

            if is_blocked:
                row_dict["_filtered"] = True
                row_dict["_filter_reason"] = "Contains blocked content"
                blocked_count += 1
            else:
                row_dict["_filtered"] = False

            filtered_rows.append(row_dict)

        context.data = pd.DataFrame(filtered_rows)
        print(f"    → Blocked {blocked_count} / {len(filtered_rows)} rows")

        # Note: Blocked rows are marked with _filtered=True but still flow to LLM.
        # In production, you may want to drop these rows entirely or skip LLM processing.
        # For this demo, we keep them to show the filtering metadata in results.

        return context

    def process(self, data):
        return data

    def validate_input(self, data):
        return True

    def estimate_cost(self, data):
        return 0.0


@stage("fact_checker")
class FactCheckerStage(PipelineStage):
    """
    Custom stage: Verify LLM outputs against known facts.

    Demonstrates post-processing after LLM invocation.
    """

    def __init__(self, fact_database: dict | None = None):
        super().__init__(name="fact_checker")
        self.fact_database = fact_database or {}

    def execute(self, context: ExecutionContext):
        """
        Check LLM outputs against fact database.

        Adds confidence scores based on fact verification.
        """
        print("  ✓ Fact-checking LLM outputs...")

        verified_rows = []
        for _, row in context.data.iterrows():
            row_dict = row.to_dict()

            # Simulate fact checking (in reality, query a knowledge base)
            confidence = "high" if len(row_dict.get("output", "")) > 10 else "low"

            row_dict["_fact_check_confidence"] = confidence
            row_dict["_fact_checked"] = True

            verified_rows.append(row_dict)

        context.data = pd.DataFrame(verified_rows)
        return context

    def process(self, data):
        return data

    def validate_input(self, data):
        return True

    def estimate_cost(self, data):
        return 0.0


def example_1_data_enrichment():
    """Example 1: Add custom stage to enrich data before processing."""
    print("\n" + "=" * 60)
    print("Example 1: Data Enrichment Stage")
    print("=" * 60)

    # Sample data
    data = pd.DataFrame(
        {
            "name": ["Product A", "Product B", "Product C"],
            "description": ["Basic laptop", "Gaming laptop", "Ultrabook"],
        }
    )

    # Build pipeline with custom enrichment stage
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data, input_columns=["name", "description"], output_columns=["category"]
        )
        # Inject custom stage BEFORE prompt formatting
        .with_stage("data_enrichment", position="before_prompt", source="product_api")
        .with_prompt(
            template="""Categorize this product based on the description and external context.

Product: {name}
Description: {description}
External Context: {enriched_context}

Category (electronics/office/other):"""
        )
        .with_llm(
            provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=10
        )
        .with_batch_size(3)
        .build()
    )

    print("\n📊 Processing with data enrichment stage...")
    result = pipeline.execute()

    print("\n✅ Results:")
    for idx, row in result.data.iterrows():
        print(f"  {idx + 1}. {row['name']}: {row['category']}")
        print(f"     Context: {row['enriched_context'][:50]}...")


def example_2_content_filtering():
    """Example 2: Filter content before LLM processing."""
    print("\n" + "=" * 60)
    print("Example 2: Content Filtering Stage")
    print("=" * 60)

    # Sample data with some problematic content
    data = pd.DataFrame(
        {
            "user_input": [
                "Please help me with this technical question",
                "This is spam content trying to sell you stuff",
                "I have a legitimate support request",
                "Offensive language and inappropriate content here",
            ]
        }
    )

    # Build pipeline with content filter
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(data, input_columns=["user_input"], output_columns=["response"])
        # Inject filter BEFORE LLM
        .with_stage(
            "content_filter",
            position="before_llm",
            block_patterns=["spam", "offensive", "inappropriate"],
        )
        .with_prompt(template="Respond to: {user_input}")
        .with_llm(
            provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=50
        )
        .build()
    )

    print("\n📊 Processing with content filtering...")
    result = pipeline.execute()

    print("\n✅ Results:")
    for idx, row in result.data.iterrows():
        status = "🚫 BLOCKED" if row.get("_filtered") else "✅ ALLOWED"
        print(f"  {idx + 1}. {status}: {row['user_input'][:50]}...")
        if row.get("_filtered"):
            print(f"     Reason: {row.get('_filter_reason')}")


def example_3_fact_checking():
    """Example 3: Verify LLM outputs after generation."""
    print("\n" + "=" * 60)
    print("Example 3: Fact Checking Stage")
    print("=" * 60)

    # Sample questions
    data = pd.DataFrame(
        {
            "question": [
                "What is Python?",
                "What is the capital of France?",
                "Who invented the lightbulb?",
            ]
        }
    )

    # Build pipeline with fact checker AFTER LLM
    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(data, input_columns=["question"], output_columns=["answer"])
        .with_prompt(template="Answer concisely: {question}")
        .with_llm(
            provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=50
        )
        # Inject fact checker AFTER LLM, BEFORE parser
        .with_stage("fact_checker", position="after_llm", fact_database={})
        .build()
    )

    print("\n📊 Processing with fact checking...")
    result = pipeline.execute()

    print("\n✅ Results:")
    for idx, row in result.data.iterrows():
        confidence = row.get("_fact_check_confidence", "unknown")
        print(f"  {idx + 1}. Q: {row['question']}")
        print(f"     A: {row.get('answer', 'N/A')[:80]}...")
        print(f"     Confidence: {confidence}")


def example_4_multiple_custom_stages():
    """Example 4: Chain multiple custom stages together."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Custom Stages (Pipeline Composition)")
    print("=" * 60)

    # Complex pipeline with multiple custom stages
    data = pd.DataFrame(
        {
            "user_query": [
                "How do I reset my password?",
                "I want to buy something with spam attached",
                "What are your business hours?",
            ]
        }
    )

    pipeline = (
        PipelineBuilder.create()
        .from_dataframe(
            data,
            input_columns=["user_query"],
            output_columns=["response", "confidence"],
        )
        # Stage 1: Enrich with user context
        .with_stage("data_enrichment", position="before_prompt", source="user_database")
        # Stage 2: Filter inappropriate content
        .with_stage("content_filter", position="before_llm", block_patterns=["spam"])
        # Process with LLM
        .with_prompt(
            template="""You are a customer support assistant.

User Query: {user_query}
User Context: {enriched_context}

Provide a JSON response:
{{
    "response": "your response",
    "confidence": "high/medium/low"
}}"""
        )
        .with_llm(
            provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=100
        )
        .with_parser(JSONParser(strict=False))
        # Stage 3: Fact check the response
        .with_stage("fact_checker", position="after_parser")
        .build()
    )

    print("\n📊 Processing with multi-stage pipeline:")
    print("  1️⃣  Data Enrichment")
    print("  2️⃣  Content Filtering")
    print("  3️⃣  LLM Processing")
    print("  4️⃣  Fact Checking")

    result = pipeline.execute()

    print("\n✅ Results:")
    for idx, row in result.data.iterrows():
        filtered = "🚫" if row.get("_filtered") else "✅"
        confidence = row.get("_fact_check_confidence", "?")
        print(f"  {idx + 1}. {filtered} Query: {row['user_query'][:40]}...")
        print(f"     Response: {str(row.get('response', 'N/A'))[:60]}...")
        print(f"     Confidence: {confidence}")


if __name__ == "__main__":
    print("\n🌾 Ondine: Custom Pipeline Stage Examples\n")

    # Note: These examples require OPENAI_API_KEY
    # Uncomment to run:
    # example_1_data_enrichment()
    # example_2_content_filtering()
    # example_3_fact_checking()
    # example_4_multiple_custom_stages()

    print("\n" + "=" * 60)
    print("📋 Custom Stages Overview")
    print("=" * 60)

    from ondine.stages.stage_registry import StageRegistry

    # List registered stages
    stages = StageRegistry.list_stages()
    print("\n📦 Registered Custom Stages:")
    for stage_name in sorted(stages.keys()):
        print(f"  - {stage_name}")

    print("\n💡 Key Takeaways:")
    print("  1. Use @stage decorator to register custom stages")
    print("  2. Inject stages at specific positions in pipeline")
    print("  3. Positions: before_prompt, before_llm, after_llm, after_parser")
    print("  4. Chain multiple stages for complex workflows")
    print("  5. No core code modification needed!")
    print("\n")
