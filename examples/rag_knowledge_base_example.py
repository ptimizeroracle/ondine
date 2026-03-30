"""
RAG Knowledge Base Example - Retrieval-augmented generation pipeline.

This example shows how to create a KnowledgeStore, ingest documents as
raw text, and build a pipeline that automatically retrieves relevant
knowledge-base context before each LLM call.
"""

import pandas as pd

from ondine import PipelineBuilder
from ondine.knowledge import KnowledgeStore

# ---------------------------------------------------------------------------
# 1. Create an in-memory knowledge base and ingest reference documents.
# ---------------------------------------------------------------------------

kb = KnowledgeStore(db_path=":memory:")

# Ingest product reference sheets as plain text.
# In production you would call kb.ingest("path/to/docs/") to load PDFs,
# Markdown, HTML, or text files from a directory.
kb.ingest_text(
    text=(
        "Return Policy: All grocery items may be returned within 30 days "
        "of purchase with a valid receipt. Perishable items must be returned "
        "within 7 days. Refunds are issued to the original payment method."
    ),
    source="policy/returns.md",
)

kb.ingest_text(
    text=(
        "Shipping: Standard shipping is free on orders over $50. "
        "Refrigerated items ship in insulated packaging with ice packs. "
        "Express delivery (next-day) is available for $9.99. "
        "Frozen items cannot be shipped to PO boxes."
    ),
    source="policy/shipping.md",
)

kb.ingest_text(
    text=(
        "Loyalty Program: Members earn 1 point per dollar spent. "
        "100 points can be redeemed for a $5 discount. Points expire "
        "after 12 months of account inactivity. Gold members (500+ points "
        "per year) receive free express shipping on all orders."
    ),
    source="policy/loyalty.md",
)

print("Knowledge base ready (ingested documents from 3 sources).")
print()

# ---------------------------------------------------------------------------
# 2. Sample customer questions to answer using the knowledge base.
# ---------------------------------------------------------------------------

data = pd.DataFrame(
    {
        "customer_question": [
            "Can I return frozen salmon I bought last week?",
            "How much does next-day delivery cost?",
            "How do I earn free shipping through the loyalty program?",
            "Do you ship frozen meals to PO boxes?",
            "I lost my receipt. Can I still get a refund?",
        ]
    }
)

print("Customer questions:")
for q in data["customer_question"]:
    print(f"  - {q}")
print()

# ---------------------------------------------------------------------------
# 3. Build pipeline with knowledge-base retrieval.
# ---------------------------------------------------------------------------

pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["customer_question"],
        output_columns=["answer"],
    )
    .with_prompt(
        "You are a helpful customer support agent. Answer the customer's "
        "question using ONLY the context provided. If the context does not "
        "contain enough information, say so.\n\n"
        "Context:\n{kb_context}\n\n"
        "Question: {customer_question}\n\n"
        "Answer:"
    )
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    # Attach the knowledge base — retrieves top_k chunks per row and
    # injects them as {kb_context} in the prompt.
    .with_knowledge_base(kb, top_k=5)
    .build()
)

# ---------------------------------------------------------------------------
# 4. Cost estimation.
# ---------------------------------------------------------------------------

print("Estimating cost...")
estimate = pipeline.estimate_cost()
print(f"  Estimated total: ${estimate.total_cost:.4f}")
print(f"  Estimated tokens: {estimate.total_tokens:,}")
print()

# ---------------------------------------------------------------------------
# 5. Execute and display results.
# ---------------------------------------------------------------------------

print("Processing questions with KB context...")
result = pipeline.execute()

print("\nResults:")
for _, row in result.data.iterrows():
    print(f"Q: {row['customer_question']}")
    print(f"A: {row['answer']}")
    print()

# ---------------------------------------------------------------------------
# 6. Metrics summary.
# ---------------------------------------------------------------------------

print(f"Processed rows: {result.metrics.processed_rows}")
print(f"Duration: {result.metrics.total_duration_seconds:.2f}s")
print(f"Total cost: ${result.costs.total_cost:.4f}")
