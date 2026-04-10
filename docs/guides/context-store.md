# Context Store and Anti-Hallucination

Ondine's context store system gives pipelines a memory of prior validated LLM responses. Each output can be stored as an evidence record, retrieved to inform future prompts, verified against source text, and checked for cross-row contradictions. Together these capabilities form a composable anti-hallucination pipeline.

## How it works

When a context store is attached to a pipeline, each processed row goes through the following lifecycle:

1. **Evidence priming** (optional, pre-LLM) — the store is searched for prior validated answers relevant to the current row. The top-k results are injected into the prompt as `_evidence_context`.
2. **LLM inference** — the model generates a response, optionally informed by primed evidence.
3. **Grounding verification** (optional, post-LLM) — the response is compared against source text using TF-IDF cosine similarity (or dense embeddings if an `embed_fn` is supplied). Results below the threshold are flagged.
4. **Contradiction detection** (optional) — rows that share the same key columns but produce different output values are flagged as contradictions.
5. **Confidence scoring** (optional) — a composite score is computed from the grounding similarity and evidence support count and written to a `confidence_score` column.
6. **Storage** — validated responses are stored as evidence records, accumulating a knowledge base for subsequent pipeline runs.

---

## Store backends

### RustContextStore

The default high-performance backend. It compiles to a native extension (`ondine._engine`) backed by SQLite with FTS5 full-text search. Supports hybrid search (TF-IDF sparse + optional dense embeddings via Reciprocal Rank Fusion), persistent storage across runs, and contradiction tracking.

```python
from ondine.context import RustContextStore

# Persistent database — survives between runs
store = RustContextStore("evidence.db")

# In-memory (Rust speed, no persistence)
store = RustContextStore(":memory:")
```

**When to use:** Production workloads, datasets processed across multiple runs, any case where evidence accumulation between runs matters.

**Requires:** The compiled Rust extension. Install with `pip install ondine` (a Rust toolchain is required only when building from source).

### ZepContextStore

A cloud-hosted knowledge graph backed by the [Zep Cloud](https://www.getzep.com/) API. Zep automatically extracts entities and relationships from stored text, enabling graph-aware semantic search with cross-encoder reranking.

```python
from ondine.context import ZepContextStore

# Uses ZEP_API_KEY environment variable
store = ZepContextStore(graph_id="my-pipeline-run")

# Explicit API key
store = ZepContextStore(graph_id="my-pipeline-run", api_key="zep-...")
```

Each `graph_id` acts as an isolated namespace. Multiple pipeline runs can share a graph (for cross-run memory) or use separate graphs (for isolation). The `graph_id` defaults to a random UUID if not specified.

**When to use:** When you need a managed cloud store, cross-run persistent knowledge graphs, or Zep's entity/relationship extraction.

**Requires:** `pip install ondine[zep]` and a `ZEP_API_KEY` environment variable (or explicit `api_key`).

**Note on availability:** `ZepContextStore.available` returns `False` if the client could not be initialized (missing package or invalid key). The store gracefully degrades — `search()` returns an empty list — so pipelines continue running even without a valid Zep connection.

### InMemoryContextStore

A pure-Python fallback with no external dependencies. Uses the same TF-IDF algorithm as the Rust backend but runs entirely in Python. All data is lost when the process exits.

```python
from ondine.context import InMemoryContextStore

store = InMemoryContextStore()
```

**When to use:** Unit tests, CI environments without a Rust toolchain, quick prototyping where persistence is not needed.

---

## Working with stores directly

All three backends implement the `ContextStore` protocol:

```python
from ondine.context import RustContextStore, EvidenceRecord

store = RustContextStore("evidence.db")

# Store an evidence record
claim_id = store.store(EvidenceRecord(
    text="Paris is the capital of France.",
    source_ref="geography-101",
    claim_type="factual",       # "factual", "opinion", etc.
    source_type="document",     # "document", "llm_response", "user_correction", "external"
    asserted_by="pipeline",
    confidence=0.95,
))

# Retrieve by ID
record = store.retrieve(claim_id)

# Search by query
results = store.search("capital of France", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.text}  (source: {r.source_ref})")

# Ground an LLM response against source sentences
groundings = store.ground(
    response_text="Paris is the capital of France.",
    source_sentences=["Paris is where the French government is based.", "France is in Western Europe."],
    threshold=0.3,
)
for g in groundings:
    print(f"grounded={g.grounded}  confidence={g.confidence:.2f}")

# Record a contradiction between two claims
store.add_contradiction(claim_id_a, claim_id_b)
contradicted_by = store.get_contradictions(claim_id_a)

store.close()
```

### EvidenceRecord fields

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | The evidence text |
| `source_ref` | `str` | `""` | Reference to the source document or URL |
| `claim_type` | `str` | `"factual"` | Semantic type of the claim |
| `source_type` | `str` | `"llm_response"` | One of `"document"`, `"llm_response"`, `"user_correction"`, `"external"` |
| `asserted_by` | `str` | `"pipeline"` | Who/what asserted the claim |
| `claim_id` | `str \| None` | `None` | If None, a UUID is assigned on store |
| `confidence` | `float \| None` | `None` | Optional confidence score (0.0–1.0) |
| `metadata` | `dict` | `{}` | Arbitrary key-value metadata |

### RetrievalResult fields

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Evidence text |
| `score` | `float` | Relevance score (higher is more relevant) |
| `claim_id` | `str` | ID of the stored claim |
| `source_ref` | `str` | Source reference |
| `support_count` | `int` | How many times this claim has been reinforced |

---

## Pipeline builder API

### with_context_store()

Attaches a context store to the pipeline. All other anti-hallucination methods require a store and will call this automatically if it has not been set.

```python
def with_context_store(
    store: ContextStore | None = None,
) -> PipelineBuilder
```

**Parameters:**

- `store` — A `ContextStore` instance. If `None`, the builder auto-detects: it tries `RustContextStore` (in-memory) first and falls back to `InMemoryContextStore` if the Rust extension is unavailable.

```python
from ondine import PipelineBuilder
from ondine.context import RustContextStore

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["question"], output_columns=["answer"])
    .with_prompt("Answer: {question}")
    .with_llm(model="openai/gpt-4o-mini")
    .with_context_store(RustContextStore("evidence.db"))
    .build()
)
```

For persistent storage across multiple pipeline runs, always pass an explicit store with a file path. The auto-detected store uses `":memory:"` which does not persist.

---

### with_evidence_priming()

Enriches each row with prior validated answers before LLM inference. The store is searched using the configured query columns, and the top-k results are written to `_evidence_context` and `_evidence_count` columns. The prompt formatter automatically includes `_evidence_context` if it appears in the template.

```python
def with_evidence_priming(
    query_columns: list[str] | None = None,
    *,
    top_k: int = 3,
    min_score: float = 0.1,
) -> PipelineBuilder
```

**Parameters:**

- `query_columns` — Which input columns to concatenate as the search query. Defaults to the pipeline's input columns when `None`.
- `top_k` — Maximum number of evidence records to retrieve per row.
- `min_score` — Minimum relevance score to include a result (0.0–1.0). Results below this are discarded to prevent injecting low-relevance noise.

**Note:** On the first pipeline run the evidence store is empty, so evidence priming is a no-op. Evidence accumulates as grounding and verification store validated claims, so subsequent runs benefit from prior answers.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["product"], output_columns=["category"])
    .with_prompt(
        "Prior evidence:\n{_evidence_context}\n\n"
        "Classify this product: {product}\n\nCategory:"
    )
    .with_llm(model="openai/gpt-4o-mini")
    .with_context_store(RustContextStore("evidence.db"))
    .with_evidence_priming(query_columns=["product"], top_k=3, min_score=0.2)
    .build()
)
```

The `_evidence_context` column contains formatted evidence with relevance scores and source references:

```
[score=0.87](source: classification-run-1) Produce
---
[score=0.72](source: classification-run-1) Snacks
```

---

### with_grounding()

Verifies each LLM response against its source text after inference. Adds `grounding_score` (float, 0–1) and `grounding_flag` (bool, `True` when the score is below the threshold) columns to the output.

```python
def with_grounding(
    threshold: float = 0.3,
    action: str = "flag",
    embed_fn: callable | None = None,
) -> PipelineBuilder
```

**Parameters:**

- `threshold` — Minimum similarity score to consider a response grounded (0.0–1.0). Responses scoring below this value are considered ungrounded.
- `action` — What to do with ungrounded responses:
  - `"flag"` (default) — add `grounding_score` and `grounding_flag` columns and continue.
  - `"retry"` — re-prompt the LLM for that row.
  - `"skip"` — drop the ungrounded row from the output.
- `embed_fn` — Optional callable with signature `(list[str]) -> list[list[float]]`. When provided, dense embedding cosine similarity is computed alongside TF-IDF. The final score is `max(tfidf_score, embedding_score)`, so embeddings can rescue claims that TF-IDF misses due to vocabulary mismatch.

```python
# Basic grounding with flag action
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize: {text}")
    .with_llm(model="openai/gpt-4o-mini")
    .with_context_store(RustContextStore("evidence.db"))
    .with_grounding(threshold=0.3, action="flag")
    .build()
)
```

```python
# Grounding with dense embeddings for better recall
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize: {text}")
    .with_llm(model="openai/gpt-4o-mini")
    .with_context_store(RustContextStore("evidence.db"))
    .with_grounding(threshold=0.4, action="flag", embed_fn=embed)
    .build()
)
```

**Threshold guidance:**

| Threshold | Behavior |
|---|---|
| `0.1` | Very permissive — only the most divergent responses are flagged |
| `0.3` | Recommended starting point for most classification tasks |
| `0.5` | Moderate — suitable for factual Q&A where response must closely mirror source |
| `0.7+` | Strict — most responses will be flagged unless they nearly quote the source |

---

### with_contradiction_detection()

Flags rows whose output conflicts with a previously seen row that shares the same key column values. Adds a `contradiction_flag` (bool) column to the output.

```python
def with_contradiction_detection(
    key_columns: list[str] | None = None,
    value_columns: list[str] | None = None,
    tolerance: int | float | None = None,
) -> PipelineBuilder
```

**Parameters:**

- `key_columns` — Columns that identify the same entity (e.g., `["product_id"]`). Rows sharing the same values in these columns are compared. Defaults to the pipeline's input columns when `None`.
- `value_columns` — Columns holding the outputs to compare. Defaults to the pipeline's output columns when `None`.
- `tolerance` — Differences smaller than or equal to this value are not flagged as contradictions. `None` (default) uses exact string equality. Useful for numeric outputs — for example, `tolerance=1` on a 0–5 rating scale ignores ±1 score differences.

```python
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_id", "product_name"],
        output_columns=["category"],
    )
    .with_prompt("Classify {product_name}. Reply with only the category.\n\nCategory:")
    .with_llm(model="openai/gpt-4o-mini", temperature=0.0)
    .with_context_store(RustContextStore("evidence.db"))
    .with_contradiction_detection(key_columns=["product_id"])
    .build()
)
```

A `contradiction_flag` value of `True` means this row's output differs from a previously seen row with the same `product_id`. Investigate these rows to determine whether the LLM is inconsistent or the input data contains genuinely different items.

---

### with_confidence_scoring()

Adds a `confidence_score` column (float, 0–1) computed from the grounding similarity and evidence support count.

```python
def with_confidence_scoring(
    include_in_output: bool = True,
    scoring_mode: str = "default",
) -> PipelineBuilder
```

**Parameters:**

- `include_in_output` — Whether to write the `confidence_score` column to the output DataFrame.
- `scoring_mode` — Formula to use:
  - `"default"` — blends grounding score and evidence support count.
  - `"sigmoid"` — applies a sigmoid transform to the grounding score only.
  - `"grounding_only"` — passes the raw grounding score through without blending.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["label"])
    .with_prompt("Label this text: {text}\n\nLabel:")
    .with_llm(model="openai/gpt-4o-mini")
    .with_context_store()
    .with_grounding(threshold=0.3)
    .with_confidence_scoring(scoring_mode="sigmoid")
    .build()
)
```

---

## Full anti-hallucination pipeline

The following example combines all four features:

```python
import pandas as pd
from ondine import PipelineBuilder
from ondine.context import RustContextStore

data = pd.DataFrame({
    "product_id":   ["SKU-001", "SKU-002", "SKU-003", "SKU-001"],
    "product_name": [
        "Organic Fuji Apples 3lb Bag",
        "Kirkland Signature Almond Butter 27oz",
        "Blue Diamond Roasted Almonds 16oz",
        "Organic Fuji Apples 3lb Bag",  # duplicate — should match first row
    ],
})

pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        data,
        input_columns=["product_id", "product_name"],
        output_columns=["category"],
    )
    .with_prompt(
        "Classify this grocery product into exactly ONE category.\n"
        "Choose from: Produce, Pantry, Snacks, Household.\n"
        "Reply with ONLY the category name.\n\n"
        "Product: {product_name}\n\n"
        "Category:"
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .with_context_store(RustContextStore("evidence.db"))
    .with_grounding(threshold=0.3, action="flag")
    .with_contradiction_detection(key_columns=["product_id"])
    .with_confidence_scoring(scoring_mode="sigmoid")
    .build()
)

result = pipeline.execute()
print(result.data[
    ["product_id", "product_name", "category",
     "grounding_score", "grounding_flag",
     "contradiction_flag", "confidence_score"]
])
```

Output columns added by the anti-hallucination stack:

| Column | Type | When present | Description |
|---|---|---|---|
| `grounding_score` | float | `with_grounding` enabled | TF-IDF (or max TF-IDF/embedding) similarity of the LLM response to its source text |
| `grounding_flag` | bool | `with_grounding` enabled | `True` when `grounding_score < threshold` |
| `contradiction_flag` | bool | `with_contradiction_detection` enabled | `True` when this row's output conflicts with a prior row sharing the same key columns |
| `confidence_score` | float | `with_confidence_scoring` enabled | Composite score (0–1) computed per the selected `scoring_mode` |
| `_evidence_context` | str | `with_evidence_priming` enabled | Formatted prior evidence injected into the prompt |
| `_evidence_count` | int | `with_evidence_priming` enabled | Number of evidence records that exceeded `min_score` |

---

## Zep Cloud setup

1. Create an account at [https://www.getzep.com/](https://www.getzep.com/) and obtain an API key.

2. Install the optional dependency:

    ```bash
    pip install "ondine[zep]"
    ```

3. Set the environment variable:

    ```bash
    export ZEP_API_KEY=your-key-here
    ```

4. Use `ZepContextStore` in your pipeline:

    ```python
    from ondine import PipelineBuilder
    from ondine.context import ZepContextStore

    # Shared graph — persists across pipeline runs
    store = ZepContextStore(graph_id="my-project-evidence")

    pipeline = (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["question"], output_columns=["answer"])
        .with_prompt("Answer: {question}")
        .with_llm(model="openai/gpt-4o-mini")
        .with_context_store(store)
        .with_grounding(threshold=0.3)
        .build()
    )
    ```

Zep graphs are created automatically on first use. Storing to the same `graph_id` from multiple pipeline runs accumulates evidence over time. Use a unique `graph_id` per project or experiment to keep evidence namespaces separate.

`ZepContextStore` does not implement `ground()` or `add_contradiction()` — those operations fall back to no-ops. For full grounding and contradiction support with cloud persistence, pair `ZepContextStore` for storage and search with `RustContextStore` for grounding operations, or use `RustContextStore` exclusively.

---

## Choosing a backend

| Requirement | Recommended backend |
|---|---|
| Production pipeline, persistent evidence | `RustContextStore("evidence.db")` |
| Multi-run evidence accumulation | `RustContextStore("evidence.db")` |
| Managed cloud storage, entity extraction | `ZepContextStore` |
| Unit tests, CI with no Rust toolchain | `InMemoryContextStore` |
| Quick prototype, no persistence needed | `with_context_store()` (auto-detects) |

---

## Related

- [RAG Knowledge Base](../../examples/rag_knowledge_base_example.py) — retrieval-augmented generation with `KnowledgeStore`
- [Context Store Example](../../examples/context_store_example.py) — full anti-hallucination pipeline
- [Cost Control](cost-control.md) — budget limits and cost estimation
- [Structured Output](structured-output.md) — type-safe LLM responses
