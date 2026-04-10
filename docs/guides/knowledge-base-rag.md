# Knowledge Base & RAG Guide

Add retrieval-augmented generation (RAG) to your pipelines: ingest documents, search them with hybrid BM25 + dense vector retrieval, and inject context into your LLM prompts automatically.

## Overview

The knowledge base system covers the full RAG lifecycle:

- **Ingestion**: Load PDF, Markdown, plain text, CSV, HTML, and image files
- **Chunking**: Split documents into semantically coherent chunks
- **Embedding**: Encode chunks as dense vectors for semantic search
- **Hybrid search**: Combine BM25 full-text search and dense vector retrieval via Reciprocal Rank Fusion (RRF)
- **Reranking**: Re-score results with a cross-encoder for higher precision
- **Query transformation**: Expand or rewrite queries before retrieval to improve recall
- **OCR**: Extract text from images and scanned PDFs
- **Evaluation**: Score RAG answers for faithfulness, relevancy, and context precision using an LLM judge

The primary entry point is `KnowledgeStore`. It exposes two public methods — `ingest()` and `search()` — and hides all intermediate steps behind that interface.

## Installation

Install the `knowledge` extra to get PDF loading and local embedding support:

```bash
pip install 'ondine[knowledge]'
```

This adds:
- `pymupdf` — PDF text extraction
- `sentence-transformers` — local embedding and cross-encoder reranking

For API-based embedders, rerankers, query transformers, and OCR, install `litellm`:

```bash
pip install litellm
```

For offline OCR alternatives:

```bash
# Tesseract (requires the tesseract binary on your PATH)
pip install pytesseract Pillow

# DocTR deep-learning OCR
pip install python-doctr
```

## Quick Start

```python
from ondine.knowledge import KnowledgeStore

# Create a persistent knowledge base (or use ":memory:" for tests)
kb = KnowledgeStore("knowledge.db")

# Ingest a directory of documents
kb.ingest("docs/")

# Search
results = kb.search("How does authentication work?", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.source}: {r.text[:120]}...")
```

### Using with the Pipeline Builder

The most common use case is attaching a `KnowledgeStore` to a processing pipeline so that every row is automatically augmented with retrieved context:

```python
import pandas as pd
from ondine import PipelineBuilder
from ondine.knowledge import KnowledgeStore

# Build and populate the knowledge base
kb = KnowledgeStore("knowledge.db")
kb.ingest("docs/")

# Build a RAG pipeline
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(
        df,
        input_columns=["question"],
        output_columns=["answer"],
    )
    .with_knowledge_base(kb, top_k=5)
    .with_prompt(
        "Answer the question using only the context provided.\n\n"
        "Context:\n{_kb_context}\n\n"
        "Question: {question}\n\nAnswer:"
    )
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

result = pipeline.execute()
```

The pipeline inserts a retrieval stage before prompt formatting. Each row's query columns are concatenated, searched against the knowledge base, and the top chunks are joined into the `{_kb_context}` template variable.

## KnowledgeStore

### Constructor

```python
KnowledgeStore(
    db_path: str = ":memory:",
    *,
    chunker: SemanticChunker | None = None,
    embedder: Embedder | str | None = None,
    reranker: Reranker | str | bool | None = None,
    query_transform: QueryTransformer | str | None = None,
    ocr: OCRProvider | str | None = None,
    extract_pdf_images: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `":memory:"` | SQLite file path or `":memory:"` for an in-process store |
| `chunker` | `SemanticChunker \| None` | `None` | Custom chunker; `None` uses `SemanticChunker()` defaults |
| `embedder` | `Embedder \| str \| None` | `None` | Embedding model; `None` auto-detects `SentenceTransformerEmbedder` if available |
| `reranker` | `Reranker \| str \| bool \| None` | `None` | Reranker to apply after retrieval; `False`/`None` disables, `True` uses the default cross-encoder |
| `query_transform` | `QueryTransformer \| str \| None` | `None` | Query expansion strategy; `None` disables |
| `ocr` | `OCRProvider \| str \| None` | `None` | OCR provider for image files; `None` skips images during ingest |
| `extract_pdf_images` | `bool` | `False` | When `True` and `ocr` is configured, also OCR embedded images in PDFs |

### Ingestion methods

```python
# Load from a file or directory (recursive)
kb.ingest(path: str | Path) -> int

# Ingest pre-loaded Document objects
kb.ingest_documents(docs: list[Document]) -> int

# Ingest raw text without file I/O
kb.ingest_text(text: str, source: str = "inline", metadata: dict | None = None) -> int
```

All three return the number of chunks stored.

### Search

```python
kb.search(query: str, limit: int = 5) -> list[SearchResult]
```

Returns a list of `SearchResult` objects:

```python
@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict
```

### Properties

```python
kb.chunk_count  # int: number of chunks currently stored
```

## Document Loading

`KnowledgeStore.ingest()` delegates to `DocumentLoader`, which dispatches by file extension:

| Extension(s) | Reader | Extra required |
|---|---|---|
| `.pdf` | PyMuPDF | `ondine[knowledge]` |
| `.md`, `.txt`, `.csv`, `.tsv`, `.json`, `.xml`, `.html`, `.htm` | stdlib text reader | — |
| `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.gif` | OCR provider | OCR configured |

When you call `kb.ingest("docs/")` on a directory, all supported files are loaded recursively.

### Loading text inline

```python
kb.ingest_text(
    text="All orders ship within 2 business days.",
    source="policy/shipping.md",
    metadata={"section": "shipping"},
)
```

### Loading pre-built Document objects

Use `ingest_documents()` when you need to control text extraction yourself:

```python
from ondine.knowledge import KnowledgeStore
from ondine.knowledge.loader import Document

docs = [
    Document(text="...", source="custom://page-1", metadata={"page": 1}),
    Document(text="...", source="custom://page-2", metadata={"page": 2}),
]

kb = KnowledgeStore(":memory:")
kb.ingest_documents(docs)
```

`Document` is a frozen dataclass with three fields: `text: str`, `source: str`, and `metadata: dict`.

## Chunking

`SemanticChunker` splits documents into coherent chunks by detecting breakpoints where sentence-level embeddings diverge. When `sentence-transformers` is not installed it falls back gracefully to fixed-size sentence-count windows.

### Default behaviour

```python
from ondine.knowledge import KnowledgeStore

# Uses SemanticChunker with default settings
kb = KnowledgeStore("knowledge.db")
```

### Customising the chunker

```python
from ondine.knowledge import KnowledgeStore
from ondine.knowledge.chunker import SemanticChunker

chunker = SemanticChunker(
    max_chunk_tokens=256,       # soft upper bound per chunk (whitespace tokens)
    breakpoint_percentile=0.20, # lower = more splits; 0.0–1.0
    model_name="all-MiniLM-L6-v2",  # sentence-transformers model for breakpoint detection
)

kb = KnowledgeStore("knowledge.db", chunker=chunker)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_tokens` | `512` | Soft upper bound on chunk size in whitespace tokens |
| `breakpoint_percentile` | `0.25` | Percentile threshold for breakpoint detection. Lower values produce more, smaller chunks |
| `model_name` | `"all-MiniLM-L6-v2"` | `sentence-transformers` model used for similarity scoring during splitting |

## Embedding Models

Embeddings enable dense vector search. Without an embedder, `KnowledgeStore` falls back to BM25 full-text search only.

### Auto-detection (default)

When `embedder=None`, the store attempts to load `SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5")`. If `sentence-transformers` is not installed, embeddings are silently disabled and only BM25 is used.

### SentenceTransformerEmbedder (local)

```python
from ondine.knowledge import KnowledgeStore, SentenceTransformerEmbedder

kb = KnowledgeStore(
    "knowledge.db",
    embedder=SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5"),
)
```

Or pass the model name as a string:

```python
kb = KnowledgeStore("knowledge.db", embedder="BAAI/bge-large-en-v1.5")
```

The model is loaded lazily on the first embed call.

### OpenAIEmbedder (API-based)

Uses `litellm` for provider-agnostic access. Works with OpenAI, Cohere, Azure, and any litellm-supported provider:

```python
from ondine.knowledge import KnowledgeStore, OpenAIEmbedder

kb = KnowledgeStore(
    "knowledge.db",
    embedder=OpenAIEmbedder(
        model="text-embedding-3-small",
        api_key="sk-...",     # optional; falls back to OPENAI_API_KEY env var  # pragma: allowlist secret
        dimensions=1536,      # optional; reduce dimensions for smaller vectors
    ),
)
```

Passing a string that contains `"text-embedding"` also selects `OpenAIEmbedder`:

```python
kb = KnowledgeStore("knowledge.db", embedder="text-embedding-3-small")
```

### Custom embedder

Any object with an `embed(texts: list[str]) -> list[list[float]]` method works:

```python
class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # your implementation
        ...

kb = KnowledgeStore("knowledge.db", embedder=MyEmbedder())
```

## Reranking

Reranking re-scores the initial hybrid-search candidates using a cross-attention model. This improves precision at the cost of an extra inference step.

### Enable with default cross-encoder

```python
kb = KnowledgeStore("knowledge.db", reranker=True)
```

This uses `CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")`.

### CrossEncoderReranker (local)

```python
from ondine.knowledge import KnowledgeStore, CrossEncoderReranker

kb = KnowledgeStore(
    "knowledge.db",
    reranker=CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=5,
    ),
)
```

Or pass the model name as a string:

```python
kb = KnowledgeStore("knowledge.db", reranker="cross-encoder/ms-marco-MiniLM-L-12-v2")
```

### JinaReranker (API-based)

```python
from ondine.knowledge import KnowledgeStore, JinaReranker

kb = KnowledgeStore(
    "knowledge.db",
    reranker=JinaReranker(
        model="jina-reranker-v2-base-multilingual",
        top_k=5,
        api_key="jina_...",   # or set JINA_API_KEY env var  # pragma: allowlist secret
    ),
)
```

Passing a string containing `"jina"` also selects `JinaReranker`:

```python
kb = KnowledgeStore("knowledge.db", reranker="jina-reranker-v2-base-multilingual")
```

### Custom reranker

Any object with a `rerank(query: str, results: list, top_k: int | None) -> list` method works:

```python
class MyReranker:
    def rerank(self, query: str, results: list, top_k: int | None = None) -> list:
        ...

kb = KnowledgeStore("knowledge.db", reranker=MyReranker())
```

## Query Transformation

Query transformers expand or rewrite the query before retrieval. The store runs hybrid search for each expanded query, deduplicates results, then optionally reranks the merged set.

All three built-in transformers require `litellm` and an API key for the underlying LLM.

### MultiQueryTransformer

Generates N rephrasings of the original query. Retrieval unions results across all variants, improving recall for ambiguous queries.

```python
from ondine.knowledge import KnowledgeStore, MultiQueryTransformer

kb = KnowledgeStore(
    "knowledge.db",
    query_transform=MultiQueryTransformer(
        model="openai/gpt-4o-mini",
        n=3,                    # number of rephrasings to generate
        api_key="sk-...",       # optional  # pragma: allowlist secret
    ),
)
```

Or pass the shortcut string:

```python
kb = KnowledgeStore("knowledge.db", query_transform="multi-query")
```

**How it works:** The transformer asks the LLM to return a JSON array of `n` alternative formulations. Each formulation is searched independently; results are merged and deduplicated before reranking.

### HyDETransformer

Hypothetical Document Embeddings. The LLM generates a short hypothetical answer to the query; that answer is used as the retrieval query. Because the hypothesis is semantically closer to relevant passages than the raw question, dense retrieval improves.

```python
from ondine.knowledge import KnowledgeStore, HyDETransformer

kb = KnowledgeStore(
    "knowledge.db",
    query_transform=HyDETransformer(
        model="openai/gpt-4o-mini",
        api_key="sk-...",       # optional  # pragma: allowlist secret
    ),
)
```

Or:

```python
kb = KnowledgeStore("knowledge.db", query_transform="hyde")
```

**Returns:** `[original_query, hypothesis]`. Both are searched; results are merged before reranking.

### StepBackTransformer

Generates a more abstract, generalised version of the query so that retrieval can surface broader context that the specific query might miss.

```python
from ondine.knowledge import KnowledgeStore, StepBackTransformer

kb = KnowledgeStore(
    "knowledge.db",
    query_transform=StepBackTransformer(
        model="openai/gpt-4o-mini",
        api_key="sk-...",       # optional  # pragma: allowlist secret
    ),
)
```

Or:

```python
kb = KnowledgeStore("knowledge.db", query_transform="step-back")
```

**Returns:** `[original_query, step_back_query]`. Both are searched; results are merged.

### Combining reranking and query transformation

```python
kb = KnowledgeStore(
    "knowledge.db",
    embedder="BAAI/bge-base-en-v1.5",
    reranker=True,
    query_transform="hyde",
)
kb.ingest("docs/")

results = kb.search("what are the rate limits for the API?", limit=5)
```

Search flow with both enabled:
1. `HyDETransformer` produces `[original_query, hypothesis]`
2. Hybrid search runs for each variant; results are deduplicated
3. `CrossEncoderReranker` re-scores the merged set and returns top-5

## OCR Support

OCR providers extract text from image files (`.png`, `.jpg`, `.webp`, etc.) and optionally from images embedded in PDF pages.

### VisionOCR (multimodal LLM)

Uses a vision-capable LLM via `litellm`. Best quality for complex layouts, charts, and tables.

```python
from ondine.knowledge import KnowledgeStore, VisionOCR

kb = KnowledgeStore(
    "knowledge.db",
    ocr=VisionOCR(
        model="gpt-4o",
        api_key="sk-...",   # optional  # pragma: allowlist secret
    ),
)
kb.ingest("scanned_docs/")
```

Or use the shortcut string `"vision"` (defaults to `gpt-4o`):

```python
kb = KnowledgeStore("knowledge.db", ocr="vision")
```

You can also pass any litellm model name directly:

```python
kb = KnowledgeStore("knowledge.db", ocr="anthropic/claude-3-5-sonnet-20241022")
```

### TesseractOCR (local, offline)

Requires the `tesseract` binary on your system PATH and the `pytesseract` and `Pillow` Python packages.

```bash
# macOS
brew install tesseract

# Debian/Ubuntu
sudo apt install tesseract-ocr
```

```python
from ondine.knowledge import KnowledgeStore, TesseractOCR

kb = KnowledgeStore(
    "knowledge.db",
    ocr=TesseractOCR(
        lang="eng",     # Tesseract language code
        config="",      # extra Tesseract CLI flags
    ),
)
```

Or use the shortcut string:

```python
kb = KnowledgeStore("knowledge.db", ocr="tesseract")
```

### DocTROCR (local, deep-learning)

High-accuracy local OCR optimised for documents and screenshots. Requires `python-doctr`.

```python
from ondine.knowledge import KnowledgeStore, DocTROCR

kb = KnowledgeStore(
    "knowledge.db",
    ocr=DocTROCR(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
    ),
)
```

Or:

```python
kb = KnowledgeStore("knowledge.db", ocr="doctr")
```

### Extracting images embedded in PDFs

Set `extract_pdf_images=True` alongside any OCR provider to also OCR images embedded within PDF pages:

```python
kb = KnowledgeStore(
    "knowledge.db",
    ocr="vision",
    extract_pdf_images=True,
)
kb.ingest("reports/")  # text pages + embedded diagrams/screenshots are all indexed
```

Chunks from embedded images carry `{"format": "pdf_image", "extraction": "ocr", "page": N, "image_index": M}` in their metadata.

### Custom OCR provider

Any object with an `extract_text(image_path: str) -> str` method works:

```python
class MyOCR:
    def extract_text(self, image_path: str) -> str:
        ...

kb = KnowledgeStore("knowledge.db", ocr=MyOCR())
```

## Pipeline Builder Integration

### `with_knowledge_base()`

```python
PipelineBuilder.with_knowledge_base(
    store: KnowledgeStore,
    *,
    query_columns: list[str] | None = None,
    top_k: int = 3,
    rerank: bool = False,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    query_transform: str | None = None,
    evaluate: bool = False,
    eval_model: str = "openai/gpt-4o-mini",
) -> PipelineBuilder
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `store` | required | A pre-built `KnowledgeStore` instance |
| `query_columns` | `None` | Input columns to concatenate as the search query. `None` uses all input columns |
| `top_k` | `3` | Number of chunks to retrieve per row |
| `rerank` | `False` | Enable cross-encoder reranking of retrieved chunks |
| `reranker_model` | `"cross-encoder/ms-marco-MiniLM-L-12-v2"` | Model for reranking (only used when `rerank=True`) |
| `query_transform` | `None` | Query expansion strategy: `"multi-query"`, `"hyde"`, `"step-back"`, or `None` |
| `evaluate` | `False` | Run LLM-as-judge evaluation; adds `_kb_eval_*` columns to results |
| `eval_model` | `"openai/gpt-4o-mini"` | LLM for evaluation (only used when `evaluate=True`) |

**The `{_kb_context}` variable** is injected into the prompt template automatically. It contains the top-k retrieved chunks joined with newlines.

### Full pipeline example

```python
from ondine import PipelineBuilder
from ondine.knowledge import KnowledgeStore

# Build the knowledge base once
kb = KnowledgeStore("support_kb.db", embedder="BAAI/bge-base-en-v1.5")
kb.ingest("help_articles/")

# Build the pipeline
pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "customer_questions.csv",
        input_columns=["question"],
        output_columns=["answer"],
    )
    .with_knowledge_base(
        kb,
        top_k=5,
        rerank=True,
        query_transform="hyde",
        evaluate=True,
    )
    .with_prompt(
        "You are a helpful support agent. Use only the context below.\n\n"
        "Context:\n{_kb_context}\n\n"
        "Question: {question}\n\nAnswer:"
    )
    .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    .build()
)

result = pipeline.execute()
print(result.data[["question", "answer"]].to_string())
```

## Evaluation

`LLMJudge` scores a RAG answer on three dimensions using an LLM as evaluator.

### Using LLMJudge directly

```python
from ondine.knowledge.eval import LLMJudge

judge = LLMJudge(
    model="openai/gpt-4o-mini",
    api_key="sk-...",    # optional; falls back to env var  # pragma: allowlist secret
    temperature=0.0,
)

result = judge.score(
    query="What is the return window for perishables?",
    answer="Perishable items must be returned within 7 days.",
    contexts=["Perishable items must be returned within 7 days of purchase."],
)

print(result.faithfulness)       # 0.0–1.0
print(result.relevancy)          # 0.0–1.0
print(result.context_precision)  # 0.0–1.0
```

`EvalResult` is a frozen dataclass:

```python
@dataclass(frozen=True)
class EvalResult:
    faithfulness: float        # is the answer grounded in the contexts?
    relevancy: float           # does the answer address the query?
    context_precision: float   # are the retrieved contexts relevant?
    metadata: dict
```

### Automated evaluation in a pipeline

Pass `evaluate=True` to `with_knowledge_base()`. The pipeline runs the judge after each LLM call and adds three columns to the output DataFrame: `_kb_eval_faithfulness`, `_kb_eval_relevancy`, and `_kb_eval_context_precision`.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("questions.csv", input_columns=["question"], output_columns=["answer"])
    .with_knowledge_base(kb, top_k=5, evaluate=True, eval_model="openai/gpt-4o-mini")
    .with_prompt("Context:\n{_kb_context}\n\nQuestion: {question}\n\nAnswer:")
    .with_llm(model="openai/gpt-4o-mini")
    .build()
)

result = pipeline.execute()
eval_cols = ["question", "answer", "_kb_eval_faithfulness", "_kb_eval_relevancy", "_kb_eval_context_precision"]
print(result.data[eval_cols].to_string())
```

### Custom evaluator

Any object with a `score(query: str, answer: str, contexts: list[str]) -> EvalResult` method satisfies the `RetrievalScorer` protocol.

## Persistent vs In-Memory Storage

```python
# In-memory: fast, no disk I/O, lost when the process exits
kb = KnowledgeStore(":memory:")

# Persistent: survives restarts, can be shared across pipelines
kb = KnowledgeStore("knowledge.db")
```

For production use, ingest once and reuse the same `db_path`:

```python
# ingest_once.py — run once to populate
kb = KnowledgeStore("knowledge.db")
kb.ingest("docs/")
print(f"Stored {kb.chunk_count} chunks")

# pipeline.py — load and search without re-ingesting
kb = KnowledgeStore("knowledge.db")  # no ingest() call needed
results = kb.search("authentication flow")
```

## Supported File Types

| Format | Extensions | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Requires `ondine[knowledge]`; each page is a separate `Document` |
| Markdown | `.md` | |
| Plain text | `.txt` | |
| CSV / TSV | `.csv`, `.tsv` | Loaded as raw text |
| JSON | `.json` | Loaded as raw text |
| XML / HTML | `.xml`, `.html`, `.htm` | Loaded as raw text |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.gif` | Requires OCR provider |

## Custom Components

All components are specified by protocol. Passing any object with the matching method signature works without subclassing.

| Protocol | Method signature | Used for |
|----------|-----------------|----------|
| `Embedder` | `embed(texts: list[str]) -> list[list[float]]` | Dense vector encoding |
| `Reranker` | `rerank(query: str, results: list, top_k: int \| None) -> list` | Result re-scoring |
| `QueryTransformer` | `transform(query: str) -> list[str]` | Query expansion |
| `OCRProvider` | `extract_text(image_path: str) -> str` | Image-to-text extraction |
| `RetrievalScorer` | `score(query: str, answer: str, contexts: list[str]) -> EvalResult` | RAG evaluation |

## Best Practices

**Chunking size:** Smaller chunks (`max_chunk_tokens=256`) improve precision but may lose surrounding context. Larger chunks (`512–1024`) preserve more context but dilute relevance scores. Start with the default of 512 and adjust based on your evaluation scores.

**Embedder choice:** `BAAI/bge-base-en-v1.5` (default) is a strong all-around local model. Use `text-embedding-3-small` if you prefer API-based embeddings and want to avoid local GPU/CPU overhead.

**Always rerank for production:** Enable `reranker=True` when retrieval quality matters. The cross-encoder runs locally with no API cost and typically improves precision significantly.

**Query transformation:** Use `"hyde"` for question-answering tasks where queries are questions and documents are factual passages. Use `"multi-query"` for broad, exploratory retrieval. Use `"step-back"` when queries are highly specific and may miss broader conceptual matches.

**Combine strategies:** Reranking and query transformation stack well. Query transformation increases recall; reranking restores precision on the merged candidate set.

**Persistent storage:** Use a file path instead of `":memory:"` in production. Re-ingesting large document sets is expensive; a persistent database allows you to separate the ingest step from the query step.

## Troubleshooting

### No results returned

- Check `kb.chunk_count` to confirm documents were ingested.
- Verify the file extension is in the supported list.
- If ingesting images and getting no results, confirm an OCR provider is configured.

### Low retrieval quality

- Enable reranking (`reranker=True`).
- Try a query transformation strategy (`query_transform="hyde"` is a good first choice).
- Increase `limit` in `search()` or `top_k` in `with_knowledge_base()` to surface more candidates.
- Reduce `breakpoint_percentile` in `SemanticChunker` to create smaller, more focused chunks.

### `ImportError: PyMuPDF is required`

Install the knowledge extra: `pip install 'ondine[knowledge]'`.

### `sentence-transformers not installed; using fixed-size chunking`

This is an info-level log, not an error. Install `sentence-transformers` (included in `ondine[knowledge]`) for semantic chunking. Without it, the chunker falls back to fixed-size windows.

### Query transformation has no effect

All query transformers require `litellm` and a valid API key. Check that `litellm` is installed and the relevant environment variable (e.g. `OPENAI_API_KEY`) is set. When a transformer call fails, it logs a warning and falls back to the original query.

## API Reference

### `KnowledgeStore`

| Method / Property | Signature | Description |
|---|---|---|
| `ingest` | `(path: str \| Path) -> int` | Load files from a path and store chunks |
| `ingest_documents` | `(docs: list[Document]) -> int` | Chunk and store pre-loaded `Document` objects |
| `ingest_text` | `(text: str, source: str, metadata: dict \| None) -> int` | Store raw text |
| `search` | `(query: str, limit: int = 5) -> list[SearchResult]` | Hybrid search with optional transform and rerank |
| `chunk_count` | `int` (property) | Number of chunks stored |

### `SearchResult`

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Stable unique identifier |
| `text` | `str` | Chunk text content |
| `source` | `str` | File path or source label |
| `score` | `float` | Retrieval or reranker score |
| `metadata` | `dict` | Document metadata (page number, format, etc.) |

### `EvalResult`

| Field | Type | Description |
|---|---|---|
| `faithfulness` | `float` | 0–1; answer grounded in retrieved context |
| `relevancy` | `float` | 0–1; answer addresses the query |
| `context_precision` | `float` | 0–1; retrieved contexts are relevant |
| `metadata` | `dict` | Evaluator metadata (model used, raw scores) |

## Examples

See `examples/rag_knowledge_base_example.py` for a complete working example:
- Ingest product policy documents as inline text
- Build a pipeline that answers customer questions with KB context
- Cost estimation before execution
- Display results and execution metrics
