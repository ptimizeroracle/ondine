"""Processing stages for the Ondine data-transformation pipeline.

Each stage implements the ``PipelineStage`` protocol and is composed into
a linear pipeline by the ``PipelineBuilder``.  Stages are executed in
order; each receives the output of the previous stage.

Core stages (in typical execution order)
-----------------------------------------
- ``DataLoaderStage`` -- Load input data from CSV, DataFrame, or other
  sources into a ``DataContainer``.
- ``EvidenceRetrievalStage`` -- Pre-LLM evidence priming: search the
  context store for previously validated answers and inject them into
  each row as ``_evidence_context`` so the LLM can leverage prior
  evidence for consistency.
- ``KnowledgeRetrievalStage`` -- RAG retrieval: search the knowledge
  base for relevant chunks and inject them as ``_kb_context``.
- ``BatchDisaggregatorStage`` -- Split a ``DataContainer`` into
  individual prompt items for parallel LLM processing.
- ``PromptFormatterStage`` -- Render prompt templates with per-row
  variables (including ``_evidence_context`` and ``_kb_context``).
- ``LLMInvocationStage`` -- Send formatted prompts to the configured
  LLM provider and collect responses.
- ``ResponseParserStage`` -- Parse raw LLM text into structured output
  using one of the built-in parsers (JSON, Pydantic, Regex, raw text).
- ``BatchAggregatorStage`` -- Re-assemble individual results into a
  single ``DataContainer``.
- ``ResultWriterStage`` -- Write final results to CSV or other sinks.

Response parsers
----------------
- ``ResponseParser`` (ABC), ``RawTextParser``, ``JSONParser``,
  ``PydanticParser``, ``RegexParser``.
- ``create_response_parser`` -- Factory function to instantiate a parser
  by name.

Batch processing utilities
--------------------------
- ``BatchProcessor`` -- Process items in configurable batches.
- ``BatchMap`` -- Map a function over batches.
- ``PromptItem`` -- A single prompt with metadata for batch processing.

Stage registry
--------------
- ``StageRegistry`` -- Discover and register stages by name.
- ``@stage`` -- Decorator to register a custom stage class.

Examples
--------
Using the builder API (most common)::

    from ondine.api import PipelineBuilder

    pipeline = (
        PipelineBuilder.create()
        .from_csv("data.csv", input_columns=["text"], output_columns=["label"])
        .with_prompt("Classify: {text}")
        .with_llm(model="openai/gpt-4o-mini")
        .with_context_store()
        .with_evidence_priming(query_columns=["text"], top_k=3)
        .to_csv("results.csv")
        .build()
    )

Direct stage composition::

    from ondine.stages import (
        DataLoaderStage,
        EvidenceRetrievalStage,
        PromptFormatterStage,
        LLMInvocationStage,
        ResponseParserStage,
        ResultWriterStage,
    )
"""

from ondine.stages.batch_aggregator_stage import BatchAggregatorStage
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.stages.batch_processor import BatchMap, BatchProcessor, PromptItem
from ondine.stages.data_loader_stage import DataLoaderStage
from ondine.stages.evidence_retrieval_stage import EvidenceRetrievalStage
from ondine.stages.knowledge_retrieval_stage import KnowledgeRetrievalStage
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.stages.parser_factory import create_response_parser
from ondine.stages.pipeline_stage import PipelineStage
from ondine.stages.prompt_formatter_stage import (
    PromptFormatterStage,
)
from ondine.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
    ResponseParserStage,
)
from ondine.stages.result_writer_stage import ResultWriterStage
from ondine.stages.stage_registry import StageRegistry, stage

__all__ = [
    "PipelineStage",
    "BatchAggregatorStage",
    "BatchDisaggregatorStage",
    "DataLoaderStage",
    "PromptFormatterStage",
    "LLMInvocationStage",
    "ResponseParserStage",
    "ResultWriterStage",
    "ResponseParser",
    "RawTextParser",
    "JSONParser",
    "PydanticParser",
    "RegexParser",
    "create_response_parser",
    # Knowledge retrieval
    "KnowledgeRetrievalStage",
    # Evidence priming
    "EvidenceRetrievalStage",
    # Stage Registry
    "StageRegistry",
    "stage",
    # Batch processing utilities
    "BatchProcessor",
    "BatchMap",
    "PromptItem",
]
