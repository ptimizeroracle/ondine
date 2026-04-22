"""
LLM Dataset Processing Engine.

An SDK for processing tabular datasets using Large Language
Models with reliability, observability, and cost control.
"""

import logging
import os
import warnings

# Suppress transformers warnings about missing deep learning frameworks
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress common dependency warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

# Suppress HTTP request logs from shared HTTP client libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__version__ = "1.9.0"

# Layer 4: High-Level API
from ondine.api.dataset_processor import DatasetProcessor
from ondine.api.pipeline import Pipeline
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.api.quick import QuickPipeline

# Context store (pluggable anti-hallucination backends)
from ondine.context import (
    ContextStore,
    EvidenceRecord,
    GroundingResult,
    RetrievalResult,
)

# Core result models
from ondine.core.models import (
    CostEstimate,
    ExecutionResult,
    ProcessingStats,
    QualityReport,
)

# Core configuration models
from ondine.core.router_strategies import RouterStrategy
from ondine.core.specifications import (
    DatasetSpec,
    LLMSpec,
    PipelineSpecifications,
    ProcessingSpec,
    PromptSpec,
)

__all__ = [
    "__version__",
    "Pipeline",
    "PipelineBuilder",
    "QuickPipeline",
    "DatasetProcessor",
    "DatasetSpec",
    "PromptSpec",
    "LLMSpec",
    "ProcessingSpec",
    "PipelineSpecifications",
    "RouterStrategy",
    "ExecutionResult",
    "QualityReport",
    "ProcessingStats",
    "CostEstimate",
    "ContextStore",
    "EvidenceRecord",
    "GroundingResult",
    "RetrievalResult",
    "RustContextStore",
    "ZepContextStore",
    "InMemoryContextStore",
]


def __getattr__(name: str):
    if name == "RustContextStore":
        from ondine.context.rust_store import RustContextStore

        return RustContextStore
    if name == "ZepContextStore":
        from ondine.context.zep_store import ZepContextStore

        return ZepContextStore
    if name == "InMemoryContextStore":
        from ondine.context.memory_store import InMemoryContextStore

        return InMemoryContextStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
