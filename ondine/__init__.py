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

# Suppress HTTP request logs from httpx/httpcore (used by llama_index)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__version__ = "1.3.1"

# Layer 4: High-Level API
from ondine.api.dataset_processor import DatasetProcessor
from ondine.api.pipeline import Pipeline
from ondine.api.pipeline_builder import PipelineBuilder
from ondine.api.quick import QuickPipeline

# Core result models
from ondine.core.models import (
    CostEstimate,
    ExecutionResult,
    ProcessingStats,
    QualityReport,
)

# Core configuration models
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
    "ExecutionResult",
    "QualityReport",
    "ProcessingStats",
    "CostEstimate",
]
