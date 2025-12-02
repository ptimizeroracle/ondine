"""Core configuration and data models."""

from ondine.core.data_container import (
    AsyncDataContainer,
    AsyncRowIterator,
    BaseDataContainer,
    DataContainer,
    ResultContainer,
    Row,
    RowIterator,
)
from ondine.core.exceptions import (
    ConfigurationError,
    InvalidAPIKeyError,
    ModelNotFoundError,
    NonRetryableError,
    QuotaExceededError,
)
from ondine.core.models import (
    CheckpointInfo,
    CostEstimate,
    ErrorInfo,
    ExecutionResult,
    LLMResponse,
    ProcessingStats,
    PromptBatch,
    ResponseBatch,
    RowMetadata,
    ValidationResult,
    WriteConfirmation,
)
from ondine.core.router_strategies import RouterStrategy
from ondine.core.specifications import (
    DatasetSpec,
    DataSourceType,
    ErrorPolicy,
    LLMProvider,
    LLMSpec,
    MergeStrategy,
    OutputSpec,
    PipelineSpecifications,
    ProcessingSpec,
    PromptSpec,
)

__all__ = [
    # Data Container Protocol
    "DataContainer",
    "AsyncDataContainer",
    "ResultContainer",
    "BaseDataContainer",
    "Row",
    "RowIterator",
    "AsyncRowIterator",
    # Specifications
    "DatasetSpec",
    "PromptSpec",
    "LLMSpec",
    "ProcessingSpec",
    "OutputSpec",
    "PipelineSpecifications",
    # Enums
    "DataSourceType",
    "LLMProvider",
    "ErrorPolicy",
    "MergeStrategy",
    "RouterStrategy",
    # Models
    "LLMResponse",
    "CostEstimate",
    "ProcessingStats",
    "ErrorInfo",
    "ExecutionResult",
    "ValidationResult",
    "WriteConfirmation",
    "CheckpointInfo",
    "RowMetadata",
    "PromptBatch",
    "ResponseBatch",
    # Exceptions
    "NonRetryableError",
    "ModelNotFoundError",
    "InvalidAPIKeyError",
    "ConfigurationError",
    "QuotaExceededError",
]
