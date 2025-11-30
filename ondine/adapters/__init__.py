"""Infrastructure adapters for external systems."""

from ondine.adapters.checkpoint_storage import (
    CheckpointStorage,
    LocalFileCheckpointStorage,
)
from ondine.adapters.data_io import (
    CSVReader,
    CSVWriter,
    DataFrameReader,
    DataReader,
    DataWriter,
    ExcelReader,
    ExcelWriter,
    ParquetReader,
    ParquetWriter,
    create_data_reader,
    create_data_writer,
)
from ondine.adapters.llm_client import LLMClient, MLXClient, create_llm_client
from ondine.adapters.provider_registry import ProviderRegistry, provider
from ondine.adapters.streaming_loader import StreamingDataLoader
from ondine.adapters.streaming_writer import (
    MultiFormatWriter,
    StreamingResultWriter,
)
from ondine.adapters.unified_litellm_client import UnifiedLiteLLMClient

__all__ = [
    # LLM Clients (PUBLIC API)
    "LLMClient",  # Abstract base - for type hints only
    "create_llm_client",  # Factory - PRIMARY API for creating clients
    "UnifiedLiteLLMClient",  # Direct LiteLLM client (for advanced usage)
    # Provider Registry (ADVANCED API - for custom providers)
    "ProviderRegistry",
    "provider",
    # Special Clients (ADVANCED API - direct usage)
    "MLXClient",  # For Apple Silicon local inference
    # Data I/O
    "DataReader",
    "DataWriter",
    "CSVReader",
    "CSVWriter",
    "ExcelReader",
    "ExcelWriter",
    "ParquetReader",
    "ParquetWriter",
    "DataFrameReader",
    "create_data_reader",
    "create_data_writer",
    # Streaming I/O (for large datasets)
    "StreamingDataLoader",
    "StreamingResultWriter",
    "MultiFormatWriter",
    # Checkpoint Storage
    "CheckpointStorage",
    "LocalFileCheckpointStorage",
]
