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
from ondine.adapters.llm_client import (
    AzureOpenAIClient,
    LLMClient,
    MLXClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from ondine.adapters.provider_registry import ProviderRegistry, provider

__all__ = [
    # LLM Clients (PUBLIC API)
    "LLMClient",  # Abstract base - for type hints only
    "create_llm_client",  # Factory - PRIMARY API for creating clients
    # Provider Registry (ADVANCED API - for custom providers)
    "ProviderRegistry",
    "provider",
    # Special Clients (ADVANCED API - direct usage)
    "AzureOpenAIClient",  # For Azure Managed Identity (enterprise)
    "MLXClient",  # For Apple Silicon local inference
    "OpenAICompatibleClient",  # For custom OpenAI-compatible endpoints
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
    # Checkpoint Storage
    "CheckpointStorage",
    "LocalFileCheckpointStorage",
]
