"""Orchestration engine for pipeline execution control."""

from ondine.orchestration.async_executor import AsyncExecutor
from ondine.orchestration.backends import (
    SUPPORTED_BATCH_PROVIDERS,
    BatchProgress,
    ExecutionBackend,
    ProviderBatchBackend,
)
from ondine.orchestration.concurrency_controller import ConcurrencyController
from ondine.orchestration.deployment_tracker import DeploymentTracker
from ondine.orchestration.execution_context import (
    ExecutionContext,
    RunProgressState,
    StageProgressSnapshot,
)
from ondine.orchestration.execution_strategy import ExecutionStrategy
from ondine.orchestration.observers import (
    CostTrackingObserver,
    ExecutionObserver,
    LoggingObserver,
    ProgressBarObserver,
)
from ondine.orchestration.progress_reporter import ProgressReporter
from ondine.orchestration.progress_tracker import (
    LoggingProgressTracker,
    ProgressTracker,
    RichProgressTracker,
    create_progress_tracker,
)
from ondine.orchestration.run_registry import (
    REGISTRY_FILENAME,
    RegistryObserver,
    RunHandle,
    RunRegistry,
    RunSpec,
    RunStatus,
)
from ondine.orchestration.state_manager import StateManager
from ondine.orchestration.streaming_executor import (
    StreamingExecutor,
    StreamingResult,
)
from ondine.orchestration.streaming_processor import (
    ChunkResult,
    StreamingProcessor,
    StreamingStats,
)
from ondine.orchestration.sync_executor import SyncExecutor

__all__ = [
    "ExecutionContext",
    "RunProgressState",
    "StageProgressSnapshot",
    "StateManager",
    "ExecutionObserver",
    "ProgressBarObserver",
    "LoggingObserver",
    "CostTrackingObserver",
    "ExecutionStrategy",
    "SyncExecutor",
    "AsyncExecutor",
    "StreamingExecutor",
    "StreamingResult",
    "ProgressTracker",
    "RichProgressTracker",
    "LoggingProgressTracker",
    "create_progress_tracker",
    # Extracted components for LLM invocation
    "ConcurrencyController",
    "DeploymentTracker",
    "ProgressReporter",
    # Run registry — persistent cross-process job index
    "RunRegistry",
    "RunHandle",
    "RunSpec",
    "RunStatus",
    "RegistryObserver",
    "REGISTRY_FILENAME",
    # Execution backends — pluggable middle of the pipeline (§3, §5)
    "ExecutionBackend",
    "BatchProgress",
    "ProviderBatchBackend",
    "SUPPORTED_BATCH_PROVIDERS",
    # Streaming processing (for large datasets)
    "StreamingProcessor",
    "StreamingStats",
    "ChunkResult",
]
