"""Execution backends — the pluggable middle of the pipeline.

* :class:`ExecutionBackend` (in :mod:`base`) is the protocol both live
  and batch backends satisfy.
* :class:`LiveBackend` (in :mod:`live`) runs the existing asyncio engine
  synchronously through the protocol's degenerate lifecycle.
* :class:`ProviderBatchBackend` (in :mod:`provider_batch`) implements
  OpenAI + Anthropic native Batch API mode.
"""

from ondine.orchestration.backends.base import (
    BatchProgress,
    ExecutionBackend,
)
from ondine.orchestration.backends.live import LiveBackend
from ondine.orchestration.backends.provider_batch import (
    SUPPORTED_BATCH_PROVIDERS,
    ProviderBatchBackend,
)

__all__ = [
    "BatchProgress",
    "ExecutionBackend",
    "LiveBackend",
    "ProviderBatchBackend",
    "SUPPORTED_BATCH_PROVIDERS",
]
