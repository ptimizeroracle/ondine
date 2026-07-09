"""Execution backends — the pluggable middle of the pipeline.

* :class:`ExecutionBackend` (in :mod:`base`) is the protocol both live
  and batch backends satisfy.
* :class:`ProviderBatchBackend` (in :mod:`provider_batch`) implements
  OpenAI + Anthropic native Batch API mode.
* ``LiveBackend`` (moving the existing asyncio engine behind the
  protocol) is a follow-up build step; until it lands, live runs keep
  using the engine directly and only batch mode routes through here.
"""

from ondine.orchestration.backends.base import BatchProgress, ExecutionBackend
from ondine.orchestration.backends.provider_batch import (
    SUPPORTED_BATCH_PROVIDERS,
    ProviderBatchBackend,
)

__all__ = [
    "BatchProgress",
    "ExecutionBackend",
    "ProviderBatchBackend",
    "SUPPORTED_BATCH_PROVIDERS",
]
