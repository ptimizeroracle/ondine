"""
Data container implementations.

This module provides various DataContainer implementations:
- StreamingCSVContainer: Default, pure Python, O(1) memory
- DictListContainer: In-memory list of dictionaries
- PolarsContainer: Polars DataFrame wrapper
- PandasContainer: Pandas DataFrame wrapper (backward compat)
- ResultContainer: Pipeline output container
"""

from ondine.adapters.containers.dict_list import DictListContainer
from ondine.adapters.containers.pandas_container import PandasContainer
from ondine.adapters.containers.polars_container import PolarsContainer
from ondine.adapters.containers.result_container import (
    ResultContainerImpl,
)
from ondine.adapters.containers.streaming_csv import StreamingCSVContainer

__all__ = [
    # Core containers
    "StreamingCSVContainer",
    "DictListContainer",
    # Framework adapters
    "PolarsContainer",
    "PandasContainer",
    # Output
    "ResultContainerImpl",
]

