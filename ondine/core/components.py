"""Live collaborators injected into a pipeline run.

Configuration belongs in :class:`PipelineSpecifications` and must stay
serializable — MCP snapshots it with ``model_dump(mode="json")``. These are not
configuration: a client, a store, a parser, a model class. They used to be
smuggled through ``specifications.metadata``, a ``dict[str, Any]`` that gives
"pass this object to a stage" and "this is part of the config" the same
spelling. Two consequences followed, both silent:

* the specs stopped being dumpable, so any run using structured output could
  not be snapshotted by MCP (#232);
* sub-pipelines start from ``model_copy(deep=True)``, so each received a deep
  *copy* of the caller's object — state it held stopped being shared (#230).

Keeping them here separates the two ideas at the type level rather than by
convention, so a new collaborator is a field on this class instead of another
entry in a dict that is expected to serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineComponents:
    """Objects a caller supplies for a run, passed by reference.

    Every field is optional; the default instance means "build everything from
    the specifications", which is the common case.
    """

    llm_client: Any | None = None
    knowledge_store: Any | None = None
    context_store: Any | None = None
    structured_output_model: Any | None = None
    custom_parser: Any | None = None
