"""Specifications must survive model_dump(mode="json") (issue #232).

Specifications are configuration, and something eventually serializes them —
today it is MCP, which snapshots a run with
``pipeline.specifications.model_dump(mode="json")`` (ondine/mcp/server.py).

Live objects used to be smuggled through ``specifications.metadata``, a
``dict[str, Any]`` in which "pass this object to a stage" and "this is part of
the configuration" look identical. So a single builder call could make a
pipeline unserializable, and nothing said so until something tried:

    .with_structured_output(Schema)
    → PydanticSerializationError: Unable to serialize unknown type:
      <class 'pydantic._internal._model_construction.ModelMetaclass'>

Any run using structured output could not be snapshotted by MCP — two features
that could not be used together.

These tests fail the moment a builder method puts an object back into the
specs, which is the only reliable guard: the failure otherwise surfaces far
from the call that caused it.
"""

import pandas as pd
import pytest
from pydantic import BaseModel

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.context.memory_store import InMemoryContextStore


class _Schema(BaseModel):
    label: str


def _builder() -> PipelineBuilder:
    return (
        PipelineBuilder.create()
        .from_dataframe(
            pd.DataFrame({"t": ["a"]}), input_columns=["t"], output_columns=["o"]
        )
        .with_prompt("Classify: {t}")
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",  # pragma: allowlist secret
        )
    )


@pytest.mark.parametrize(
    ("label", "configure"),
    [
        ("plain", lambda b: b),
        ("structured_output", lambda b: b.with_structured_output(_Schema)),
        ("context_store", lambda b: b.with_context_store(InMemoryContextStore())),
        ("grounding", lambda b: b.with_grounding()),
        ("evidence_priming", lambda b: b.with_evidence_priming(query_columns=["t"])),
        (
            "structured_plus_context",
            lambda b: b.with_structured_output(_Schema).with_context_store(),
        ),
    ],
)
def test_specifications_survive_json_dump(label, configure):
    """Whatever the builder configured, the specs must still dump to JSON."""
    pipeline = configure(_builder()).build()

    dumped = pipeline.specifications.model_dump(mode="json")

    assert isinstance(dumped, dict)


def test_live_objects_are_not_in_metadata():
    """The objects reach the run, but through components — not the specs.

    Asserting both halves matters: dropping them entirely would also make the
    specs serializable, and would break every feature that needs them.
    """
    pipeline = (
        _builder()
        .with_structured_output(_Schema)
        .with_context_store(InMemoryContextStore())
        .build()
    )

    metadata = pipeline.specifications.metadata or {}
    for key in (
        "structured_output_model",
        "custom_parser",
        "knowledge_store",
        "context_store",
        "custom_llm_client",
    ):
        assert key not in metadata, f"{key} is back in the serializable specs"

    assert pipeline._components.structured_output_model is _Schema
    assert pipeline._components.context_store is not None
    assert pipeline._components.custom_parser is not None
