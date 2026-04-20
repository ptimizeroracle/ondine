"""Tests for ``PipelineBuilder.with_adaptive_concurrency`` wiring.

Before this change, ``LLMInvocationStage(adaptive_concurrency=True)``
existed but was unreachable — nothing passed the flag from specs to
the stage. These tests lock in the end-to-end plumb so the feature
stays reachable:

* ``ProcessingSpec.adaptive_concurrency`` is part of the spec shape.
* ``PipelineBuilder.with_adaptive_concurrency()`` flips it.
* The constructed ``LLMInvocationStage`` sees the flag.
"""

from __future__ import annotations

import pandas as pd

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.specifications import ProcessingSpec


class TestProcessingSpecHasField:
    """The spec is the contract between builder and pipeline — if the
    field disappears, everything silently degrades."""

    def test_default_is_false(self):
        spec = ProcessingSpec()
        assert spec.adaptive_concurrency is False

    def test_can_be_set_true(self):
        spec = ProcessingSpec(adaptive_concurrency=True)
        assert spec.adaptive_concurrency is True


class TestBuilderMethod:
    """Public API surface — users call this to opt in."""

    def _base_builder(self) -> PipelineBuilder:
        df = pd.DataFrame({"text": ["hello"]})
        return (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["out"])
            .with_prompt("Classify: {text}")
            .with_llm(model="mock", provider="openai")
        )

    def test_with_adaptive_concurrency_sets_spec_flag(self):
        builder = self._base_builder().with_adaptive_concurrency()
        pipeline = builder.build()
        assert pipeline.specifications.processing.adaptive_concurrency is True

    def test_default_off_when_builder_method_not_called(self):
        builder = self._base_builder()
        pipeline = builder.build()
        assert pipeline.specifications.processing.adaptive_concurrency is False

    def test_can_disable_explicitly(self):
        builder = self._base_builder().with_adaptive_concurrency(False)
        pipeline = builder.build()
        assert pipeline.specifications.processing.adaptive_concurrency is False


class TestStageReceivesFlag:
    """Plumbing regression — the flag must flow from spec into the
    stage that actually uses it. Verifies by capturing the kwargs
    passed to ``LLMInvocationStage`` during pipeline execution."""

    def test_pipeline_forwards_adaptive_flag_to_llm_stage(self, monkeypatch):
        import ondine.api.pipeline as pipeline_mod

        captured: dict = {}
        real_init = pipeline_mod.LLMInvocationStage.__init__

        def capturing_init(self, *args, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            # Stop the init chain; we only need the kwargs snapshot.
            raise _StopExecutionError

        monkeypatch.setattr(pipeline_mod.LLMInvocationStage, "__init__", capturing_init)

        df = pd.DataFrame({"text": ["hello"]})
        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["out"])
            .with_prompt("Classify: {text}")
            .with_llm(model="mock", provider="openai")
            .with_adaptive_concurrency()
            .build()
        )

        try:
            pipeline.execute()
        except _StopExecutionError:
            pass

        # Restore for any other tests that share the class — monkeypatch
        # auto-restores at teardown, but be explicit about intent.
        pipeline_mod.LLMInvocationStage.__init__ = real_init

        assert captured.get("adaptive_concurrency") is True


class _StopExecutionError(Exception):
    """Sentinel to short-circuit pipeline.execute() once we've
    captured the kwargs we care about."""
