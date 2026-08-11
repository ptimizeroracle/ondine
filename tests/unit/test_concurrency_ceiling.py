"""The concurrency ceiling must be the same through every front door."""

import pytest
from pydantic import ValidationError

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.specifications import ProcessingSpec


def test_the_spec_accepts_the_concurrency_the_builder_documents():
    """Catches the two front doors disagreeing.

    `with_concurrency(64)` was accepted while `ProcessingSpec(concurrency=64)`
    raised, so a value the builder's own docstring recommends was rejected on
    the YAML path — and a large run was capped at 20 without saying so.
    """
    assert ProcessingSpec(concurrency=64).concurrency == 64
    assert (
        PipelineBuilder.create().with_concurrency(64)._processing_spec.concurrency == 64
    )


def test_concurrency_above_the_documented_ceiling_is_still_rejected():
    """The limit moved; it did not disappear."""
    with pytest.raises(ValidationError):
        ProcessingSpec(concurrency=101)
