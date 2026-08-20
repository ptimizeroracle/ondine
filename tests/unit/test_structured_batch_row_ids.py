"""Unit tests for structured-batch row-id injection (#255).

The disaggregator matches structured ``items`` to rows by position. When a
model reorders its items that silently misassigns answers, so ondine augments
the schema with a row id the model echoes and realigns by it. These tests pin
the augmentation contract in isolation from the pipeline: which models get an
id, which are left alone, and that the id is actually required on each item.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel

from ondine.strategies.structured_batch import ROW_ID_FIELD, with_row_ids


class Answer(BaseModel):
    token: str
    category: str


class AnswerBatch(BaseModel):
    items: list[Answer]


def _item_type(batch_model: type[BaseModel]) -> type[BaseModel]:
    return typing.get_args(batch_model.model_fields["items"].annotation)[0]


def test_items_gain_a_required_row_id():
    """The augmented item carries a mandatory id the original lacked.

    Without this the structured path has no id to sort on, so a reordered
    batch is trusted in arrival order — the whole bug. The id must be required,
    not optional: a model that omits it should fail parsing loudly rather than
    fall back to position.
    """
    indexed = with_row_ids(AnswerBatch)

    assert indexed is not None
    item = _item_type(indexed)
    assert ROW_ID_FIELD in item.model_fields
    assert item.model_fields[ROW_ID_FIELD].is_required()
    # The caller's own fields survive the augmentation.
    assert {"token", "category"} <= set(item.model_fields)


def test_original_model_is_left_untouched():
    """Augmentation must not mutate the caller's schema in place.

    The caller keeps using their model elsewhere; adding an id to it globally
    would leak ondine's batching internals into the caller's type.
    """
    with_row_ids(AnswerBatch)

    assert ROW_ID_FIELD not in Answer.model_fields


def test_a_schema_that_already_has_an_id_is_not_reaugmented():
    """A caller-supplied id is authoritative; do not shadow or renumber it."""

    class Identified(BaseModel):
        id: int
        value: str

    class IdentifiedBatch(BaseModel):
        items: list[Identified]

    assert with_row_ids(IdentifiedBatch) is None


def test_a_non_items_schema_cannot_be_augmented():
    """A single-object schema has no list to index; the caller is warned and
    falls back rather than crashing."""

    class Flat(BaseModel):
        value: str

    assert with_row_ids(Flat) is None
