"""Row-id injection for structured batch output.

Structured output lets the caller hand ondine a Pydantic model with an
``items`` list and get one parsed object per row back. The disaggregator then
maps ``items[i]`` onto row ``i`` — it trusts arrival order. Models do not
preserve order reliably: they re-sort, group and renumber batch responses, and
structured output makes that *more* likely, because the schema frames the task
as "produce a list" rather than "answer N questions in order". A reordered
response then gives every row a neighbour's answer, with every cell populated
and every counter clean — the one corruption no post-hoc check can catch (#255).

The plain-JSON batch path never had this problem: each item carries a 1-based
``id`` the model echoes, and the parser sorts by it and raises on gaps. This
module closes the asymmetry by giving the *structured* path the same id — added
to the schema ondine sends, so the caller's own model stays untouched. Once the
items carry an id, the existing id-sorting / ``PartialParseError`` machinery in
``JsonBatchStrategy`` realigns a reordered batch and fails a truncated one for
free.
"""

from __future__ import annotations

import typing
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, create_model

# The field ondine adds to each batch item. Underscore-free and short because a
# provider has to echo it: the batch prompt already instructs "IDs must match
# the input IDs (1 to N)", and a schema field named ``id`` is what lets a
# structured model actually carry that back.
ROW_ID_FIELD = "id"


def _items_element_type(batch_model: type[BaseModel]) -> type[BaseModel] | None:
    """The ``X`` in a ``items: list[X]`` field, or None if there is no such field.

    Returns None (rather than raising) for any shape we cannot augment — no
    ``items`` field, a non-list annotation, or a list of non-models — so the
    caller can warn and fall back to the legacy positional behaviour instead of
    breaking a pipeline that was working, if unsafely.
    """
    field = batch_model.model_fields.get("items")
    if field is None:
        return None
    annotation = field.annotation
    origin = typing.get_origin(annotation)
    if origin not in (list, typing.List):  # noqa: UP006 - runtime origin check
        return None
    args = typing.get_args(annotation)
    if not args:
        return None
    element = args[0]
    if isinstance(element, type) and issubclass(element, BaseModel):
        return element
    return None


@lru_cache(maxsize=128)
def with_row_ids(batch_model: type[BaseModel]) -> type[BaseModel] | None:
    """A copy of ``batch_model`` whose items carry a row ``id``, or None.

    None means the model is not an augmentable ``items: list[Model]`` batch, or
    its items already carry an ``id`` — in both cases the caller should use the
    original model unchanged. The result is cached so the derived model is built
    once per input model, not once per batch.
    """
    element = _items_element_type(batch_model)
    if element is None:
        return None
    if ROW_ID_FIELD in element.model_fields:
        # The caller already gives their items an id; the existing id-sorting
        # path handles them. Do not shadow or renumber the caller's field.
        return None

    id_field: dict[str, Any] = {ROW_ID_FIELD: (int, ...)}
    indexed_element = create_model(
        f"RowIndexed{element.__name__}",
        __base__=element,
        **id_field,
    )
    return create_model(
        f"RowIndexed{batch_model.__name__}",
        __base__=batch_model,
        items=(list[indexed_element], ...),  # type: ignore[valid-type]
    )
