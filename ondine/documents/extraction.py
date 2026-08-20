"""Deep facade for schema-driven document extraction.

The facade owns stable task planning, task-result alignment, and final Pydantic
validation.  Parsers and model execution remain replaceable ports so callers
do not inherit vendor-specific concepts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ondine.documents.models import (
    DocumentExtractionResult,
    DocumentSource,
    ExtractionCandidate,
    ExtractionTask,
    FieldStatus,
    ParsedDocument,
    SchemaT,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ondine.documents.protocols import DocumentParser, ExtractionTaskRunner


_PLANNER_VERSION = "1"
_SCHEMA_READY_STATUSES = {
    FieldStatus.FOUND,
    FieldStatus.ABSENT_IN_DOCUMENT,
    FieldStatus.NOT_APPLICABLE,
}


def extract_document(
    source: DocumentSource,
    *,
    schema: type[SchemaT],
    parser: DocumentParser,
    task_runner: ExtractionTaskRunner,
) -> DocumentExtractionResult[SchemaT]:
    """Extract one source into a validated schema while retaining evidence.

    The parser and runner are explicit in this first vertical slice.  A later
    convenience layer can select defaults without changing the extraction
    contract or coupling this module to a particular PDF library or provider.
    """
    document = parser.parse(source)
    _validate_source_graph(source, document)
    tasks = _plan_tasks(document, schema, task_runner.execution_fingerprint)
    candidates = _align_candidates(tasks, task_runner.run(document, tasks))
    candidates = _validate_evidence(document, candidates)
    payload = _found_values(tasks, candidates)
    data = None
    if all(candidate.status in _SCHEMA_READY_STATUSES for candidate in candidates):
        data, candidates = _validate_payload(schema, payload, candidates)
    return DocumentExtractionResult(document=document, data=data, fields=candidates)


def _validate_source_graph(
    source: DocumentSource,
    document: ParsedDocument,
) -> None:
    if document.document_id != source.document_id:
        raise ValueError(
            f"parser returned document {document.document_id!r}; "
            f"expected {source.document_id!r}"
        )

    if document.page_count < 1:
        raise ValueError("page_count must be at least 1")

    page_numbers = [page.page_number for page in document.pages]
    duplicates = sorted(
        number for number, count in Counter(page_numbers).items() if count > 1
    )
    if duplicates:
        numbers = ", ".join(str(number) for number in duplicates)
        raise ValueError(f"duplicate page manifest entries: {numbers}")

    expected = set(range(1, document.page_count + 1))
    present = set(page_numbers)
    missing = sorted(expected - present)
    if missing:
        numbers = ", ".join(str(number) for number in missing)
        raise ValueError(f"missing page manifest entries: {numbers}")

    unexpected = sorted(present - expected)
    if unexpected:
        numbers = ", ".join(str(number) for number in unexpected)
        raise ValueError(f"unexpected page manifest entries: {numbers}")

    for page in document.pages:
        element_ids = [element.element_id for element in page.elements]
        duplicate_elements = sorted(
            element_id
            for element_id, count in Counter(element_ids).items()
            if count > 1
        )
        if duplicate_elements:
            raise ValueError(
                f"duplicate element ids on page {page.page_number}: "
                f"{', '.join(duplicate_elements)}"
            )
        for element in page.elements:
            if element.page_number != page.page_number:
                raise ValueError(
                    f"element {element.element_id!r} says page "
                    f"{element.page_number} but belongs to page {page.page_number}"
                )


def _plan_tasks(
    document: ParsedDocument,
    schema: type[BaseModel],
    execution_fingerprint: str,
) -> tuple[ExtractionTask, ...]:
    schema_definition = schema.model_json_schema(by_alias=False)
    schema_fingerprint = _stable_hash(schema_definition)
    source_graph_fingerprint = _source_graph_fingerprint(document)
    properties = schema_definition.get("properties", {})
    tasks: list[ExtractionTask] = []

    for field_name in schema.model_fields:
        field_path = (field_name,)
        field_schema = properties.get(field_name, {})
        if "$defs" in schema_definition and "#/$defs/" in _canonical_json(field_schema):
            field_schema = {
                "$defs": schema_definition["$defs"],
                **field_schema,
            }
        identity = {
            "document_id": document.document_id,
            "field_path": field_path,
            "planner_version": _PLANNER_VERSION,
            "schema_fingerprint": schema_fingerprint,
            "source_graph_fingerprint": source_graph_fingerprint,
            "execution_fingerprint": execution_fingerprint,
        }
        tasks.append(
            ExtractionTask(
                task_id=f"task:{_stable_hash(identity)}",
                document_id=document.document_id,
                field_path=field_path,
                schema_json=_canonical_json(field_schema),
            )
        )

    return tuple(tasks)


def _source_graph_fingerprint(document: ParsedDocument) -> str:
    graph = {
        "document_id": document.document_id,
        "page_count": document.page_count,
        "parser": {
            "name": document.parser_name,
            "version": document.parser_version,
        },
        "pages": [
            {
                "number": page.page_number,
                "status": page.status,
                "width": page.width,
                "height": page.height,
                "error": page.error,
                "elements": [
                    {
                        "id": element.element_id,
                        "page_number": element.page_number,
                        "kind": element.kind,
                        "text": element.text,
                        "bbox": element.bbox,
                    }
                    for element in page.elements
                ],
            }
            for page in document.pages
        ],
    }
    return _stable_hash(graph)


def _found_values(
    tasks: tuple[ExtractionTask, ...],
    candidates: tuple[ExtractionCandidate, ...],
) -> dict[str, object]:
    candidates_by_id = {candidate.task_id: candidate for candidate in candidates}
    payload: dict[str, object] = {}

    for task in tasks:
        candidate = candidates_by_id[task.task_id]
        if candidate.field_path != task.field_path:
            raise ValueError(
                f"Task {task.task_id!r} returned field {candidate.field_path!r}; "
                f"expected {task.field_path!r}"
            )
        if candidate.status == FieldStatus.FOUND:
            payload[task.field_path[0]] = candidate.value

    return payload


def _align_candidates(
    tasks: tuple[ExtractionTask, ...],
    candidates: tuple[ExtractionCandidate, ...],
) -> tuple[ExtractionCandidate, ...]:
    expected_ids = {task.task_id for task in tasks}
    unknown_ids = sorted({candidate.task_id for candidate in candidates} - expected_ids)
    if unknown_ids:
        raise ValueError(f"unknown task result: {', '.join(unknown_ids)}")

    candidates_by_id: dict[str, list[ExtractionCandidate]] = {}
    for candidate in candidates:
        candidates_by_id.setdefault(candidate.task_id, []).append(candidate)

    aligned: list[ExtractionCandidate] = []
    for task in tasks:
        matches = candidates_by_id.get(task.task_id, [])
        if len(matches) == 1:
            aligned.append(matches[0])
        else:
            status = (
                FieldStatus.NOT_EXAMINED if not matches else FieldStatus.CONFLICTING
            )
            aligned.append(
                ExtractionCandidate(
                    task_id=task.task_id,
                    field_path=task.field_path,
                    value=None,
                    status=status,
                )
            )

    return tuple(aligned)


def _validate_evidence(
    document: ParsedDocument,
    candidates: tuple[ExtractionCandidate, ...],
) -> tuple[ExtractionCandidate, ...]:
    elements_by_page = {
        page.page_number: {element.element_id for element in page.elements}
        for page in document.pages
    }
    text_by_page = {
        page.page_number: "\n".join(element.text for element in page.elements)
        for page in document.pages
    }
    checked: list[ExtractionCandidate] = []

    for candidate in candidates:
        if candidate.status != FieldStatus.FOUND:
            checked.append(candidate)
            continue

        evidence_is_valid = bool(candidate.evidence) and all(
            evidence.document_id == document.document_id
            and evidence.page_number in elements_by_page
            and set(evidence.element_ids) <= elements_by_page[evidence.page_number]
            and (
                evidence.quote is None
                or (
                    bool(evidence.quote.strip())
                    and evidence.quote in text_by_page[evidence.page_number]
                )
            )
            for evidence in candidate.evidence
        )
        checked.append(
            candidate
            if evidence_is_valid
            else replace(candidate, value=None, status=FieldStatus.CONFLICTING)
        )

    return tuple(checked)


def _validate_payload(
    schema: type[SchemaT],
    payload: dict[str, object],
    candidates: tuple[ExtractionCandidate, ...],
) -> tuple[SchemaT | None, tuple[ExtractionCandidate, ...]]:
    try:
        return schema.model_validate(payload, by_name=True), candidates
    except ValidationError as error:
        invalid_fields = {
            str(detail["loc"][0]) for detail in error.errors() if detail["loc"]
        }
        if not invalid_fields:
            invalid_fields = {
                candidate.field_path[0]
                for candidate in candidates
                if candidate.status == FieldStatus.FOUND
            }
        invalid = tuple(
            replace(candidate, value=None, status=FieldStatus.CONFLICTING)
            if candidate.status == FieldStatus.FOUND
            and candidate.field_path[0] in invalid_fields
            else candidate
            for candidate in candidates
        )
        return None, invalid


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
