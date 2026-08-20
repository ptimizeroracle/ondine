"""Stable runtime contracts for document extraction.

These value objects preserve source identity and evidence across parser and
model boundaries.  They intentionally contain no parser, provider, or Ondine
pipeline implementation details.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)
FieldPath = tuple[str, ...]
BoundingBox = tuple[float, float, float, float]


class IncompleteDocumentError(RuntimeError):
    """Raised when a caller requires data from an incomplete extraction."""


class PageStatus(str, Enum):
    """Parser outcome for one physical page in the source document."""

    PARSED = "parsed"
    BLANK = "blank"
    FAILED = "failed"


class FieldStatus(str, Enum):
    """What extraction established about one requested schema field."""

    FOUND = "found"
    ABSENT_IN_DOCUMENT = "absent_in_document"
    NOT_EXAMINED = "not_examined"
    PARSER_FAILED = "parser_failed"
    CONFLICTING = "conflicting"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class DocumentSource:
    """Immutable input bytes with a content-derived document identity.

    ``document_id`` is stable across file renames and process restarts, which
    allows later journal implementations to key work by content rather than by
    a temporary path.
    """

    content: bytes = field(repr=False)
    name: str
    media_type: str
    document_id: str = field(init=False)

    def __post_init__(self) -> None:
        """Bind identity to content so callers cannot forge cache aliases."""
        digest = hashlib.sha256(self.content).hexdigest()
        object.__setattr__(self, "document_id", f"sha256:{digest}")

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        name: str,
        media_type: str,
    ) -> DocumentSource:
        """Create a source whose identity is the SHA-256 of its exact bytes."""
        return cls(
            content=content,
            name=name,
            media_type=media_type,
        )


@dataclass(frozen=True)
class DocumentElement:
    """One parser-produced source element with stable page-local provenance."""

    element_id: str
    page_number: int
    kind: str
    text: str
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class DocumentPage:
    """Manifest entry for one physical page, including blank or failed pages."""

    page_number: int
    status: PageStatus
    elements: tuple[DocumentElement, ...] = ()
    width: float | None = None
    height: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Provider-neutral parser output consumed by extraction planning."""

    document_id: str
    parser_name: str
    parser_version: str
    page_count: int
    pages: tuple[DocumentPage, ...]


@dataclass(frozen=True)
class FieldEvidence:
    """Source locations supporting one extracted field value.

    The page number is the minimum anchor. Element IDs, an exact quote, and a
    bounding box add precision when the selected parser can provide them.
    """

    document_id: str
    page_number: int
    element_ids: tuple[str, ...] = ()
    quote: str | None = None
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class ExtractionTask:
    """A bounded field request with a stable, execution-versioned identity."""

    task_id: str
    document_id: str
    field_path: FieldPath
    schema_json: str


@dataclass(frozen=True)
class ExtractionCandidate:
    """One task result before cross-task reconciliation and schema validation."""

    task_id: str
    field_path: FieldPath
    value: Any
    status: FieldStatus
    evidence: tuple[FieldEvidence, ...] = ()


@dataclass(frozen=True)
class DocumentExtractionResult(Generic[SchemaT]):
    """Validated document data together with its source evidence."""

    document: ParsedDocument
    data: SchemaT | None
    fields: tuple[ExtractionCandidate, ...]
    success: bool = True

    @property
    def is_complete(self) -> bool:
        """True when every planned field was examined and the schema validated."""
        incomplete = {
            FieldStatus.NOT_EXAMINED,
            FieldStatus.PARSER_FAILED,
            FieldStatus.CONFLICTING,
        }
        return (
            self.data is not None
            and all(candidate.status not in incomplete for candidate in self.fields)
            and all(page.status != PageStatus.FAILED for page in self.document.pages)
        )

    def evidence_for(self, field: str | FieldPath) -> tuple[FieldEvidence, ...]:
        """Return all evidence attached to a schema field path."""
        path = (field,) if isinstance(field, str) else field
        return tuple(
            evidence
            for candidate in self.fields
            if candidate.field_path == path
            for evidence in candidate.evidence
        )

    @property
    def partial_data(self) -> dict[str, Any]:
        """Return one record with unresolved planned fields represented by None."""
        if self.data is not None:
            return self.data.model_dump(mode="json")
        return {
            candidate.field_path[0]: (
                candidate.value if candidate.status == FieldStatus.FOUND else None
            )
            for candidate in self.fields
        }

    def require_complete(self) -> SchemaT:
        """Return validated data or identify fields that remain unresolved."""
        if self.data is not None and self.is_complete:
            return self.data

        unresolved_fields = [
            ".".join(candidate.field_path)
            for candidate in self.fields
            if candidate.status
            in {
                FieldStatus.NOT_EXAMINED,
                FieldStatus.PARSER_FAILED,
                FieldStatus.CONFLICTING,
            }
        ]
        if self.data is None and not unresolved_fields:
            unresolved_fields = [
                ".".join(candidate.field_path) for candidate in self.fields
            ]
        failed_pages = [
            str(page.page_number)
            for page in self.document.pages
            if page.status == PageStatus.FAILED
        ]
        problems: list[str] = []
        if unresolved_fields:
            problems.append(f"unresolved fields: {', '.join(unresolved_fields)}")
        if failed_pages:
            problems.append(f"failed pages: {', '.join(failed_pages)}")
        detail = "; ".join(problems) or "validated data is unavailable"
        raise IncompleteDocumentError(f"document extraction is incomplete; {detail}")

    def to_pandas(self) -> Any:
        """Return one row, using ``None`` for unresolved planned fields."""
        import pandas as pd

        return pd.DataFrame([self.partial_data])
