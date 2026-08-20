"""Provider-neutral document-to-structured-data extraction."""

from ondine.documents.extraction import extract_document
from ondine.documents.models import (
    DocumentElement,
    DocumentExtractionResult,
    DocumentPage,
    DocumentSource,
    ExtractionCandidate,
    ExtractionTask,
    FieldEvidence,
    FieldStatus,
    IncompleteDocumentError,
    PageStatus,
    ParsedDocument,
)
from ondine.documents.protocols import DocumentParser, ExtractionTaskRunner

__all__ = [
    "DocumentElement",
    "DocumentExtractionResult",
    "DocumentPage",
    "DocumentParser",
    "DocumentSource",
    "ExtractionCandidate",
    "ExtractionTask",
    "ExtractionTaskRunner",
    "FieldEvidence",
    "FieldStatus",
    "IncompleteDocumentError",
    "PageStatus",
    "ParsedDocument",
    "extract_document",
]
