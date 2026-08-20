"""Dependency-inversion ports for document parsing and task execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ondine.documents.models import (  # noqa: TC001
    DocumentSource,
    ExtractionCandidate,
    ExtractionTask,
    ParsedDocument,
)


@runtime_checkable
class DocumentParser(Protocol):
    """Turn source bytes into a complete, provider-neutral page manifest."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        """Return a complete source graph for ``source``."""
        ...


@runtime_checkable
class ExtractionTaskRunner(Protocol):
    """Execute bounded extraction tasks without owning reconciliation rules."""

    @property
    def execution_fingerprint(self) -> str:
        """Identify the model, prompt, and normalization execution contract."""
        ...

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        """Return zero or more identity-bearing task results."""
        ...
