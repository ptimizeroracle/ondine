"""Document loading for PDF, Markdown, and plain text files.

Uses PyMuPDF for PDF extraction (optional dependency) and stdlib
for text-based formats. Each loader returns a list of Document
dataclasses that downstream chunkers consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """A loaded document with its text content and provenance metadata."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Load documents from files or directories.

    Dispatches to the appropriate reader based on file extension:
    - .pdf  → PyMuPDF (``pymupdf`` extra required)
    - .md   → stdlib text reader
    - .txt  → stdlib text reader

    Calling code never needs to know which reader is used — the loader
    absorbs that complexity (Ousterhout: pull complexity downward).
    """

    _SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        """Load one file or recursively load all supported files in a directory."""
        p = Path(path)
        if p.is_dir():
            return self._load_directory(p)
        return self._load_file(p)

    def _load_directory(self, directory: Path) -> list[Document]:
        docs: list[Document] = []
        for ext in self._SUPPORTED_EXTENSIONS:
            for fp in sorted(directory.rglob(f"*{ext}")):
                docs.extend(self._load_file(fp))
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported file type: %s", ext)
            return []

        if ext == ".pdf":
            return self._load_pdf(path)
        return self._load_text(path)

    def _load_pdf(self, path: Path) -> list[Document]:
        try:
            import pymupdf  # noqa: F811
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install with: pip install 'ondine[knowledge]'"
            ) from None

        docs: list[Document] = []
        with pymupdf.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(
                        Document(
                            text=text,
                            source=str(path),
                            metadata={"page": page_num + 1, "format": "pdf"},
                        )
                    )
        logger.info("Loaded %d pages from %s", len(docs), path)
        return docs

    def _load_text(self, path: Path) -> list[Document]:
        text = path.read_text(encoding=self._encoding)
        if not text.strip():
            return []
        return [
            Document(
                text=text,
                source=str(path),
                metadata={"format": path.suffix.lstrip(".")},
            )
        ]
