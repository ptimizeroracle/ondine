"""Document loading for PDF, Markdown, plain text, and image files.

Uses PyMuPDF for PDF extraction (optional dependency), stdlib for
text-based formats, and pluggable OCR providers for images. Each
loader returns a list of Document dataclasses that downstream
chunkers consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.knowledge.protocols import OCRProvider

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".tsv", ".json", ".xml", ".html", ".htm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"}


@dataclass(frozen=True)
class Document:
    """A loaded document with its text content and provenance metadata."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Load documents from files or directories.

    Dispatches to the appropriate reader based on file extension:
    - .pdf                → PyMuPDF (``pymupdf`` extra required)
    - .md / .txt / etc.   → stdlib text reader
    - .png / .jpg / etc.  → OCR provider (if configured)

    When ``ocr`` is provided, images are converted to text via the
    OCR provider and then treated identically to any other document.
    When ``extract_pdf_images=True``, embedded images in PDFs are
    also extracted and OCR'd alongside the regular text content.

    Calling code never needs to know which reader is used — the loader
    absorbs that complexity (Ousterhout: pull complexity downward).

    Args:
        encoding: Text encoding for text-based files.
        ocr: An ``OCRProvider``-protocol object, a shortcut string
            (``"vision"``, ``"tesseract"``, ``"doctr"``), or ``None``
            to skip image files.
        extract_pdf_images: When ``True`` and ``ocr`` is configured,
            also extract and OCR embedded images from PDF pages.
    """

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        ocr: OCRProvider | str | None = None,
        extract_pdf_images: bool = False,
    ) -> None:
        self._encoding = encoding
        self._extract_pdf_images = extract_pdf_images

        from ondine.knowledge.ocr import resolve_ocr

        self._ocr: OCRProvider | None = resolve_ocr(ocr)

    @property
    def supported_extensions(self) -> set[str]:
        """All file extensions this loader can handle."""
        exts = {".pdf"} | _TEXT_EXTENSIONS
        if self._ocr is not None:
            exts |= _IMAGE_EXTENSIONS
        return exts

    def load(self, path: str | Path) -> list[Document]:
        """Load one file or recursively load all supported files in a directory."""
        p = Path(path)
        if p.is_dir():
            return self._load_directory(p)
        return self._load_file(p)

    def _load_directory(self, directory: Path) -> list[Document]:
        docs: list[Document] = []
        for ext in sorted(self.supported_extensions):
            for fp in sorted(directory.rglob(f"*{ext}")):
                docs.extend(self._load_file(fp))
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)

        if ext in _IMAGE_EXTENSIONS:
            return self._load_image(path)

        if ext in _TEXT_EXTENSIONS:
            return self._load_text(path)

        logger.warning("Unsupported file type: %s", ext)
        return []

    def _load_pdf(self, path: Path) -> list[Document]:
        try:
            import pymupdf
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

                if self._extract_pdf_images and self._ocr is not None:
                    docs.extend(
                        self._extract_images_from_pdf_page(page, path, page_num + 1)
                    )

        logger.info("Loaded %d documents from %s", len(docs), path)
        return docs

    def _extract_images_from_pdf_page(
        self, page, pdf_path: Path, page_num: int
    ) -> list[Document]:
        """Extract embedded images from a PDF page and OCR them."""
        import tempfile

        docs: list[Document] = []
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                import pymupdf

                pix = pymupdf.Pixmap(page.parent, xref)
                if pix.n > 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pix.save(tmp.name)
                    text = self._ocr.extract_text(tmp.name)  # type: ignore[union-attr]

                if text.strip():
                    docs.append(
                        Document(
                            text=text,
                            source=str(pdf_path),
                            metadata={
                                "page": page_num,
                                "image_index": img_idx,
                                "format": "pdf_image",
                                "extraction": "ocr",
                            },
                        )
                    )
            except Exception:
                logger.debug(
                    "Failed to extract image %d from page %d of %s",
                    img_idx,
                    page_num,
                    pdf_path,
                    exc_info=True,
                )
        return docs

    def _load_image(self, path: Path) -> list[Document]:
        """Extract text from an image file using the configured OCR provider."""
        if self._ocr is None:
            logger.info(
                "Skipping image %s — no OCR provider configured. "
                "Pass ocr='vision' or ocr='tesseract' to enable.",
                path,
            )
            return []

        try:
            text = self._ocr.extract_text(str(path))
        except Exception:
            logger.warning("OCR failed for %s", path, exc_info=True)
            return []

        if not text.strip():
            logger.info("No text extracted from %s", path)
            return []

        return [
            Document(
                text=text,
                source=str(path),
                metadata={"format": path.suffix.lstrip("."), "extraction": "ocr"},
            )
        ]

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
