"""Tests for DocumentLoader image and OCR integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ondine.knowledge.loader import DocumentLoader


class _FakeOCR:
    """Minimal OCRProvider for testing."""

    def __init__(self, text: str = "OCR extracted text"):
        self._text = text

    def extract_text(self, image_path: str) -> str:
        return self._text


# ── Image loading with OCR ────────────────────────────────────────


def test_image_file_loaded_with_ocr():
    ocr = _FakeOCR("Invoice total: $500")
    loader = DocumentLoader(ocr=ocr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        f.flush()
        docs = loader.load(f.name)

    assert len(docs) == 1
    assert docs[0].text == "Invoice total: $500"
    assert docs[0].metadata["format"] == "png"
    assert docs[0].metadata["extraction"] == "ocr"


def test_image_file_skipped_without_ocr():
    loader = DocumentLoader()  # no OCR configured

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG" + b"\x00" * 100)
        f.flush()
        docs = loader.load(f.name)

    assert docs == []


def test_image_with_empty_ocr_result():
    ocr = _FakeOCR("")
    loader = DocumentLoader(ocr=ocr)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        f.flush()
        docs = loader.load(f.name)

    assert docs == []


def test_ocr_failure_returns_empty():
    ocr = MagicMock()
    ocr.extract_text.side_effect = RuntimeError("OCR crashed")
    loader = DocumentLoader(ocr=ocr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG" + b"\x00" * 100)
        f.flush()
        docs = loader.load(f.name)

    assert docs == []


# ── Directory loading with mixed file types ───────────────────────


def test_directory_loads_images_and_text():
    ocr = _FakeOCR("Image text content")

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "notes.txt").write_text("Text file content")
        img_path = Path(tmpdir) / "screenshot.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        loader = DocumentLoader(ocr=ocr)
        docs = loader.load(tmpdir)

    texts = {d.text for d in docs}
    assert "Text file content" in texts
    assert "Image text content" in texts


def test_directory_skips_images_without_ocr():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "notes.txt").write_text("Text only")
        (Path(tmpdir) / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        loader = DocumentLoader()  # no OCR
        docs = loader.load(tmpdir)

    assert len(docs) == 1
    assert docs[0].text == "Text only"


# ── supported_extensions property ─────────────────────────────────


def test_supported_extensions_without_ocr():
    loader = DocumentLoader()
    exts = loader.supported_extensions
    assert ".pdf" in exts
    assert ".txt" in exts
    assert ".md" in exts
    assert ".png" not in exts
    assert ".jpg" not in exts


def test_supported_extensions_with_ocr():
    loader = DocumentLoader(ocr=_FakeOCR())
    exts = loader.supported_extensions
    assert ".png" in exts
    assert ".jpg" in exts
    assert ".jpeg" in exts
    assert ".webp" in exts
    assert ".tiff" in exts
    assert ".bmp" in exts
    assert ".pdf" in exts
    assert ".txt" in exts


# ── String shortcut for OCR ───────────────────────────────────────


def test_string_shortcut_resolves_ocr():
    loader = DocumentLoader(ocr="tesseract")
    from ondine.knowledge.ocr import TesseractOCR

    assert isinstance(loader._ocr, TesseractOCR)


def test_vision_string_shortcut():
    loader = DocumentLoader(ocr="vision")
    from ondine.knowledge.ocr import VisionOCR

    assert isinstance(loader._ocr, VisionOCR)


# ── KnowledgeStore integration ────────────────────────────────────


def test_knowledge_store_accepts_ocr_param():
    """KnowledgeStore passes ocr= through to DocumentLoader."""
    from ondine.knowledge.store import KnowledgeStore

    ocr = _FakeOCR("test")
    kb = KnowledgeStore(":memory:", ocr=ocr)
    assert kb._loader._ocr is ocr


def test_knowledge_store_ingest_image():
    """Full integration: image → OCR → chunk → store."""
    from ondine.knowledge.store import KnowledgeStore

    ocr = _FakeOCR("The quarterly revenue was $2.5 billion, up 15% year-over-year.")
    kb = KnowledgeStore(":memory:", ocr=ocr)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        f.flush()
        n = kb.ingest(f.name)

    assert n >= 1
    assert kb.chunk_count >= 1

    results = kb.search("quarterly revenue")
    assert len(results) >= 1
    assert "revenue" in results[0].text.lower()
