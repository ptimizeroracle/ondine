"""Tests for OCR protocol, implementations, and resolve factory."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ondine.knowledge.ocr import (
    DocTROCR,
    TesseractOCR,
    VisionOCR,
    is_image_file,
    resolve_ocr,
)
from ondine.knowledge.protocols import OCRProvider

# ── Protocol conformance ──────────────────────────────────────────


class _MockOCR:
    def extract_text(self, image_path: str) -> str:
        return "mock text"


def test_mock_satisfies_ocr_protocol():
    assert isinstance(_MockOCR(), OCRProvider)


def test_vision_ocr_satisfies_protocol():
    assert isinstance(VisionOCR(), OCRProvider)


def test_tesseract_ocr_satisfies_protocol():
    assert isinstance(TesseractOCR(), OCRProvider)


def test_doctr_ocr_satisfies_protocol():
    assert isinstance(DocTROCR(), OCRProvider)


# ── is_image_file ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("photo.png", True),
        ("photo.PNG", True),
        ("doc.jpg", True),
        ("doc.jpeg", True),
        ("img.webp", True),
        ("scan.tiff", True),
        ("scan.tif", True),
        ("icon.bmp", True),
        ("anim.gif", True),
        ("report.pdf", False),
        ("notes.txt", False),
        ("data.csv", False),
    ],
)
def test_is_image_file(path, expected):
    assert is_image_file(path) == expected


# ── VisionOCR ─────────────────────────────────────────────────────


@patch("litellm.completion")
def test_vision_ocr_calls_litellm(mock_completion):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "  Extracted invoice text  "
    mock_completion.return_value = mock_response

    ocr = VisionOCR(model="gpt-4o", api_key="sk-test")  # pragma: allowlist secret

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        f.flush()
        result = ocr.extract_text(f.name)

    assert result == "Extracted invoice text"
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"
    assert call_kwargs.kwargs["api_key"] == "sk-test"  # pragma: allowlist secret

    messages = call_kwargs.kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert any(c["type"] == "image_url" for c in content)


def test_vision_ocr_repr():
    ocr = VisionOCR(model="gemini-2.0-flash")
    assert "gemini-2.0-flash" in repr(ocr)


# ── TesseractOCR ──────────────────────────────────────────────────


def test_tesseract_ocr_calls_pytesseract():
    import sys

    mock_tess = MagicMock()
    mock_tess.image_to_string.return_value = "  Receipt total: $42.50  "

    mock_pil_image = MagicMock()
    mock_img = MagicMock()
    mock_pil_image.open.return_value = mock_img

    with patch.dict(
        sys.modules,
        {
            "pytesseract": mock_tess,
            "PIL": MagicMock(Image=mock_pil_image),
            "PIL.Image": mock_pil_image,
        },
    ):
        ocr = TesseractOCR(lang="eng")
        result = ocr.extract_text("/tmp/receipt.png")

    assert result == "Receipt total: $42.50"
    mock_pil_image.open.assert_called_once_with("/tmp/receipt.png")
    mock_tess.image_to_string.assert_called_once_with(mock_img, lang="eng", config="")


def test_tesseract_ocr_repr():
    ocr = TesseractOCR(lang="fra")
    assert "fra" in repr(ocr)


# ── DocTROCR ──────────────────────────────────────────────────────


def test_doctr_ocr_calls_predictor():
    import sys

    mock_doctr_io = MagicMock()
    mock_doctr_io.DocumentFile.from_images.return_value = "fake_doc"

    mock_model = MagicMock()
    mock_model.return_value.render.return_value = "Document content here"

    mock_doctr_models = MagicMock()
    mock_doctr_models.ocr_predictor.return_value = mock_model

    with patch.dict(
        sys.modules,
        {
            "doctr": MagicMock(),
            "doctr.io": mock_doctr_io,
            "doctr.models": mock_doctr_models,
        },
    ):
        ocr = DocTROCR()
        result = ocr.extract_text("/tmp/scan.png")

    assert result == "Document content here"
    mock_doctr_io.DocumentFile.from_images.assert_called_once_with("/tmp/scan.png")
    mock_model.assert_called_once_with("fake_doc")


def test_doctr_ocr_repr():
    ocr = DocTROCR(det_arch="db_mobilenet_v3_large")
    assert "db_mobilenet_v3_large" in repr(ocr)


# ── resolve_ocr factory ──────────────────────────────────────────


def test_resolve_none_returns_none():
    assert resolve_ocr(None) is None


def test_resolve_tesseract_string():
    result = resolve_ocr("tesseract")
    assert isinstance(result, TesseractOCR)


def test_resolve_doctr_string():
    result = resolve_ocr("doctr")
    assert isinstance(result, DocTROCR)


def test_resolve_vision_string():
    result = resolve_ocr("vision")
    assert isinstance(result, VisionOCR)


def test_resolve_model_name_string():
    result = resolve_ocr("openrouter/google/gemini-2.0-flash-001")
    assert isinstance(result, VisionOCR)
    assert result._model == "openrouter/google/gemini-2.0-flash-001"


def test_resolve_existing_provider():
    mock = _MockOCR()
    result = resolve_ocr(mock)
    assert result is mock
