"""OCR implementations for image-to-text extraction.

Satisfies the ``OCRProvider`` protocol via structural subtyping.

Three built-in strategies:
* ``VisionOCR`` — uses a multimodal LLM (via *litellm*) to describe
  and extract text from images.  Best quality, requires API access.
* ``TesseractOCR`` — wraps ``pytesseract`` for local, offline OCR.
* ``DocTROCR`` — wraps ``doctr`` (Document Text Recognition) for
  high-accuracy local OCR on documents and screenshots.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ondine.knowledge.protocols import OCRProvider

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"}


def is_image_file(path: str) -> bool:
    """Check if a file path has an image extension."""
    from pathlib import Path

    return Path(path).suffix.lower() in _IMAGE_EXTENSIONS


def _image_to_data_uri(path: str) -> str:
    """Read an image file and return a base64 data URI."""
    from pathlib import Path

    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class VisionOCR:
    """Extract text from images using a multimodal LLM via litellm.

    Works with any vision-capable model: GPT-4o, Gemini 2.0 Flash,
    Claude 3.5 Sonnet, etc. The model receives the image and a prompt
    asking it to extract all visible text faithfully.

    Args:
        model: litellm model identifier (e.g. ``"gpt-4o"``).
        api_key: Optional API key override.
        prompt: System prompt guiding text extraction.
    """

    _DEFAULT_PROMPT = (
        "Extract ALL text visible in this image. Preserve the original "
        "structure (headings, paragraphs, tables, lists) as faithfully as "
        "possible using plain text or Markdown. If the image contains a "
        "table, reproduce it. If there are charts or diagrams, describe "
        "the data they represent. Do NOT add commentary — output only "
        "the extracted content."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._prompt = prompt or self._DEFAULT_PROMPT

    def extract_text(self, image_path: str) -> str:
        import litellm

        data_uri = _image_to_data_uri(image_path)

        kwargs: dict = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 4096,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = litellm.completion(**kwargs)
        return str(response.choices[0].message.content).strip()

    def __repr__(self) -> str:
        return f"VisionOCR(model={self._model!r})"


class TesseractOCR:
    """Extract text from images using Tesseract via pytesseract.

    Fully offline, no API calls. Requires ``tesseract`` binary
    installed on the system and ``pytesseract`` Python package.

    Args:
        lang: Tesseract language code (e.g. ``"eng"``, ``"fra"``).
        config: Extra Tesseract CLI config flags.
    """

    def __init__(self, lang: str = "eng", *, config: str = "") -> None:
        self._lang = lang
        self._config = config

    def extract_text(self, image_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "TesseractOCR requires pytesseract and Pillow. "
                "Install with: pip install pytesseract Pillow"
            ) from None

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=self._lang, config=self._config)
        return str(text).strip()

    def __repr__(self) -> str:
        return f"TesseractOCR(lang={self._lang!r})"


class DocTROCR:
    """Extract text from images using doctr (Document Text Recognition).

    High-accuracy local OCR optimized for documents, receipts, and
    screenshots. No API calls required.

    Args:
        det_arch: Detection model architecture.
        reco_arch: Recognition model architecture.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
    ) -> None:
        self._det_arch = det_arch
        self._reco_arch = reco_arch
        self._model = None

    def extract_text(self, image_path: str) -> str:
        model = self._load()

        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        return str(result.render())

    def _load(self):
        if self._model is not None:
            return self._model
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self._det_arch,
            reco_arch=self._reco_arch,
            pretrained=True,
        )
        logger.info(
            "Loaded DocTR OCR: det=%s, reco=%s", self._det_arch, self._reco_arch
        )
        return self._model

    def __repr__(self) -> str:
        return f"DocTROCR(det={self._det_arch!r}, reco={self._reco_arch!r})"


# ── Factory ───────────────────────────────────────────────────────


def resolve_ocr(spec: OCRProvider | str | None) -> OCRProvider | None:
    """Resolve an OCR specification to a concrete instance.

    Accepts:
    * ``None`` → no OCR (images will be skipped during ingestion).
    * ``"vision"`` or ``"gpt-4o"`` or any litellm model → ``VisionOCR``.
    * ``"tesseract"`` → ``TesseractOCR``.
    * ``"doctr"`` → ``DocTROCR``.
    * An existing ``OCRProvider``-compatible object → returned as-is.
    """
    if spec is None:
        return None

    if isinstance(spec, str):
        low = spec.lower()
        if low == "tesseract":
            return TesseractOCR()
        if low == "doctr":
            return DocTROCR()
        if low == "vision":
            return VisionOCR()
        # Assume it's a litellm model name for VisionOCR
        return VisionOCR(model=spec)

    return spec  # already an OCRProvider-compatible object
