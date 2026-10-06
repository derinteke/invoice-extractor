"""Turn uploaded files (images or PDFs) into bytes an extractor can send."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MAX_SIDE = 2000  # px; larger scans cost more tokens without reading better


@dataclass
class Payload:
    data: bytes
    mime_type: str


def image_to_jpeg(img: Image.Image, max_side: int = MAX_SIDE) -> bytes:
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def pdf_to_images(data: bytes, max_pages: int = 3, dpi: int = 150) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    return [pdf[i].render(scale=dpi / 72).to_pil() for i in range(min(len(pdf), max_pages))]


def load_payload(data: bytes, filename: str) -> Payload:
    """PDFs are passed through (Gemini reads them natively); images are normalised to JPEG."""
    if Path(filename).suffix.lower() == ".pdf":
        return Payload(data, "application/pdf")
    return Payload(image_to_jpeg(Image.open(io.BytesIO(data))), "image/jpeg")


def payload_images(payload: Payload) -> list[bytes]:
    """JPEG pages for extractors that only accept images (local models)."""
    if payload.mime_type == "application/pdf":
        return [image_to_jpeg(p) for p in pdf_to_images(payload.data)]
    return [payload.data]
