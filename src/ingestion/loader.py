"""Load a resume from PDF / image / text into clean text.

Strategy (per page):
  1. Try the embedded text layer (pdfplumber). Good for digital PDFs.
  2. If a page has too little text (scanned PDF, image-only, photo) -> rasterise & OCR.
Images (png/jpg) go straight to OCR.
"""
import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber
from PIL import Image

from .ocr import get_engine

MIN_CHARS_PER_PAGE = 80      # below this the page is treated as scanned
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class LoadedDocument:
    text: str
    pages: List[dict] = field(default_factory=list)  # {page, method, chars, confidence}
    ocr_engine: Optional[str] = None

    @property
    def used_ocr(self):
        return any(p["method"] == "ocr" for p in self.pages)


def clean_text(text: str) -> str:
    """Normalise unicode, bullets and whitespace while preserving line structure."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25CF\u27A2\u2756\uf0b7\uf0a7]", "\n", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\S\n]{3,}", "\n", text)           # wide gaps = column/table break
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"(?<=\w)\s*@\s*(?=\w)", "@", text)           # OCR often splits e-mails: "a.b @x.com"
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def _render_pdf_page(pdf_bytes: bytes, page_idx: int, scale: float = 2.5) -> Image.Image:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        return pdf[page_idx].render(scale=scale).to_pil()
    finally:
        pdf.close()


def load_document(data: bytes, filename: str = "", ocr_engine: str = "auto",
                  force_ocr: bool = False) -> LoadedDocument:
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    engine = None

    def ocr(img):
        nonlocal engine
        engine = engine or get_engine(ocr_engine)
        return engine.run(img)

    if ext in IMAGE_EXT:
        res = ocr(Image.open(io.BytesIO(data)))
        return LoadedDocument(clean_text(res.text),
                              [{"page": 1, "method": "ocr", "chars": len(res.text),
                                "confidence": res.mean_confidence}], res.engine)

    if ext in (".txt", ".md"):
        return LoadedDocument(clean_text(data.decode("utf-8", "ignore")),
                              [{"page": 1, "method": "text", "chars": len(data), "confidence": None}])

    pages, texts = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = "" if force_ocr else (page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
            if len(txt.strip()) >= MIN_CHARS_PER_PAGE:
                pages.append({"page": i + 1, "method": "text-layer", "chars": len(txt), "confidence": None})
            else:
                res = ocr(_render_pdf_page(data, i))
                txt = res.text
                pages.append({"page": i + 1, "method": "ocr", "chars": len(txt),
                              "confidence": res.mean_confidence})
            texts.append(txt)
    return LoadedDocument(clean_text("\n".join(texts)), pages, engine.name if engine else None)
