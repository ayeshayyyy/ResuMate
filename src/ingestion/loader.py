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
    tables: List[dict] = field(default_factory=list)

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


def _line_groups(words, y_tolerance=4):
    lines = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > y_tolerance:
            lines.append([word])
        else:
            lines[-1].append(word)
    return [sorted(line, key=lambda item: item["x0"]) for line in lines]


def layout_reading_order(words, page_width):
    """Read detected columns top-to-bottom, left column before right column."""
    if not words:
        return ""
    intervals = []
    for word in sorted(words, key=lambda item: item["x0"]):
        if intervals and word["x0"] - intervals[-1][1] <= 18:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], word["x1"]))
        else:
            intervals.append((word["x0"], word["x1"]))
    gaps = [(right[0] - left[1], (left[1] + right[0]) / 2)
            for left, right in zip(intervals, intervals[1:])]
    split = None
    if gaps:
        gap, center = max(gaps)
        left_count = sum(word["x1"] <= center for word in words)
        right_count = sum(word["x0"] >= center for word in words)
        if gap >= max(30, page_width * 0.05) and min(left_count, right_count) >= 2:
            split = center
    lines = _line_groups(words)
    if split is None:
        return "\n".join(" ".join(word["text"] for word in line) for line in lines)

    output, left_block, right_block = [], [], []
    for line in lines:
        crosses = any(word["x0"] < split < word["x1"] for word in line)
        left_words = [word for word in line if word["x1"] <= split]
        right_words = [word for word in line if word["x0"] >= split]
        has_both = bool(left_words and right_words)
        bridge_gap = (min(word["x0"] for word in right_words) -
                      max(word["x1"] for word in left_words)) if has_both else page_width
        spans_gutter = has_both and bridge_gap <= max(24, page_width * 0.04)
        if crosses or spans_gutter:
            output.extend(" ".join(word["text"] for word in line)
                          for block in (left_block, right_block) for line in block)
            left_block, right_block = [], []
            output.append(" ".join(word["text"] for word in line))
        elif has_both:
            left_block.append(left_words)
            right_block.append(right_words)
        elif right_words:
            right_block.append(right_words)
        else:
            left_block.append(left_words or line)
    output.extend(" ".join(word["text"] for word in line)
                  for block in (left_block, right_block) for line in block)
    return "\n".join(output)


def load_document(data: bytes, filename: str = "", ocr_engine: str = "auto",
                  force_ocr: bool = False) -> LoadedDocument:
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    engine = None

    def ocr(img):
        nonlocal engine
        engine = engine or get_engine(ocr_engine)
        return engine.run(img)

    if ext in IMAGE_EXT:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        res = ocr(image)
        return LoadedDocument(clean_text(res.text),
                              [{"page": 1, "method": "ocr", "chars": len(res.text),
                    "confidence": res.mean_confidence, "width": image.width,
                    "height": image.height, "coordinate_width": image.width,
                    "coordinate_height": image.height, "word_boxes": res.word_boxes,
                    "tables": []}], res.engine)

    if ext in (".txt", ".md"):
        return LoadedDocument(clean_text(data.decode("utf-8", "ignore")),
                      [{"page": 1, "method": "text", "chars": len(data), "confidence": None,
                    "word_boxes": [], "tables": []}])

    pages, texts, tables = [], [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables() or []
            tables.extend({"page": i + 1, "rows": table} for table in page_tables if table)
            txt = "" if force_ocr else (page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
            if len(txt.strip()) >= MIN_CHARS_PER_PAGE:
                word_boxes = [{"text": w["text"], "x0": float(w["x0"]), "top": float(w["top"]),
                               "x1": float(w["x1"]), "bottom": float(w["bottom"])}
                              for w in page.extract_words(x_tolerance=2, y_tolerance=3)]
                txt = layout_reading_order(word_boxes, page.width) or txt
                pages.append({"page": i + 1, "method": "text-layer", "chars": len(txt), "confidence": None})
            else:
                ocr_image = _render_pdf_page(data, i)
                res = ocr(ocr_image)
                txt = res.text
                word_boxes = res.word_boxes
                pages.append({"page": i + 1, "method": "ocr", "chars": len(txt),
                              "confidence": res.mean_confidence,
                              "coordinate_width": ocr_image.width, "coordinate_height": ocr_image.height})
            pages[-1].update({"width": float(page.width), "height": float(page.height),
                              "word_boxes": word_boxes, "tables": page_tables})
            pages[-1].setdefault("coordinate_width", float(page.width))
            pages[-1].setdefault("coordinate_height", float(page.height))
            texts.append(txt)
    return LoadedDocument(clean_text("\n".join(texts)), pages, engine.name if engine else None, tables)
