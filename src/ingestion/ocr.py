"""OCR engines with a common interface.

Preferred engine: PaddleOCR (PP-OCR, pretrained).  Fallback: Tesseract (pytesseract).
Both return text with reading order preserved (words grouped into lines by position).
"""
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    engine: str
    mean_confidence: Optional[float]  # 0-1, None if engine gives no scores
    n_boxes: int
    word_boxes: List[dict] = field(default_factory=list)


def _group_into_lines(items, y_tol=0.6):
    """items: list of (x_left, y_center, height, text). Group by y, order by x."""
    items = sorted(items, key=lambda t: t[1])
    lines, cur, cur_y, cur_h = [], [], None, None
    for x, y, h, txt in items:
        if cur and abs(y - cur_y) > y_tol * max(h, cur_h):
            lines.append(cur)
            cur = []
        if not cur:
            cur_y, cur_h = y, h
        cur.append((x, txt))
    if cur:
        lines.append(cur)
    return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)


class PaddleEngine:
    name = "paddleocr"

    def __init__(self, lang="en"):
        from paddleocr import PaddleOCR  # noqa: heavy optional import
        try:                                   # PaddleOCR >= 3.x
            self.ocr = PaddleOCR(lang=lang, use_textline_orientation=True)
            self.v3 = True
        except TypeError:                      # PaddleOCR 2.x
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
            self.v3 = False

    def run(self, img: Image.Image) -> OCRResult:
        arr = np.array(img.convert("RGB"))
        items, confs, boxes = [], [], []
        if self.v3:
            for page in self.ocr.predict(arr):
                texts, scores, polys = page["rec_texts"], page["rec_scores"], page["rec_polys"]
                for t, s, poly in zip(texts, scores, polys):
                    poly = np.asarray(poly)
                    items.append((poly[:, 0].min(), poly[:, 1].mean(), np.ptp(poly[:, 1]), t))
                    confs.append(float(s))
                    boxes.append({"text": t, "x0": float(poly[:, 0].min()), "top": float(poly[:, 1].min()),
                                  "x1": float(poly[:, 0].max()), "bottom": float(poly[:, 1].max())})
        else:
            res = self.ocr.ocr(arr, cls=True) or [[]]
            for box, (t, s) in (res[0] or []):
                poly = np.asarray(box)
                items.append((poly[:, 0].min(), poly[:, 1].mean(), np.ptp(poly[:, 1]), t))
                confs.append(float(s))
                boxes.append({"text": t, "x0": float(poly[:, 0].min()), "top": float(poly[:, 1].min()),
                          "x1": float(poly[:, 0].max()), "bottom": float(poly[:, 1].max())})
        return OCRResult(_group_into_lines(items), self.name,
                     float(np.mean(confs)) if confs else None, len(items), boxes)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang="eng"):
        import pytesseract
        if not shutil.which("tesseract"):
            win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_default):
                pytesseract.pytesseract.tesseract_cmd = win_default
        self.pt, self.lang = pytesseract, lang
        self.pt.get_tesseract_version()  # raises if binary missing

    def run(self, img: Image.Image) -> OCRResult:
        d = self.pt.image_to_data(img.convert("RGB"), lang=self.lang, config="--psm 4",
                                  output_type=self.pt.Output.DICT)
        lines, confs, boxes, n = {}, [], [], 0
        for i, word in enumerate(d["text"]):
            if not word.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append((d["left"][i], word))
            boxes.append({"text": word, "x0": float(d["left"][i]), "top": float(d["top"][i]),
                          "x1": float(d["left"][i] + d["width"][i]),
                          "bottom": float(d["top"][i] + d["height"][i])})
            c = float(d["conf"][i])
            if c >= 0:
                confs.append(c / 100)
            n += 1
        text = "\n".join(" ".join(w for _, w in sorted(ws)) for _, ws in sorted(lines.items()))
        return OCRResult(text, self.name, float(np.mean(confs)) if confs else None, n, boxes)


def get_engine(prefer: str = "auto"):
    """prefer: 'paddle' | 'tesseract' | 'auto' (paddle, then tesseract)."""
    errors = []
    if prefer in ("auto", "paddle"):
        try:
            return PaddleEngine()
        except Exception as e:  # not installed / model download failed
            errors.append(f"paddleocr: {e}")
    if prefer in ("auto", "tesseract"):
        try:
            return TesseractEngine()
        except Exception as e:
            errors.append(f"tesseract: {e}")
    raise RuntimeError("No OCR engine available. " + " | ".join(errors))