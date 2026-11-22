"""Multimodal robustness: how much does scanning + OCR hurt extraction?

For each of the 30 hand-labelled synthetic resumes we
  1. render the text to a PDF page and rasterise it (image-only, like a scan),
  2. degrade it (small rotation, blur, noise) to mimic a phone scan,
  3. OCR it (PaddleOCR if installed, else Tesseract) and run the same extractor,
and compare Character Error Rate (CER) and entity F1 against the clean-text input.

    python -m evaluation.ocr_robustness [--engine auto|paddle|tesseract] [--model PATH]
"""
import argparse
import io
import json
import random

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from evaluation.metrics import evaluate
from evaluation.run_eval import load_synthetic_test
from src.config import ENTITY_TYPES, FINETUNED_DIR, resolve_model_path, RESULTS_DIR
from src.ingestion.loader import clean_text
from src.ingestion.ocr import get_engine


def render(text: str) -> Image.Image:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.splitlines():
        while len(line) > 95:
            c.setFont("Helvetica", 11); c.drawString(50, y, line[:95]); line, y = line[95:], y - 15
        c.setFont("Helvetica-Bold" if line.isupper() else "Helvetica", 11)
        c.drawString(50, y, line); y -= 15
    c.save()
    return pdfium.PdfDocument(buf.getvalue())[0].render(scale=2).to_pil().convert("RGB")


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.RandomState(rng.randrange(10**6)).normal(0, 12, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def cer(ref: str, hyp: str) -> float:
    ref, hyp = " ".join(ref.split()), " ".join(hyp.split())
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--model", default=resolve_model_path())
    args = ap.parse_args()

    from src.extraction.pipeline import ResumeAssistant
    assistant = ResumeAssistant(model_path=args.model)
    predict = (lambda t: assistant.process_text(t)["entities"])
    print("Extractor:", assistant.mode)

    engine, rng = get_engine(args.engine), random.Random(42)
    docs = load_synthetic_test()
    clean_in, ocr_in, cers = [], [], []
    for text, _ in docs:
        ref = clean_text(text)
        hyp = clean_text(engine.run(degrade(render(text), rng)).text)
        clean_in.append(ref); ocr_in.append(hyp); cers.append(cer(ref, hyp))
    golds = [g for _, g in docs]
    res = {}
    for name, texts in (("clean text", clean_in), ("scanned + OCR", ocr_in)):
        t, _ = evaluate(golds, [predict(x) for x in texts], ENTITY_TYPES, "partial")
        res[name] = t["micro"]["f1"]
    out = {"engine": engine.name, "extractor": assistant.mode, "mean_CER": float(np.mean(cers)),
           "micro_F1_partial": res}
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS_DIR / "ocr_robustness.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
