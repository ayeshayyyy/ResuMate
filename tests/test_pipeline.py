import io
from datetime import date

import pypdfium2 as pdfium
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.extraction import rules
from src.extraction.pipeline import ResumeAssistant
from src.extraction.skills import extract_skills
from src.ingestion.loader import load_document

LINES = ["Jane Doe", "jane.doe@mail.com | +1 415 555 2671", "", "EXPERIENCE",
         "Data Scientist - Acme Corp   Mar 2019 - Present", "Analyst - Beta Corp   2016 - 2018", "",
         "EDUCATION", "BS Computer Science - MIT", "", "SKILLS: Python, SQL, PyTorch"]


def _pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for l in LINES:
        c.setFont("Helvetica-Bold" if l.isupper() else "Helvetica", 13)
        c.drawString(60, y, l)
        y -= 24
    c.save()
    return buf.getvalue()


def _scan(pdf_bytes):
    img = pdfium.PdfDocument(pdf_bytes)[0].render(scale=2).to_pil().convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_rules_contact_and_dates():
    t = "\n".join(LINES)
    assert rules.extract_emails(t) == ["jane.doe@mail.com"]
    assert rules.extract_phones(t) == ["+1 415 555 2671"]
    r = rules.extract_date_ranges(t, today=date(2025, 1, 1))
    assert {x["text"] for x in r} >= {"Mar 2019 - Present", "2016 - 2018"}
    assert rules.total_experience_years(t, today=date(2025, 1, 1)) == 8.9   # (71 + 36) months = 8.9 y; education excluded


def test_no_phone_false_positive_on_years():
    assert rules.extract_phones("Worked 2016-2018 and 2019 - 2021") == []


def test_skills_boundaries():
    s = extract_skills("Skills: C++, C#, Node.js, Go, spring boot. I will go home and rest.")
    flat = {x.lower() for v in s.values() for x in v}
    assert {"c++", "c#", "node.js", "spring boot"} <= flat
    assert "rest" not in flat


def test_digital_pdf_uses_text_layer():
    d = load_document(_pdf(), "cv.pdf")
    assert not d.used_ocr and "Jane Doe" in d.text


def test_scanned_image_uses_ocr_and_pipeline_extracts():
    pytest.importorskip("pytesseract")
    scan = _scan(_pdf())
    out = ResumeAssistant(model_path="/nonexistent").process_file(scan, "scan.png")
    assert out["document"]["used_ocr"] is True
    assert out["contact"]["emails"] == ["jane.doe@mail.com"]       # OCR e-mail repair works
    assert out["contact"]["name"] == "Jane Doe"
    assert "python" in [s.lower() for s in out["skills"].get("Programming", [])]
