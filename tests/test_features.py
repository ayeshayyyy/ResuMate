import io
from datetime import date
from unittest.mock import patch

import pdfplumber
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from src.extraction import heuristics
from src.extraction.pipeline import ResumeAssistant
from src.extraction.rules import analyze_timeline
from src.extraction.skills import extract_skills
from src.ingestion.loader import layout_reading_order, load_document
from src.ingestion.visual import locate_page_fields, render_page_with_marks
from src.matching.matcher import rank_candidates, requirement_evidence
from src.privacy import create_flattened_redacted_pdf, score_invariance


def _pdf(lines):
    output = io.BytesIO()
    page = canvas.Canvas(output, pagesize=A4)
    y = 800
    for line in lines:
        page.drawString(60, y, line)
        y -= 24
    page.save()
    return output.getvalue()


def test_two_column_layout_reads_left_column_before_right():
    words = []
    for top, left, right in ((10, "Left", "Right"), (30, "column", "column"),
                             (50, "continues", "continues")):
        words.extend([{"text": left, "x0": 20, "x1": 70, "top": top, "bottom": top + 8},
                      {"text": right, "x0": 320, "x1": 380, "top": top, "bottom": top + 8}])
    text = layout_reading_order(words, 600)
    assert text.index("continues") < text.rindex("Right")
    assert text.splitlines() == ["Left", "column", "continues", "Right", "column", "continues"]


def test_uppercase_name_and_customer_service_skills():
    text = """KERRY WHITAKER
PROFESSIONAL SUMMARY
Cashier with a record of exceeding sales targets.
SKILLS
Cash handling
Customer service
POS systems
Team leadership
Product recommendations
Attention to detail
Communication
Time management
"""
    assert heuristics.predict_strings(text)["Name"] == ["KERRY WHITAKER"]
    skills = {skill.lower() for values in extract_skills(text).values() for skill in values}
    assert {"cash handling", "customer service", "pos systems", "team leadership",
            "product recommendations", "attention to detail", "communication", "time management"} <= skills


def test_retail_resume_transcript_pipeline_extracts_candidate_skills_and_experience():
    text = """KERRY WHITAKER
PROFESSIONAL SUMMARY
Results-driven cashier providing outstanding service.
SKILLS
Cash handling
Customer service
POS systems
Team leadership
Product recommendations
Attention to detail
Communication
Time management
WORK HISTORY
Burlington - Cashier
Inglewood, CA - 01/2022 - Current
Big Lots Store - Checkout Associate
Inglewood, CA - 09/2018 - 12/2021
Walgreens - Customer Service Associate
Inglewood, CA - 06/2015 - 05/2018
EDUCATION
Los Angeles City College
"""
    output = ResumeAssistant(model_path="/nonexistent").process_text(text)
    assert output["contact"]["name"] == "KERRY WHITAKER"
    assert output["experience_years"] > 0
    found_skills = {skill.lower() for values in output["skills"].values() for skill in values}
    assert {"cash handling", "customer service", "pos systems"} <= found_skills


def test_provenance_matches_words_and_renders_overlay():
    words = [{"text": "Jane", "x0": 10, "x1": 40, "top": 10, "bottom": 20},
             {"text": "Doe", "x0": 42, "x1": 65, "top": 10, "bottom": 20}]
    marks = locate_page_fields(words, {"Name": ["Jane Doe"]})
    assert marks[0]["bbox"] == [10, 10, 65, 20]
    image = render_page_with_marks(_pdf(["Jane Doe"]), "resume.pdf", 0,
                                  {"width": 595, "height": 842, "coordinate_width": 595,
                                   "coordinate_height": 842}, marks)
    assert image.size[0] > 0


def test_provenance_can_locate_repeated_pii_occurrences():
    words = [{"text": value, "x0": index * 30, "x1": index * 30 + 20,
              "top": index * 20, "bottom": index * 20 + 10}
             for index, value in enumerate(["Jane", "Doe", "Jane", "Doe"])]
    marks = locate_page_fields(words, {"Name": ["Jane Doe"]}, all_occurrences=True)
    assert len(marks) == 2


def test_pdf_tables_are_returned_as_rows():
    output = io.BytesIO()
    page = canvas.Canvas(output, pagesize=A4)
    table = Table([["Skill", "Level"], ["Python", "Advanced"]], colWidths=[180, 180])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    table.wrapOn(page, 400, 200)
    table.drawOn(page, 60, 700)
    page.drawString(60, 600, "Candidate resume document with enough content for text extraction threshold.")
    page.save()
    loaded = load_document(output.getvalue(), "table.pdf")
    assert loaded.tables
    assert loaded.tables[0]["rows"][1][0] == "Python"


def test_timeline_flags_overlaps_gaps_future_dates_and_claim_drift():
    text = """EXPERIENCE
Engineer A Jan 2018 - Dec 2018
Engineer B Jan 2020 - Dec 2021
Engineer C Jan 2021 - Present
Engineer D Jan 2028 - Present
Engineer E 13/2020 - 01/2021
Claims 5 years of experience
Claims 3 years of Java development experience
"""
    result = analyze_timeline(text, today=date(2025, 1, 1))
    assert result["overlaps"]
    assert result["gaps"]
    assert result["impossible_dates"]
    assert any("month" in item["reason"] for item in result["impossible_dates"])
    assert result["claims"][0]["consistent_within_one_year"] is False
    assert len(result["claims"]) == 2


def test_requirement_verdict_quotes_supporting_sentence():
    result = requirement_evidence("Built APIs using Python and Flask. Managed SQL deployments.",
                                  "Requires Python and Flask experience. Must know SQL.")
    assert [item["verdict"] for item in result["requirements"]] == ["supported", "supported"]
    assert result["requirements"][0]["evidence"] == "Built APIs using Python and Flask."


def test_candidate_ranking_uses_auditable_evidence_score():
    results = [{"candidate": "B", "evidence_score": 0.4, "semantic_similarity": 0.9},
               {"candidate": "A", "evidence_score": 0.8, "semantic_similarity": 0.2}]
    with patch("src.matching.matcher.match", side_effect=results):
        ranked = rank_candidates([{"name": "B", "resume_text": "b"},
                                  {"name": "A", "resume_text": "a"}], "JD")
    assert [item["candidate"] for item in ranked] == ["A", "B"]
    assert [item["rank"] for item in ranked] == [1, 2]


def test_blind_screening_redacts_pii_and_preserves_job_evidence_score():
    text = ("Jane Doe lives in Boston. Email jane@example.com. DOB: 01/02/1990. "
            "Address: 14 Main Street, Boston. Portfolio https://jane.example.com. Python developer with Flask.")
    report = score_invariance(text, "Requires Python and Flask.",
                              {"entities": {"Name": ["Jane Doe"], "Location": ["Boston"]}})
    assert "Jane Doe" not in report["redacted_text"]
    assert "jane@example.com" not in report["redacted_text"]
    assert "01/02/1990" not in report["redacted_text"]
    assert "14 Main Street" not in report["redacted_text"]
    assert "jane.example.com" not in report["redacted_text"]
    assert report["invariant"]


def test_redacted_pdf_is_flattened_and_contains_opaque_masks():
    original = _pdf(["Jane Doe jane@example.com Python developer"])
    loaded = load_document(original, "resume.pdf")
    redaction = score_invariance("Jane Doe jane@example.com Python developer", "Python developer",
                                {"entities": {"Name": ["Jane Doe"]}})
    flattened = create_flattened_redacted_pdf(original, "resume.pdf", redaction["fields"], loaded.pages)
    with pdfplumber.open(io.BytesIO(flattened)) as pdf:
        assert len(pdf.pages) == 1
        assert not (pdf.pages[0].extract_text() or "").strip()
    marks = locate_page_fields(loaded.pages[0]["word_boxes"], redaction["fields"], all_occurrences=True)
    image = pdfium.PdfDocument(flattened)[0].render(scale=1).to_pil().convert("RGB")
    x0, y0, x1, y1 = marks[0]["bbox"]
    sx = image.width / loaded.pages[0]["coordinate_width"]
    sy = image.height / loaded.pages[0]["coordinate_height"]
    assert image.getpixel((round((x0 + x1) * sx / 2), round((y0 + y1) * sy / 2))) == (0, 0, 0)