"""Blind-screening redaction helpers."""
import io
import re

from PIL import Image, ImageDraw

from src.extraction import rules
from src.ingestion.visual import locate_page_fields
from src.matching.matcher import requirement_evidence

ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9.'#-]+\s+){1,6}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way)\b", re.I)
DOB_RE = re.compile(
    r"\b(?:date\s+of\s+birth|dob|born)\s*[:\-]?\s*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})", re.I)
WEB_URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;]+", re.I)


def pii_fields(text, extracted=None):
    extracted = extracted or {}
    entities = extracted.get("entities", {})
    contact = extracted.get("contact", {})
    fields = {
        "Name": list(entities.get("Name", [])) + ([contact["name"]] if contact.get("name") else []),
        "Location": list(entities.get("Location", [])) + ([contact["location"]] if contact.get("location") else []),
        "Email": rules.extract_emails(text),
        "Phone": rules.extract_phones(text),
        "Profile link": rules.extract_links(text),
        "Address": ADDRESS_RE.findall(text),
        "Birth date": DOB_RE.findall(text),
        "Website": [value.rstrip(".!?)]}") for value in WEB_URL_RE.findall(text)],
    }
    return {label: list(dict.fromkeys(value for value in values if value)) for label, values in fields.items()}


def redact_text(text, extracted=None):
    """Redact detected names, locations, contact details, and profile links."""
    fields = pii_fields(text, extracted)
    redacted = text
    replacements = []
    items = sorted(((label, value) for label, values in fields.items() for value in values),
                   key=lambda item: len(item[1]), reverse=True)
    for label, value in items:
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.I)
        redacted, count = pattern.subn(f"[REDACTED {label.upper()}]", redacted)
        if count:
            replacements.append({"field": label, "count": count})
    return redacted, {"fields": fields, "replacements": replacements}


def create_flattened_redacted_pdf(data, filename, fields, page_metadata=None):
    """Rasterise source pages, mask detected PII boxes, and emit image-only PDF pages."""
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    page_metadata = page_metadata or []
    if ext == ".pdf":
        import pypdfium2 as pdfium
        source = pdfium.PdfDocument(data)
        images = []
        try:
            for index, page in enumerate(source):
                image = page.render(scale=2).to_pil().convert("RGB")
                metadata = page_metadata[index] if index < len(page_metadata) else {}
                boxes = metadata.get("word_boxes", [])
                marks = locate_page_fields(boxes, fields, all_occurrences=True)
                sx = image.width / (metadata.get("coordinate_width") or metadata.get("width") or image.width)
                sy = image.height / (metadata.get("coordinate_height") or metadata.get("height") or image.height)
                draw = ImageDraw.Draw(image)
                for mark in marks:
                    x0, y0, x1, y1 = mark["bbox"]
                    draw.rectangle((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)), fill="black")
                images.append(image)
        finally:
            source.close()
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        metadata = page_metadata[0] if page_metadata else {}
        marks = locate_page_fields(metadata.get("word_boxes", []), fields, all_occurrences=True)
        draw = ImageDraw.Draw(image)
        for mark in marks:
            draw.rectangle(tuple(round(value) for value in mark["bbox"]), fill="black")
        images = [image]
    else:
        raise ValueError("Flattened redacted PDF requires a PDF or image input")

    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    for image in images:
        pdf.setPageSize((image.width, image.height))
        pdf.drawImage(ImageReader(image), 0, 0, width=image.width, height=image.height)
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def score_invariance(text, jd_text, extracted=None):
    """Compare evidence scores before and after PII redaction."""
    redacted, details = redact_text(text, extracted)
    before = requirement_evidence(text, jd_text)["evidence_score"]
    after = requirement_evidence(redacted, jd_text)["evidence_score"]
    return {"before": before, "after": after, "delta": after - before,
            "invariant": abs(after - before) <= 1e-12, "redacted_text": redacted, **details}