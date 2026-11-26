"""Visual field provenance and safe raster rendering helpers."""
import io
import re

from PIL import Image, ImageDraw


def _tokens(value):
    return [token.strip(".,:;()[]{}'").lower() for token in
            re.findall(r"[A-Za-z0-9@._+#-]+", str(value)) if token.strip(".,:;()[]{}'")]


def locate_page_fields(word_boxes, fields, all_occurrences=False):
    """Associate extracted field values with contiguous page word boxes."""
    words = sorted(word_boxes, key=lambda item: (item["top"], item["x0"]))
    word_tokens = [_tokens(word.get("text", "")) for word in words]
    marks, seen = [], set()
    for label, values in fields.items():
        for value in values:
            target = _tokens(value)
            if not target:
                continue
            for start in range(max(0, len(words) - len(target) + 1)):
                if word_tokens[start:start + len(target)] != [[token] for token in target]:
                    continue
                chosen = words[start:start + len(target)]
                bbox = [min(word["x0"] for word in chosen), min(word["top"] for word in chosen),
                        max(word["x1"] for word in chosen), max(word["bottom"] for word in chosen)]
                key = (label, str(value).lower(), tuple(round(number, 1) for number in bbox))
                if key not in seen:
                    marks.append({"field": label, "value": str(value), "bbox": bbox})
                    seen.add(key)
                if not all_occurrences:
                    break
    return marks


def render_page_with_marks(data, filename, page_index, page_info, marks):
    """Render a source page and overlay its extraction evidence boxes."""
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == ".pdf":
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(data)
        try:
            image = document[page_index].render(scale=1.5).to_pil().convert("RGB")
        finally:
            document.close()
    else:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    source_width = page_info.get("coordinate_width") or page_info.get("width") or image.width
    source_height = page_info.get("coordinate_height") or page_info.get("height") or image.height
    sx, sy = image.width / source_width, image.height / source_height
    for mark in marks:
        x0, y0, x1, y1 = mark["bbox"]
        box = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
        draw.rectangle(box, outline=(0, 170, 125), width=max(2, image.width // 900))
    return image
