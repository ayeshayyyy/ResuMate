"""Rule-based extractors: contact fields, dates, experience duration, sections.

Everything here is deterministic (regex + gazetteers) - no model training.
"""
import re
from datetime import date

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w/])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w/])")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in|github\.com)/[A-Za-z0-9_\-/.]+", re.I)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_YEAR = r"(?:19|20)\d{2}"


def _tok(s):
    return (rf"(?:(?P<mon{s}>{_MON})\.?,?\s+(?P<ya{s}>{_YEAR})"
            rf"|(?P<mm{s}>\d{{1,2}})[/.](?P<yb{s}>{_YEAR})"
            rf"|(?P<yc{s}>{_YEAR}))")


_PRESENT = r"(?P<pres>Present|Current|Now|Till\s+date|Today|Ongoing)"
DATE_RANGE_RE = re.compile(
    rf"(?<![\d/])" + _tok("1") + r"\s*(?:-|\u2013|\u2014|to|until|till)\s*(?:" + _tok("2") + r"|" + _PRESENT + r")(?![\d])",
    re.I,
)

HEADERS = {
    "experience": ["experience", "work experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic", "qualifications", "academics"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "projects": ["projects", "personal projects"],
    "summary": ["summary", "objective", "profile", "about me"],
    "certifications": ["certifications", "certificates", "awards", "achievements"],
}


def extract_emails(text):
    return list(dict.fromkeys(EMAIL_RE.findall(text)))


def extract_phones(text):
    out = []
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 9 <= len(digits) <= 14 and not re.fullmatch(rf"{_YEAR}{_YEAR}", digits):
            out.append(m.group(0).strip())
    return list(dict.fromkeys(out))


def extract_links(text):
    return list(dict.fromkeys(m.group(0) for m in URL_RE.finditer(text)))


def _to_month_index(mon, year):
    return int(year) * 12 + (MONTHS[mon[:3].lower()] if mon else 1) - 1


def extract_date_ranges(text, today=None):
    """Return list of dicts {text, start, end, months, current} for every date range."""
    today = today or date.today()
    now_idx = today.year * 12 + today.month - 1
    out = []
    for m in DATE_RANGE_RE.finditer(text):
        g = m.groupdict()
        s_year = g["ya1"] or g["yb1"] or g["yc1"]
        s_mon = g["mon1"] or (list(MONTHS)[int(g["mm1"]) - 1] if g["mm1"] and 1 <= int(g["mm1"]) <= 12 else None)
        start = _to_month_index(s_mon, s_year)
        current = bool(g["pres"])
        if current:
            end = now_idx
        else:
            e_year = g["ya2"] or g["yb2"] or g["yc2"]
            e_mon = g["mon2"] or (list(MONTHS)[int(g["mm2"]) - 1] if g["mm2"] and 1 <= int(g["mm2"]) <= 12 else None)
            end = _to_month_index(e_mon, e_year)
            if not (g["mon2"] or g["mm2"]):
                end += 11  # year-only end => end of that year
        if end < start or end - start > 12 * 50:
            continue
        out.append({"text": m.group(0).strip(), "start": start, "end": end,
                    "months": end - start + 1, "current": current})
    return out


def split_sections(text):
    """Split resume text into {section_name: text} using header lines."""
    sections, current, buf = {}, "header", []
    for line in text.splitlines():
        key = _header_key(line)
        if key:
            sections[current] = "\n".join(buf).strip()
            current, buf = key, []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v}


def _header_key(line):
    clean = re.sub(r"[^a-z ]", "", line.lower()).strip()
    if not clean or len(clean) > 30 or len(line.split()) > 4:
        return None
    for key, variants in HEADERS.items():
        if clean in variants:
            return key
    return None


def total_experience_years(text, today=None):
    """Union of date ranges in the experience part of the resume (overlaps counted once)."""
    sections = split_sections(text)
    if "experience" in sections:
        scope = sections["experience"]
    else:
        scope = "\n".join(v for k, v in sections.items() if k not in ("education", "certifications"))
    ranges = extract_date_ranges(scope, today)
    if not ranges:
        return 0.0
    ranges.sort(key=lambda r: r["start"])
    total, cur_s, cur_e = 0, ranges[0]["start"], ranges[0]["end"]
    for r in ranges[1:]:
        if r["start"] <= cur_e + 1:
            cur_e = max(cur_e, r["end"])
        else:
            total += cur_e - cur_s + 1
            cur_s, cur_e = r["start"], r["end"]
    total += cur_e - cur_s + 1
    return round(total / 12, 1)
