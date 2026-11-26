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
CLAIMED_EXPERIENCE_RE = re.compile(
    r"\b(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:(?:of|in)\s+)?"
    r"(?:[A-Za-z][A-Za-z0-9+#./-]*\s+){0,4}(?:experience|exp)\b", re.I)

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


def _range_months(match, today):
    groups = match.groupdict()
    start_year = groups["ya1"] or groups["yb1"] or groups["yc1"]
    start_month = groups["mon1"] or (list(MONTHS)[int(groups["mm1"]) - 1]
                                      if groups["mm1"] and 1 <= int(groups["mm1"]) <= 12 else None)
    start = _to_month_index(start_month, start_year)
    current = bool(groups["pres"])
    if current:
        end = today.year * 12 + today.month - 1
    else:
        end_year = groups["ya2"] or groups["yb2"] or groups["yc2"]
        end_month = groups["mon2"] or (list(MONTHS)[int(groups["mm2"]) - 1]
                                        if groups["mm2"] and 1 <= int(groups["mm2"]) <= 12 else None)
        end = _to_month_index(end_month, end_year)
        if not (groups["mon2"] or groups["mm2"]):
            end += 11
    return start, end, current


def _month_label(index):
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}"


def analyze_timeline(text, today=None):
    """Report dated-role overlaps, gaps, impossible dates, and experience-claim drift."""
    today = today or date.today()
    sections = split_sections(text)
    scope = sections.get("experience") or "\n".join(
        value for key, value in sections.items() if key not in ("education", "certifications"))
    now_idx = today.year * 12 + today.month - 1
    intervals, impossible = [], []
    for match in DATE_RANGE_RE.finditer(scope):
        groups = match.groupdict()
        invalid_month = any(groups[key] and not 1 <= int(groups[key]) <= 12 for key in ("mm1", "mm2"))
        if invalid_month:
            impossible.append({"text": match.group(0), "reason": "numeric month is outside 1-12"})
            continue
        start, end, current = _range_months(match, today)
        reason = None
        if start > now_idx or (not current and end > now_idx):
            reason = "date is in the future"
        elif end < start:
            reason = "end date precedes start date"
        elif end - start > 12 * 50 or start < 1900 * 12:
            reason = "date range is outside the plausible 1900-to-50-year window"
        if reason:
            impossible.append({"text": match.group(0), "reason": reason})
        else:
            intervals.append({"text": match.group(0).strip(), "start": start, "end": end,
                              "current": current})

    intervals.sort(key=lambda item: (item["start"], item["end"]))
    overlaps, gaps = [], []
    previous = None
    for interval in intervals:
        if previous:
            if interval["start"] <= previous["end"]:
                overlaps.append({"earlier": previous["text"], "later": interval["text"],
                                 "overlap_months": previous["end"] - interval["start"] + 1})
            elif interval["start"] > previous["end"] + 1:
                gaps.append({"from": _month_label(previous["end"] + 1),
                             "to": _month_label(interval["start"] - 1),
                             "months": interval["start"] - previous["end"] - 1,
                             "between": [previous["text"], interval["text"]]})
        if previous is None or interval["end"] > previous["end"]:
            previous = interval

    claims = [float(match.group("years")) for match in CLAIMED_EXPERIENCE_RE.finditer(text)]
    dated_years = total_experience_years(text, today)
    claim_deltas = [{"claimed_years": years, "dated_years": dated_years,
                     "difference_years": round(years - dated_years, 1),
                     "consistent_within_one_year": abs(years - dated_years) <= 1.0}
                    for years in claims]
    return {"dated_history_years": dated_years, "intervals": intervals, "overlaps": overlaps,
            "gaps": gaps, "impossible_dates": impossible, "claims": claim_deltas}
