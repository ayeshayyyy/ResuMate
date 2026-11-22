"""Rule/heuristic NER baseline (no ML). Used as the 'lower bound' in the evaluation."""
import re
from typing import Dict, List

from src.config import ENTITY_TYPES

ROLE_WORDS = r"(?:engineer|developer|manager|analyst|consultant|administrator|specialist|designer|director|" \
             r"architect|scientist|lead|associate|intern|officer|executive|programmer|tester|coordinator|head)"
DEGREE_RE = re.compile(
    r"\b(?:B\.?\s?(?:Tech|E|Sc|S|A|Com|CA|BA)|M\.?\s?(?:Tech|E|Sc|S|A|Com|CA|BA)|Ph\.?D|MBA|BBA|BCA|MCA|"
    r"Bachelor(?:s)?(?: of [A-Za-z ]+)?|Master(?:s)?(?: of [A-Za-z ]+)?|Diploma(?: in [A-Za-z ]+)?)\b"
    r"(?:\.?\s?(?:in|of)\s[A-Z][A-Za-z&]+(?:\s[A-Za-z&]+){0,3})?")
COLLEGE_RE = re.compile(r"[A-Z][A-Za-z.&']*(?:\s[A-Za-z.&']+){0,5}\s(?:University|College|Institute|School|Academy)"
                        r"(?:\sof\s[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})?|"
                        r"(?:University|Institute|College) of [A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3}")
COMPANY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.]+\s){0,3}(?:Inc|LLC|Ltd|Limited|Corp|Corporation|Technologies|"
                        r"Solutions|Systems|Services|Labs|Pvt|Group|Consulting)\b\.?")
DESIG_RE = re.compile(rf"\b(?:(?:Senior|Junior|Lead|Principal|Staff|Associate|Assistant|Software|Data|Machine Learning|"
                      rf"Full ?Stack|Front ?end|Back ?end|Web|Mobile|DevOps|QA|Project|Product|Program|Systems?|"
                      rf"Application|Java|Python|Cloud|Network|Business|Technical)\s+){{0,3}}{ROLE_WORDS}\b", re.I)
LOC_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s(?:[A-Z]{2}\b|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")


def predict_strings(text: str) -> Dict[str, List[str]]:
    out = {t: [] for t in ENTITY_TYPES}
    lines = [l for l in text.splitlines() if l.strip()]
    for line in lines[:5]:  # name = first short, letters-only, non-heading line
        words = line.strip().split()
        if 2 <= len(words) <= 4 and re.fullmatch(r"[A-Za-z.' -]+", line.strip()) \
                and not DESIG_RE.search(line) and not line.strip().isupper():
            out["Name"] = [line.strip()]
            break
    pats = {"Degree": DEGREE_RE, "College": COLLEGE_RE, "Company": COMPANY_RE,
            "Designation": DESIG_RE, "Location": LOC_RE}
    for label, pat in pats.items():
        vals = []
        for m in pat.finditer(text):
            v = m.group(0).strip(" .,-")
            if len(v) > 2 and v.lower() not in (x.lower() for x in vals):
                vals.append(v)
        out[label] = vals
    return out
