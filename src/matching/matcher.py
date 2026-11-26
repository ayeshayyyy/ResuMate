"""Resume <-> job-description matching with a pretrained sentence-embedding model (no training)."""
from functools import lru_cache
import re

from src.config import EMBEDDING_MODEL
from src.extraction.skills import flat_skills

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have", "in",
              "is", "it", "must", "of", "on", "or", "our", "the", "to", "using", "with", "you",
              "experience", "require", "required", "requires", "years"}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_similarity(resume_text: str, jd_text: str) -> float:
    from sentence_transformers import util
    m = _model()
    a, b = m.encode([resume_text[:3000], jd_text[:3000]], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a, b))


def skill_gap(resume_text: str, jd_text: str) -> dict:
    have = {s.lower() for s in flat_skills(resume_text)}
    need = {s.lower() for s in flat_skills(jd_text)}
    return {"matched": sorted(need & have), "missing": sorted(need - have),
            "coverage": (len(need & have) / len(need)) if need else None}


def _sentences(text):
    return [part.strip(" \t-*") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _contains_term(text, term):
    escaped = re.escape(term.strip())
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, re.I))


def _requirement_terms(requirement):
    terms = list(dict.fromkeys(skill.lower() for skill in flat_skills(requirement)))
    if terms:
        return terms
    words = [word.strip(".") for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{2,}", requirement.lower())]
    return list(dict.fromkeys(word for word in words if word not in _STOPWORDS))


def requirement_evidence(resume_text: str, jd_text: str) -> dict:
    """Return a verdict and a supporting source sentence for every JD sentence."""
    resume_sentences = _sentences(resume_text)
    requirements = []
    total_score = 0.0
    for requirement in _sentences(jd_text):
        terms = _requirement_terms(requirement)
        matches = [term for term in terms if _contains_term(resume_text, term)]
        ratio = len(matches) / len(terms) if terms else 0.0
        verdict = "supported" if ratio == 1.0 and terms else "partial" if ratio > 0 else "not_found"
        evidence = max(resume_sentences,
                       key=lambda sentence: sum(_contains_term(sentence, term) for term in matches),
                       default=None)
        if evidence and not any(_contains_term(evidence, term) for term in matches):
            evidence = None
        evidence_terms = ([term for term in matches if _contains_term(evidence, term)] if evidence else [])
        total_score += ratio
        requirements.append({"requirement": requirement, "terms": terms, "matched_terms": matches,
                             "verdict": verdict, "evidence": evidence,
                             "evidence_terms": evidence_terms, "coverage": ratio})
    evidence_score = total_score / len(requirements) if requirements else 0.0
    return {"requirements": requirements, "evidence_score": evidence_score}


def match(resume_text: str, jd_text: str) -> dict:
    out = skill_gap(resume_text, jd_text)
    out.update(requirement_evidence(resume_text, jd_text))
    try:
        out["semantic_similarity"] = semantic_similarity(resume_text, jd_text)
    except Exception as e:  # sentence-transformers not installed
        out["semantic_similarity"] = None
        out["note"] = f"semantic similarity unavailable: {e}"
    return out


def rank_candidates(candidates, jd_text: str) -> list:
    """Rank candidate mappings with `name` and `resume_text` by evidence score."""
    ranked = []
    for candidate in candidates:
        result = match(candidate["resume_text"], jd_text)
        ranked.append({"candidate": candidate.get("name") or "Candidate", **result})
    ranked.sort(key=lambda result: (-result["evidence_score"],
                                   -(result["semantic_similarity"] or 0.0),
                                   result["candidate"].casefold()))
    for position, result in enumerate(ranked, 1):
        result["rank"] = position
    return ranked
