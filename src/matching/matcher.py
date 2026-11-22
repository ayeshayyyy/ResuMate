"""Resume <-> job-description matching with a pretrained sentence-embedding model (no training)."""
from functools import lru_cache

from src.config import EMBEDDING_MODEL
from src.extraction.skills import flat_skills


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


def match(resume_text: str, jd_text: str) -> dict:
    out = skill_gap(resume_text, jd_text)
    try:
        out["semantic_similarity"] = semantic_similarity(resume_text, jd_text)
    except Exception as e:  # sentence-transformers not installed
        out["semantic_similarity"] = None
        out["note"] = f"semantic similarity unavailable: {e}"
    return out
