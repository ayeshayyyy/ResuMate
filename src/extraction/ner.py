"""Transformer NER (Hugging Face token-classification) used for Name/Company/Designation/
Location/College/Degree.

The same wrapper serves two models so they can be compared fairly:
  * off-the-shelf pretrained NER (dslim/bert-base-NER, PER/ORG/LOC only)  -> "base"
  * DistilBERT fine-tuned on resume data (this project)                   -> "fine-tuned"
"""
import re
from typing import Dict, List, Optional

from src.config import ENTITY_TYPES


def _windows(text: str, max_chars: int = 1200):
    """Split on line boundaries into windows (<=512 tokens) and yield (offset, chunk)."""
    pos, buf, start = 0, [], 0
    for line in text.splitlines(keepends=True):
        if buf and sum(len(b) for b in buf) + len(line) > max_chars:
            yield start, "".join(buf)
            start, buf = pos, []
        buf.append(line)
        pos += len(line)
    if buf:
        yield start, "".join(buf)


_JUNK_EDGES = re.compile(r"^[\s,.:;|()\-\u2013]+|[\s,.:;|()\-\u2013]+$")


def _tidy(s: str) -> str:
    return _JUNK_EDGES.sub("", s).strip()


class TransformerNER:
    def __init__(self, model_id: str, label_map: Optional[Dict[str, str]] = None,
                 device: Optional[int] = None, min_score: float = 0.0):
        from transformers import pipeline
        import torch
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.model_id = model_id
        self.label_map = label_map
        self.min_score = min_score
        self.pipe = pipeline("token-classification", model=model_id, tokenizer=model_id,
                             aggregation_strategy="simple", device=device)

    def predict(self, text: str) -> Dict[str, List[dict]]:
        """Return {entity_type: [{text, score, start, end}]} (deduplicated)."""
        out = {t: [] for t in ENTITY_TYPES}
        seen = set()
        for offset, chunk in _windows(text):
            for ent in self.pipe(chunk):
                label = ent["entity_group"]
                label = self.label_map.get(label) if self.label_map else label
                if label not in out or ent["score"] < self.min_score:
                    continue
                s, e = ent["start"], ent["end"]
                value = _tidy(chunk[s:e])
                if len(value) < 2 or (label, value.lower()) in seen:
                    continue
                seen.add((label, value.lower()))
                out[label].append({"text": value, "score": float(ent["score"]),
                                   "start": offset + s, "end": offset + e})
        return out

    def predict_strings(self, text: str) -> Dict[str, List[str]]:
        return {k: [x["text"] for x in v] for k, v in self.predict(text).items()}
