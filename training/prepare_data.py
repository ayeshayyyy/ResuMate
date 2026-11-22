"""Convert the DataTurks Resume-NER dump into clean, seeded train/val/test splits.

The TEST split is never used for training or model selection.

Usage:  python -m training.prepare_data
"""
import json
import random
from pathlib import Path

from src.config import DATASET_LABEL_MAP, RAW_NER_FILE, SEED, SPLIT_DIR


def _locate(text: str, point: dict):
    """Return (start, end) char span for an annotation point.

    Offsets in this dataset are a mix of end-exclusive and end-inclusive, so we
    verify against the annotated string and fall back to a text search.
    """
    s, e, t = point["start"], point["end"], point["text"]
    if text[s:e] == t:
        return s, e
    if text[s:e + 1] == t:
        return s, e + 1
    idx = text.find(t)
    return (idx, idx + len(t)) if idx >= 0 else None


def load_rows(path=RAW_NER_FILE):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            text = raw["content"]
            spans = []
            for ann in raw.get("annotation") or []:
                labels = ann.get("label") or []
                if not labels or labels[0] not in DATASET_LABEL_MAP:
                    continue
                label = DATASET_LABEL_MAP[labels[0]]
                for p in ann["points"]:
                    loc = _locate(text, p)
                    if loc and text[loc[0]:loc[1]].strip():
                        spans.append([loc[0], loc[1], label])
            spans = _resolve_overlaps(spans)
            if spans:
                rows.append({"text": text, "entities": spans})
    return rows


def _resolve_overlaps(spans):
    """Keep the longest span when annotations overlap."""
    taken, kept = set(), []
    for s, e, l in sorted(spans, key=lambda x: x[0] - x[1]):  # longest first
        if any(i in taken for i in range(s, e)):
            continue
        taken.update(range(s, e))
        kept.append([s, e, l])
    return sorted(kept)


def split_rows(rows, seed=SEED, val=0.1, test=0.1):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test, n_val = int(n * test), int(n * val)
    return {
        "test": rows[:n_test],
        "val": rows[n_test:n_test + n_val],
        "train": rows[n_test + n_val:],
    }


def main():
    rows = load_rows()
    splits = split_rows(rows)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(SPLIT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5s}: {len(data)} resumes")


if __name__ == "__main__":
    main()
