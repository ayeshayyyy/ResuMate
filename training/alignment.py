"""Pure-python span -> BIO alignment (no torch needed, unit-tested)."""
from typing import List, Sequence, Tuple

from src.config import LABEL2ID


def align_labels(offsets: Sequence[Tuple[int, int]], entities: Sequence[Sequence], ignore_id: int = -100,
                 window_start: int = 0, window_end: int = None) -> List[int]:
    """Map character-level entity spans to per-token BIO label ids.

    offsets  : (start, end) char offsets of each token ((0, 0) = special token -> ignored)
    entities : [(start, end, label), ...] non-overlapping, in the same char coordinates
    Every sub-word of an entity gets a label (B- for the first token, I- afterwards), which matches
    how the Hugging Face `aggregation_strategy="simple"` pipeline merges tokens at inference time.
    """
    ents = sorted(entities, key=lambda e: e[0])
    labels, started = [], set()
    for s, e in offsets:
        if s == e:
            labels.append(ignore_id)
            continue
        hit = next((i for i, (es, ee, _) in enumerate(ents) if s < ee and e > es), None)
        if hit is None:
            labels.append(LABEL2ID["O"])
            continue
        prefix = "I" if hit in started else "B"
        started.add(hit)
        labels.append(LABEL2ID[f"{prefix}-{ents[hit][2]}"])
    return labels
