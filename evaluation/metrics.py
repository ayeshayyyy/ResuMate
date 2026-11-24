"""Entity-level evaluation: per-document sets of (label, normalised string).

exact   : normalised strings must be identical
partial : strings match if one contains the other (boundary errors forgiven)
"""
import re
from collections import defaultdict
from typing import Dict, Iterable, List


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w+#.\s]", " ", s)
    s = re.sub(r"(?<!\w)\.|\.(?!\w)", " ", s)     
    return re.sub(r"\s+", " ", s).strip()


def _match(a: str, b: str, mode: str) -> bool:
    if mode == "exact":
        return a == b
    return a == b or (len(a) > 2 and len(b) > 2 and (a in b or b in a))


def match_doc(gold: List[str], pred: List[str], mode: str):
    """Greedy one-to-one matching -> (tp, fp_list, fn_list)."""
    g = list(dict.fromkeys(norm(x) for x in gold if norm(x)))
    p = list(dict.fromkeys(norm(x) for x in pred if norm(x)))
    unmatched_g, tp, used = [], 0, set()
    for gi in g:
        hit = next((j for j, pj in enumerate(p) if j not in used and _match(gi, pj, mode)), None)
        if hit is None:
            unmatched_g.append(gi)
        else:
            used.add(hit)
            tp += 1
    fp = [pj for j, pj in enumerate(p) if j not in used]
    return tp, fp, unmatched_g


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(gold_docs: Iterable[Dict[str, List[str]]], pred_docs: Iterable[Dict[str, List[str]]],
             labels: List[str], mode: str = "exact"):
    """Return (table, errors).
    table  : {label: {precision, recall, f1, support, tp, fp, fn}, 'micro': ..., 'macro': ...}
    errors : {label: {'fp': [(doc_idx, str)], 'fn': [(doc_idx, str)]}}
    """
    counts = {l: [0, 0, 0] for l in labels}
    errors = {l: {"fp": [], "fn": []} for l in labels}
    for i, (g, p) in enumerate(zip(gold_docs, pred_docs)):
        for l in labels:
            tp, fp, fn = match_doc(g.get(l, []), p.get(l, []), mode)
            counts[l][0] += tp
            counts[l][1] += len(fp)
            counts[l][2] += len(fn)
            errors[l]["fp"] += [(i, x) for x in fp]
            errors[l]["fn"] += [(i, x) for x in fn]
    table = {}
    for l, (tp, fp, fn) in counts.items():
        p, r, f = prf(tp, fp, fn)
        table[l] = dict(precision=p, recall=r, f1=f, support=tp + fn, tp=tp, fp=fp, fn=fn)
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in range(3))
    p, r, f = prf(tp, fp, fn)
    table["micro"] = dict(precision=p, recall=r, f1=f, support=tp + fn, tp=tp, fp=fp, fn=fn)
    active = [l for l in labels if table[l]["support"] > 0]
    table["macro"] = dict(
        precision=sum(table[l]["precision"] for l in active) / max(len(active), 1),
        recall=sum(table[l]["recall"] for l in active) / max(len(active), 1),
        f1=sum(table[l]["f1"] for l in active) / max(len(active), 1),
        support=sum(table[l]["support"] for l in active), tp=0, fp=0, fn=0)
    return table, errors


def per_doc_counts(gold_docs, pred_docs, labels, mode="exact"):
    """List of (tp, fp, fn) per document, summed over `labels` (for bootstrap CIs)."""
    out = []
    for g, p in zip(gold_docs, pred_docs):
        tp = fp = fn = 0
        for l in labels:
            a, b, c = match_doc(g.get(l, []), p.get(l, []), mode)
            tp, fp, fn = tp + a, fp + len(b), fn + len(c)
        out.append((tp, fp, fn))
    return out


def _micro_f1(counts):
    tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
    return prf(tp, fp, fn)[2]


def paired_bootstrap(counts_a, counts_b, n_boot=2000, seed=42):
    """Bootstrap over documents: returns (f1_a, f1_b, diff_mean, ci_low, ci_high, p_b_not_better).
    diff = F1(b) - F1(a)."""
    import random
    rng = random.Random(seed)
    n = len(counts_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_micro_f1([counts_b[i] for i in idx]) - _micro_f1([counts_a[i] for i in idx]))
    diffs.sort()
    return (_micro_f1(counts_a), _micro_f1(counts_b), sum(diffs) / n_boot,
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)],
            sum(d <= 0 for d in diffs) / n_boot)
