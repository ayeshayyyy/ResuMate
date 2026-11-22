import json
from tokenizers import BertWordPieceTokenizer

from src.config import ID2LABEL, SPLIT_DIR
from training.alignment import align_labels


def _tok():
    texts = [json.loads(l)["text"] for l in open(SPLIT_DIR / "train.jsonl", encoding="utf-8")][:300]
    tk = BertWordPieceTokenizer(lowercase=True)
    tk.train_from_iterator(texts, vocab_size=3000, min_frequency=2)
    return tk


def test_bio_alignment_roundtrip():
    text = "Priya Singh\nData Scientist at Amazon, Seattle"
    ents = [[0, 11, "Name"], [12, 26, "Designation"], [30, 36, "Company"], [38, 45, "Location"]]
    enc = _tok().encode(text)
    offsets = [(0, 0)] + list(enc.offsets) + [(0, 0)]          # emulate [CLS] / [SEP]
    labels = align_labels(offsets, ents)
    # special tokens ignored, every entity starts with exactly one B- tag
    assert labels[0] == -100 and labels[-1] == -100
    tags = [ID2LABEL[l] for l in labels if l != -100]
    assert sum(t.startswith("B-") for t in tags) == 4
    # decode entities back from tags -> must reproduce gold char spans (token granularity)
    spans, cur = [], None
    for (s, e), l in zip(offsets, labels):
        if l == -100:
            continue
        tag = ID2LABEL[l]
        if tag.startswith("B-"):
            cur = [s, e, tag[2:]]; spans.append(cur)
        elif tag.startswith("I-") and cur:
            cur[1] = e
        else:
            cur = None
    assert [text[s:e] for s, e, _ in spans] == ["Priya Singh", "Data Scientist", "Amazon", "Seattle"]


def test_real_training_rows_align():
    tk = _tok()
    rows = [json.loads(l) for l in open(SPLIT_DIR / "train.jsonl", encoding="utf-8")][:50]
    n_ent = 0
    for r in rows:
        enc = tk.encode(r["text"])
        labels = align_labels(enc.offsets, r["entities"])
        assert len(labels) == len(enc.offsets)
        n_ent += sum(ID2LABEL[l].startswith("B-") for l in labels if l != -100)
    gold = sum(len(r["entities"]) for r in rows)
    assert n_ent >= 0.9 * gold   # a few entities may be lost to window/tokenisation edge cases
