"""Fine-tune a pretrained DistilBERT for resume NER (token classification).

NOT training from scratch: we start from `distilbert-base-uncased` (pretrained on English text) and only
adapt it - either full fine-tuning (default) or parameter-efficient LoRA (`--lora`).

Runs on a free Colab / Kaggle GPU in a few minutes (1,774 resumes, ~5 epochs).

    python -m training.prepare_data
    python -m training.finetune --epochs 5
    python -m evaluation.run_eval --methods heuristics,base,finetuned
"""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from src.config import BASE_CHECKPOINT, FINETUNED_DIR, ID2LABEL, LABEL2ID, LABELS, SEED, SPLIT_DIR
from training.alignment import align_labels


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_dataset(rows, tokenizer, max_len, stride):
    """Tokenise with sliding windows so long resumes are not truncated."""
    from datasets import Dataset
    feats = {"input_ids": [], "attention_mask": [], "labels": []}
    for r in rows:
        enc = tokenizer(r["text"], truncation=True, max_length=max_len, stride=stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
        for i in range(len(enc["input_ids"])):
            feats["input_ids"].append(enc["input_ids"][i])
            feats["attention_mask"].append(enc["attention_mask"][i])
            feats["labels"].append(align_labels(enc["offset_mapping"][i], r["entities"]))
    return Dataset.from_dict(feats)


def token_f1(eval_pred):
    """Token-level micro P/R/F1 over non-O labels (model selection only; final eval is entity-level)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    p, l = preds[mask], labels[mask]
    o = LABEL2ID["O"]
    tp = int(((p == l) & (l != o)).sum())
    fp = int(((p != l) & (p != o)).sum())
    fn = int(((p != l) & (l != o)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "accuracy": float((p == l).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_CHECKPOINT)
    ap.add_argument("--out", default=str(FINETUNED_DIR))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--lora", action="store_true", help="use LoRA (PEFT) instead of full fine-tuning")
    ap.add_argument("--max_train", type=int, default=None, help="debug: limit training resumes")
    args = ap.parse_args()

    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.TOKEN_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["q_lin", "v_lin"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    train_rows = read_jsonl(SPLIT_DIR / "train.jsonl")[: args.max_train]
    val_rows = read_jsonl(SPLIT_DIR / "val.jsonl")[: args.max_train]
    train_ds = build_dataset(train_rows, tok, args.max_len, args.stride)
    val_ds = build_dataset(val_rows, tok, args.max_len, args.stride)
    print(f"train windows: {len(train_ds)} | val windows: {len(val_ds)}   (test split is NOT touched)")

    ta_kwargs = dict(output_dir=str(Path(args.out) / "checkpoints"), num_train_epochs=args.epochs,
                     learning_rate=args.lr, per_device_train_batch_size=args.batch,
                     per_device_eval_batch_size=32, weight_decay=0.01, warmup_ratio=0.1,
                     save_strategy="epoch", logging_steps=25, load_best_model_at_end=True,
                     metric_for_best_model="f1", greater_is_better=True, save_total_limit=1,
                     report_to="none", seed=SEED)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    ta_kwargs[key] = "epoch"
    trainer_kwargs = dict(model=model, args=TrainingArguments(**ta_kwargs), train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=DataCollatorForTokenClassification(tok),
                          compute_metrics=token_f1)
    tk = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[tk] = tok
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    final = trainer.model
    if args.lora:
        final = final.merge_and_unload()          # plain HF model -> same inference code path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    final.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    log = trainer.state.log_history
    json.dump(log, open(Path(args.out) / "training_log.json", "w"), indent=1)
    _plot(log, Path(args.out).parent.parent / "results" / "training_curves.png")
    print(f"Saved fine-tuned model to {args.out}")


def _plot(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(x["epoch"], x["loss"]) for x in log if "loss" in x]
    ev = [(x["epoch"], x["eval_loss"], x.get("eval_f1")) for x in log if "eval_loss" in x]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(*zip(*tr), label="train loss")
    ax[0].plot([e[0] for e in ev], [e[1] for e in ev], marker="o", label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot([e[0] for e in ev], [e[2] for e in ev], marker="o", color="green")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Validation token-F1")
    path.parent.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
