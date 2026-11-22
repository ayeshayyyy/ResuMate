"""Central configuration: paths, label schema, model ids."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "ner" / "splits"
RAW_NER_FILE = DATA_DIR / "ner" / "dataturks_resume_ner.jsonl"
SYNTHETIC_TEST_FILE = DATA_DIR / "synthetic_test" / "test_resumes.json"
RESULTS_DIR = ROOT / "results"
FINETUNED_DIR = ROOT / "models" / "resumate-distilbert-ner"
# Hugging Face Hub repo id of the fine-tuned model, e.g. "your-username/resumate-distilbert-ner".

HF_MODEL_ID = "ayeshayyyyyyy/resumate-distilbert-ner"


def resolve_model_path() -> str:
    """Priority: RESUMATE_MODEL env var -> local models/ folder -> Hugging Face Hub id."""
    import os
    env = os.getenv("RESUMATE_MODEL")
    if env:
        return env
    if FINETUNED_DIR.exists() and any(FINETUNED_DIR.glob("config.json")):
        return str(FINETUNED_DIR)
    return HF_MODEL_ID or str(FINETUNED_DIR)

# Entity types predicted by the transformer (contact fields/dates/skills come from rules).
ENTITY_TYPES = ["Name", "Company", "Designation", "Location", "College", "Degree"]

# Raw dataset label -> our label
DATASET_LABEL_MAP = {
    "Name": "Name",
    "Companies worked at": "Company",
    "Designation": "Designation",
    "Location": "Location",
    "College Name": "College",
    "Degree": "Degree",
}

# BIO label space (shared by training and inference)
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Pretrained checkpoints
BASE_CHECKPOINT = "distilbert-base-uncased"          # starting point for fine-tuning
OFF_THE_SHELF_NER = "dslim/bert-base-NER"            # generic pretrained NER (baseline)
OFF_THE_SHELF_MAP = {"PER": "Name", "ORG": "Company", "LOC": "Location"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
