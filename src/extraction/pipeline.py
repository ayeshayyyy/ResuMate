"""End-to-end resume understanding: file (PDF/scan/image) -> structured JSON."""
import os
from pathlib import Path
from typing import Optional

from src.config import ENTITY_TYPES, FINETUNED_DIR, resolve_model_path
from src.extraction import heuristics, rules
from src.extraction.skills import extract_skills
from src.ingestion.loader import load_document


class ResumeAssistant:
    def __init__(self, model_path: Optional[str] = None, ocr_engine: str = "auto"):
        self.ocr_engine = ocr_engine
        self.ner = None
        self.ner_error = None
        model_path = model_path or resolve_model_path()
        if Path(model_path).exists() or "/" in model_path and not model_path.startswith(("/", ".")):
            try:
                from src.extraction.ner import TransformerNER
                self.ner = TransformerNER(model_path)
                self.model_path = model_path
            except Exception as e: 
                self.ner_error = str(e)
        else:
            self.ner_error = f"Fine-tuned model not found at {model_path} (run training/finetune.py, or set HF_MODEL_ID in src/config.py / RESUMATE_MODEL)"

    @property
    def mode(self):
        return "fine-tuned DistilBERT + rules" if self.ner else "rules/heuristics only (no NER model loaded)"

    def process_file(self, data: bytes, filename: str = "", force_ocr: bool = False) -> dict:
        doc = load_document(data, filename, ocr_engine=self.ocr_engine, force_ocr=force_ocr)
        return self.process_text(doc.text, doc)

    def process_text(self, text: str, doc=None) -> dict:
        if self.ner:
            ents = self.ner.predict(text)
            entities = {k: [e["text"] for e in v] for k, v in ents.items()}
            confidence = {k: [round(e["score"], 3) for e in v] for k, v in ents.items()}
            method = "transformer"
        else:
            entities = heuristics.predict_strings(text)
            confidence, method = {}, "heuristic"
        if not entities["Name"]: 
            entities["Name"] = heuristics.predict_strings(text)["Name"]

        ranges = rules.extract_date_ranges(text)
        return {
            "document": {
                "pages": doc.pages if doc else [], "used_ocr": bool(doc and doc.used_ocr),
                "ocr_engine": doc.ocr_engine if doc else None, "n_chars": len(text),
            },
            "contact": {"name": (entities["Name"] or [None])[0],
                        "emails": rules.extract_emails(text), "phones": rules.extract_phones(text),
                        "links": rules.extract_links(text),
                        "location": (entities["Location"] or [None])[0]},
            "entities": entities,
            "entity_confidence": confidence,
            "entity_method": method,
            "skills": extract_skills(text),
            "date_ranges": [{"text": r["text"], "months": r["months"], "current": r["current"]} for r in ranges],
            "experience_years": rules.total_experience_years(text),
            "sections": list(rules.split_sections(text).keys()),
            "text": text,
        }
