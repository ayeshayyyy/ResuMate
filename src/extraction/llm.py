"""Optional LLM extractor (few-shot, JSON output) for comparison.

Works with any OpenAI-compatible endpoint: OpenAI, or a local Ollama server
(OPENAI_BASE_URL=http://localhost:11434/v1, OPENAI_API_KEY=ollama, LLM_MODEL=llama3.2).
"""
import json
import os
import re
from typing import Dict, List

from src.config import ENTITY_TYPES

PROMPT = """Extract entities from the resume below. Return ONLY a JSON object with keys {keys}. Each value is a list of
exact substrings copied from the resume (Name: the candidate's own name; Company: employers; Designation: job titles;
Location: cities/regions; College: schools/universities; Degree: degree names). Use [] when absent.

RESUME:
\"\"\"
{text}
\"\"\"
"""


class LLMExtractor:
    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY", "none"))
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def predict_strings(self, text: str) -> Dict[str, List[str]]:
        msg = PROMPT.format(keys=", ".join(ENTITY_TYPES), text=text[:6000])
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, messages=[{"role": "user", "content": msg}])
        raw = r.choices[0].message.content
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        return {k: [str(x) for x in data.get(k, []) if x] for k in ENTITY_TYPES}
