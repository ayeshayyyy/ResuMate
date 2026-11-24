"""Gazetteer-based skill extraction (no training involved)."""
import re

SKILLS = {
    "Programming": ["python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", "golang", "php",
                    "rust", "scala", "swift", "kotlin", "r", "matlab", "html", "css", "sql", "bash", "shell",
                    "powershell", "objective-c", "dart"],
    "Frameworks": ["django", "flask", "fastapi", "spring boot", "spring", "express", "express.js", "react",
                   "react.js", "react native", "angular", "vue", "vue.js", "next.js", "node.js", "rails",
                   "flutter", "redux", "svelte", ".net", "laravel", "bootstrap", "tailwind"],
    "ML/NLP": ["tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
               "xgboost", "opencv", "hugging face", "huggingface", "transformers", "spacy", "nltk", "langchain",
               "llamaindex", "nlp", "natural language processing", "computer vision", "machine learning",
               "deep learning", "generative ai", "llm", "rag", "fine-tuning", "bert", "prompt engineering"],
    "Databases": ["mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "oracle", "mariadb",
                  "elasticsearch", "cassandra", "dynamodb", "firebase", "nosql"],
    "Cloud/DevOps": ["aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins", "terraform",
                     "ansible", "ci/cd", "linux", "git", "github", "gitlab", "heroku"],
    "Tools": ["jira", "tableau", "power bi", "excel", "figma", "kafka", "spark", "hadoop", "airflow", "rabbitmq",
              "selenium", "junit", "postman"],
    "Practices": ["agile", "scrum", "kanban", "rest", "restful apis", "graphql", "microservices", "oop", "tdd"],
}

_CANON = {s: cat for cat, items in SKILLS.items() for s in items}
# Longest-first so "spring boot" wins over "spring"
_ALTERNATION = "|".join(re.escape(s) for s in sorted(_CANON, key=len, reverse=True))
# Custom boundaries: \b breaks on c++ / c# / .net, so use look-arounds instead.
_SKILL_RE = re.compile(rf"(?<![A-Za-z0-9+#])(?:{_ALTERNATION})(?![A-Za-z0-9+#])", re.I)

# Short ambiguous tokens only count when they look like a skill listing
_AMBIGUOUS = {"r", "go", "rag", "spring", "express", "rest", "git"}


def extract_skills(text: str) -> dict:
    """Return {category: [skills]} found in text (deduplicated, original order)."""
    found = {}
    for m in _SKILL_RE.finditer(text):
        raw = m.group(0)
        key = raw.lower()
        if key in _AMBIGUOUS:
            ctx = text[max(0, m.start() - 20): m.end() + 20]
            if not (raw[0].isupper() and raw.isupper() or "," in ctx or "|" in ctx or ":" in ctx):
                continue
        cat = _CANON[key]
        bucket = found.setdefault(cat, [])
        if key not in (s.lower() for s in bucket):
            bucket.append(raw if raw.isupper() and len(raw) > 1 else key)
    return found


def flat_skills(text: str) -> list:
    return [s for items in extract_skills(text).values() for s in items]
