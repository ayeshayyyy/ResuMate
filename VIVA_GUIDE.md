# ResuMate Viva Guide

A concise project overview, evaluation summary, and viva revision sheet. This guide follows the current workspace README.

## Project At A Glance

**ResuMate** is a resume document assistant for digital PDFs, scanned PDFs, images, and text files. It extracts structured resume information and can compare one or more candidates with a job description. It is a human-review aid, not an automated hiring decision system.

The project is a hybrid NLP application. It uses pretrained OCR and language models, fine-tunes DistilBERT on resume data, and uses rules/gazetteers where those are more practical. It does not train a neural model from scratch. 

**Fields:** Name, Company, Designation, Location, College, Degree, email, phone, links, date ranges, experience duration, skills, and detected sections. Monetary amounts are not applicable to resumes.

## Architecture And Flow

```text
PDF / scan / image / text
  -> PDF text extraction or rasterization + OCR
  -> layout ordering, table extraction, and text cleanup
  -> DistilBERT NER + regex rules + skill gazetteer
  -> structured resume data and source-page evidence
  -> timeline checks, job matching/ranking, optional PII redaction
  -> Streamlit interface and JSON / flattened redacted PDF
```

1. `pdfplumber` reads a PDF text layer. Pages with fewer than 80 extracted characters, or pages where OCR is forced, are rasterized by `pypdfium2` and passed to OCR.
2. Tesseract or optional PaddleOCR returns text, confidence, and word positions. Images go directly to OCR; TXT/Markdown files are decoded and cleaned.
3. Text cleaning normalizes Unicode, bullets, whitespace, and common OCR-split email addresses.
4. Fine-tuned DistilBERT extracts six entity types. Rules extract contact data, links, date ranges, sections, and experience duration. A skill gazetteer detects skills.
5. The result is displayed in Streamlit. Optional workflows highlight source evidence, inspect timeline consistency, compare candidates, and redact PII.

## Five Added Features

- **Visual provenance:** extracted fields are matched to positioned PDF/OCR words and highlighted on rendered pages. This is best-effort; unmatched fields may have no highlight.
- **Layout and tables:** a gutter-based heuristic reads simple two-column PDFs left column then right. `pdfplumber` extracts machine-readable PDF tables. Scanned tables and complex layouts are not reliably reconstructed.
- **Timeline integrity:** reports overlaps, gaps between dated entries, future/implausible dates, invalid numeric months, and differences between stated experience claims and dated history. A gap is not proof of unemployment; overlapping jobs can be legitimate.
- **Evidence-based job matching:** each job-description sentence gets `supported`, `partial`, or `not_found`, matched terms, and the strongest supporting resume sentence. Multiple candidates are ranked primarily by average requirement-term coverage; semantic similarity is a secondary tie-breaker.
- **Blind screening:** redacts detected names, locations, emails, phones, profile links, portfolio URLs, labeled birth dates, and street addresses. PDF/image output is rasterized with opaque masks and has no selectable text layer. The UI checks whether the requirement-evidence score is unchanged after redaction. This is not a guarantee of anonymisation; review the output.

## Models And Data

### Models

- **Resume NER:** `distilbert-base-uncased`, fine-tuned for Name, Company, Designation, Location, College, and Degree using BIO token labels.
- **Generic NER baseline:** `dslim/bert-base-NER`; PER/ORG/LOC map to Name/Company/Location only.
- **Job similarity:** pretrained `sentence-transformers/all-MiniLM-L6-v2`, using cosine similarity over up to the first 3,000 characters of each text.
- **OCR:** pretrained Tesseract, or optional pretrained PaddleOCR.

Model lookup order is `RESUMATE_MODEL`, local `models/resumate-distilbert-ner/`, then the Hugging Face model ID configured in `src/config.py`. The optional LLM extractor is implemented but is not part of reported results.

Configured Hub ID: `ayeshayyyyyyy/resumate-distilbert-ner`.

### Dataset

- **DataTurks Resume-NER:** about 2,216 annotated resumes. Seeded 80/10/10 split: 1,774 train, 221 validation, 221 held-out test.
- **Synthetic test set:** 30 hand-written, hand-labeled resumes used only to test domain shift, not for training.
- DataTurks labels map to six project entity types. The preparation code handles inconsistent annotation offsets and resolves overlapping spans by keeping the longest.

The test split is not used for training, early stopping, or model selection. Validation token-F1 selects the checkpoint.

### Fine-tuning

Starting from pretrained DistilBERT is transfer learning, not training from scratch. Character spans are aligned to tokenizer offsets and converted to BIO labels; special tokens use ignore label `-100`. Long resumes use overlapping windows.

Default settings: 5 epochs, learning rate `5e-5`, batch size 16, maximum length 256 tokens, stride 64, weight decay 0.01, warmup ratio 0.1, seed 42. Optional LoRA uses rank 16, alpha 32, dropout 0.1, targeting query/value layers. Final evaluation is entity-level, not token-level.

## Evaluation And Results

### Evaluation design

Three reported methods are compared: rule/gazetteer heuristics, generic pretrained NER, and resume-fine-tuned DistilBERT. The optional LLM method was not evaluated.

Entity strings are normalized and matched one-to-one. **Exact** requires normalized equality; **partial** allows one string to contain the other when both are longer than two characters. Precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 is their harmonic mean. Micro-F1 pools counts across types; macro-F1 averages type scores.

A paired bootstrap resamples documents 2,000 times to estimate a 95% confidence interval for base-versus-fine-tuned F1 differences. The shared-label comparison (Name, Company, Location) is fairest because the generic model cannot predict the other three types.

### Held-out DataTurks test (221 resumes)

| Method | Exact micro-F1 | Partial micro-F1 |
|---|---:|---:|
| Heuristics | 31.9% | 54.8% |
| Generic NER | 25.0% | 31.9% |
| Fine-tuned DistilBERT | **93.9%** | **95.8%** |

Fine-tuned exact F1 by type: Name 99.8%, Company 91.5%, Designation 90.6%, Location 96.8%, College 91.5%, Degree 93.4%. For shared labels, exact micro-F1 rises from 30.7% to 96.1%; the bootstrap interval for the gain is +62.6 to +68.0 points.

Across all six labels, the gain is 25.0% to 93.9% (95% interval +67.1 to +70.7 points).

### Synthetic domain-shift test (30 resumes)

| Method | Exact micro-F1 | Partial micro-F1 |
|---|---:|---:|
| Heuristics | 29.3% | 59.9% |
| Generic NER | 35.9% | 44.4% |
| Fine-tuned DistilBERT | **50.6%** | **69.9%** |

Performance falls versus DataTurks. The fine-tuned model beats generic NER across all six labels, but not significantly on the three shared labels given the small sample. Company exact F1 is especially weak at 13.0%.

Bootstrap differences: all six labels +14.5 points (95% interval +1.7 to +28.3); shared labels -15.6 points (95% interval -34.9 to +4.2), not statistically significant.

### OCR robustness

On 30 synthetic resumes rendered as image pages with mild rotation, blur, and noise, Tesseract mean character error rate is 0.71%.

| Extractor | Clean text partial F1 | Scanned + OCR partial F1 |
|---|---:|---:|
| Heuristics | 59.9% | 53.4% |
| Fine-tuned DistilBERT + rules | 69.9% | 68.5% |

These are synthetic pages, not real phone photos or complex multi-column scans.

### Main error patterns

- Fragmented or boundary-mismatched names, companies, and schools; this explains why partial F1 is higher than exact F1.
- Prose-style synthetic resumes differ from the mostly list-style training data.
- Company recall drops on unseen employers; the generic NER can instead overpredict tools as companies.
- The uncased model may struggle with short capitalized abbreviations, though this has not been isolated in an ablation.
- DataTurks contains questionable labels, so exact scores are imperfect measures of true quality.

Examples include fragmented company names (`Infosys` -> `info`), truncated schools (`University of`), prose-style false positives, missed unseen employers such as Oracle/IBM, and questionable gold labels such as a job title marked as a degree.

The synthetic drop is consistent with domain shift, but the small test set and absence of an isolating ablation limit that conclusion. Report only reproducible results from the current evaluation; earlier unreproducible claims were removed from the README.

## Run And Inspect

```powershell
pip install -r requirements.txt
python -m pytest -q
python -m training.prepare_data
python -m training.finetune --epochs 5
python -m evaluation.run_eval --methods heuristics,base,finetuned
python -m evaluation.ocr_robustness --engine tesseract
streamlit run app.py
```

Tesseract also requires its system binary. Training is intended for a GPU such as Colab/Kaggle; the notebook is `training/colab_finetune.ipynb`. On Windows, set `$env:PYTHONUTF8=1` for evaluation if needed. Evaluation outputs are written under `results/`. Running evaluation with only a subset of methods overwrites result files; regenerate with all three methods for the report.

Project map: `app.py` (Streamlit); `src/ingestion/` (loader, OCR, provenance); `src/extraction/` (NER, rules, skills, heuristics, optional LLM, pipeline); `src/matching/matcher.py` (evidence, ranking, MiniLM); `src/privacy.py` (PII redaction); `training/` (data prep and fine-tuning); `evaluation/` (metrics and studies); `data/` (training and tests); `tests/`; `results/`.

## Limitations And Responsible Use

- Tables are most reliable in text-based PDFs. Scanned tables, nested cells, and complex columns can still be scrambled.
- Training data is mostly Indian IT resumes in list format; the synthetic test set is small and differently written.
- OCR testing uses mild synthetic degradation. PaddleOCR is supported, but reported OCR results use Tesseract.
- Annotation noise, fixed skill vocabulary, date-format assumptions, and best-effort visual matching constrain output quality.
- PII redaction can miss identifiers. Inspect the flattened PDF before sharing.
- Resume data is sensitive and screening may encode bias. Use human oversight; do not make employment decisions solely from model scores.

Future work: evaluate real scans, add diverse resume styles, use a layout-aware model for complex tables/columns, test a cased or larger NER model, improve skill understanding, audit fairness, and evaluate the optional LLM baseline.

## Viva Quick Answers

**What is the project?** A hybrid resume parsing assistant that accepts digital and image documents, extracts structured fields, and provides evidence-based job matching for human review.

**Why combine models and rules?** Transformers handle contextual entities; regex is predictable for contacts and dates; a gazetteer is simple for a bounded skill list. The combination is practical and reduces unnecessary model calls.

**What is BIO tagging?** `B` starts an entity, `I` continues it, and `O` marks tokens outside entities. It trains token classification from character-level spans.

**Why fine-tune DistilBERT?** Generic NER supports only common person/organization/location types. Resume fine-tuning learns six resume-specific labels from annotated examples.

**Why report exact and partial F1?** Exact score penalizes boundary errors; partial score shows whether the right entity was partly recovered. Both precision and recall matter because entity labels are sparse.

**What is the main result?** Fine-tuned DistilBERT reaches 93.9% exact micro-F1 on held-out DataTurks data, then drops to 50.6% on the 30-item synthetic domain-shift set.

**How does the timeline checker work?** It parses dated ranges, merges intervals for total experience, and flags overlaps, gaps, implausible dates, and drift from explicit experience claims. Flags are prompts for review, not proof of dishonesty.

**What does evidence-based matching mean?** Each JD sentence is checked for terms in the resume and paired with the strongest matching resume sentence. Ranking is a transparent lexical coverage score, not proof of qualification.

**What does score invariance test?** It compares the requirement-evidence score before and after text PII redaction. It does not guarantee complete anonymisation or test every possible fairness effect.

**What are the key limitations?** Domain shift, noisy annotations, OCR/layout errors, incomplete skill vocabulary, and privacy/fairness risks.

## Short Presentation Outline

1. Problem: resumes are varied, semi-structured documents and manual review takes time.
2. Method: PDF extraction/OCR, layout handling, DistilBERT NER, deterministic rules, and a skill gazetteer.
3. Added workflows: visual evidence, timeline checks, requirement-level matching/ranking, and blind-screen redaction.
4. Evaluation: 221 held-out DataTurks resumes, 30 synthetic domain-shift resumes, exact/partial metrics, bootstrap comparison, and synthetic OCR study.
5. Results and caveats: 93.9% exact micro-F1 on DataTurks; 50.6% on synthetic data; human review remains essential.
