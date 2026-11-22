# ResuMate - Multimodal Resume Document Assistant (NLP course project)

Repository: https://github.com/ayeshayyyy/ResuMate

**Track:** *Multimodal Document Assistant* - one specific document type: **resumes / CVs**
(digital PDFs, scanned PDFs, phone photos).

The system reads a resume (PDF text layer, or **OCR** when the page is an image), extracts structured fields
(**names, organisations, job titles, locations, schools, degrees, e-mails, phones, links, date ranges,
years of experience, skills**), and optionally matches the resume to a job description.

## Requirement checklist

| Requirement | How it is met |
|---|---|
| Real-world NLP application | Resume parsing / screening from messy PDFs and scans |
| One category, one document type | Multimodal Document Assistant -> resumes only |
| Pretrained models, **no training from scratch** | OCR: Tesseract (PaddleOCR supported as an optional engine). NER: `distilbert-base-uncased` **fine-tuned** (optionally with LoRA). Matching: pretrained `all-MiniLM-L6-v2`. An earlier from-scratch LSTM, CRF and TF-IDF classifier were **removed** |
| Existing, small, clean dataset | DataTurks Resume-NER (2,216 resumes, in `data/ner/`) + 30 hand-labelled synthetic resumes. The DataTurks annotations are noisy (see Limitations) |
| Show methodology and results, not just UI | `evaluation/` - held-out test sets, 3 methods compared (an optional LLM baseline is implemented but was not run), exact/partial metrics, bootstrap significance, error analysis, OCR robustness study |
| Base vs fine-tuned on held-out test set | `evaluation/run_eval.py` (`base` = off-the-shelf NER, `finetuned` = ours); the test split is never used for training or model selection |
| Simple Streamlit interface | `app.py` |
| Extract names, dates, amounts, tables, important fields | Names / organisations / titles / locations / schools / degrees via NER; contact details, date ranges and years of experience via rules; skills via gazetteer. Resumes contain no monetary amounts. **Table extraction is not implemented** |

## Pipeline

```
PDF / scan / photo
   |  pdfplumber text layer  --(page has < 80 chars?)-->  rasterise (pypdfium2) -> OCR (Tesseract | PaddleOCR)
   v
clean text (unicode, bullets, OCR e-mail repair)
   |-- fine-tuned DistilBERT NER  -> Name, Company, Designation, Location, College, Degree  (+ confidence)
   |-- regex rules                -> e-mail, phone, links, date ranges, years of experience, sections
   |-- skill gazetteer            -> skills by category
   v
structured JSON  ->  Streamlit UI  (+ job-description match with MiniLM embeddings)
```

## Quick start

```bash
pip install -r requirements.txt
# Tesseract binary is also required for OCR:
#   Windows: winget install --id UB-Mannheim.TesseractOCR -e   (then add it to PATH)
#   Linux:   sudo apt install tesseract-ocr
# Optional OCR engine: pip install paddlepaddle paddleocr

python -m pytest -q                      # 11 unit tests (rules, metrics, label alignment, OCR path)
python -m training.prepare_data          # seeded 80/10/10 split -> data/ner/splits (test = 221 resumes)

# GPU (free Colab/Kaggle T4) - see training/colab_finetune.ipynb
python -m training.finetune --epochs 5   # add --lora for the PEFT variant
python -m evaluation.run_eval --methods heuristics,base,finetuned     # (+ ,llm if you have an API/Ollama)
python -m evaluation.ocr_robustness --engine tesseract

streamlit run app.py
```

On Windows, set `$env:PYTHONUTF8=1` before running the evaluation scripts.

Outputs land in `results/`: `comparison.md`, `error_analysis.md`, `metrics.json`, `f1_by_label.png`,
`training_curves.png`, `ocr_robustness.json`.

> Note: running `run_eval` with a subset of methods (e.g. `--methods heuristics`) overwrites these result files.
> Always regenerate them with all three methods: `--methods heuristics,base,finetuned`.

## Fine-tuned model

The fine-tuned DistilBERT weights are hosted on the **Hugging Face Hub** (they are too large for git).
Model: `HF_MODEL_ID` (set in `src/config.py`; replace with `ayeshayyyyyyy/resumate-distilbert-ner`).
The app and evaluation scripts load the model in this order: `RESUMATE_MODEL` environment variable ->
local folder `models/resumate-distilbert-ner/` -> the Hub id above, so no manual download is needed.
Training is reproducible with `training/finetune.py` (or `training/colab_finetune.ipynb`).

## Evaluation design

* **Test set 1 - DataTurks held-out (221 real resumes).** Seeded split; train 1,774 / val 221 / test 221. The test
  split is not used for training, early stopping or model selection.
* **Test set 2 - synthetic (30 resumes, different style).** Hand-written ground truth; measures domain shift.
  Because the same team wrote both the resumes and the labels, some exact-match failures reflect format
  disagreements rather than model errors.
* **Methods compared**
  1. `heuristics` - regex/gazetteer baseline (no ML).
  2. `base` - *off-the-shelf pretrained NER* (`dslim/bert-base-NER`, only PER/ORG/LOC -> Name/Company/Location).
     This is the "base model" for the fine-tuning question: a generic NER with no resume adaptation.
  3. `finetuned` - `distilbert-base-uncased` fine-tuned on the training split.
  4. `llm` (optional, implemented in `src/extraction/llm.py`, **not evaluated in this report**).
* **Metrics.** Entity-level precision/recall/F1 on per-document sets of strings, *exact* and *partial*
  (containment) matching, per type + micro/macro. A paired bootstrap over documents (2,000 resamples) gives a 95%
  CI on the F1 difference base -> fine-tuned, evaluated both on the 3 shared labels and on all 6.
  The base model can only predict Name, Company and Location, so the shared-label comparison is the fair one;
  the all-six comparison favours the fine-tuned model by construction.
* **Error analysis.** `results/error_analysis.md` lists false positives / false negatives per type and method.

## Results

### 1. Held-out DataTurks test set (221 resumes)

| Method | Exact micro-F1 | Partial micro-F1 |
|---|---|---|
| Rule/heuristic baseline | 31.9 | 54.8 |
| Off-the-shelf NER (base, 3 labels only) | 25.0 | 31.9 |
| **Fine-tuned DistilBERT** | **93.9** | **95.8** |

Per-type exact-match F1 of the fine-tuned model: Name 99.8, Company 91.5, Designation 90.6, Location 96.8,
College 91.5, Degree 93.4.

Does fine-tuning help? On the three labels the base model can predict (Name, Company, Location), exact micro-F1
rises from 30.7 to 96.1 (paired bootstrap 95% CI on the difference: +62.6 to +68.0 points). Over all six types
the gain is 25.0 -> 93.9 (CI +67.1 to +70.7). Name is easy in this dataset (the rule baseline also reaches 99.8,
because the name is almost always the first line), so the real gains come from Company, Location, College and
Degree.

### 2. Out-of-distribution synthetic test set (30 resumes)

| Method | Exact micro-F1 | Partial micro-F1 |
|---|---|---|
| Rule/heuristic baseline | 29.3 | 59.9 |
| Off-the-shelf NER (base, 3 labels only) | 35.9 | 44.4 |
| **Fine-tuned DistilBERT** | **50.6** | **69.9** |

Performance drops sharply compared with the DataTurks test set. Over all six types the fine-tuned model still
beats the base model (+14.5 points, CI +1.7 to +28.3), but on the three shared labels it does not (-15.6 points,
CI -34.9 to +4.2; not statistically significant with n = 30). Company is the weakest type (exact F1 13.0).

### 3. Error analysis (fine-tuned model)

Full lists are in `results/error_analysis.md`. The main patterns:

1. **Fragment / boundary errors** (both test sets): `sy`, `info` (Infosys), `hang` (Hangzhou), `nets` (Netsol),
   `boo` / `allen hamilton` (Booz Allen Hamilton), `end developer`, `university of`, `british columbia`. Many of
   these are partly correct, which is why partial match is much higher than exact match (e.g. synthetic Name
   recall is 100% under partial match). The base model produces similar fragments, so the token-merging step
   (`aggregation_strategy="simple"`) may contribute as well as the training.
2. **Prose-style resumes.** Some synthetic resumes read as sentences ("Mark Stone is a ...", "She serves as ...",
   "He holds a BS ..."). The model tags words such as `mark stone is`, `she`, `he`, `serves`, `graduated`,
   `earned`, `from`, because DataTurks consists of list-style Indeed resumes.
3. **Company recall is very low out of distribution** (3 of 34 exact). Misses include `oracle`, `ibm`,
   `deloitte`, `barclays`, `tencent`, `alibaba` in patterns like "Senior Java Developer at Oracle" and
   "Data Scientist @ Amazon". The base model finds more companies (64.7% recall) but with 22% precision, tagging
   tools such as `kafka` and `aws` as companies.
4. **Short capitalised abbreviations** are missed (`MIT`, `USC`, `FAST NUCES`, `PUCIT`, `BS CS`). The model is
   `distilbert-base-uncased`, which discards capitalisation; this may contribute but was not tested (a cased
   model would be the natural ablation).
5. **Annotation noise.** Some DataTurks gold labels are questionable (`engineering director` labelled as a
   Degree; an `indeed.com/r/...` string labelled as a Location), so exact match understates true quality
   slightly.

We interpret the drop on the synthetic set as domain shift from list-style Indian-IT resumes to differently
formatted text. This is a hypothesis supported by the examples above, not isolated by an ablation, and the small
test set limits the strength of the conclusion.

### 4. OCR robustness (multimodal part)

`evaluation/ocr_robustness.py` renders the 30 synthetic resumes as image-only pages, applies small rotation, blur
and noise, runs Tesseract, and compares against clean-text input.

| Extractor | Clean text (partial micro-F1) | Scanned + OCR (partial micro-F1) |
|---|---|---|
| Rule/heuristic baseline | 59.9 | 53.4 |
| Fine-tuned DistilBERT + rules | 69.9 | 68.5 |

Mean character error rate: 0.71% (Tesseract). The fine-tuned model is fairly robust to OCR noise on these pages.
Caveat: they are clean rendered text pages with mild synthetic degradation, not real phone photos or multi-column
layouts, so real-world robustness is likely lower.

> **Note on the earlier README.** The previous version claimed 93.9% F1 / 100% precision from a model and
> evaluation script that were not reproducible from the repository (no weights, partly circular "ground truth"
> generated by the extractor itself). Those claims were removed. Every number in this README is regenerated by
> the commands above.

## Limitations / discussion points

* **No table extraction.** Resumes are processed as text lines; tables and multi-column layouts can scramble
  reading order in text extraction and OCR.
* DataTurks annotations are noisy (inconsistent boundaries, mixed offset conventions), so exact match
  understates quality; partial match is reported alongside.
* Training data is mostly Indian-IT-industry text in list style; the model transfers poorly to prose-style or
  differently formatted resumes (see the synthetic results). The synthetic set is small (30), so conclusions
  about it are tentative.
* OCR was tested on synthetic scans only. PaddleOCR support is implemented but the reported results use Tesseract.
* Skills use a fixed gazetteer, so unseen skills are missed; short ambiguous tokens (`R`, `Go`) use a context
  filter.
* Experience duration is computed from date ranges in the experience section; overlapping jobs are merged,
  undated jobs are ignored.
* Resumes contain personal data, and NER-based screening can encode bias; the tool should assist a human, not
  decide.

## Possible future work

* Cased model (`distilbert-base-cased`) or a larger model, and checking the span-merging strategy using the
  validation split.
* Adding a few prose-style or multi-format resumes to the training data and re-testing on a fresh unseen set.
* Layout-aware extraction (tables, columns) with a document-layout model.
* The optional LLM baseline for comparison.

## Project layout

```
app.py                       Streamlit UI
src/ingestion/               loader (text layer vs OCR), ocr engines (Tesseract / PaddleOCR)
src/extraction/              ner.py (HF NER wrapper), rules.py, skills.py, heuristics.py, llm.py, pipeline.py
src/matching/matcher.py      MiniLM similarity + skill gap
training/                    prepare_data.py, alignment.py, finetune.py, colab_finetune.ipynb
evaluation/                  metrics.py, run_eval.py, ocr_robustness.py
data/                        ner/ (DataTurks + splits), synthetic_test/, samples/
tests/                       pytest suite
results/                     generated tables / plots
```