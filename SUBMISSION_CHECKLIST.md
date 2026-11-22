# Submission checklist

- [ ] `HF_MODEL_ID` set in `src/config.py` and in README ("Fine-tuned model" section) to the Hugging Face repo id
- [ ] `pip install -r requirements.txt` and `python -m pytest -q` -> 11 passed
- [ ] `streamlit run app.py` shows "fine-tuned DistilBERT + rules" in the sidebar (model loads from the Hub)
- [ ] App tested: digital PDF, scanned PDF / "Force OCR", image, job-description match (screenshots taken)
- [ ] Project idea added to the shared Excel sheet (no similar idea from another group)
- [ ] Presentation / evaluation slides prepared
- [ ] Tesseract is the OCR engine used for the reported results (state PaddleOCR only if you ran it)
