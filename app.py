"""Streamlit UI:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.extraction.pipeline import ResumeAssistant  # noqa: E402
from src.matching.matcher import match  # noqa: E402

st.set_page_config(page_title="ResuMate - Resume Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_assistant(ocr_engine: str):
    return ResumeAssistant(ocr_engine=ocr_engine)


st.title("📄 ResuMate - Multimodal Resume Assistant")
st.caption("Upload a digital PDF, a scanned PDF or a photo of a resume. The system reads it (text layer or OCR), "
           "extracts structured fields with a fine-tuned DistilBERT + rules, and can match it to a job description.")

with st.sidebar:
    st.header("Settings")
    ocr_engine = st.selectbox("OCR engine", ["auto", "paddle", "tesseract"],
                              help="auto = PaddleOCR if installed, else Tesseract")
    force_ocr = st.checkbox("Force OCR (ignore PDF text layer)")
    assistant = get_assistant(ocr_engine)
    st.markdown(f"**Extractor:** {assistant.mode}")
    if assistant.ner_error:
        st.warning(assistant.ner_error)

file = st.file_uploader("Resume", type=["pdf", "png", "jpg", "jpeg", "txt"])
jd = st.text_area("Job description (optional)", height=120)

if file:
    with st.spinner("Reading and analysing document..."):
        try:
            out = assistant.process_file(file.getvalue(), file.name, force_ocr=force_ocr)
        except Exception as e:
            st.error(f"Could not process file: {e}")
            st.stop()

    d, c = out["document"], out["contact"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Candidate", c["name"] or "not found")
    k2.metric("Years of experience", out["experience_years"])
    k3.metric("Skills found", sum(len(v) for v in out["skills"].values()))
    conf = [p["confidence"] for p in d["pages"] if p["confidence"] is not None]
    k4.metric("Read via", ("OCR · " + d["ocr_engine"]) if d["used_ocr"] else "PDF text layer",
              delta=f"OCR conf {sum(conf)/len(conf):.0%}" if conf else None)

    t1, t2, t3, t4, t5 = st.tabs(["Entities", "Contact & dates", "Skills", "Job match", "Raw"])
    with t1:
        rows = []
        for label, vals in out["entities"].items():
            confs = out["entity_confidence"].get(label, [])
            for i, v in enumerate(vals):
                rows.append({"Type": label, "Text": v, "Confidence": confs[i] if i < len(confs) else None})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No entities found.")
        st.caption(f"Entity method: {out['entity_method']}")
    with t2:
        st.json({k: v for k, v in c.items()})
        if out["date_ranges"]:
            st.dataframe(pd.DataFrame(out["date_ranges"]), use_container_width=True, hide_index=True)
        st.write("Sections detected:", ", ".join(out["sections"]) or "none")
    with t3:
        for cat, items in out["skills"].items():
            st.markdown(f"**{cat}:** " + ", ".join(items))
        if not out["skills"]:
            st.info("No skills from the gazetteer were found.")
    with t4:
        if jd.strip():
            m = match(out["text"], jd)
            if m.get("semantic_similarity") is not None:
                st.metric("Semantic similarity (MiniLM)", f"{m['semantic_similarity']:.0%}")
            if m["coverage"] is not None:
                st.metric("Required-skill coverage", f"{m['coverage']:.0%}")
            st.write("✅ Matched:", ", ".join(m["matched"]) or "-")
            st.write("⚠️ Missing:", ", ".join(m["missing"]) or "-")
        else:
            st.info("Paste a job description above to see the match.")
    with t5:
        st.download_button("Download JSON", json.dumps({k: v for k, v in out.items() if k != "text"}, indent=2),
                           file_name="resume.json")
        with st.expander("Extracted text"):
            st.text(out["text"])
