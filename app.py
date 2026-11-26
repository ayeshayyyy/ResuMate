"""Streamlit UI:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.extraction.pipeline import ResumeAssistant  # noqa: E402
from src.matching.matcher import match, rank_candidates  # noqa: E402
from src.ingestion.visual import render_page_with_marks  # noqa: E402
from src.privacy import create_flattened_redacted_pdf, score_invariance  # noqa: E402

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
candidate_files = st.file_uploader("Candidate pool for ranking", type=["pdf", "png", "jpg", "jpeg", "txt"],
                                   accept_multiple_files=True)

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

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["Entities", "Contact & dates", "Skills", "Job match", "Evidence", "Blind screen", "Raw"])
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
        st.subheader("Timeline integrity")
        timeline = out["timeline"]
        st.metric("Dated history", f"{timeline['dated_history_years']:.1f} years")
        if timeline["overlaps"]:
            st.warning(f"Overlapping dated roles: {len(timeline['overlaps'])}")
            st.dataframe(pd.DataFrame(timeline["overlaps"]), use_container_width=True, hide_index=True)
        if timeline["gaps"]:
            st.info("Gaps are between dated entries and do not imply unemployment.")
            st.dataframe(pd.DataFrame(timeline["gaps"]), use_container_width=True, hide_index=True)
        if timeline["impossible_dates"]:
            st.error("Implausible date ranges detected")
            st.dataframe(pd.DataFrame(timeline["impossible_dates"]), use_container_width=True, hide_index=True)
        if timeline["claims"]:
            st.dataframe(pd.DataFrame(timeline["claims"]), use_container_width=True, hide_index=True)
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
            if m["requirements"]:
                st.subheader("Requirement evidence")
                st.dataframe(pd.DataFrame([{"Requirement": item["requirement"], "Verdict": item["verdict"],
                                            "Evidence": item["evidence"] or "No supporting sentence",
                                            "Matched terms": ", ".join(item["matched_terms"])}
                                           for item in m["requirements"]]), use_container_width=True, hide_index=True)
        else:
            st.info("Paste a job description above to see the match.")
    with t5:
        st.caption("Boxes show text spans associated with extracted fields.")
        if file.name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
            for index, page in enumerate(d["pages"]):
                marks = out["provenance"][index]["marks"] if index < len(out["provenance"]) else []
                image = render_page_with_marks(file.getvalue(), file.name, index, page, marks)
                st.image(image, caption=f"Page {index + 1} · {len(marks)} field spans highlighted",
                         use_container_width=True)
        else:
            st.info("Visual provenance is available for PDF and image inputs.")
        if d["tables"]:
            st.subheader("Extracted tables")
            for table_index, table in enumerate(d["tables"], 1):
                st.caption(f"Table {table_index} · page {table['page']}")
                st.dataframe(pd.DataFrame(table["rows"]), use_container_width=True, hide_index=True)
        else:
            st.info("No machine-readable tables were detected.")
    with t6:
        redaction = score_invariance(out["text"], jd, out)
        st.text_area("Redacted resume text", redaction["redacted_text"], height=220)
        st.caption(f"PII replacements applied: {sum(item['count'] for item in redaction['replacements'])}")
        if jd.strip():
            st.metric("Evidence score unchanged after redaction", "Yes" if redaction["invariant"] else "No",
                      delta=f"score delta {redaction['delta']:+.3f}")
        if file.name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
            redacted_pdf = create_flattened_redacted_pdf(file.getvalue(), file.name,
                                                         redaction["fields"], d["pages"])
            st.download_button("Download flattened redacted PDF", redacted_pdf,
                               file_name="blind-screened-resume.pdf", mime="application/pdf")
        st.warning("Review the redacted output before sharing. Text extraction can miss identifying details.")
    with t7:
        st.download_button("Download JSON", json.dumps({k: v for k, v in out.items() if k != "text"}, indent=2),
                           file_name="resume.json")
        with st.expander("Extracted text"):
            st.text(out["text"])

if candidate_files and jd.strip():
    st.subheader("Candidate ranking by requirement evidence")
    with st.spinner("Parsing and comparing candidates..."):
        pool = []
        for candidate_file in candidate_files:
            candidate_result = assistant.process_file(candidate_file.getvalue(), candidate_file.name,
                                                      force_ocr=force_ocr)
            pool.append({"name": candidate_result["contact"]["name"] or candidate_file.name,
                         "resume_text": candidate_result["text"]})
        ranking = rank_candidates(pool, jd)
    st.dataframe(pd.DataFrame([{"Rank": item["rank"], "Candidate": item["candidate"],
                                "Evidence score": item["evidence_score"],
                                "Skill coverage": item["coverage"],
                                "Semantic similarity": item["semantic_similarity"]}
                               for item in ranking]), use_container_width=True, hide_index=True)
    for item in ranking:
        with st.expander(f"#{item['rank']} {item['candidate']} · evidence {item['evidence_score']:.0%}"):
            st.dataframe(pd.DataFrame([{"Requirement": row["requirement"], "Verdict": row["verdict"],
                                        "Evidence": row["evidence"] or "No supporting sentence"}
                                       for row in item["requirements"]]), use_container_width=True, hide_index=True)
