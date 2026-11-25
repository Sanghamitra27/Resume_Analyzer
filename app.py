"""Steps 2, 3, 8: Streamlit interface that ties everything together."""
import os
import streamlit as st

from src.text_utils import extract_pdf_text
from src.analyzer import similarity_score, find_skill_gaps
from src.llm import get_suggestions

st.set_page_config(page_title="Resume & Job Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Resume & Job Compatibility Analyzer")
st.caption("Uploaded resumes are processed locally and are not stored.")


@st.cache_data
def load_vocab():
    path = "data/skills_vocab.txt"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


vocab = load_vocab()
if not vocab:
    st.warning("data/skills_vocab.txt not found. Run `python build_vocab.py` first.")

with st.sidebar:
    st.header("Settings")
    use_ai = st.checkbox("Generate AI suggestions (needs Ollama)", value=True)
    model = st.text_input("Ollama model", "llama3.2")

# ---------- Step 2: inputs ----------
col1, col2 = st.columns(2)
resume_text = ""
with col1:
    st.subheader("1. Your resume")
    pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    if pdf is not None:
        # ---------- Step 3: extract text ----------
        try:
            resume_text = extract_pdf_text(pdf)
        except Exception as e:
            st.error(f"Could not read this PDF: {e}")
        if pdf is not None and len(resume_text.strip()) < 50:
            st.error("No readable text found. The PDF may be a scanned image (needs OCR).")
            resume_text = ""
        else:
            with st.expander("Check extracted text"):
                st.text_area("Extracted resume text", resume_text, height=250)
with col2:
    st.subheader("2. Job description")
    job_text = st.text_area("Paste the job description", height=300)

# ---------- Step 8: results ----------
if st.button("Analyze", type="primary"):
    if not resume_text:
        st.error("Please upload a readable resume PDF.")
    elif len(job_text.strip()) < 50:
        st.error("Please paste a job description (at least a few sentences).")
    else:
        score = similarity_score(resume_text, job_text)
        matched, missing = find_skill_gaps(resume_text, job_text, vocab)

        st.divider()
        st.subheader("Text similarity")
        st.progress(min(int(score), 100))
        st.metric("TF-IDF similarity", f"{score:.1f} / 100")
        st.caption("This measures word overlap between the two texts. "
                   "It is NOT a probability of getting the job.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"✅ Matched skills ({len(matched)})")
            st.write(", ".join(matched) if matched else "None detected.")
        with c2:
            st.subheader(f"⚠️ Possible gaps ({len(missing)})")
            st.write(", ".join(missing) if missing else "None detected.")

        if use_ai:
            st.subheader("💡 AI suggestions")
            with st.spinner("Asking the local language model..."):
                st.markdown(get_suggestions(resume_text, job_text,
                                            matched, missing, model))
