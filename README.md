# AI Resume & Job Compatibility Analyzer

Compares a resume (PDF) with a job description using TF-IDF + cosine similarity,
finds matched / missing skills, and (optionally) asks a local LLM (Ollama) for
truthful improvement suggestions.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
# put job_resume_fit.csv inside data/
python build_vocab.py        # builds data/skills_vocab.txt
python evaluate.py           # optional: test on the dataset
streamlit run app.py
```
Optional AI suggestions: install Ollama, then `ollama pull llama3.2`.

## Limitations
TF-IDF measures word overlap, not meaning. Scanned PDFs need OCR.
The score is not a hiring probability. Do not commit real resumes.
