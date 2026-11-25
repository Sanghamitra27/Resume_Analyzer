"""Step 7: ask a local LLM (Ollama) for truthful resume suggestions."""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def build_prompt(resume, job, matched, missing) -> str:
    return f"""You are a careful resume coach.
Use ONLY the information in the resume below. Never invent experience,
qualifications, tools or skills. If a missing skill is not in the resume,
tell the candidate to add it ONLY if they truly have it, or to learn it.

RESUME:
{resume[:4000]}

JOB DESCRIPTION:
{job[:3000]}

MATCHED SKILLS: {', '.join(matched) or 'none'}
MISSING SKILLS: {', '.join(missing) or 'none'}

Give 5 short, practical suggestions as a numbered list."""


def get_suggestions(resume, job, matched, missing, model="llama3.2") -> str:
    try:
        r = requests.post(OLLAMA_URL, timeout=180, json={
            "model": model,
            "prompt": build_prompt(resume, job, matched, missing),
            "stream": False})
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        return ("Could not reach Ollama. Install it, run `ollama serve`, "
                f"and `ollama pull {model}`.")
    except Exception as e:
        return f"AI suggestion failed: {e}"
