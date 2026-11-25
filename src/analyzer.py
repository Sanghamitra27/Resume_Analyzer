"""Step 5 & 6: similarity score + skill gap detection."""
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_utils import clean_text

# keeps tokens such as c++, c#, node.js, sql
TOKEN_PATTERN = r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]"


def similarity_score(resume: str, job: str) -> float:
    """TF-IDF + cosine similarity, returned as 0-100 (text overlap, NOT hiring chance)."""
    docs = [clean_text(resume), clean_text(job)]
    if not docs[0] or not docs[1]:
        return 0.0
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                          stop_words="english")
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:               # e.g. text has only stop words
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0] * 100)


def parse_skill_list(value) -> list:
    """The CSV stores skills as the text "['a', 'b']" -> convert to a real list."""
    try:
        return [s.strip().lower() for s in ast.literal_eval(value)]
    except (ValueError, SyntaxError):
        return []


def contains_term(text: str, term: str) -> bool:
    """Whole-word match so 'java' does not match inside 'javascript'."""
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def find_skill_gaps(resume: str, job: str, vocabulary: list):
    """Skills in the job text that are (matched) / are not (missing) in the resume."""
    r, j = clean_text(resume), clean_text(job)
    matched, missing = [], []
    for skill in vocabulary:
        if contains_term(j, skill):
            (matched if contains_term(r, skill) else missing).append(skill)
    return sorted(matched), sorted(missing)
