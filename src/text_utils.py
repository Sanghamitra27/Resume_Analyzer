"""Step 3 & 4: read PDF text and clean text."""
import re
from pypdf import PdfReader


def extract_pdf_text(file) -> str:
    """Read all pages of a PDF (path or uploaded file object) and return text."""
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Lowercase, remove noise, but KEEP useful symbols like C++, C#, .NET."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove links
    text = re.sub(r"\S+@\S+", " ", text)                    # remove emails
    text = re.sub(r"[^a-z0-9+#.\s/&-]", " ", text)          # keep + # . / & -
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", text)  # drop stray dots
    text = re.sub(r"\s+", " ", text).strip()                # normalise whitespace
    return text
