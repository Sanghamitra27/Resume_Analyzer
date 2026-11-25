"""Step 6 (prep): build a clean skill vocabulary from the CSV -> data/skills_vocab.txt

Keeps (a) every skill the 23 job descriptions require and
(b) resume skills that appear in at least MIN_COUNT different resumes
    (rare, one-off phrases are mostly noise).
"""
import sys
from collections import Counter
import pandas as pd
from src.analyzer import parse_skill_list

MIN_COUNT = 15
csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/job_resume_fit.csv"
df = pd.read_csv(csv_path, usecols=["job_required_skills", "resume_skill_list"])

job_skills = set()
for value in df["job_required_skills"].drop_duplicates():
    job_skills.update(parse_skill_list(value))

counter = Counter()
for value in df["resume_skill_list"]:
    counter.update(set(parse_skill_list(value)))      # count once per resume
common = {s for s, c in counter.items() if c >= MIN_COUNT}

skills = {s for s in (job_skills | common) if 2 <= len(s) <= 40}
with open("data/skills_vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(skills)))
print(f"job skills: {len(job_skills)}, common resume skills: {len(common)}")
print(f"Saved {len(skills)} skills to data/skills_vocab.txt")
