"""Step 9: test the analyzer on the CSV.

Test A - correlation: does our TF-IDF score move with the dataset's existing scores?
Test B - sanity check: a resume should score higher with its OWN category's job
         than with a job from a different category.
"""
import sys
import random
import pandas as pd
from src.analyzer import similarity_score

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/job_resume_fit.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600     # rows to test (None = all)

df = pd.read_csv(csv_path)
df = df[df["resume_text"].str.len() > 500]             # drop broken/very short resumes
df = df.sample(n=min(N, len(df)), random_state=42).reset_index(drop=True)

# the 23 unique job descriptions, one per category
jobs = df.drop_duplicates("category").set_index("category")["job_text"].to_dict()
categories = list(jobs)
random.seed(42)

df["tfidf_score"] = [similarity_score(r, j) for r, j in zip(df.resume_text, df.job_text)]
df["mismatch_score"] = [
    similarity_score(r, jobs[random.choice([c for c in categories if c != cat])])
    for r, cat in zip(df.resume_text, df.category)]

print("\n=== Test A: correlation with existing scores (Spearman) ===")
cols = ["tfidf_score", "ai_match_score", "skill_string_match_score", "fuzzy_match_score"]
print(df[cols].corr(method="spearman")["tfidf_score"].round(3).to_string())

print("\n=== Test B: matched job vs mismatched job ===")
print(f"Average score, correct job   : {df.tfidf_score.mean():.2f}")
print(f"Average score, wrong job     : {df.mismatch_score.mean():.2f}")
print(f"Correct job scored higher in : {(df.tfidf_score > df.mismatch_score).mean()*100:.1f}% of resumes")

print("\n=== Average TF-IDF score by category ===")
print(df.groupby("category").tfidf_score.mean().round(2).sort_values().to_string())

df.drop(columns=["resume_text", "job_text"]).to_csv("data/evaluation_results.csv", index=False)
print("\nSaved data/evaluation_results.csv")
