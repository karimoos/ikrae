import pandas as pd
import numpy as np
import zipfile
import io
import requests
import os
from pathlib import Path

# Output directory
OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)

# EdNet download links
KT3_URL = "https://ednet-kt3.s3.ap-northeast-2.amazonaws.com/KT3.zip"
CONTENT_URL = "https://ednet-kt3.s3.ap-northeast-2.amazonaws.com/contents.zip"


# ----------------------------------------------------
# 1. Download + extract ZIP files in memory
# ----------------------------------------------------

def download_and_extract_zip(url):
    """Handle CI-mode (dummy data) and full download locally."""
    
    # ------------------------------------------------
    # CI MODE (GitHub Actions)
    # ------------------------------------------------
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        print("[CI MODE] Skipping EdNet download for:", url)

        if "KT3" in url:
            # Fake KT3.csv (4 rows)
            df = pd.DataFrame({
                "user_id": [1, 1, 2, 2],
                "question_id": [10, 11, 10, 11],
                "correct_answer": [1, 0, 1, 1],
                "user_answer": [1, 0, 1, 1],
                "elapsed_time": [30000, 40000, 25000, 35000],
                "timestamp": [1000, 2000, 1000, 3000]
            })
            return {"KT3.csv": df.to_csv(index=False).encode()}

        else:
            # Fake questions/lectures metadata
            questions = pd.DataFrame({
                "question_id": [10, 11],
                "tags": ["A", "B"]
            })
            return {
                "questions.csv": questions.to_csv(index=False).encode(),
                "lectures.csv": pd.DataFrame({"dummy": []}).to_csv(index=False).encode()
            }

    # ------------------------------------------------
    # NORMAL MODE → real online download
    # ------------------------------------------------
    print(f"[Download] Fetching: {url}")
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    return {name: z.read(name) for name in z.namelist()}


# ----------------------------------------------------
# 2. Load KT3 + content CSVs directly from zip files
# ----------------------------------------------------

def load_kt3(sample_rows=None):
    data = download_and_extract_zip(KT3_URL)
    csv_bytes = data["KT3.csv"]
    df = pd.read_csv(io.BytesIO(csv_bytes))

    if sample_rows:
        df = df.sample(sample_rows, random_state=42)

    print(f"[KT3] Loaded {len(df):,} rows")
    return df


def load_questions():
    data = download_and_extract_zip(CONTENT_URL)
    df = pd.read_csv(io.BytesIO(data["questions.csv"]))
    print(f"[questions] Loaded {len(df):,} questions")
    return df


def load_lectures():
    data = download_and_extract_zip(CONTENT_URL)
    df = pd.read_csv(io.BytesIO(data["lectures.csv"]))
    print(f"[lectures] Loaded {len(df):,} lectures")
    return df


# ----------------------------------------------------
# 3. Build learning objects table
# ----------------------------------------------------

def build_learning_objects(kt3_df, questions_df):
    print("[Build] Constructing learning objects...")

    questions_df = questions_df.rename(columns={"question_id": "lo_id"})

    stats = kt3_df.groupby("question_id").agg(
        duration_min=("elapsed_time", lambda x: x.mean() / 60000),
        accuracy=("user_answer", lambda s: (s == kt3_df.loc[s.index, "correct_answer"]).mean())
    ).reset_index().rename(columns={"question_id": "lo_id"})

    lo = questions_df.merge(stats, on="lo_id", how="left")

    lo["duration_min"] = lo["duration_min"].fillna(lo["duration_min"].median())
    lo["accuracy"] = lo["accuracy"].fillna(0.5)

    lo["type"] = "question"
    lo["language"] = "en"
    lo["requires_mastery"] = np.clip(1 - lo["accuracy"], 0.0, 1.0)
    lo["pedagogical_weight"] = 1 - lo["accuracy"]

    print("[Build] Learning objects:", len(lo))
    return lo


# ----------------------------------------------------
# 4. REAL prerequisite graph (from sequential transitions)
# ----------------------------------------------------

def build_prerequisite_edges_real(kt3_df):
    print("[Prereq] Building real EdNet prerequisite graph...")

    kt3_df = kt3_df.sort_values(["user_id", "timestamp"])

    transitions = []
    for uid, group in kt3_df.groupby("user_id"):
        seq = group["question_id"].astype(str).tolist()
        for i in range(len(seq) - 1):
            transitions.append((seq[i], seq[i + 1]))

    trans_df = pd.DataFrame(transitions, columns=["src", "dst"])
    freq = trans_df.groupby(["src", "dst"]).size().reset_index(name="count")
    freq = freq[freq["count"] >= 1]  # in CI, small dataset

    print(f"[Prereq] Final edges: {len(freq)}")
    return freq[["src", "dst"]]


# ----------------------------------------------------
# 5. Export everything for IKRAE pipeline
# ----------------------------------------------------

def export_online_ednet(sample_rows=None):
    print("=== IKRAE EdNet Online Loader ===")

    kt3 = load_kt3(sample_rows=sample_rows)
    questions = load_questions()
    lectures = load_lectures()

    lo_df = build_learning_objects(kt3, questions)
    lo_df.to_csv(OUT / "learning_objects.csv", index=False)
    print("[Save] learning_objects.csv")

    edges_df = build_prerequisite_edges_real(kt3)
    edges_df.to_csv(OUT / "prerequisites.csv", index=False)
    print("[Save] prerequisites.csv")

    print("=== Done: EdNet extraction complete ===")


# ----------------------------------------------------
# Run directly
# ----------------------------------------------------

if __name__ == "__main__":
    # Use tiny sample in CI for speed
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        export_online_ednet(sample_rows=1000)
    else:
        export_online_ednet(sample_rows=500000)
