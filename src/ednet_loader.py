import pandas as pd
import numpy as np
import zipfile
import os
from pathlib import Path
import urllib.request

# ==========================================
# CONFIG
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_ZIP = DATA_DIR / "ikrae_kt3_clean.zip"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CI_MODE = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))

# Zenodo permanent dataset link
ZENODO_URL = "https://zenodo.org/record/17664110/files/ikrae_kt3_clean.zip?download=1"


# ==========================================
# Helpers
# ==========================================

def download_kt3_zip():
    """Download the KT3 zip from Zenodo."""
    print("[Download] KT3 dataset missing. Downloading from Zenodo...")

    DATA_DIR.mkdir(exist_ok=True)

    try:
        urllib.request.urlretrieve(ZENODO_URL, DATA_ZIP)
    except Exception as e:
        raise RuntimeError(f"Failed to download KT3 dataset from Zenodo: {e}")

    print(f"[Download] Saved to: {DATA_ZIP}")


def extract_zip():
    """Extract KT3 zip into /data."""
    print("[Extract] Extracting KT3 zip...")
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("[Extract] Extraction complete.")


# ==========================================
# 1. Load KT3 dataset
# ==========================================

def load_kt3(sample_rows=None):
    """Load KT3 interactions (local zip or CI dummy)."""

    if CI_MODE:
        print("[CI MODE] Using tiny KT3 synthetic dataset")
        df = pd.DataFrame({
            "user_id":   [1, 1, 2, 2],
            "timestamp": [1000, 2000, 1000, 3000],
            "item_id":   ["Q10", "Q11", "Q10", "Q11"],
            "user_answer": [1, 0, 1, 1],
        })
        return df

    # Auto-download if missing
    if not DATA_ZIP.exists():
        download_kt3_zip()
        extract_zip()

    print(f"[Local ZIP] Loading KT3 from {DATA_ZIP}")
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        assert csv_files, "ERROR: ZIP contains no CSV file!"
        csv_name = csv_files[0]
        print(f"[Local ZIP] Using CSV: {csv_name}")
        df = pd.read_csv(z.open(csv_name))

    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
        print(f"[KT3] Sampled to: {len(df):,} rows")

    print(f"[KT3] Loaded total rows: {len(df):,}")
    return df


# ==========================================
# 2. Build minimal questions table
# ==========================================

def build_questions_from_kt3(kt3_df):
    print("[Questions] Building questions table")

    qid_col = "question_id" if "question_id" in kt3_df.columns else "item_id"
    question_ids = kt3_df[qid_col].astype(str).unique()

    df = pd.DataFrame({
        "question_id": question_ids,
        "tags": [["generic"]] * len(question_ids)
    })

    print(f"[Questions] Total questions: {len(df):,}")
    return df


# ==========================================
# 3. Build Learning Objects (LO table)
# ==========================================

def build_learning_objects(kt3_df, questions_df):
    print("[Build] Constructing Learning Objects...")

    qid_src = "question_id" if "question_id" in kt3_df.columns else "item_id"

    # Duration column guess
    time_col = None
    for c in ["elapsed_time", "duration", "time_ms"]:
        if c in kt3_df.columns:
            time_col = c
            break

    # Accuracy columns
    correct_col = None
    if "correct_answer" in kt3_df.columns:
        correct_col = "correct_answer"
    elif "correct" in kt3_df.columns:
        correct_col = "correct"

    user_col = "user_answer" if "user_answer" in kt3_df.columns else None

    # Aggregations
    def duration_agg(x):
        if time_col:
            return x.mean() / 60000.0  # convert ms → minutes
        return 1.0  # fallback default

    def accuracy_agg(s):
        if correct_col and user_col:
            return (kt3_df.loc[s.index, user_col] ==
                    kt3_df.loc[s.index, correct_col]).mean()
        return 0.7  # fallback baseline

    stats = kt3_df.groupby(qid_src).agg(
        duration_min=(time_col if time_col else qid_src, duration_agg),
        accuracy=(qid_src, accuracy_agg),
    ).reset_index().rename(columns={qid_src: "lo_id"})

    # Merge with minimal question table
    q_copy = questions_df.rename(columns={"question_id": "lo_id"})
    lo = q_copy.merge(stats, on="lo_id", how="left")

    # Fill missing values
    lo["duration_min"] = lo["duration_min"].fillna(lo["duration_min"].median())
    lo["accuracy"] = lo["accuracy"].fillna(0.5)

    lo["type"] = "question"
    lo["language"] = "en"
    lo["requires_mastery"] = np.clip(1 - lo["accuracy"], 0.0, 1.0)
    lo["pedagogical_weight"] = 1 - lo["accuracy"]

    print(f"[Build] Final LO count: {len(lo):,}")
    return lo


# ==========================================
# 4. Build REAL prerequisite edges
# ==========================================

def build_prerequisite_edges_real(kt3_df):
    print("[Prereq] Building prerequisite graph...")

    qid_col = "question_id" if "question_id" in kt3_df.columns else "item_id"

    kt3_df = kt3_df.sort_values(["user_id", "timestamp"])

    transitions = []
    for uid, group in kt3_df.groupby("user_id"):
        seq = group[qid_col].astype(str).tolist()
        for i in range(len(seq) - 1):
            transitions.append((seq[i], seq[i + 1]))

    df = pd.DataFrame(transitions, columns=["src", "dst"])
    freq = df.groupby(["src", "dst"]).size().reset_index(name="count")

    print(f"[Prereq] Total edges: {len(freq):,}")
    return freq[["src", "dst"]]


# ==========================================
# 5. EXPORT FUNCTION (CALLED BY PIPELINE)
# ==========================================

def export_ednet(sample_rows=None):
    """Main entry point to generate:
    - learning_objects.csv
    - prerequisites.csv
    """

    print("=== IKRAE Local EdNet Loader ===")

    kt3 = load_kt3(sample_rows=sample_rows)
    questions = build_questions_from_kt3(kt3)

    lo_df = build_learning_objects(kt3, questions)
    lo_df.to_csv(OUT / "learning_objects.csv", index=False)
    print("[Save] learning_objects.csv")

    edges_df = build_prerequisite_edges_real(kt3)
    edges_df.to_csv(OUT / "prerequisites.csv", index=False)
    print("[Save] prerequisites.csv")

    print("=== EdNet export complete ===")
