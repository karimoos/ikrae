import pandas as pd
import numpy as np
import zipfile
import os
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT  / "data"
DATA_ZIP = DATA_DIR / "ikrae_kt3_clean.zip"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CI_MODE = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))

# Google Drive file ID for KT3
GDRIVE_ID = "1uyVS796ulJTfND1Oz0yJgjRwj6wKsksh"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"


# ==========================================
# Helpers
# ==========================================

def download_kt3_zip():
    """Download the KT3 zip from Google Drive into data/ folder."""
    print("[Download] KT3 zip not found. Downloading from Google Drive...")

    try:
        import gdown
    except ImportError:
        raise ImportError("Missing dependency: gdown. Add 'gdown' to requirements.txt")

    DATA_DIR.mkdir(exist_ok=True)
    gdown.download(GDRIVE_URL, str(DATA_ZIP), quiet=False)
    print(f"[Download] Saved to {DATA_ZIP}")


def extract_zip():
    """Extract KT3 zip."""
    print("[Extract] Extracting KT3 dataset...")
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("[Extract] Done.")


# ==========================================
# 1. Load KT3 (Local or CI)
# ==========================================

def load_kt3(sample_rows=None):
    """Load KT3 interactions.

    - In CI: use small synthetic dataset.
    - Locally: auto-download + extract from Google Drive if missing.
    """
    if CI_MODE:
        print("[CI MODE] Using tiny synthetic KT3 dataset")
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

    # Load from ZIP
    print(f"[Local ZIP] Loading KT3 from {DATA_ZIP}")
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found inside ikrae_kt3_clean.zip")

        csv_name = csv_names[0]
        print(f"[Local ZIP] Using internal file: {csv_name}")
        df = pd.read_csv(z.open(csv_name))

    # Optional sampling
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
        print(f"[KT3] Sampled {len(df):,} rows")

    print(f"[KT3] Loaded {len(df):,} rows")
    return df


# ==========================================
# 2. Minimal questions table
# ==========================================

def build_questions_from_kt3(kt3_df):
    print("[Questions] Building minimal questions table from KT3")

    qid_col = "question_id" if "question_id" in kt3_df.columns else "item_id"
    unique_ids = kt3_df[qid_col].astype(str).unique()

    questions = pd.DataFrame({
        "question_id": unique_ids,
        "tags": [["generic"]] * len(unique_ids)
    })

    print(f"[Questions] Created {len(questions):,} rows")
    return questions


# ==========================================
# 3. Build learning objects table
# ==========================================

def build_learning_objects(kt3_df, questions_df):
    print("[Build] Constructing learning objects...")

    qid_src = "question_id" if "question_id" in kt3_df.columns else "item_id"

    time_col = None
    for cand in ["elapsed_time", "duration", "time_ms"]:
        if cand in kt3_df.columns:
            time_col = cand
            break

    correct_col = None
    if "correct_answer" in kt3_df.columns:
        correct_col = "correct_answer"
    elif "correct" in kt3_df.columns:
        correct_col = "correct"

    user_col = "user_answer" if "user_answer" in kt3_df.columns else None

    def duration_agg(x):
        if time_col is None:
            return 1.0
        return x.mean() / 60000.0

    def accuracy_agg(s):
        if correct_col and user_col and correct_col in kt3_df.columns and user_col in kt3_df.columns:
            return (kt3_df.loc[s.index, user_col] == kt3_df.loc[s.index, correct_col]).mean()
        if correct_col in kt3_df.columns:
            return kt3_df.loc[s.index, correc]()_
