import pandas as pd
import numpy as np
import zipfile
import io
import os
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = ROOT / "data" / "ikrae_kt3_clean.zip"   # local dataset
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CI_MODE = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


# ==========================================
# 1. Load KT3 (local zip or CI dummy)
# ==========================================

def load_kt3(sample_rows=None):
    """Load KT3 interactions.

    - In CI: tiny synthetic dataset (fast, no big files).
    - Locally: read from data/ikrae_kt3_clean.zip.
    """

    if CI_MODE:
        print("[CI MODE] Using tiny synthetic KT3 dataset")
        df = pd.DataFrame({
            "user_id":   [1, 1, 2, 2],
            "timestamp": [1_000, 2_000, 1_000, 3_000],
            "item_id":   ["Q10", "Q11", "Q10", "Q11"],
            "user_answer": [1, 0, 1, 1],
        })
        return df

    if not DATA_ZIP.exists():
        raise FileNotFoundError(f"KT3 zip not found at {DATA_ZIP}")

    print(f"[Local ZIP] Loading KT3 from {DATA_ZIP}")
    with zipfile.ZipFile(DATA_ZIP, "r") as z:
        # pick the first CSV inside the zip
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV file found inside ikrae_kt3_clean.zip")
        csv_name = csv_names[0]
        print(f"[Local ZIP] Using internal file: {csv_name}")
        df = pd.read_csv(z.open(csv_name))

    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
        print(f"[KT3] Sampled {len(df):,} rows")

    print(f"[KT3] Loaded {len(df):,} rows")
    return df


# ==========================================
# 2. Minimal questions table from KT3
# ==========================================

def build_questions_from_kt3(kt3_df):
    """Create a minimal questions table from KT3 item ids."""
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

    # Figure out column names
    qid_src = "question_id" if "question_id" in kt3_df.columns else "item_id"
    time_col = None
    for cand in ["elapsed_time", "duration", "time_ms"]:
        if cand in kt3_df.columns:
            time_col = cand
            break

    # Accuracy sources
    correct_col = None
    if "correct_answer" in kt3_df.columns:
        correct_col = "correct_answer"
    elif "correct" in kt3_df.columns:
        correct_col = "correct"

    user_col = "user_answer" if "user_answer" in kt3_df.columns else None

    # Build stats
    def duration_agg(x):
        if time_col is None:
            return 1.0  # fallback 1 minute
        return x.mean() / 60000.0

    def accuracy_agg(s):
        if correct_col and user_col and correct_col in kt3_df.columns and user_col in kt3_df.columns:
            return (kt3_df.loc[s.index, user_col] == kt3_df.loc[s.index, correct_col]).mean()
        if correct_col and correct_col in kt3_df.columns:
            return kt3_df.loc[s.index, correct_col].mean()
        # fallback default accuracy
        return 0.7

    stats = kt3_df.groupby(qid_src).agg(
        duration_min=(time_col if time_col else qid_src, duration_agg),
        accuracy=(qid_src, accuracy_agg),
    ).reset_index().rename(columns={qid_src: "lo_id"})

    questions_df = questions_df.rename(columns={"question_id": "lo_id"})

    lo = questions_df.merge(stats, on="lo_id", how="left")

    lo["duration_min"] = lo["duration_min"].fillna(lo["duration_min"].median())
    lo["accuracy"] = lo["accuracy"].fillna(0.5)

    lo["type"] = "question"
    lo["language"] = "en"
    lo["requires_mastery"] = np.clip(1 - lo["accuracy"], 0.0, 1.0)
    lo["pedagogical_weight"] = 1 - lo["accuracy"]

    print(f"[Build] Learning objects: {len(lo):,}")
    return lo


# ==========================================
# 4. REAL prerequisite graph (from sequential transitions)
# ==========================================

def build_prerequisite_edges_real(kt3_df):
    print("[Prereq] Building real EdNet prerequisite graph...")

    qid_col = "question_id" if "question_id" in kt3_df.columns else "item_id"

    kt3_df = kt3_df.sort_values(["user_id", "timestamp"])

    transitions = []
    for uid, group in kt3_df.groupby("user_id"):
        seq = group[qid_col].astype(str).tolist()
        for i in range(len(seq) - 1):
            transitions.append((seq[i], seq[i + 1]))

    trans_df = pd.DataFrame(transitions, columns=["src", "dst"])
    freq = trans_df.groupby(["src", "dst"]).size().reset_index(name="count")

    print(f"[Prereq] Final edges: {len(freq):,}")
    return freq[["src", "dst"]]


# ==========================================
# 5. Export everything for IKRAE pipeline
# ==========================================

def export_ednet(sample_rows=None):
    print("=== IKRAE Local EdNet Loader ===")

    kt3 = load_kt3(sample_rows=sample_rows)
    questions = build_questions_from_kt3(kt3)

    lo_df = build_learning_objects(kt3, questions)
    lo_df.to_csv(OUT / "learning_objects.csv", index=False)
    print("[Save] learning_objects.csv")

    edges_df = build_prerequisite_edges_real(kt3)
    edges_df.to_csv(OUT / "prerequisites.csv", index=False)
    print("[Save] prerequisites.csv")

    print("=== Done: EdNet extraction complete ===")


# ==========================================
# Run directly
# ==========================================

if __name__ == "__main__":
    # Keep sampling for local runs; CI uses tiny synthetic data anyway
    export_ednet(sample_rows=500_000)
