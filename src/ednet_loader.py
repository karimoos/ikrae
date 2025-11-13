"""
ednet_loader.py
---------------
Online EdNet-KT3 loader for IKRAE.

- Downloads EdNet-KT3 and Content zip files from official links.
- Extracts them under data/ednet/{KT3,content}
- Builds:
    experiments/results/learning_objects.csv
    experiments/results/prerequisites.csv

Notes:
- This is a reasonable default pipeline.
"""

import io
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Official download links from the EdNet GitHub README :contentReference[oaicite:1]{index=1}
EDNET_DOWNLOADS = {
    "KT3": "https://bit.ly/ednet-kt3",
    "content": "https://bit.ly/ednet-content",
}

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "ednet"
KT3_DIR = DATA_ROOT / "KT3"
CONTENT_DIR = DATA_ROOT / "content"
OUT_DIR = ROOT / "experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# 1. Download + extract helpers
# -------------------------------------------------------------------

def download_and_extract(name: str, url: str, target_dir: Path) -> None:
    """
    Download a zip from URL and extract to target_dir (if not already there).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # If there are CSVs already, assume it's done
    if any(target_dir.rglob("*.csv")):
        print(f"[{name}] Found existing CSV files in {target_dir}, skipping download.")
        return

    print(f"[{name}] Downloading from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = Path(tmp.name)

    print(f"[{name}] Extracting zip to {target_dir} ...")
    with zipfile.ZipFile(tmp_path, "r") as zf:
        zf.extractall(target_dir)

    tmp_path.unlink(missing_ok=True)
    print(f"[{name}] Done. Extracted into {target_dir}")


def ensure_ednet_kt3_available() -> None:
    download_and_extract("KT3", EDNET_DOWNLOADS["KT3"], KT3_DIR)


def ensure_content_available() -> None:
    download_and_extract("content", EDNET_DOWNLOADS["content"], CONTENT_DIR)


# -------------------------------------------------------------------
# 2. Loading and LO statistics
# -------------------------------------------------------------------

def load_kt3_sample(max_users: int = 50000, max_rows_per_user: Optional[int] = None) -> pd.DataFrame:
    """
    Load a sample of EdNet-KT3 per-user CSVs from KT3_DIR.
    Assumes each file is {user_id}.csv (as per EdNet spec).
    """
    ensure_ednet_kt3_available()

    csv_files = sorted(KT3_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No KT3 CSV files found in {KT3_DIR}")

    dfs = []
    for f in csv_files[:max_users]:
        try:
            df = pd.read_csv(f)
            # Attach user_id from filename if not present
            if "user_id" not in df.columns:
                df["user_id"] = int(f.stem)
            if max_rows_per_user is not None:
                df = df.head(max_rows_per_user)
            dfs.append(df)
        except Exception as e:
            print(f"[KT3] Skipping {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No KT3 data could be loaded.")
    full = pd.concat(dfs, ignore_index=True)
    print(f"[KT3] Loaded {len(full)} interaction rows from {len(dfs)} users.")
    return full


def load_content_questions() -> pd.DataFrame:
    """
    Load questions.csv from the EdNet content zip.
    """
    ensure_content_available()
    q_path = next(CONTENT_DIR.rglob("questions.csv"), None)
    if q_path is None:
        raise FileNotFoundError(f"questions.csv not found under {CONTENT_DIR}")
    questions = pd.read_csv(q_path)
    print(f"[content] Loaded {len(questions)} questions.")
    return questions


def load_content_lectures() -> pd.DataFrame:
    """
    Load lectures.csv from the EdNet content zip, if present.
    """
    ensure_content_available()
    l_path = next(CONTENT_DIR.rglob("lectures.csv"), None)
    if l_path is None:
        print("[content] lectures.csv not found. Proceeding without lecture LOs.")
        return pd.DataFrame()
    lectures = pd.read_csv(l_path)
    print(f"[content] Loaded {len(lectures)} lectures.")
    return lectures


def build_lo_table(kt3_df: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build IKRAE-compatible learning_objects table.

    This function assumes EdNet-style columns, e.g.:
    - kt3_df contains: question_id, elapsed_time, user_answer, correct_answer (or equivalent)
    - questions_df contains: question_id, correct_answer, tags, etc.

    If your KT3 CSV schema differs, adapt the column names accordingly.
    """
    # Attempt to map/guess correct-answer column
    if "correct_answer" in questions_df.columns:
        correct_series = questions_df.set_index("question_id")["correct_answer"]
        kt3_df = kt3_df.merge(
            correct_series, left_on="question_id", right_index=True, how="left"
        )
        # bool: user_answer == correct_answer
        if "user_answer" in kt3_df.columns:
            kt3_df["is_correct"] = kt3_df["user_answer"] == kt3_df["correct_answer"]
        elif "correct" in kt3_df.columns:
            kt3_df["is_correct"] = kt3_df["correct"].astype(int) == 1
        else:
            kt3_df["is_correct"] = 0.5  # fallback
    else:
        # If content does not have correct_answer, fallback
        kt3_df["is_correct"] = 0.5

    # Duration: EdNet elapsed_time is millis; approximate minutes if exists
    if "elapsed_time" in kt3_df.columns:
        kt3_df["duration_min"] = kt3_df["elapsed_time"].astype(float) / 60000.0
    else:
        kt3_df["duration_min"] = 1.0

    stats = kt3_df.groupby("question_id").agg(
        duration_min=("duration_min", "mean"),
        accuracy=("is_correct", "mean"),
    ).reset_index()

    stats["pedagogical_weight"] = stats["accuracy"].fillna(0.5)

    lo_df = questions_df[["question_id"]].rename(columns={"question_id": "lo_id"})
    lo_df = lo_df.merge(stats, left_on="lo_id", right_on="question_id", how="left")
    lo_df = lo_df.drop(columns=["question_id"])

    lo_df["duration_min"] = lo_df["duration_min"].fillna(lo_df["duration_min"].median())
    lo_df["pedagogical_weight"] = lo_df["pedagogical_weight"].fillna(0.5)
    lo_df["type"] = "question"
    lo_df["language"] = "en"
    lo_df["requires_mastery"] = 0.0

    print(f"[LO] Built learning_objects table with {len(lo_df)} rows.")
    return lo_df


def build_prerequisite_edges(lo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple prerequisite chain as a baseline:
        lo_1 -> lo_2 -> lo_3 -> ...

    You can later replace this with a real skill DAG.
    """
    lo_ids = lo_df["lo_id"].astype(str).tolist()
    if len(lo_ids) < 2:
        return pd.DataFrame(columns=["src", "dst"])

    src = lo_ids[:-1]
    dst = lo_ids[1:]
    edges = pd.DataFrame({"src": src, "dst": dst})
    print(f"[Prereq] Built {len(edges)} prerequisite edges (simple chain).")
    return edges


def export_online_ednet(sample_users: int = 50000, max_rows_per_user: Optional[int] = None) -> None:
    """
    High-level function:
    - Loads KT3 & content online (if not cached)
    - Builds learning_objects + prerequisites
    - Exports CSVs under experiments/results/
    """
    kt3 = load_kt3_sample(max_users=sample_users, max_rows_per_user=max_rows_per_user)
    questions = load_content_questions()
    lectures = load_content_lectures()  # currently unused but kept for future extensions

    lo_df = build_lo_table(kt3, questions)
    lo_path = OUT_DIR / "learning_objects.csv"
    lo_df.to_csv(lo_path, index=False)

    edges_df = build_prerequisite_edges(lo_df)
    edges_path = OUT_DIR / "prerequisites.csv"
    edges_df.to_csv(edges_path, index=False)

    print(f"[Export] Saved:\n  - {lo_path}\n  - {edges_path}")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Online EdNet-KT3 loader for IKRAE")
    parser.add_argument("--sample_users", type=int, default=5000,
                        help="Number of KT3 user CSVs to sample (approx).")
    parser.add_argument("--max_rows_per_user", type=int, default=None,
                        help="Optional limit per-user rows for fast tests.")
    args = parser.parse_args()

    export_online_ednet(sample_users=args.sample_users,
                        max_rows_per_user=args.max_rows_per_user)
