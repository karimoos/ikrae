# src/ednet_loader.py
"""
EdNet Loader for IKRAE
----------------------
Loads KT3 CSVs, computes LO statistics (duration, pedagogical weight),
and exports ready-to-use files for the IKRAE pipeline.
"""

import os
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

EDNET_PATH = Path("../data/ednet")
OUTPUT_PATH = Path("../experiments/results")


def load_ednet_kt3(sample_users=50000):
    """Load EdNet-KT3 user traces with sampling."""
    kt3_dir = EDNET_PATH / "KT3"
    if not kt3_dir.exists():
        raise FileNotFoundError("Place EdNet KT3 CSVs in data/ednet/KT3/")

    files = list(kt3_dir.glob("*.csv"))[:sample_users]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["user_id"] = int(f.stem)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True)


def build_lo_stats(kt3_df, questions_df):
    """Compute average duration and accuracy per question."""
    q_correct = questions_df.set_index("question_id")["correct_answer"]
    kt3_df = kt3_df.merge(q_correct, left_on="question_id", right_index=True, how="left")

    stats = kt3_df.groupby("question_id").agg(
        duration_min=("elapsed_time", lambda x: x.mean() / 60000.0),
        accuracy=("user_answer", lambda s: (s == kt3_df.loc[s.index, "correct_answer"]).mean())
    ).reset_index()

    stats["pedagogical_weight"] = stats["accuracy"].fillna(0.5)
    return stats


def build_skill_dag(kt3_df, questions_df, threshold=0.15):
    """Build a prerequisite DAG from skill co-occurrence."""
    q_to_skills = questions_df.set_index("question_id")["tags"].fillna("").str.split(";").to_dict()
    skill_pairs = defaultdict(int)

    for uid, group in kt3_df.groupby("user_id"):
        q_seq = group.sort_values("timestamp")["question_id"].tolist()
        if len(q_seq) < 2:
            continue
        skills = []
        for q in q_seq:
            skills.extend([s.strip() for s in q_to_skills.get(q, []) if s])
        for i in range(len(skills) - 1):
            skill_pairs[(skills[i], skills[i + 1])] += 1

    total = sum(skill_pairs.values())
    G = nx.DiGraph()
    for (s1, s2), count in skill_pairs.items():
        if total > 0 and (count / total) > threshold:
            G.add_edge(s1, s2)

    return G


def export_for_ikrae(kt3_df, questions_df, lectures_df, output_dir=OUTPUT_PATH):
    """Export data in the schema expected by the IKRAE optimizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = build_lo_stats(kt3_df, questions_df)

    # Merge stats with questions
    lo_q = questions_df[["question_id"]].rename(columns={"question_id": "lo_id"})
    lo_q = lo_q.merge(stats, left_on="lo_id", right_on="question_id", how="left").drop(columns="question_id")
    lo_q["type"] = "question"
    lo_q["language"] = "en"
    lo_q["requires_mastery"] = 0.0

    # Optional lectures
    if not lectures_df.empty and {"lecture_id", "video_length"}.issubset(lectures_df.columns):
        lo_l = lectures_df[["lecture_id", "video_length"]].rename(
            columns={"lecture_id": "lo_id", "video_length": "duration_min"}
        )
        lo_l["pedagogical_weight"] = 0.5
        lo_l["type"] = "video"
        lo_l["language"] = "en"
        lo_l["requires_mastery"] = 0.0
        lo_df = pd.concat([lo_q, lo_l], ignore_index=True)
    else:
        lo_df = lo_q

    # Save LOs
    lo_df.to_csv(output_dir / "learning_objects.csv", index=False)

    # Dummy prerequisite edges (to be replaced by build_skill_dag if needed)
    edges_df = pd.DataFrame({"src": lo_df["lo_id"].shift(1).dropna(), "dst": lo_df["lo_id"][1:]})
    edges_df.to_csv(output_dir / "prerequisites.csv", index=False)

    print("Exported {len(lo_df)} LOs and {len(edges_df)} edges to {output_dir}")
