# src/ednet_loader.py
import os
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import json
from collections import defaultdict
import networkx as nx

EDNET_PATH = Path("../data/ednet")
OUTPUT_PATH = Path("../experiments/results")

def load_ednet_kt3(sample_users=50000):
    """Load EdNet-KT3 with sampling for scalability"""
    kt3_files = list(EDNET_PATH.glob("KT3/*.csv"))
    if len(kt3_files) == 0:
        raise FileNotFoundError("Place EdNet KT3 CSVs in data/ednet/KT3/")
    
    # Sample users
    user_ids = [int(f.stem) for f in kt3_files[:sample_users]]
    dfs = []
    for uid in user_ids:
        try:
            df = pd.read_csv(EDNET_PATH / "KT3" / f"{uid}.csv")
            df['user_id'] = uid
            dfs.append(df)
        except:
            continue
    return pd.concat(dfs, ignore_index=True)

def build_lo_stats(kt3_df, questions_df):
    """Compute duration, pedagogical weight per LO"""
    # Merge with correct answers
    q_correct = questions_df.set_index('question_id')['correct_answer']
    kt3_df = kt3_df.merge(q_correct, left_on='question_id', right_index=True, how='left')
    
    stats = kt3_df.groupby('question_id').agg(
        duration_min=('elapsed_time', lambda x: x.mean() / 60000),
        accuracy=('user_answer', lambda x: (x == x.name).mean())
    ).reset_index()
    stats['pedagogical_weight'] = stats['accuracy']
    return stats

def build_skill_dag(kt3_df, questions_df, threshold=0.15):
    """Build prerequisite DAG from skill co-occurrence"""
    # Map question → skills
    q_to_skills = questions_df.set_index('question_id')['tags'].str.split(';').to_dict()
    
    G = nx.DiGraph()
    skill_pairs = defaultdict(int)
    
    for _, group in kt3_df.groupby(['user_id', 'timestamp']):
        q_seq = group['question_id'].tolist()
        if len(q_seq) < 2: continue
        skills = []
        for q in q_seq:
            skills.extend([s.strip() for s in q_to_skills.get(q, [])])
        for i in range(len(skills)-1):
            skill_pairs[(skills[i], skills[i+1])] += 1
    
    total = sum(skill_pairs.values())
    for (s1, s2), count in skill_pairs.items():
        if count / total > threshold / 100:
            G.add_edge(s1, s2)
    
    return G

def export_for_ikrae(kt3_df, questions_df, lectures_df, output_dir=OUTPUT_PATH):
    """Export JSON + CSV for IKRAE pipeline"""
    output_dir.mkdir(exist_ok=True)
    
    # LO metadata
    lo_df = pd.concat([
        questions_df[['question_id', 'tags']].rename(columns={'question_id': 'lo_id'}),
        lectures_df[['lecture_id', 'video_length']].assign(tags='lecture').rename(columns={'lecture_id': 'lo_id', 'video_length': 'duration_min'})
    ])
    lo_df.to_csv(output_dir / "learning_objects.csv", index=False)
    
    # Sample interactions
    kt3_df.sample(100000).to_csv(output_dir / "interactions_sample.csv", index=False)
    
    print(f"Exported to {output_dir}")
