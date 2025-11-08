import pandas as pd
import numpy as np
from zipfile import ZipFile
from tqdm import tqdm
from itertools import combinations
import networkx as nx

# === CONFIGURATION ===
ZIP_FILE = "EdNet-KT3.zip"
INTERACTIONS_FILE = "interactions.csv"   # or "train.csv" depending on version
QUESTIONS_FILE = "questions.csv"
SAMPLE_LEARNERS = 50000
MIN_PROB_EDGE = 0.15  # prerequisite threshold
CHUNK_SIZE = 500000   # for memory-safe reading

# === STEP 1: LOAD QUESTIONS METADATA ===
with ZipFile(ZIP_FILE) as z:
    with z.open(QUESTIONS_FILE) as f:
        questions = pd.read_csv(f)
print(f"Questions loaded: {len(questions):,}")

# === STEP 2: Ikrae SAMPLE LEARNERS ===
learner_ids = set()
with ZipFile(ZIP_FILE) as z:
    with z.open(INTERACTIONS_FILE) as f:
        for chunk in pd.read_csv(f, usecols=["user_id"], chunksize=CHUNK_SIZE):
            learner_ids.update(chunk["user_id"].unique())
learner_ids = list(learner_ids)[:SAMPLE_LEARNERS]
print(f"Selected {len(learner_ids):,} learners")

# === STEP 3: LOAD INTERACTIONS FOR SAMPLE ===
interactions = []
with ZipFile(ZIP_FILE) as z:
    with z.open(INTERACTIONS_FILE) as f:
        for chunk in tqdm(pd.read_csv(f, chunksize=CHUNK_SIZE)):
            chunk = chunk[chunk["user_id"].isin(learner_ids)]
            interactions.append(chunk)
interactions = pd.concat(interactions, ignore_index=True)
print(f"Interactions loaded: {len(interactions):,}")

# === STEP 4: COMPUTE IKrae LEARNING OBJECT FEATURES ===
# Estimated duration (mean per content_id)
duration = interactions.groupby("content_id")["elapsed_time"].mean() / 60000
# Pedagogical weight (1 - error rate)
weight = 1 - interactions.groupby("content_id")["correct"].mean()

comp = pd.Series(["en/desktop"] * len(duration), index=duration.index)

lo_df = pd.DataFrame({
    "id": duration.index,
    "d": duration.values,
    "w": weight.values,
    "comp": comp.values
})

print(f"Learning objects: {len(lo_df):,}")

edges = []
for uid, group in tqdm(interactions.groupby("user_id")):
    seq = group.sort_values("timestamp")["content_id"].tolist()
    for (a, b) in zip(seq[:-1], seq[1:]):
        edges.append((a, b))

edges_df = pd.DataFrame(edges, columns=["src", "dst"])
prob_df = edges_df.groupby(["src", "dst"]).size().reset_index(name="count")

total_src = prob_df.groupby("src")["count"].transform("sum")
prob_df["prob"] = prob_df["count"] / total_src

E = prob_df[prob_df["prob"] > MIN_PROB_EDGE][["src", "dst"]]
print(f"Prerequisite edges: {len(E):,}")

G = nx.from_pandas_edgelist(E, source="src", target="dst", create_using=nx.DiGraph())
is_dag = nx.is_directed_acyclic_graph(G)
print(f"Graph acyclic: {is_dag}")
print(f"Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

lo_df.to_csv("learning_objects.csv", index=False)
E.to_csv("prerequisites.csv", index=False)

print("Data extraction complete:")
print(" - learning_objects.csv (LO features)")
print(" - prerequisites.csv (edges for G=(V,E))")
