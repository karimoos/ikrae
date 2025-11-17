import networkx as nx
import numpy as np
import random
import os
import pickle

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------
SIZES = [200, 1000, 5000, 10000, 50000, 100000]
NUM_TOPICS = 20
EDGE_PROB = 0.006   # Controls graph density
OUT_DIR = "experiments/graphs"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_graph(num_los):
    """Generate a synthetic LO graph with topics + difficulty + DAG edges."""
    G = nx.DiGraph()

    # ------------------------------
    # Nodes
    # ------------------------------
    for i in range(num_los):
        node_id = f"LO{i}"
        topic = random.randint(0, NUM_TOPICS - 1)
        difficulty = float(np.clip(np.random.normal(0.5, 0.15), 0, 1))

        G.add_node(node_id, topic=topic, difficulty=difficulty)

    # ------------------------------
    # Edges (DAG forward direction)
    # ------------------------------
    nodes = list(G.nodes())
    for i in range(num_los):
        for j in range(i + 1, num_los):
            if random.random() < EDGE_PROB:
                G.add_edge(nodes[i], nodes[j])

    return G


def generate_suite():
    print("=== IKRAE Synthetic Graph Suite Generator ===")

    for size in SIZES:
        print(f"\nGenerating graph with {size} learning objects...")

        G = generate_graph(size)

        path = os.path.join(OUT_DIR, f"base_graph_{size}.pkl")
        with open(path, "wb") as f:
            pickle.dump(G, f)

        print(f"Saved: {path}   ({len(G.edges())} edges)")

    print("\nAll synthetic graphs generated successfully.")


if __name__ == "__main__":
    generate_suite()
