import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

USER_CONTEXT_PATH = Path("experiments/user_context.json")

# Cost weights
ALPHA = 0.4  # duration
BETA = 0.3   # difficulty
GAMMA = 0.3  # context penalty


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def load_user_context():
    with open(USER_CONTEXT_PATH, "r") as f:
        return json.load(f)


def load_feasible_los():
    lo_path = RESULTS_DIR / "learning_objects_feasible.csv"
    return pd.read_csv(lo_path)


def load_edges():
    edges_path = RESULTS_DIR / "prerequisites.csv"
    return pd.read_csv(edges_path)


def load_infeasible():
    path = RESULTS_DIR / "infeasible_los.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


# ----------------------------------------------------------
# Cost model
# ----------------------------------------------------------

def context_penalty(lo_row, user):
    """
    Penalty based on mismatch with user context.
    You can refine this to match your paper exactly.
    """
    penalty = 0.0

    # Example: mobile & long lecture
    if user["device"] == "mobile" and lo_row["type"] == "lecture" and lo_row["duration_min"] > 10:
        penalty += 10.0

    # Example: low bandwidth & lecture
    if user["bandwidth"] == "low" and lo_row["type"] == "lecture":
        penalty += 8.0

    # Example: time budget proximity (long LOs near limit)
    if lo_row["duration_min"] > 0.5 * user["time_budget_min"]:
        penalty += 5.0

    return penalty


def edge_cost(lo_row, user):
    """
    Total cost of including this LO in the path.
    duration + difficulty + context penalty.
    """
    duration = float(lo_row["duration_min"])
    difficulty = 1.0 - float(lo_row.get("accuracy", 0.5))
    penalty = context_penalty(lo_row, user)

    total = ALPHA * duration + BETA * difficulty + GAMMA * penalty
    return total, duration, difficulty, penalty


# ----------------------------------------------------------
# Graph construction
# ----------------------------------------------------------

def build_graph(lo_df, edges_df, user):
    G = nx.DiGraph()

    # Add LO nodes
    for _, row in lo_df.iterrows():
        lo_id = str(row["lo_id"])
        G.add_node(lo_id, **row.to_dict())

    # Add START and GOAL
    G.add_node("START")
    G.add_node("GOAL")

    # Add prerequisite edges
    for _, e in edges_df.iterrows():
        src = str(e["src"])
        dst = str(e["dst"])
        if src in G.nodes and dst in G.nodes:
            # cost attached to the destination LO
            lo_row = lo_df[lo_df["lo_id"].astype(str) == dst].iloc[0]
            total, duration, difficulty, penalty = edge_cost(lo_row, user)
            G.add_edge(src, dst, weight=total, duration=duration,
                       difficulty=difficulty, penalty=penalty)

    # Connect START to all nodes with no incoming edges
    candidates_start = [n for n in G.nodes if n not in ("START", "GOAL")]
    for n in candidates_start:
        if G.in_degree(n) == 0:
            G.add_edge("START", n, weight=0.0, duration=0.0,
                       difficulty=0.0, penalty=0.0)

    # Connect nodes with no outgoing edges to GOAL
    for n in candidates_start:
        if G.out_degree(n) == 0:
            G.add_edge(n, "GOAL", weight=0.0, duration=0.0,
                       difficulty=0.0, penalty=0.0)

    return G


# ----------------------------------------------------------
# Path optimization
# ----------------------------------------------------------

def compute_shortest_path(G):
    try:
        path = nx.shortest_path(G, source="START", target="GOAL", weight="weight")
        cost = nx.shortest_path_length(G, source="START", target="GOAL", weight="weight")
        return path, cost
    except nx.NetworkXNoPath:
        return None, float("inf")


# ----------------------------------------------------------
# Explanation builder
# ----------------------------------------------------------

def build_explanation(path, cost, G, user, infeasible, runtime_ms):
    explanation = {
        "user_context": user,
        "runtime_ms": runtime_ms,
        "total_cost": cost,
        "primary_path": path,
        "edges": [],
        "excluded_los": infeasible,
    }

    if path is None:
        explanation["error"] = "No feasible path from START to GOAL"
        return explanation

    # Collect edge-level details
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v, default={})
        explanation["edges"].append({
            "from": u,
            "to": v,
            "duration": data.get("duration", 0.0),
            "difficulty": data.get("difficulty", 0.0),
            "penalty": data.get("penalty", 0.0),
            "weight": data.get("weight", 0.0),
        })

    return explanation


# ----------------------------------------------------------
# Main entry
# ----------------------------------------------------------

def run_optimizer():
    start = time.time()

    user = load_user_context()
    lo_df = load_feasible_los()
    edges_df = load_edges()
    infeasible = load_infeasible()

    print(f"[Optimizer] Feasible LOs: {len(lo_df)}")
    print(f"[Optimizer] Edges: {len(edges_df)}")

    G = build_graph(lo_df, edges_df, user)
    path, cost = compute_shortest_path(G)

    runtime_ms = (time.time() - start) * 1000.0
    print(f"[Optimizer] Runtime: {runtime_ms:.2f} ms")

    explanation = build_explanation(path, cost, G, user, infeasible, runtime_ms)

    out_path = RESULTS_DIR / "path_trace.json"
    with open(out_path, "w") as f:
        json.dump(explanation, f, indent=2)

    print(f"[Optimizer] Saved path explanation to {out_path}")


if __name__ == "__main__":
    run_optimizer()
