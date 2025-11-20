import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default context (used if run standalone)
DEFAULT_USER_CONTEXT = Path("experiments/contexts/user_context.json")

# Cost weights
ALPHA = 0.4  # duration
BETA  = 0.3  # difficulty
GAMMA = 0.3  # context penalty


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def load_user_context(path=None):
    path = Path(path) if path else DEFAULT_USER_CONTEXT
    with open(path, "r") as f:
        return json.load(f)


def load_lo_table(path=None):
    if path:
        return pd.read_csv(path)
    return pd.read_csv(RESULTS_DIR / "learning_objects_feasible.csv")


def load_edges(path=None):
    if path:
        return pd.read_csv(path)
    return pd.read_csv(RESULTS_DIR / "prerequisites.csv")


def load_infeasible(path=None):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    fallback = RESULTS_DIR / "infeasible_los.json"
    if fallback.exists():
        with open(fallback, "r") as f:
            return json.load(f)
    return []


# ----------------------------------------------------------
# Cost model
# ----------------------------------------------------------

def context_penalty(lo_row, user):
    penalty = 0.0

    # Mobile & long lecture
    if user["device"] == "mobile" and lo_row["type"] == "lecture" and lo_row["duration_min"] > 10:
        penalty += 10.0

    # Low bandwidth & lecture
    if user["bandwidth"] == "low" and lo_row["type"] == "lecture":
        penalty += 8.0

    # LO duration relative to user's time budget
    if lo_row["duration_min"] > 0.5 * user["time_budget_min"]:
        penalty += 5.0

    return penalty


def edge_cost(lo_row, user):
    duration   = float(lo_row["duration_min"])
    difficulty = 1.0 - float(lo_row.get("accuracy", 0.5))
    penalty    = context_penalty(lo_row, user)

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

    # Add meta nodes
    G.add_node("START")
    G.add_node("GOAL")

    # Add prerequisite edges
    for _, e in edges_df.iterrows():
        src = str(e["src"])
        dst = str(e["dst"])

        if src not in G.nodes or dst not in G.nodes:
            continue

        lo_row = lo_df[lo_df["lo_id"].astype(str) == dst].iloc[0]
        total, duration, difficulty, penalty = edge_cost(lo_row, user)

        G.add_edge(src, dst,
                   weight=total,
                   duration=duration,
                   difficulty=difficulty,
                   penalty=penalty)

    # START → nodes with no incoming edges
    candidates = [n for n in G.nodes if n not in ("START", "GOAL")]
    for n in candidates:
        if G.in_degree(n) == 0:
            G.add_edge("START", n, weight=0.0)

    # nodes with no outgoing edges → GOAL
    for n in candidates:
        if G.out_degree(n) == 0:
            G.add_edge(n, "GOAL", weight=0.0)

    return G


# ----------------------------------------------------------
# Path optimization
# ----------------------------------------------------------

def compute_shortest_path(G):
    try:
        path = nx.shortest_path(G, "START", "GOAL", weight="weight")
        cost = nx.shortest_path_length(G, "START", "GOAL", weight="weight")
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
        "real_time_compliant": path is not None,
    }

    if not path:
        explanation["error"] = "No feasible path from START to GOAL"
        return explanation

    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v, {})
        explanation["edges"].append({
            "from": u,
            "to": v,
            "duration": data.get("duration", 0),
            "difficulty": data.get("difficulty", 0),
            "penalty": data.get("penalty", 0),
            "weight": data.get("weight", 0),
        })

    return explanation


# ----------------------------------------------------------
# PIPELINE FUNCTION
# ----------------------------------------------------------

def run_optimizer(
    lo_csv=None,
    edges_csv=None,
    user_json=None,
    infeasible_json=None,
    output_json=None,
    k=1
):
    """
    Main optimizer entry point.
    Matches the signature used by run_experiments.py
    and run_pipeline.sh.
    """

    start = time.time()

    # Load data (from file paths or defaults)
    user      = load_user_context(user_json)
    lo_df     = load_lo_table(lo_csv)
    edges_df  = load_edges(edges_csv)
    infeasible = load_infeasible(infeasible_json)

    print(f"[Optimizer] Feasible LOs: {len(lo_df)}")
    print(f"[Optimizer] Edges: {len(edges_df)}")

    # Build graph + compute path
    G = build_graph(lo_df, edges_df, user)
    path, cost = compute_shortest_path(G)

    runtime_ms = (time.time() - start) * 1000
    print(f"[Optimizer] Runtime: {runtime_ms:.2f} ms")

    # Create explanation
    explanation = build_explanation(path, cost, G, user, infeasible, runtime_ms)

    # Determine output path
    out_path = Path(output_json) if output_json else (RESULTS_DIR / "path_trace.json")

    # Save results
    with open(out_path, "w") as f:
        json.dump(explanation, f, indent=2)

    print(f"[Optimizer] Saved path explanation to {out_path}")

    return explanation


# ----------------------------------------------------------
# Standalone CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    run_optimizer()
