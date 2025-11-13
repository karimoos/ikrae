"""
ikrae_optimizer.py
------------------
IKRAE graph-based optimizer:

- Input:
    experiments/results/learning_objects_feasible.csv
    experiments/results/prerequisites.csv
    experiments/user_context.json

- Output:
    experiments/results/path_trace.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.4  # duration weight
BETA = 0.3   # difficulty (1 - pedagogical weight)
GAMMA = 0.3  # context penalty weight
REAL_TIME_THRESHOLD_MS = 200.0


def compute_context_penalty(lo: Dict, user_context: Dict) -> float:
    penalty = 0.0

    lo_type = lo.get("type", "question")
    bandwidth = user_context.get("bandwidth", "high")
    device = user_context.get("device", "desktop")
    language = user_context.get("language", "en")
    mastery = float(user_context.get("mastery_level", 1.0))
    time_budget = float(user_context.get("time_budget_min", 60.0))

    duration = float(lo.get("duration_min", 5.0))
    requires_mastery = float(lo.get("requires_mastery", 0.0))
    lo_lang = lo.get("language", "en")

    # Low bandwidth + video
    if lo_type == "video" and bandwidth == "low":
        penalty += 15.0

    # Mobile + long video
    if lo_type == "video" and device == "mobile" and duration > 15.0:
        penalty += 10.0

    # Language mismatch
    if lo_lang and lo_lang != language:
        penalty += 20.0

    # Mastery margin
    if requires_mastery > mastery:
        penalty += 50.0

    # Exceeding budget fraction
    if duration > 0.7 * time_budget:
        penalty += 10.0

    return penalty


def build_weighted_dag(
    lo_df: pd.DataFrame,
    prereq_edges: List[Tuple[str, str]],
    user_context: Dict,
) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in lo_df.iterrows():
        lo_id = str(row["lo_id"])
        G.add_node(lo_id, **row.to_dict())

    # Add edges from prerequisites
    for src, dst in prereq_edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)

    # Add START / GOAL
    G.add_node("START")
    G.add_node("GOAL")

    entry_los = [n for n in G.nodes if n not in ("START", "GOAL") and G.in_degree(n) == 0]
    exit_los = [n for n in G.nodes if n not in ("START", "GOAL") and G.out_degree(n) == 0]

    for lo in entry_los:
        G.add_edge("START", lo, weight=0.0)
    for lo in exit_los:
        G.add_edge(lo, "GOAL", weight=0.0)

    # Compute weights
    for u, v in G.edges():
        if u == "START" or v == "GOAL":
            G[u][v]["weight"] = 0.0
            G[u][v]["cost_breakdown"] = {
                "duration": 0.0,
                "difficulty": 0.0,
                "penalty": 0.0,
                "total": 0.0,
            }
            continue

        lo_v = G.nodes[v]
        duration = float(lo_v.get("duration_min", 5.0))
        ped_weight = float(lo_v.get("pedagogical_weight", 0.5))
        difficulty = 1.0 - ped_weight
        penalty = compute_context_penalty(lo_v, user_context)

        total = ALPHA * duration + BETA * difficulty + GAMMA * penalty
        total = max(total, 1e-6)

        G[u][v]["weight"] = total
        G[u][v]["cost_breakdown"] = {
            "duration": duration,
            "difficulty": difficulty,
            "penalty": penalty,
            "total": total,
        }

    return G


def load_edges_from_csv(edges_csv: Optional[Path], lo_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if edges_csv is None:
        # fallback: simple chain over feasible LOs
        lo_ids = lo_df["lo_id"].astype(str).tolist()
        return list(zip(lo_ids[:-1], lo_ids[1:]))

    if not edges_csv.exists():
        raise FileNotFoundError(f"Prerequisite CSV not found: {edges_csv}")

    e_df = pd.read_csv(edges_csv)
    if not {"src", "dst"}.issubset(e_df.columns):
        raise ValueError("prerequisites.csv must contain columns: src, dst")

    return list(zip(e_df["src"].astype(str), e_df["dst"].astype(str)))


def optimize_paths(G: nx.DiGraph, k: int = 1) -> List[Dict]:
    try:
        if k <= 1:
            path = nx.shortest_path(G, "START", "GOAL", weight="weight")
            cost = nx.shortest_path_length(G, "START", "GOAL", weight="weight")
            return [{"rank": 1, "path": path, "cost": float(cost)}]

        paths = list(nx.shortest_simple_paths(G, "START", "GOAL", weight="weight"))[:k]
        results = []
        for i, path in enumerate(paths, start=1):
            cost = 0.0
            for u, v in zip(path[:-1], path[1:]):
                cost += float(G[u][v]["weight"])
            results.append({"rank": i, "path": path, "cost": float(cost)})
        return results

    except nx.NetworkXNoPath:
        return [{"rank": 1, "path": None, "cost": float("inf"), "error": "No feasible path"}]


def build_explanation(
    G: nx.DiGraph,
    paths: List[Dict],
    user_context: Dict,
    infeasible_los_trace: Optional[List[Dict]] = None,
    runtime_ms: float = 0.0,
) -> Dict:
    primary = paths[0]
    explanation = {
        "user_context": user_context,
        "runtime_ms": runtime_ms,
        "real_time_compliant": runtime_ms < REAL_TIME_THRESHOLD_MS,
        "paths": paths,
        "primary_path": primary,
        "primary_cost_breakdown": [],
        "excluded_los": infeasible_los_trace or [],
    }

    path = primary.get("path")
    if not path or path[0] is None:
        explanation["error"] = primary.get("error", "No path found")
        return explanation

    for u, v in zip(path[:-1], path[1:]):
        bd = G[u][v].get("cost_breakdown", {})
        explanation["primary_cost_breakdown"].append(
            {"from": u, "to": v, "details": bd}
        )

    return explanation


def run_optimizer(
    lo_csv: Path,
    edges_csv: Optional[Path],
    user_json: Path,
    infeasible_json: Optional[Path],
    output_json: Path,
    k: int = 1,
) -> Dict:
    start = time.time()

    lo_df = pd.read_csv(lo_csv)
    with open(user_json, encoding="utf-8") as f:
        user_ctx = json.load(f)

    if infeasible_json is not None and infeasible_json.exists():
        with open(infeasible_json, encoding="utf-8") as f:
            infeasible_trace = json.load(f)
    else:
        infeasible_trace = []

    print(f"[Optimizer] Loaded {len(lo_df)} feasible LOs.")
    edges = load_edges_from_csv(edges_csv, lo_df)
    print(f"[Optimizer] Using {len(edges)} prerequisite edges.")

    G = build_weighted_dag(lo_df, edges, user_ctx)
    paths = optimize_paths(G, k=k)

    runtime_ms = (time.time() - start) * 1000.0
    explanation = build_explanation(
        G,
        paths,
        user_ctx,
        infeasible_los_trace=infeasible_trace,
        runtime_ms=runtime_ms,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    print(f"[Optimizer] Done in {runtime_ms:.1f} ms → {output_json}")
    return explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IKRAE Optimizer")
    parser.add_argument(
        "--lo_csv",
        default=str(RESULTS_DIR / "learning_objects_feasible.csv"),
        help="Feasible LOs CSV",
    )
    parser.add_argument(
        "--edges_csv",
        default=str(RESULTS_DIR / "prerequisites.csv"),
        help="Prerequisites CSV (src,dst)",
    )
    parser.add_argument(
        "--user_json",
        default=str(ROOT / "experiments" / "user_context.json"),
        help="User context JSON",
    )
    parser.add_argument(
        "--infeasible_json",
        default=str(RESULTS_DIR / "infeasible_los.json"),
        help="Infeasible LOs trace JSON (from reasoner)",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "path_trace.json"),
        help="Output explanation JSON",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of alternative paths")

    args = parser.parse_args()

    run_optimizer(
        lo_csv=Path(args.lo_csv),
        edges_csv=Path(args.edges_csv) if args.edges_csv else None,
        user_json=Path(args.user_json),
        infeasible_json=Path(args.infeasible_json) if args.infeasible_json else None,
        output_json=Path(args.output),
        k=args.k,
    )
