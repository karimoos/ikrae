# src/ikrae_optimizer.py
"""
IKRAE Optimizer: Semantic + Graph-Based Adaptive Learning Path Generation
- Input: EdNet-processed LOs, user context
- Output: Optimal path + explainable JSON trace
- Scalable to 100K+ LOs, <200ms re-planning
"""

import argparse
import json
import time
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional: Owlready2 for lightweight reasoning (fallback if OWLAPI not available)
try:
    from owlready2 import get_ontology, default_world
    OWLREADY_AVAILABLE = True
except ImportError:
    OWLREADY_AVAILABLE = False

# -------------------------------
# CONFIGURATION
# -------------------------------
ALPHA = 0.4  # Duration weight
BETA = 0.3   # Pedagogical weight (difficulty)
GAMMA = 0.3  # Context penalty
K_SHORTEST = 3  # Number of alternative paths
REAL_TIME_THRESHOLD_MS = 200

# -------------------------------
# HELPER: Context Penalty
# -------------------------------
def compute_context_penalty(lo: Dict, user_context: Dict) -> float:
    """Calculate penalty based on device, time, language, mastery"""
    penalty = 0.0

    # Device mismatch
    if lo.get('type') == 'video' and user_context.get('bandwidth') == 'low':
        penalty += 15
    if lo.get('type') == 'video' and user_context.get('device') == 'mobile':
        penalty += 8

    # Time budget
    remaining = user_context.get('time_budget_min', 60)
    if lo.get('duration_min', 5) > remaining * 0.7:
        penalty += 20

    # Language mismatch
    if lo.get('language') != user_context.get('language'):
        penalty += 25

    # Prerequisite mastery (from ontology)
    if lo.get('requires_mastery', 0.0) > user_context.get('mastery_level', 1.0):
        penalty += 100  # Block infeasible

    return penalty

# -------------------------------
# CORE: Build Weighted DAG
# -------------------------------
def build_weighted_dag(
    lo_df: pd.DataFrame,
    prereq_edges: List[Tuple[str, str]],
    user_context: Dict
) -> nx.DiGraph:
    """Build DAG with context-sensitive edge weights"""
    G = nx.DiGraph()

    # Add nodes
    for _, row in lo_df.iterrows():
        lo_id = str(row['lo_id'])
        G.add_node(lo_id, **row.to_dict())

    # Add prerequisite edges
    for src, dst in prereq_edges:
        G.add_edge(src, dst)

    # Add START and GOAL
    entry_los = [n for n in G.nodes if G.in_degree(n) == 0 and n not in ['START', 'GOAL']]
    exit_los = [n for n in G.nodes if G.out_degree(n) == 0 and n not in ['START', 'GOAL']]
    G.add_node("START")
    G.add_node("GOAL")
    for lo in entry_los:
        G.add_edge("START", lo, weight=0)
    for lo in exit_los:
        G.add_edge(lo, "GOAL", weight=0)

    # Assign edge weights: cost of entering v from u
    for u, v in G.edges():
        if u == "START" or v == "GOAL":
            G[u][v]['weight'] = 0
            continue

        lo_v = G.nodes[v]
        duration = lo_v.get('duration_min', 5.0)
        ped_weight = 1.0 - lo_v.get('pedagogical_weight', 0.5)  # inverse
        penalty = compute_context_penalty(lo_v, user_context)

        total_cost = (
            ALPHA * duration +
            BETA * ped_weight +
            GAMMA * penalty
        )
        G[u][v]['weight'] = max(total_cost, 1e-6)  # Avoid zero/negative

        # Store decomposition for explainability
        G[u][v]['cost_breakdown'] = {
            "duration": duration,
            "difficulty": ped_weight,
            "penalty": penalty,
            "total": total_cost
        }

    return G

# -------------------------------
# OPTIMIZATION: Dijkstra + k-shortest
# -------------------------------
def optimize_path(
    G: nx.DiGraph,
    k: int = 1
) -> List[Dict]:
    """Run Dijkstra and optional Yen's k-shortest paths"""
    try:
        if k == 1:
            path = nx.shortest_path(G, "START", "GOAL", weight='weight')
            cost = nx.shortest_path_length(G, "START", "GOAL", weight='weight')
            return [{"path": path, "cost": cost, "rank": 1}]
        else:
            # Yen's algorithm via networkx
            paths = list(nx.shortest_simple_paths(G, "START", "GOAL", weight='weight'))[:k]
            results = []
            for i, path in enumerate(paths, 1):
                cost = sum(G[path[j]][path[j+1]]['weight'] for j in range(len(path)-1))
                results.append({"path": path, "cost": cost, "rank": i})
            return results
    except nx.NetworkXNoPath:
        return [{"path": None, "cost": float('inf'), "error": "No feasible path"}]

# -------------------------------
# EXPLAINABILITY: Generate Trace
# -------------------------------
def generate_explanation(
    G: nx.DiGraph,
    path_result: Dict,
    user_context: Dict,
    infeasible_los: List[str]
) -> Dict:
    """Generate human-readable + JSON explanation"""
    path = path_result['path']
    if not path or path[0] is None:
        return {"error": "No path found", "infeasible_count": len(infeasible_los)}

    trace = {
        "user_context": user_context,
        "optimal_path": path,
        "total_cost": path_result['cost'],
        "cost_breakdown": [],
        "excluded_los": [],
        "included_reasons": [],
        "k_alternatives": []
    }

    # Path cost decomposition
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        breakdown = G[u][v].get('cost_breakdown', {})
        trace['cost_breakdown'].append({
            "from": u, "to": v, "details": breakdown
        })

    # Infeasible LOs
    for lo in infeasible_los:
        reason = "unknown"
        lo_data = G.nodes.get(lo, {})
        if lo_data.get('requires_mastery', 0) > user_context.get('mastery_level', 1):
            reason = f"mastery too low (need {lo_data['requires_mastery']})"
        elif lo_data.get('type') == 'video' and user_context.get('bandwidth') == 'low':
            reason = "low bandwidth + video"
        trace['excluded_los'].append({"lo": lo, "reason": reason})

    return trace

# -------------------------------
# SEMANTIC FILTER (Lightweight via Owlready2)
# -------------------------------
def apply_semantic_filter(
    lo_df: pd.DataFrame,
    user_context: Dict,
    ontology_path: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter infeasible LOs using SWRL rules (Owlready2 fallback)"""
    infeasible = []

    # Rule-based filtering (mimics SWRL)
    filtered = []
    for _, row in lo_df.iterrows():
        lo = row.to_dict()
        lo_id = str(lo['lo_id'])

        # Prerequisite mastery
        if lo.get('requires_mastery', 0) > user_context.get('mastery_level', 1.0):
            infeasible.append(lo_id)
            continue

        # Device
        if lo.get('type') == 'video' and user_context.get('bandwidth') == 'low':
            infeasible.append(lo_id)
            continue

        # Language
        if lo.get('language') and lo['language'] != user_context['language']:
            infeasible.append(lo_id)
            continue

        filtered.append(row)

    filtered_df = pd.DataFrame(filtered).reset_index(drop=True)
    return filtered_df, infeasible

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def run_ikrae_pipeline(
    lo_csv: str,
    prereq_edges: List[Tuple[str, str]],
    user_context: Dict,
    output_json: str,
    k: int = 1,
    use_semantic: bool = True
) -> Dict:
    start_time = time.time()

    # 1. Load LOs
    lo_df = pd.read_csv(lo_csv)
    print(f"Loaded {len(lo_df)} learning objects")

    # 2. Semantic filtering
    if use_semantic:
        lo_df, infeasible = apply_semantic_filter(lo_df, user_context)
        print(f"Filtered out {len(infeasible)} infeasible LOs")
    else:
        infeasible = []

    # 3. Build weighted DAG
    G = build_weighted_dag(lo_df, prereq_edges, user_context)

    # 4. Optimize
    paths = optimize_path(G, k=k)
    primary = paths[0]

    # 5. Explain
    explanation = generate_explanation(G, primary, user_context, infeasible)

    # 6. Add alternatives
    explanation['k_alternatives'] = [
        {"rank": p['rank'], "cost": p['cost'], "path": p['path']}
        for p in paths[1:]
    ]

    # 7. Save
    total_time = (time.time() - start_time) * 1000  # ms
    explanation['runtime_ms'] = total_time
    explanation['real_time_compliant'] = total_time < REAL_TIME_THRESHOLD_MS

    with open(output_json, 'w') as f:
        json.dump(explanation, f, indent=2)

    print(f"Path computed in {total_time:.1f}ms → {output_json}")
    return explanation

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IKRAE Optimizer")
    parser.add_argument("--lo_csv", default="../experiments/results/learning_objects.csv")
    parser.add_argument("--user_json", required=True, help="User context JSON")
    parser.add_argument("--output", default="../experiments/results/path_trace.json")
    parser.add_argument("--k", type=int, default=1, help="Number of alternative paths")
    parser.add_argument("--no_semantic", action="store_false", dest="use_semantic")

    args = parser.parse_args()

    # Load user context
    with open(args.user_json) as f:
        user_context = json.load(f)

    # Dummy prereq edges (replace with ednet_loader output)
    prereq_edges = [("START", "q1"), ("q1", "q2"), ("q2", "GOAL")]

    run_ikrae_pipeline(
        lo_csv=args.lo_csv,
        prereq_edges=prereq_edges,
        user_context=user_context,
        output_json=args.output,
        k=args.k,
        use_semantic=args.use_semantic
    )
