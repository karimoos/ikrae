import json
import os
import argparse
from datetime import datetime
import networkx as nx

from src.semantic_reasoner import build_feasible_graph
from src.optimizer import compute_k_shortest_paths
from src.evaluation import compute_metrics


def load_user_context(path):
    """Load user JSON context file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_base_graph():
    """Loads your synthetic or real learning object graph."""
    graph_path = "experiments/graphs/base_graph.pkl"

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    import pickle
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def run_ikrae_experiment(context_path, k_paths=3):
    """Run full IKRAE pipeline for a given user context."""
    context = load_user_context(context_path)
    base_graph = load_base_graph()

    print("=== Running IKRAE Experiment ===")
    print("Context:", context)

    # === STEP 1: Semantic Reasoning (feasible graph Gf) ===
    print("\n[1] Building feasible graph...")
    Gf = build_feasible_graph(base_graph, context)
    print(f"Feasible graph: {len(Gf.nodes())} nodes, {len(Gf.edges())} edges")

    # Pick start and end nodes
    start_node = list(Gf.nodes())[0]
    end_node   = list(Gf.nodes())[-1]

    # === STEP 2: Optimization (K-shortest paths) ===
    print("\n[2] Computing K-shortest paths...")
    paths = compute_k_shortest_paths(Gf, start_node, end_node, k=k_paths)
    print(f"Generated {len(paths)} paths")

    # === STEP 3: Evaluation ===
    print("\n[3] Computing evaluation metrics...")
    results = compute_metrics(Gf, paths)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # === Save results ===
    save_dir = "experiments/results"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(save_dir, f"result_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump({
            "context": context,
            "paths": paths,
            "metrics": results
        }, f, indent=2)

    print(f"\n➡ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IKRAE experiment with a user context JSON.")
    parser.add_argument("--context", type=str, required=True,
                        help="Path to JSON user context file (e.g., experiments/context_high_mastery.json)")
    parser.add_argument("--k", type=int, default=3, help="Number of alternative paths to compute")

    args = parser.parse_args()

    run_ikrae_experiment(args.context, args.k)
