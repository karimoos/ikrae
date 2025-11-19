"""
run_experiments.py
------------------
IKRAE Experiment Runner

- Loads EdNet (local or auto-download) through ednet_loader.export_ednet()
- Runs semantic filtering (OWL/SWRL → feasible learning objects)
- Runs graph optimization (Dijkstra + k-shortest paths)
- Saves results to experiments/results/
"""

import time
from pathlib import Path

from src.ednet_loader import export_ednet
from src.ikrae_reasoner import run_reasoner
from src.ikrae_optimizer import run_optimizer


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def run_single_experiment(sample_rows: int = 500_000, k_paths: int = 3) -> None:
    """Runs the full IKRAE pipeline using local EdNet data."""

    # Input/output paths
    lo_raw = RESULTS_DIR / "learning_objects.csv"
    lo_feasible = RESULTS_DIR / "learning_objects_feasible.csv"
    infeasible_json = RESULTS_DIR / "infeasible_los.json"
    edges_csv = RESULTS_DIR / "prerequisites.csv"
    user_json = ROOT / "experiments" / "user_context.json"
    path_trace = RESULTS_DIR / "path_trace.json"

    # 1) Load & export EdNet
    t0 = time.time()
    export_ednet(sample_rows=sample_rows)
    t1 = time.time()

    # 2) Semantic filtering
    run_reasoner(
        lo_csv=lo_raw,
        user_json=user_json,
        feasible_csv=lo_feasible,
        infeasible_json=infeasible_json,
    )
    t2 = time.time()

    # 3) Graph optimization + k-shortest paths
    explanation = run_optimizer(
        lo_csv=lo_feasible,
        edges_csv=edges_csv,
        user_json=user_json,
        infeasible_json=infeasible_json,
        output_json=path_trace,
        k=k_paths,
    )
    t3 = time.time()

    # Summary
    print("\n=== Experiment Summary ===")
    print(f"EdNet load + export: {1000*(t1 - t0):.1f} ms")
    print(f"Semantic reasoning:  {1000*(t2 - t1):.1f} ms")
    print(f"Optimization:        {1000*(t3 - t2):.1f} ms")
    print(f"Total:               {1000*(t3 - t0):.1f} ms")
    print(f"Real-time compliant? {explanation.get('real_time_compliant')}")


if __name__ == "__main__":
    run_single_experiment(sample_rows=500_000, k_paths=3)
