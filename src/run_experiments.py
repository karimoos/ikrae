"""
run_experiments.py
------------------
Example experiment script for IKRAE:

- Runs online EdNet loading
- Runs semantic filter
- Runs optimizer
- Prints runtime and basic stats
"""

from pathlib import Path
import time
import json

from ednet_loader import export_online_ednet
from ikrae_reasoner import run_reasoner
from ikrae_optimizer import run_optimizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"


def run_single_experiment(sample_users: int = 5000, k_paths: int = 3) -> None:
    user_json = ROOT / "experiments" / "user_context.json"
    lo_raw = RESULTS_DIR / "learning_objects.csv"
    lo_feasible = RESULTS_DIR / "learning_objects_feasible.csv"
    infeasible_json = RESULTS_DIR / "infeasible_los.json"
    edges_csv = RESULTS_DIR / "prerequisites.csv"
    path_trace = RESULTS_DIR / "path_trace.json"

    t0 = time.time()
    export_online_ednet(sample_users=sample_users)
    t1 = time.time()
    run_reasoner(lo_csv=lo_raw, user_json=user_json,
                 feasible_csv=lo_feasible, infeasible_json=infeasible_json)
    t2 = time.time()
    explanation = run_optimizer(
        lo_csv=lo_feasible,
        edges_csv=edges_csv,
        user_json=user_json,
        infeasible_json=infeasible_json,
        output_json=path_trace,
        k=k_paths,
    )
    t3 = time.time()

    print("\n=== Experiment Summary ===")
    print(f"EdNet load + export: {1000*(t1 - t0):.1f} ms")
    print(f"Semantic reasoning:  {1000*(t2 - t1):.1f} ms")
    print(f"Optimization:        {1000*(t3 - t2):.1f} ms")
    print(f"Total:               {1000*(t3 - t0):.1f} ms")
    print(f"Real-time compliant? {explanation.get('real_time_compliant')}")


if __name__ == "__main__":
    run_single_experiment(sample_users=2000, k_paths=3)
