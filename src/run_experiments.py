"""
IKRAE-EdNet Experimental Benchmark
----------------------------------
Runs full preprocessing → optimization pipeline
and measures scalability across learning object (LO) sizes.

Outputs:
 - experiments/results/scalability.csv
 - experiments/results/scalability.png
 - experiments/results/path_trace.json
"""

import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ednet_loader import load_ednet_kt3, build_lo_stats, export_for_ikrae

RESULTS = Path("../experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1.  pipeline runner
# ---------------------------------------------------------------------
def run_ikrae_pipeline(lo_csv, user_json, out_json, edges_csv=None, k=3):
    """Run the full IKRAE optimizer pipeline and return runtime in seconds."""
    start = time.time()

    cmd = (
        f"python src/ikrae_optimizer.py "
        f"--lo_csv {lo_csv} "
        f"--user_json {user_json} "
        f"--output {out_json} "
        f"--k {k} "
    )

    # Optional prerequisite edges
    if edges_csv:
        cmd += f"--edges_csv {edges_csv} "

    subprocess.run(cmd, shell=True, check=True)
    return time.time() - start


# ---------------------------------------------------------------------
# 2. Scalability benchmark
# ---------------------------------------------------------------------
def benchmark_scalability(lo_csv, user_json, edges_csv, out_dir=RESULTS):
    """Run the optimizer for increasing LO sizes and record runtime."""
    sizes = [200, 1000, 5000, 10000, 50000, 100000]
    times = []

    full_df = pd.read_csv(lo_csv)
    for size in sizes:
        # Sample subset of LOs to simulate different graph sizes
        sampled_csv = out_dir / f"learning_objects_{size}.csv"
        full_df.sample(n=min(size, len(full_df)), random_state=42).to_csv(sampled_csv, index=False)

        out_json = out_dir / f"path_trace_{size}.json"
        print(f"\nRunning pipeline with {size} learning objects...")
        elapsed = run_ikrae_pipeline(str(sampled_csv), user_json, str(out_json), edges_csv)
        times.append(elapsed)
        print(f"{size} LOs completed in {elapsed:.3f}s")

    # Save results
    df = pd.DataFrame({"LOs": sizes, "Time_s": times})
    df.to_csv(out_dir / "scalability.csv", index=False)

    # Plot scalability curve
    plt.figure(figsize=(8, 5))
    plt.plot(df['LOs'], df['Time_s'], marker='o')
    plt.xlabel("Number of Learning Objects")
    plt.ylabel("Path Computation Time (s)")
    plt.title("IKRAE-EdNet Scalability Benchmark")
    plt.grid(True)
    plt.savefig(out_dir / "scalability.png", dpi=300, bbox_inches='tight')
    print(f"\nScalability results saved to {out_dir}/scalability.png")


# ---------------------------------------------------------------------
# 3. Main experimental entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== IKRAE-EdNet Experiments ===")

    # -----------------------------------------------------------------
    # Step 1. Load and export EdNet data
    # -----------------------------------------------------------------
    print("Loading EdNet-KT3 subset...")
    try:
        questions = pd.read_csv("../data/ednet/content/questions.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find '../data/ednet/content/questions.csv' — please place EdNet files under data/ednet/content/")

    kt3 = load_ednet_kt3(sample_users=10000)

    # Dummy lectures dataset (optional)
    lectures = pd.DataFrame(columns=["lecture_id", "video_length"])

    print("Exporting data for IKRAE...")
    export_for_ikrae(kt3, questions, lectures, output_dir=RESULTS)

    lo_csv = str(RESULTS / "learning_objects.csv")
    edges_csv = str(RESULTS / "prerequisites.csv") if (RESULTS / "prerequisites.csv").exists() else None
    user_json = "../experiments/user_context.json"
    out_json = str(RESULTS / "path_trace.json")

    # -----------------------------------------------------------------
    # Step 2. Run initial pipeline for baseline path generation
    # -----------------------------------------------------------------
    print("\nRunning baseline IKRAE pipeline...")
    t = run_ikrae_pipeline(lo_csv, user_json, out_json, edges_csv)
    print(f"Baseline path generation completed in {t:.2f} s")

    # -----------------------------------------------------------------
    # Step 3. Scalability benchmark
    # -----------------------------------------------------------------
    print("\nRunning scalability benchmark...")
    benchmark_scalability(lo_csv, user_json, edges_csv)

    print(" All experiments completed. Results are available in:")
    print(f"   {RESULTS.resolve()}")
