"""
IKRAE-EdNet Experimental Benchmark
----------------------------------
Runs full preprocessing → optimization pipeline
and measures scalability across learning object (LO) sizes.
"""

import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ednet_loader import load_ednet_kt3, export_for_ikrae

RESULTS = Path("../experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def run_ikrae_pipeline(lo_csv, user_json, out_json, edges_csv=None, k=3):
    """Run the IKRAE optimizer and return runtime in seconds."""
    start = time.time()
    cmd = (
        f"python src/ikrae_optimizer.py "
        f"--lo_csv {lo_csv} "
        f"--user_json {user_json} "
        f"--output {out_json} "
        f"--k {k} "
    )
    if edges_csv:
        cmd += f"--edges_csv {edges_csv} "
    subprocess.run(cmd, shell=True, check=True)
    return time.time() - start


def benchmark_scalability(lo_csv, user_json, edges_csv, out_dir=RESULTS):
    """Run optimizer for increasing LO sizes and record runtime."""
    sizes = [200, 1000, 5000, 10000, 50000, 100000]
    times = []
    full_df = pd.read_csv(lo_csv)

    for size in sizes:
        sampled_csv = out_dir / f"learning_objects_{size}.csv"
        full_df.sample(n=min(size, len(full_df)), random_state=42).to_csv(sampled_csv, index=False)
        out_json = out_dir / f"path_trace_{size}.json"
        print(f"\n▶ Running pipeline with {size} LOs...")
        elapsed = run_ikrae_pipeline(str(sampled_csv), user_json, str(out_json), edges_csv)
        times.append(elapsed)
        print(f"⏱ {size} LOs completed in {elapsed:.3f}s")

    df = pd.DataFrame({"LOs": sizes, "Time_s": times})
    df.to_csv(out_dir / "scalability.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["LOs"], df["Time_s"], marker="o")
    plt.xlabel("Number of Learning Objects")
    plt.ylabel("Computation Time (s)")
    plt.title("IKRAE-EdNet Scalability Benchmark")
    plt.grid(True)
    plt.savefig(out_dir / "scalability.png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Scalability results saved to {out_dir}/scalability.png")


if __name__ == "__main__":
    print("=== IKRAE-EdNet Experiments ===")
    questions = pd.read_csv("../data/ednet/content/questions.csv")
    kt3 = load_ednet_kt3(sample_users=10000)
    lectures = pd.DataFrame(columns=["lecture_id", "video_length"])
    export_for_ikrae(kt3, questions, lectures, output_dir=RESULTS)

    lo_csv = str(RESULTS / "learning_objects.csv")
    edges_csv = str(RESULTS / "prerequisites.csv")
    user_json = "../experiments/user_context.json"
    out_json = str(RESULTS / "path_trace.json")

    print("\n▶ Running baseline IKRAE pipeline...")
    t = run_ikrae_pipeline(lo_csv, user_json, out_json, edges_csv)
    print(f"✅ Baseline path generation: {t:.2f}s")

    print("\n▶ Running scalability benchmark...")
    benchmark_scalability(lo_csv, user_json, edges_csv)

    print(f"\n✅ All experiments done. Results in {RESULTS.resolve()}")
