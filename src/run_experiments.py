# src/run_experiments.py
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ednet_loader import load_ednet_kt3, build_lo_stats, export_for_ikrae

RESULTS = Path("../experiments/results")
RESULTS.mkdir(exist_ok=True)

def run_ikrae_pipeline(lo_count):
    start = time.time()
    # Simulate IKRAE (replace with actual Java/Python call)
    cmd = f"python ikrae_optimizer.py --lo_count {lo_count} --mode dijkstra"
    subprocess.run(cmd, shell=True, check=True)
    return time.time() - start

def benchmark_scalability():
    sizes = [200, 1000, 5000, 10000, 50000, 100000]
    times = []
    for size in sizes:
        t = run_ikrae_pipeline(size)
        times.append(t)
        print(f"{size} LOs: {t:.3f}s")
    
    df = pd.DataFrame({"LOs": sizes, "Time_s": times})
    df.to_csv(RESULTS / "scalability.csv", index=False)
    
    plt.figure(figsize=(8,5))
    plt.plot(df['LOs'], df['Time_s'], marker='o')
    plt.xlabel("Number of Learning Objects")
    plt.ylabel("Path Computation Time (s)")
    plt.title("IKRAE-EdNet Scalability")
    plt.savefig(RESULTS / "scalability.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    print("Loading EdNet...")
    questions = pd.read_csv("../data/ednet/content/questions.csv")
    kt3 = load_ednet_kt3(sample_users=10000)
    export_for_ikrae(kt3, questions, pd.DataFrame())  # lectures optional
    
    print("Running scalability benchmark...")
    benchmark_scalability()
    
    print("Done! Results in experiments/results/")
