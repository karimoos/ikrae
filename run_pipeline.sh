#!/usr/bin/env bash
set -e

echo "======================================"
echo "        IKRAE FULL PIPELINE           "
echo "======================================"

# 1/3: Build learning_objects.csv + prerequisites.csv from local KT3 zip
echo "[1/3] Loading EdNet and building LO tables..."
python3 src/ednet_loader.py

# 2/3: Semantic filtering (OWL/SWRL → feasible graph Gf)
echo "[2/3] Running semantic reasoner..."
python3 src/ikrae_reasoner.py

# 3/3: Graph optimization experiments (Dijkstra + k-shortest paths)
echo "[3/3] Running optimization experiments..."
python3 src/run_experiments.py

echo "======================================"
echo "IKRAE pipeline finished successfully."
echo "======================================"
