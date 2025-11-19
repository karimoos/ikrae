#!/usr/bin/env bash
set -e

echo "======================================"
echo "        IKRAE FULL PIPELINE           "
echo "======================================"

# 1/3: Build learning_objects.csv + prerequisites.csv from local KT3 zip
echo "[1/3] Loading EdNet and building LO tables..."
python3 -m src.ednet_loader

# 2/3: Semantic filtering (OWL/SWRL → feasible graph Gf)
echo "[2/3] Running semantic reasoner..."
python3 -m src.ikrae_reasoner

# 3/3: Graph optimization experiments (Dijkstra + k-shortest paths)
echo "[3/3] Running optimization experiments..."
python3 -m src.run_experiments

echo "======================================"
echo "IKRAE pipeline finished successfully."
echo "======================================"
