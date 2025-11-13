#!/bin/bash

echo "======================================"
echo "        IKRAE FULL PIPELINE           "
echo "======================================"

# Exit on error
set -e

# 1. Online EdNet extraction
echo "[1/3] Loading EdNet and building LO tables..."
python3 src/ednet_loader.py

# 2. Semantic filtering
echo "[2/3] Applying semantic filtering..."
python3 src/ikrae_reasoner.py

# 3. Optimization
echo "[3/3] Running graph optimizer..."
python3 src/ikrae_optimizer.py

echo "======================================"
echo "     Pipeline completed successfully   "
echo "======================================"

echo "Generated files in: experiments/results/"
echo " - learning_objects.csv"
echo " - prerequisites.csv"
echo " - learning_objects_feasible.csv"
echo " - infeasible_los.json"
echo " - path_trace.json"
