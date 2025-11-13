#!/usr/bin/env bash
set -e

# Resolve repo root
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

echo "[1/3] Online EdNet → learning_objects.csv & prerequisites.csv"
python src/ednet_loader.py --sample_users 5000

echo "[2/3] Semantic reasoning → feasible LOs + infeasible trace"
python src/ikrae_reasoner.py \
  --lo_csv experiments/results/learning_objects.csv \
  --user_json experiments/user_context.json \
  --feasible_csv experiments/results/learning_objects_feasible.csv \
  --infeasible_json experiments/results/infeasible_los.json

echo "[3/3] Optimization → path_trace.json"
python src/ikrae_optimizer.py \
  --lo_csv experiments/results/learning_objects_feasible.csv \
  --edges_csv experiments/results/prerequisites.csv \
  --user_json experiments/user_context.json \
  --infeasible_json experiments/results/infeasible_los.json \
  --output experiments/results/path_trace.json \
  --k 3

echo "IKRAE pipeline finished. See experiments/results/path_trace.json"
