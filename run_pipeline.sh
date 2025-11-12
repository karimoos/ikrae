#!/bin/bash
echo "=== Starting IKRAE Full Pipeline ==="

# Activate environment if needed
# source venv/bin/activate

# Ensure output folder exists
mkdir -p experiments/results

echo "[1/3] Loading and exporting EdNet data..."
python src/ednet_loader.py || {
    echo "❌ EdNet loader failed"
    exit 1
}

echo "[2/3] Running semantic reasoning..."
python src/ikrae_reasoner.py --user_json experiments/user_context.json || {
    echo "❌ Semantic reasoning failed"
    exit 1
}

echo "[3/3] Optimizing adaptive learning paths..."
python src/ikrae_optimizer.py \
  --lo_csv experiments/results/learning_objects.csv \
  --edges_csv experiments/results/prerequisites.csv \
  --user_json experiments/user_context.json \
  --output experiments/results/path_trace.json \
  --k 3 || {
    echo "❌ Optimization failed"
    exit 1
}

echo "Pipeline complete! Results available in experiments/results/"
