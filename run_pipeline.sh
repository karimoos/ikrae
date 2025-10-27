#!/bin/bash
echo "Starting IKRAE-EdNet Pipeline..."

# 1. Load EdNet
python src/ednet_loader.py

# 2. Run reasoning
python src/ikrae_reasoner.py --user_json experiments/user_context.json

# 3. Optimize
python src/ikrae_optimizer.py \
  --lo_csv experiments/results/learning_objects.csv \
  --user_json experiments/user_context.json \
  --output experiments/results/path_trace.json

echo "Done! Check experiments/results/"
