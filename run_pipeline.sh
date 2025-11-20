#!/usr/bin/env bash
set -e

echo "======================================"
echo "        IKRAE FULL PIPELINE           "
echo "======================================"

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT/data"
ZIP_FILE="$DATA_DIR/ikrae_kt3_clean.zip"
ZENODO_URL="https://zenodo.org/record/17664110/files/ikrae_kt3_clean.zip?download=1"

mkdir -p "$DATA_DIR"

# ---------------------------------------
# Step 0: Fetch dataset if missing
# ---------------------------------------
if [ ! -f "$ZIP_FILE" ]; then
    echo "[0/3] Dataset missing → downloading from Zenodo..."
    wget -O "$ZIP_FILE" "$ZENODO_URL"
    echo "[0/3] Extracting dataset..."
    unzip -o "$ZIP_FILE" -d "$DATA_DIR"
else
    echo "[0/3] Dataset already present ✓"
fi

# ---------------------------------------
# Step 1: Build LO + prerequisites
# ---------------------------------------
echo "[1/3] Loading EdNet and building LO tables..."
python3 -m src.ednet_loader

# ---------------------------------------
# Step 2: Semantic Reasoning
# ---------------------------------------
echo "[2/3] Running semantic reasoner..."
python3 -m src.ikrae_reasoner

# ---------------------------------------
# Step 3: Graph Optimization
# ---------------------------------------
echo "[3/3] Running optimization experiments..."
python3 -m src.run_experiments

echo "======================================"
echo "IKRAE pipeline finished successfully."
echo "======================================"
