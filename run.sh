#!/usr/bin/env bash
set -euo pipefail

# AIgnition Forecaster — scored pipeline entry point.
#   run.sh [DATA_DIR] [MODEL_PATH] [OUTPUT_PATH] [BUDGETS_JSON]
# Runs feature generation + prediction end-to-end. No network access, no
# interactivity. The AI insights layer lives OUTSIDE this scored path
# (src/anomalies.py + src/llm_insights.py, used by the demo dashboard only).

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"
# Optional: per-channel budget multipliers as JSON, e.g. '{"google": 1.2}'
BUDGETS="${4:-}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=========================================="
echo "AIgnition Forecaster - Starting Pipeline"
echo "=========================================="
echo "Data dir:     $DATA_DIR"
echo "Model path:   $MODEL_PATH"
echo "Output path:  $OUTPUT_PATH"
echo "Budgets:      ${BUDGETS:-none}"
echo "=========================================="

echo ""
echo "[1/2] Generating features..."
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet \
    --health-out "$(dirname "$OUTPUT_PATH")/data_health.json"

echo ""
echo "[2/2] Running forecasts and predictions..."
if [ -n "$BUDGETS" ]; then
    python src/predict.py \
        --features features.parquet \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_PATH" \
        --budgets "$BUDGETS"
else
    python src/predict.py \
        --features features.parquet \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_PATH"
fi

echo ""
echo "=========================================="
echo "Done. Predictions written to $OUTPUT_PATH"
echo "=========================================="
