#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"
# Optional: per-channel budget multipliers as JSON string
# Example: '{"google": 1.2, "meta": 0.9}'
BUDGETS="${4:-}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$MODEL_PATH")"

echo "=========================================="
echo "AIgnition Forecaster - Starting Pipeline"
echo "=========================================="
echo "Data dir:     $DATA_DIR"
echo "Model path:   $MODEL_PATH"
echo "Output path:  $OUTPUT_PATH"
echo "Budgets:      ${BUDGETS:-none}"
echo "=========================================="

# 1. Generate features from raw data
echo ""
echo "[1/3] Generating features..."
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

# 2. Run forecasting + prediction, save model and predictions
echo ""
echo "[2/3] Running forecasts and predictions..."
BUDGET_ARG=""
if [ -n "$BUDGETS" ]; then
    BUDGET_ARG="--budgets '$BUDGETS'"
fi

python src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    ${BUDGET_ARG}

# 3. Generate AI insights (optional - won't fail the pipeline if API key missing)
echo ""
echo "[3/3] Generating AI insights..."
python src/llm_insights.py || echo "  (AI insights skipped - check API key)"

echo ""
echo "=========================================="
echo "Done. Predictions written to $OUTPUT_PATH"
echo "=========================================="