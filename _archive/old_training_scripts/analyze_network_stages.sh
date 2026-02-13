#!/bin/bash
# Analyze network stages: visualize intermediate features

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
# Use baseline model (12-03_21-12) for analysis
MODEL_PATH="/gscratch/scrubbed/june0604/vindr/outputs/VerSe_COARSE_FINE_12-05_13-22/stage1/best_model.pth"
DATA_DIR="/gscratch/scrubbed/june0604/vindr/VerSe/processed"
OUTPUT_DIR="/gscratch/scrubbed/june0604/vindr/outputs/analysis/network_stages_small_vrm1"
NUM_SAMPLES=3
DEVICE="cuda:1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Network Stages Analysis"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "=========================================="

cd "$PROJECT_DIR" && python workspace/utils/visualize_network_stages.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --model_size "small" \
    --use_vrm 1

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

