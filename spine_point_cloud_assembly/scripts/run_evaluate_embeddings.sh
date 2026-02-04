#!/bin/bash
# Run embedding quality evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
#/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/embeddings/2026-01-11_17-50-10

MODEL_PATH="$PROJECT_ROOT/outputs/embeddings/2026-01-11_17-50-10/best_model.pth"
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"
OUTPUT_DIR="$PROJECT_ROOT/outputs/embeddings/evaluation"

echo "=========================================="
echo "Encoder Embedding Quality Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Point cloud directory: $POINT_CLOUD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

python evaluate_embeddings.py \
    --model_path "$MODEL_PATH" \
    --point_cloud_dir "$POINT_CLOUD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --use_amp \
    --tsne_samples 5000 \
    --rotation_test_samples 100

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

