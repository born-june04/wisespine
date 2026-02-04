#!/bin/bash
# Run assembly model evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
## Baseline
# MODEL_PATH="$PROJECT_ROOT/outputs/assembly/2026-01-12_13-38-26/best_model.pth"
## Spinal Field
MODEL_PATH="$PROJECT_ROOT/outputs/assembly/2026-01-12_21-07-15/best_model.pth"

EMBEDDING_DIR="$PROJECT_ROOT/outputs/assembly_embeddings"
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"
OUTPUT_DIR="$PROJECT_ROOT/outputs/assembly/evaluation"
SPLIT="test"
BATCH_SIZE=16
NUM_WORKERS=0
USE_AMP=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --embedding_dir)
            EMBEDDING_DIR="$2"
            shift 2
            ;;
        --point_cloud_dir)
            POINT_CLOUD_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --no_amp)
            USE_AMP=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Assembly Model Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Embedding directory: $EMBEDDING_DIR"
echo "Point cloud directory: $POINT_CLOUD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

python evaluate_assembly.py \
    --model_path "$MODEL_PATH" \
    --embedding_dir "$EMBEDDING_DIR" \
    --point_cloud_dir "$POINT_CLOUD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    $( [[ "$USE_AMP" -eq 1 ]] && echo "--use_amp" ) \
    --num_workers "$NUM_WORKERS"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

