#!/bin/bash
# Extract encoder embeddings for assembly training

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"
ENCODER_PATH="$PROJECT_ROOT/outputs/embeddings/2026-01-11_17-50-10/best_model.pth"
OUTPUT_DIR="$PROJECT_ROOT/outputs/assembly_embeddings"
MAX_POINTS=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --point_cloud_dir)
            POINT_CLOUD_DIR="$2"
            shift 2
            ;;
        --encoder_path)
            ENCODER_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_points)
            MAX_POINTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "============================================================"
echo "Extracting Encoder Embeddings for Assembly Training"
echo "============================================================"
echo "Point cloud directory: $POINT_CLOUD_DIR"
echo "Encoder path: $ENCODER_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Max points: $MAX_POINTS"
echo "============================================================"
echo ""

# Run extraction
python scripts/extract_assembly_embeddings.py \
    --point_cloud_dir "$POINT_CLOUD_DIR" \
    --encoder_path "$ENCODER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_points "$MAX_POINTS" \
    --device cuda

echo ""
echo "============================================================"
echo "Extraction Complete!"
echo "============================================================"
echo "Embeddings saved to: $OUTPUT_DIR"
echo "You can now run assembly training with:"
echo "  ./scripts/run_train_assembly.sh --embedding_dir $OUTPUT_DIR"
echo "============================================================"

