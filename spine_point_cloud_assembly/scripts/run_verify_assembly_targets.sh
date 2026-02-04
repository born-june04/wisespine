#!/bin/bash
#
# Verify Assembly Training Targets
# Checks if translation targets are correctly computed from original points
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
EMBEDDING_DIR="$PROJECT_ROOT/outputs/assembly_embeddings"
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"
NUM_SAMPLES=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --embedding_dir)
            EMBEDDING_DIR="$2"
            shift 2
            ;;
        --point_cloud_dir)
            POINT_CLOUD_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
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
echo "Verifying Assembly Training Targets"
echo "============================================================"
echo "Embedding directory: $EMBEDDING_DIR"
echo "Point cloud directory: $POINT_CLOUD_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "============================================================"
echo ""

python scripts/verify_assembly_targets.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --point_cloud_dir "$POINT_CLOUD_DIR" \
    --num_samples "$NUM_SAMPLES"

echo ""
echo "============================================================"
echo "Verification complete!"
echo "============================================================"

