#!/bin/bash

# Script to visualize PLY files with matplotlib

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
VISUALIZATION_DIR="${VISUALIZATION_DIR:-$PROJECT_ROOT/outputs/assembly/visualization}"
PLY_FILE="${PLY_FILE:-}"  # Specify PLY file path, or will use first available

echo "=========================================="
echo "PLY File Visualization (Matplotlib)"
echo "=========================================="
echo "Visualization Dir: $VISUALIZATION_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"

# Find PLY file if not specified
if [ -z "$PLY_FILE" ]; then
    # Find first PLY file in visualization directory
    PLY_FILE=$(find "$VISUALIZATION_DIR" -name "*.ply" -type f | head -1)
    if [ -z "$PLY_FILE" ]; then
        echo "ERROR: No PLY file found in $VISUALIZATION_DIR"
        echo "Please specify PLY file with: PLY_FILE=/path/to/file.ply $0"
        exit 1
    fi
    echo "Using first available PLY file: $PLY_FILE"
else
    echo "Using specified PLY file: $PLY_FILE"
fi

# Run visualization
python scripts/visualize_ply_matplotlib.py \
    --ply_path "$PLY_FILE" \
    --output_dir "$VISUALIZATION_DIR" \
    --use_mesh \
    --show_individual \
    --show_assembled

echo ""
echo "=========================================="
echo "Visualization Complete!"
echo "=========================================="
echo "Images saved to: $VISUALIZATION_DIR"
echo "  - *_individual_vertebrae.png (one vertebra per subplot)"
echo "  - *_assembled.png (complete spine)"
echo "=========================================="

