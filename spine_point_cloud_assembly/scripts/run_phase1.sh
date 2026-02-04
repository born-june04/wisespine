#!/bin/bash
# Phase 1: Geometry Pipeline - Complete workflow

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# Configuration
MASK_DIR="${1:-$PARENT_ROOT/VerSe/processed}"
OUTPUT_BASE="${2:-$PROJECT_ROOT/outputs}"
SPACING="1.0 1.0 1.0"
NUM_POINTS=2048
SPLIT="${3:-train}"
SAMPLE_FRACTION="${4:-1.0}"

# Convert to absolute paths
MASK_DIR="$(cd "$MASK_DIR" 2>/dev/null && pwd || echo "$MASK_DIR")"
OUTPUT_BASE="$(mkdir -p "$OUTPUT_BASE" && cd "$OUTPUT_BASE" && pwd)"

echo "=========================================="
echo "Phase 1: Geometry Pipeline"
echo "=========================================="
echo "Mask directory: $MASK_DIR"
echo "Output base: $OUTPUT_BASE"
echo "Split: $SPLIT"
echo "Sample fraction: $SAMPLE_FRACTION"
echo ""

# Change to script directory to ensure relative imports work
cd "$SCRIPT_DIR"

# Step 1: Extract meshes
echo "Step 1/3: Extracting meshes from segmentation masks..."
python extract_meshes.py \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_BASE/meshes" \
    --spacing $SPACING \
    --split "$SPLIT" \
    --sample_fraction "$SAMPLE_FRACTION"

echo ""
echo "Step 2/3: Sampling point clouds from meshes..."
python sample_points.py \
    --mesh_dir "$OUTPUT_BASE/meshes" \
    --output_dir "$OUTPUT_BASE/point_clouds" \
    --num_points $NUM_POINTS \
    --method uniform

echo ""
echo "Step 3/3: Computing directional features..."
python compute_features.py \
    --point_cloud_dir "$OUTPUT_BASE/point_clouds" \
    --output_dir "$OUTPUT_BASE/point_clouds" \
    --compute_normals \
    --compute_curvature \
    --k_nn 20

echo ""
echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo "Meshes: $OUTPUT_BASE/meshes"
echo "Point clouds: $OUTPUT_BASE/point_clouds"
echo ""

