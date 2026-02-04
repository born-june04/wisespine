#!/bin/bash

# Script to visualize assembled spine using PyVista

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
ENCODER_PATH="${ENCODER_PATH:-$PROJECT_ROOT/outputs/embeddings/2026-01-11_17-50-10/best_model.pth}"
ASSEMBLY_PATH="${ASSEMBLY_PATH:-$PROJECT_ROOT/outputs/assembly/2026-01-15_17-05-30/best_model.pth}"
POINT_CLOUD_DIR="${POINT_CLOUD_DIR:-$PROJECT_ROOT/outputs/point_clouds}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$PROJECT_ROOT/outputs/assembly_embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/assembly/visualization}"
MESH_DIR="${MESH_DIR:-$PROJECT_ROOT/outputs/meshes}"
CT_DIR="${CT_DIR:-/gscratch/scrubbed/june0604/vindr/VerSe/processed}"
SPLIT="${SPLIT:-test}"
# Multiple subjects to visualize (diverse: some with many vertebrae, some with few)
# Auto-select diverse subjects if not specified
if [ -z "$SUBJECT_IDS" ]; then
    METADATA_FILE="$EMBEDDING_DIR/metadata.json"
    if [ -f "$METADATA_FILE" ]; then
        echo "Auto-selecting diverse subjects from metadata..."
        SUBJECT_IDS=$(python scripts/select_diverse_subjects.py "$METADATA_FILE" 5)
        echo "Selected subjects: $SUBJECT_IDS"
    else
        # Fallback to default subjects
        SUBJECT_IDS="sub-verse808 sub-verse001 sub-verse050 sub-verse100 sub-verse200"
        echo "WARNING: metadata.json not found, using default subjects"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Assembly Visualization"
echo "=========================================="
echo "Encoder: $ENCODER_PATH"
echo "Assembly Model: $ASSEMBLY_PATH"
echo "Point Cloud Dir: $POINT_CLOUD_DIR"
echo "Embedding Dir: $EMBEDDING_DIR"
echo "Mesh Dir: $MESH_DIR"
if [ -n "$CT_DIR" ]; then
    echo "CT Dir: $CT_DIR"
fi
echo "Output Dir: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo "Subjects to visualize: $SUBJECT_IDS"
echo "=========================================="

cd "$PROJECT_ROOT"

# Process each subject
for SUBJECT_ID in $SUBJECT_IDS; do
    echo ""
    echo "=========================================="
    echo "Processing subject: $SUBJECT_ID"
    echo "=========================================="
    
    # Run visualization for this subject
    python scripts/visualize_assembly.py \
        --encoder_path "$ENCODER_PATH" \
        --assembly_path "$ASSEMBLY_PATH" \
        --point_cloud_dir "$POINT_CLOUD_DIR" \
        --embedding_dir "$EMBEDDING_DIR" \
        --mesh_dir "$MESH_DIR" \
        ${CT_DIR:+--ct_dir "$CT_DIR"} \
        --split "$SPLIT" \
        --subject_id "$SUBJECT_ID" \
        --output_path "$OUTPUT_DIR/assembled_spine_${SUBJECT_ID}" \
        --save_format ply \
        --save_sagittal \
        --no_show \
        --device cuda || echo "WARNING: Failed to process $SUBJECT_ID"
done

echo ""
echo "=========================================="
echo "Visualization Complete!"
echo "=========================================="
echo "Point cloud files saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
for SUBJECT_ID in $SUBJECT_IDS; do
    if [ -f "$OUTPUT_DIR/assembled_spine_${SUBJECT_ID}.ply" ]; then
        echo "  ✓ assembled_spine_${SUBJECT_ID}.ply"
        echo "    assembled_spine_${SUBJECT_ID}.json"
    fi
done
echo ""
echo "To view locally:"
echo "  import pyvista as pv"
echo "  mesh = pv.read('$OUTPUT_DIR/assembled_spine_<SUBJECT_ID>.ply')"
echo "  mesh.plot(scalars='vertebra_type', cmap='tab20')"
echo "=========================================="

