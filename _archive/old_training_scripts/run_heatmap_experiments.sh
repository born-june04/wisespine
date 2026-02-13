#!/bin/bash
#
# Heatmap-based Two-Stage Experiments
# Phase 1: Heatmap + VRM vs Heatmap - VRM
#
# MICCAI 2026 Research - SpineMedNeXt
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE=2
LEARNING_RATE=1e-4
EPOCHS=256
MODEL_SIZE="small"  # tiny, small, base
USE_BFLOAT16=1
NUM_WORKERS=4
SAVE_VALIDATION_DETAILS=1

# ============================================================================
# Experiment 1: Heatmap + VRM
# ============================================================================

echo "============================================================"
echo "Experiment 1: Heatmap-based Two-Stage + VRM"
echo "============================================================"

USE_VRM=1
EXPERIMENT_NAME="HEATMAP_VRM1"

python workspace/trainers/coarse_fine_trainer.py \
    --processed_dir "$PROJECT_ROOT/VerSe/processed" \
    --save_path "$PROJECT_ROOT/outputs" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --model_size $MODEL_SIZE \
    --use_vrm $USE_VRM \
    --use_bfloat16 $USE_BFLOAT16 \
    --num_workers $NUM_WORKERS \
    --save_validation_details $SAVE_VALIDATION_DETAILS \
    2>&1 | tee "outputs/${EXPERIMENT_NAME}_train.log"

echo ""
echo "Experiment 1 Complete!"
echo ""

# ============================================================================
# Experiment 2: Heatmap - VRM (Ablation)
# ============================================================================

echo "============================================================"
echo "Experiment 2: Heatmap-based Two-Stage - VRM (Ablation)"
echo "============================================================"

USE_VRM=0
EXPERIMENT_NAME="HEATMAP_VRM0"

python workspace/trainers/coarse_fine_trainer.py \
    --processed_dir "$PROJECT_ROOT/VerSe/processed" \
    --save_path "$PROJECT_ROOT/outputs" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --model_size $MODEL_SIZE \
    --use_vrm $USE_VRM \
    --use_bfloat16 $USE_BFLOAT16 \
    --num_workers $NUM_WORKERS \
    --save_validation_details $SAVE_VALIDATION_DETAILS \
    2>&1 | tee "outputs/${EXPERIMENT_NAME}_train.log"

echo ""
echo "Experiment 2 Complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================"
echo "All Experiments Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - outputs/HEATMAP_VRM1_*"
echo "  - outputs/HEATMAP_VRM0_*"
echo ""
echo "Next steps:"
echo "  1. Check validation details in experiment directories"
echo "  2. Run inference with visualization:"
echo "     python workspace/inference/full_volume_segmentation.py \\"
echo "       --input_path <path_to_ct> \\"
echo "       --experiment_dir outputs/HEATMAP_VRM1_* \\"
echo "       --output_dir outputs/inference_results/test \\"
echo "       --model_size $MODEL_SIZE \\"
echo "       --use_vrm 1"
echo ""

