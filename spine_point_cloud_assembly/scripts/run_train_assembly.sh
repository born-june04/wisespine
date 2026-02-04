#!/bin/bash
#
# Phase 3: Assembly Training Script
# Supports both single-GPU and multi-GPU (DDP) training
#
# Usage:
#   Single GPU: bash scripts/run_train_assembly.sh [OPTIONS]
#   Multi GPU:  torchrun --nproc_per_node=2 scripts/train_assembly.py [OPTIONS]
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
EMBEDDING_DIR="$PROJECT_ROOT/outputs/assembly_embeddings"  # Pre-extracted embeddings
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"  # Original (non-centered) point clouds
OUTPUT_DIR="$PROJECT_ROOT/outputs/assembly"
CHECKPOINT_PATH=""  # Path to checkpoint to resume from (optional)
PRETRAIN_PATH="$PROJECT_ROOT/outputs/assembly/2026-01-18_21-06-08/best_model.pth"  # Path to checkpoint to load weights only (optional)
BATCH_SIZE=8
NUM_EPOCHS=512
LEARNING_RATE=5e-5
HIDDEN_DIM=256
NUM_LAYERS=6
NUM_HEADS=8
MAX_VERTEBRAE=30
NUM_WORKERS=2  # Set to 0 to avoid CUDA multiprocessing issues
ORDERING_WEIGHT=0.0
ASSEMBLY_WEIGHT=0.5
MISSING_WEIGHT=0.0
BSPLINE_WEIGHT=0.1
BSPLINE_SMOOTH_WEIGHT=0.05
BSPLINE_K=8
S_MONOTONIC_WEIGHT=1.0
S_SMOOTH_WEIGHT=0.1
DELTA_POSE_T_WEIGHT=0.1
DELTA_POSE_ROT_WEIGHT=0.1
DELTA_CURRICULUM_EPOCHS=50
ABS_POSE_AFTER=0.1
DELTA_POSE_WARMUP=0.1
SPLINE_LATERAL_WEIGHT=0.1
SPLINE_TANGENT_SMOOTH_WEIGHT=0.1
ROOT_ANCHOR_T_WEIGHT=1.0
ROOT_ANCHOR_ROT_WEIGHT=0.5
NORMALIZE_TRANSLATION=1
FIRST_CYCLE_STEPS=20
WARMUP_STEPS=5
MAX_LR=1e-3
MIN_LR=1e-7
NUM_GPUS=1
MODEL_TYPE="spinal_field"  # "baseline" or "spinal_field"
ENABLE_DELTA_POSE=1  # 0 = disabled, 1 = enabled (only for spinal_field)
FREEZE_ORDERING_HEAD=1
TRAIN_POSE_ONLY=0

# Find available port for DDP
find_available_port() {
    local port=$1
    if [ -z "$port" ]; then
        port=12356
    fi
    while netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
        if [ $port -gt 65535 ]; then
            echo "Error: No available port found" >&2
            exit 1
        fi
    done
    echo $port
}
MASTER_PORT=$(find_available_port 12356)

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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint|--resume)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --pretrain)
            PRETRAIN_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --ordering_weight)
            ORDERING_WEIGHT="$2"
            shift 2
            ;;
        --assembly_weight)
            ASSEMBLY_WEIGHT="$2"
            shift 2
            ;;
        --missing_weight)
            MISSING_WEIGHT="$2"
            shift 2
            ;;
        --bspline_weight)
            BSPLINE_WEIGHT="$2"
            shift 2
            ;;
        --bspline_smooth_weight)
            BSPLINE_SMOOTH_WEIGHT="$2"
            shift 2
            ;;
        --bspline_k)
            BSPLINE_K="$2"
            shift 2
            ;;
        --s_monotonic_weight)
            S_MONOTONIC_WEIGHT="$2"
            shift 2
            ;;
        --s_smooth_weight)
            S_SMOOTH_WEIGHT="$2"
            shift 2
            ;;
        --delta_pose_t_weight)
            DELTA_POSE_T_WEIGHT="$2"
            shift 2
            ;;
        --delta_pose_rot_weight)
            DELTA_POSE_ROT_WEIGHT="$2"
            shift 2
            ;;
        --delta_curriculum_epochs)
            DELTA_CURRICULUM_EPOCHS="$2"
            shift 2
            ;;
        --abs_pose_after)
            ABS_POSE_AFTER="$2"
            shift 2
            ;;
        --delta_pose_warmup)
            DELTA_POSE_WARMUP="$2"
            shift 2
            ;;
        --spline_lateral_weight)
            SPLINE_LATERAL_WEIGHT="$2"
            shift 2
            ;;
        --spline_tangent_smooth_weight)
            SPLINE_TANGENT_SMOOTH_WEIGHT="$2"
            shift 2
            ;;
        --root_anchor_t_weight)
            ROOT_ANCHOR_T_WEIGHT="$2"
            shift 2
            ;;
        --root_anchor_rot_weight)
            ROOT_ANCHOR_ROT_WEIGHT="$2"
            shift 2
            ;;
        --normalize_translation)
            NORMALIZE_TRANSLATION=1
            shift
            ;;
        --no_normalize_translation)
            NORMALIZE_TRANSLATION=0
            shift
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --enable_delta_pose)
            ENABLE_DELTA_POSE=1
            shift
            ;;
        --freeze_ordering_head)
            FREEZE_ORDERING_HEAD=1
            shift
            ;;
        --train_pose_only)
            TRAIN_POSE_ONLY=1
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
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
export CUDA_VISIBLE_DEVICES="0"

# Create output directory for logging
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "============================================================"
echo "Starting Assembly Training"
echo "============================================================"
echo "Embedding directory: $EMBEDDING_DIR"
echo "Point cloud directory: $POINT_CLOUD_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Resume from checkpoint: $CHECKPOINT_PATH"
fi
if [ -n "$PRETRAIN_PATH" ]; then
    echo "Pretrain weights from: $PRETRAIN_PATH"
fi
echo "Log file: $LOG_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Number of workers: $NUM_WORKERS"
echo "Model type: $MODEL_TYPE"
if [ "$MODEL_TYPE" = "spinal_field" ]; then
    echo "Delta pose enabled: $([ "$ENABLE_DELTA_POSE" = "1" ] && echo "Yes" || echo "No")"
fi
echo "Freeze ordering head: $([ "$FREEZE_ORDERING_HEAD" = "1" ] && echo "Yes" || echo "No")"
echo "Train pose only: $([ "$TRAIN_POSE_ONLY" = "1" ] && echo "Yes" || echo "No")"
echo "B-spline Weight: $BSPLINE_WEIGHT"
echo "B-spline Smooth Weight: $BSPLINE_SMOOTH_WEIGHT"
echo "B-spline K: $BSPLINE_K"
echo "s-monotonic Weight: $S_MONOTONIC_WEIGHT"
echo "s-smooth Weight: $S_SMOOTH_WEIGHT"
echo "Delta Pose t Weight: $DELTA_POSE_T_WEIGHT"
echo "Delta Pose rot Weight: $DELTA_POSE_ROT_WEIGHT"
echo "Delta Curriculum Epochs: $DELTA_CURRICULUM_EPOCHS"
echo "Abs Pose After: $ABS_POSE_AFTER"
echo "Delta Pose Warmup: $DELTA_POSE_WARMUP"
echo "Spline Lateral Weight: $SPLINE_LATERAL_WEIGHT"
echo "Spline Tangent Smooth Weight: $SPLINE_TANGENT_SMOOTH_WEIGHT"
echo "Root Anchor t Weight: $ROOT_ANCHOR_T_WEIGHT"
echo "Root Anchor rot Weight: $ROOT_ANCHOR_ROT_WEIGHT"
echo "Normalize Translation: $NORMALIZE_TRANSLATION"
echo "============================================================"
echo ""

# Run training

echo "Using multi-GPU training (DDP, $NUM_GPUS GPUs)"
echo "Master port: $MASTER_PORT"

torchrun \
    --nnode=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr="localhost" \
    --master_port="$MASTER_PORT" \
        scripts/train_assembly.py \
        --embedding_dir "$EMBEDDING_DIR" \
        --point_cloud_dir "$POINT_CLOUD_DIR" \
        --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --max_vertebrae "$MAX_VERTEBRAE" \
    --num_workers "$NUM_WORKERS" \
    --ordering_weight "$ORDERING_WEIGHT" \
    --assembly_weight "$ASSEMBLY_WEIGHT" \
    --missing_weight "$MISSING_WEIGHT" \
    --bspline_weight "$BSPLINE_WEIGHT" \
    --bspline_smooth_weight "$BSPLINE_SMOOTH_WEIGHT" \
    --bspline_k "$BSPLINE_K" \
        --s_monotonic_weight "$S_MONOTONIC_WEIGHT" \
        --s_smooth_weight "$S_SMOOTH_WEIGHT" \
        --delta_pose_t_weight "$DELTA_POSE_T_WEIGHT" \
        --delta_pose_rot_weight "$DELTA_POSE_ROT_WEIGHT" \
        --delta_curriculum_epochs "$DELTA_CURRICULUM_EPOCHS" \
        --abs_pose_after "$ABS_POSE_AFTER" \
        --delta_pose_warmup "$DELTA_POSE_WARMUP" \
        --spline_lateral_weight "$SPLINE_LATERAL_WEIGHT" \
        --spline_tangent_smooth_weight "$SPLINE_TANGENT_SMOOTH_WEIGHT" \
        --root_anchor_t_weight "$ROOT_ANCHOR_T_WEIGHT" \
        --root_anchor_rot_weight "$ROOT_ANCHOR_ROT_WEIGHT" \
        $( [[ "$NORMALIZE_TRANSLATION" -eq 1 ]] && echo "--normalize_translation" ) \
    --first_cycle_steps "$FIRST_CYCLE_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
        --max_lr "$MAX_LR" \
        --min_lr "$MIN_LR" \
        --model_type "$MODEL_TYPE" \
        ${ENABLE_DELTA_POSE:+--enable_delta_pose} \
        ${FREEZE_ORDERING_HEAD:+--freeze_ordering_head} \
        ${TRAIN_POSE_ONLY:+--train_pose_only} \
        ${CHECKPOINT_PATH:+--resume "$CHECKPOINT_PATH"} \
        ${PRETRAIN_PATH:+--pretrain "$PRETRAIN_PATH"}


echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "Log file saved to: $LOG_FILE"
echo "Check the log file for detailed training information."
echo "============================================================"

