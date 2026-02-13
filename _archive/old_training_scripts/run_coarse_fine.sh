#!/bin/bash
#
# Coarse-to-Fine Vertebra Segmentation Training
# MICCAI 2026 Research - SpineMedNeXt
#
# Usage:
#   bash scripts/run_coarse_fine.sh [OPTIONS]
#
# Experiments:
#   1. Full SpineMedNeXt (with VRM)
#   2. Ablation: without VRM
#   3. Different backbones comparison
#

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Default parameters
BATCH_SIZE=4
LEARNING_RATE=1e-4
EPOCHS=1000
MODEL_SIZE="tiny"  # tiny, small, base
USE_VRM=1          # 1: use VRM, 0: no VRM (for ablation)
USE_BFLOAT16=1
NUM_WORKERS=4
NUM_GPUS=2         # Number of GPUs to use (for multi-GPU training)

# Loss weights (for improved supervision)
# All set to 1.0 for equal importance - can be adjusted based on loss scales
ORDER_WEIGHT=1.0
LABEL_WEIGHT=1.0
DISTANCE_WEIGHT=0.3  # Distance loss weight for inter-vertebra spacing
MIN_DISTANCE=12.5    # Minimum distance between adjacent vertebrae (in 5mm space)
MAX_DISTANCE=125.0   # Maximum distance between adjacent vertebrae (in 5mm space)

# NEW: Anatomy-aware embedding (for cross-stage learning)
USE_ANATOMY_EMBEDDING=0  # 0 = baseline, 1 = with shared embedding table
CONSISTENCY_WEIGHT=1.0    # Weight for cross-stage consistency loss

# NEW: Structure Module (AlphaFold-style iterative refinement)
USE_STRUCTURE_MODULE=0  # 0 = baseline heatmap, 1 = structure module
NUM_REFINEMENT_ITERATIONS=2  # Number of refinement iterations (1-2 for simplified)
USE_PHYSIOLOGICAL_CONSTRAINTS=1  # Load-aware, Size-aware constraints

# NEW: Fast ablation mode (Stage 1 only)
TRAIN_STAGE1_ONLY=1  # 0 = full pipeline, 1 = Stage 1 only (faster for ablation)

# NEW: Physics-Informed Module (SINDy + Contrastive Learning)
USE_PHYSICS_INFORMED=0  # 0 = baseline, 1 = with physics-informed features
PHYSICS_LAWS_PATH="physics/discovered_laws/discovered_spine_laws.json"
CONTRASTIVE_WEIGHT=1.0  # Weight for contrastive loss
CONTRASTIVE_TEMPERATURE=0.1  # Temperature for contrastive loss

# NEW: Checkpoint loading
CHECKPOINT_PATH=""  # Path to checkpoint file to resume training (empty = start from scratch)

## Baseline
# CHECKPOINT_PATH="$PROJECT_ROOT/outputs/experiments/VerSe_COARSE_FINE_12-05_13-22/stage1/best_model.pth"

## Structure Module
# CHECKPOINT_PATH="$PROJECT_ROOT/outputs/experiments/VerSe_COARSE_FINE_12-05_23-31/stage1/best_model.pth"

## Physics-Informed Module
# CHECKPOINT_PATH="$PROJECT_ROOT/outputs/experiments/VerSe_COARSE_FINE_12-05_23-35/stage1/best_model.pth"

# Experiment name
EXPERIMENT="full"  # full, no_vrm, tiny, base



# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --use_vrm)
            USE_VRM="$2"
            shift 2
            ;;
        --use_anatomy_embedding)
            USE_ANATOMY_EMBEDDING="$2"
            shift 2
            ;;
        --use_structure_module)
            USE_STRUCTURE_MODULE="$2"
            shift 2
            ;;
        --num_refinement_iterations)
            NUM_REFINEMENT_ITERATIONS="$2"
            shift 2
            ;;
        --use_physiological_constraints)
            USE_PHYSIOLOGICAL_CONSTRAINTS="$2"
            shift 2
            ;;
        --use_physics_informed)
            USE_PHYSICS_INFORMED="$2"
            shift 2
            ;;
        --physics_laws_path)
            PHYSICS_LAWS_PATH="$2"
            shift 2
            ;;
        --contrastive_weight)
            CONTRASTIVE_WEIGHT="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            export CUDA_VISIBLE_DEVICES="$GPU_ID"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment NAME   Experiment preset (full, no_vrm, tiny, base)"
            echo "  --batch_size N      Batch size (default: 2)"
            echo "  --epochs N          Number of epochs (default: 100)"
            echo "  --model_size SIZE   Model size (tiny, small, base)"
            echo "  --use_vrm 0/1       Use VRM (default: 1)"
            echo "  --num_gpus N        Number of GPUs (default: 1)"
            echo "  --gpu ID            GPU ID to use (single GPU, default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Experiment Presets
# ============================================================================

case $EXPERIMENT in
    "full")
        # Full SpineMedNeXt with VRM
        MODEL_SIZE="small"
        USE_VRM=1
        ;;
    "no_vrm")
        # Ablation: without VRM
        MODEL_SIZE="small"
        USE_VRM=0
        ;;
    "tiny")
        # Tiny model for quick testing
        MODEL_SIZE="tiny"
        USE_VRM=1
        EPOCHS=10
        ;;
    "base")
        # Base model (larger)
        MODEL_SIZE="base"
        USE_VRM=1
        ;;
esac

# ============================================================================
# Setup
# ============================================================================

cd "$PROJECT_ROOT"

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

# Check available GPUs
echo "Checking available GPUs..."
ACTUAL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [[ "$ACTUAL_GPUS" -eq 0 ]]; then
    echo "Warning: nvidia-smi not available, assuming 1 GPU"
    ACTUAL_GPUS=1
fi

# Check if requested number of GPUs is available
if [[ "$NUM_GPUS" -gt "$ACTUAL_GPUS" ]]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $ACTUAL_GPUS are available."
    echo "Adjusting to use $ACTUAL_GPUS GPU(s) instead."
    NUM_GPUS=$ACTUAL_GPUS
fi

# Generate GPU IDs based on NUM_GPUS
GPU_IDS=""
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ $i -eq 0 ]]; then
        GPU_IDS="$i"
    else
        GPU_IDS="$GPU_IDS,$i"
    fi
done

# Setup logging
LOG_DIR="$PROJECT_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
# Log file with experiment name for parallel runs
EXP_SUFFIX="${MODEL_SIZE}_vrm${USE_VRM}"
if [ "$USE_ANATOMY_EMBEDDING" -eq 1 ]; then
    EXP_SUFFIX="${EXP_SUFFIX}_anatomy"
fi
if [ "$USE_STRUCTURE_MODULE" -eq 1 ]; then
    EXP_SUFFIX="${EXP_SUFFIX}_struct${NUM_REFINEMENT_ITERATIONS}"
    if [ "$USE_PHYSIOLOGICAL_CONSTRAINTS" -eq 1 ]; then
        EXP_SUFFIX="${EXP_SUFFIX}_physio"
    fi
fi
if [ "$USE_PHYSICS_INFORMED" -eq 1 ]; then
    EXP_SUFFIX="${EXP_SUFFIX}_physics"
fi
if [ "$TRAIN_STAGE1_ONLY" -eq 1 ]; then
    EXP_SUFFIX="${EXP_SUFFIX}_s1only"
fi
# Add timestamp to prevent log file overlap
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="$LOG_DIR/train_coarse_fine_${EXP_SUFFIX}.${TIMESTAMP}.log"

# Find available port for DDP
find_available_port() {
    local port=$1
    if [ -z "$port" ]; then
        port=12355
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
MASTER_PORT=$(find_available_port 12355)
MASTER_ADDR="localhost"

echo "============================================================"
echo "Coarse-to-Fine Vertebra Segmentation Training"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Experiment: $EXPERIMENT"
echo "Model Size: $MODEL_SIZE"
echo "Use VRM: $USE_VRM"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"
echo "Batch Size: $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Loss Weights:"
echo "  - Order: $ORDER_WEIGHT"
echo "  - Label: $LABEL_WEIGHT"
echo "  - Distance: $DISTANCE_WEIGHT"
echo "  - Min Distance: $MIN_DISTANCE mm (2mm space)"
echo "  - Max Distance: $MAX_DISTANCE mm (2mm space)"
echo "Use Anatomy Embedding: $USE_ANATOMY_EMBEDDING"
echo "Consistency Weight: $CONSISTENCY_WEIGHT"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint Path: $CHECKPOINT_PATH (Resuming training)"
else
    echo "Checkpoint Path: None (Starting from scratch)"
fi
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Log File: $LOG_FILE"
echo "============================================================"

# ============================================================================
# Run Training
# ============================================================================

# Set GPU IDs as environment variable
# export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_VISIBLE_DEVICES="1,2"

# Use torchrun for multi-GPU support (similar to run.sh)
if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "Using torchrun for multi-GPU training ($NUM_GPUS GPUs)..."
    torchrun \
        --nnode 1 \
        --nproc_per_node "$NUM_GPUS" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        workspace/trainers/coarse_fine_trainer.py \
            --save_path "$PROJECT_ROOT/outputs" \
            --processed_dir "$PROJECT_ROOT/VerSe/processed" \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --epochs $EPOCHS \
            --model_size $MODEL_SIZE \
            --use_vrm $USE_VRM \
            --use_bfloat16 $USE_BFLOAT16 \
            --num_workers $NUM_WORKERS \
            --save_validation_details 1 \
            --order_weight $ORDER_WEIGHT \
            --label_weight $LABEL_WEIGHT \
            --distance_weight $DISTANCE_WEIGHT \
            --min_distance $MIN_DISTANCE \
            --max_distance $MAX_DISTANCE \
            --use_anatomy_embedding $USE_ANATOMY_EMBEDDING \
            --consistency_weight $CONSISTENCY_WEIGHT \
            --use_structure_module $USE_STRUCTURE_MODULE \
            --num_refinement_iterations $NUM_REFINEMENT_ITERATIONS \
            --use_physiological_constraints $USE_PHYSIOLOGICAL_CONSTRAINTS \
            --use_physics_informed $USE_PHYSICS_INFORMED \
            --physics_laws_path "$PHYSICS_LAWS_PATH" \
            --contrastive_weight $CONTRASTIVE_WEIGHT \
            --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
            --train_stage1_only $TRAIN_STAGE1_ONLY \
            ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"} \
            2>&1 | tee -a "$LOG_FILE"
else
    echo "Using single GPU training..."
    python workspace/trainers/coarse_fine_trainer.py \
        --save_path "$PROJECT_ROOT/outputs" \
        --processed_dir "$PROJECT_ROOT/VerSe/processed" \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --model_size $MODEL_SIZE \
        --use_vrm $USE_VRM \
        --use_bfloat16 $USE_BFLOAT16 \
        --num_workers $NUM_WORKERS \
        --save_validation_details 1 \
        --order_weight $ORDER_WEIGHT \
        --label_weight $LABEL_WEIGHT \
        --distance_weight $DISTANCE_WEIGHT \
        --min_distance $MIN_DISTANCE \
        --max_distance $MAX_DISTANCE \
            --use_anatomy_embedding $USE_ANATOMY_EMBEDDING \
            --consistency_weight $CONSISTENCY_WEIGHT \
            --use_structure_module $USE_STRUCTURE_MODULE \
            --num_refinement_iterations $NUM_REFINEMENT_ITERATIONS \
            --use_physiological_constraints $USE_PHYSIOLOGICAL_CONSTRAINTS \
            --use_physics_informed $USE_PHYSICS_INFORMED \
            --physics_laws_path "$PHYSICS_LAWS_PATH" \
            --contrastive_weight $CONTRASTIVE_WEIGHT \
            --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
            --train_stage1_only $TRAIN_STAGE1_ONLY \
            ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"} \
        2>&1 | tee -a "$LOG_FILE"
fi

echo "============================================================"
echo "Training Complete!"
echo "Results saved to: $LOG_FILE"
echo "============================================================"

