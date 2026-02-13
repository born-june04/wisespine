#!/bin/bash
#
# SpineCLUE Training Script
# Automatic Vertebrae Identification Using Contrastive Learning and Uncertainty Estimation
# MICCAI 2026 Research - SpineMedNeXt
#
# Usage:
#   bash scripts/run_spineclue.sh [OPTIONS]
#
# 3-stage pipeline:
#   1. Localization: RetinaNet ResNet50-FPN (2D slices) → Dual-factor clustering
#   2. Segmentation: TransUNet (128×128×128 bounding boxes)
#   3. Identification: 3D ResNet-101 + Contrastive Learning + Uncertainty

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Set cache directories to avoid home directory quota issues
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

# Default parameters
BATCH_SIZE=1
LEARNING_RATE=1e-4
EPOCHS=128
NUM_WORKERS=4  # Reduced from 8 to prevent OOM (limited to 2 in dataloader)
NUM_GPUS=2

# Stage selection
TRAIN_STAGE="localization"  # 'localization', 'segmentation', 'identification', 'all'

# Localization Model Selection
LOCALIZATION_MODEL="fasterrcnn"  # 'retinanet' or 'fasterrcnn'
LOCALIZATION_EPOCHS=128
LOCALIZATION_LR=1e-4
LOCALIZATION_BATCH_SIZE=4  # Reduced from 10 to prevent OOM (4 for fasterrcnn)
USE_PRETRAINED=1  # Use COCO pretrained weights
LOCALIZATION_WEIGHTS=""  # Path to custom weights (optional)
SCORE_THRESH=0.1  # Detection score threshold
NMS_THRESH=0.5    # NMS IoU threshold

# Segmentation (TransUNet or Faster R-CNN Pipeline)
SEGMENTATION_EPOCHS=128
SEGMENTATION_LR=1e-4
SEGMENTATION_BATCH_SIZE=4
USE_FASTERRCNN_PIPELINE=0  # Use Faster R-CNN + Segmentation pipeline (0=False, 1=True)
# If empty, will auto-detect most recent Faster R-CNN model
FASTERRCNN_LOCALIZATION_WEIGHTS=""  # Path to trained Faster R-CNN localization model weights (leave empty for auto-detect)

# Identification (3D ResNet-101)
IDENTIFICATION_EPOCHS=1
IDENTIFICATION_LR=1e-4
IDENTIFICATION_BATCH_SIZE=32

# Loss weights
CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_TEMPERATURE=0.1
SEQUENCE_WEIGHT=1.0

# Physics-guided features
USE_PHYSICS_CLUSTERING=0  # For clustering stage
USE_PHYSICS_LOSS=0        # For physics-guided representation learning (NEW!)
PHYSICS_LOSS_WEIGHT=0.5   # Weight for physics loss during training
PHYSICS_LAWS_PATH="physics/discovered_laws/discovered_spine_laws.json"
PHYSICS_WEIGHT=0.5
PHYSICS_THRESHOLD=15.0

# Uncertainty and Message Fusion
USE_UNCERTAINTY=1
USE_MESSAGE_FUSION=1
MC_DROPOUT_SAMPLES=20
MESSAGE_FUSION_HOPS=3
MESSAGE_FUSION_WEIGHT=0.1

# Data paths
PROCESSED_DIR="VerSe/processed"
# CSV_PATH: Leave empty to use default (processed_dir/preprocessed_data_subject.csv)
CSV_PATH=""
SAMPLE_FRACTION=1.0  # Fraction of dataset to use (1.0 = full dataset, 0.01 = 1% for quick testing)

# Experiment name (will` be auto-generated based on model and physics settings)
# Format: {model}_{physics_suffix}
# Example: retinanet_2026-01-05_12-42-21, retinanet_physics_2026-01-05_12-42-21
EXPERIMENT=""  # Leave empty for auto-generation
MANUAL_EXPERIMENT=""  # Set to override auto-generation

# GPU settings
MASTER_ADDR="localhost"

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

# Default GPU (can be overridden via --cuda_devices)
export CUDA_VISIBLE_DEVICES="0,1"

# Logging
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="$PROJECT_ROOT/outputs/spineclue_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spineclue_${EXPERIMENT}${EXP_SUFFIX}_${TIMESTAMP}.log"

# Checkpoint path - empty by default (start from scratch)
# Can be overridden via --checkpoint_path argument to resume training
# WARNING: Checkpoint must match the current model type (RetinaNet/Faster R-CNN)
CHECKPOINT_PATH="/gscratch/scrubbed/june0604/vindr/outputs/spineclue_experiments/retinanet_2026-01-02_00-00-00/best_model.pth"  # Leave empty for new experiments

# Test mode: 0 = train then test, 1 = test only (requires checkpoint)
TEST_MODE=0

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            MANUAL_EXPERIMENT="$2"
            shift 2
            ;;
        --model)
            LOCALIZATION_MODEL="$2"
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
        --train_stage)
            TRAIN_STAGE="$2"
            shift 2
            ;;
        --use_physics_clustering)
            USE_PHYSICS_CLUSTERING="$2"
            shift 2
            ;;
        --use_physics_loss)
            USE_PHYSICS_LOSS="$2"
            shift 2
            ;;
        --physics_loss_weight)
            PHYSICS_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --physics_laws_path)
            PHYSICS_LAWS_PATH="$2"
            shift 2
            ;;
        --processed_dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --cuda_devices)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --use_pretrained)
            USE_PRETRAINED="$2"
            shift 2
            ;;
        --localization_weights)
            LOCALIZATION_WEIGHTS="$2"
            shift 2
            ;;
        --score_thresh)
            SCORE_THRESH="$2"
            shift 2
            ;;
        --nms_thresh)
            NMS_THRESH="$2"
            shift 2
            ;;
        --localization_batch_size)
            LOCALIZATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --localization_epochs)
            LOCALIZATION_EPOCHS="$2"
            shift 2
            ;;
        --localization_lr)
            LOCALIZATION_LR="$2"
            shift 2
            ;;
        --sample_fraction)
            SAMPLE_FRACTION="$2"
            shift 2
            ;;
        --segmentation_batch_size)
            SEGMENTATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --segmentation_epochs)
            SEGMENTATION_EPOCHS="$2"
            shift 2
            ;;
        --segmentation_lr)
            SEGMENTATION_LR="$2"
            shift 2
            ;;
        --use_fasterrcnn_pipeline)
            USE_FASTERRCNN_PIPELINE="$2"
            shift 2
            ;;
        --fasterrcnn_localization_weights)
            FASTERRCNN_LOCALIZATION_WEIGHTS="$2"
            shift 2
            ;;
        --identification_batch_size)
            IDENTIFICATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --identification_epochs)
            IDENTIFICATION_EPOCHS="$2"
            shift 2
            ;;
        --identification_lr)
            IDENTIFICATION_LR="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --test_mode)
            TEST_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Auto-generate experiment name (model + physics suffix + timestamp)
# ============================================================================

# Generate experiment name if not manually set
if [ -n "$MANUAL_EXPERIMENT" ]; then
    # User provided custom experiment name
    EXPERIMENT="$MANUAL_EXPERIMENT"
else
    # Auto-generate: {model}_{physics}_{timestamp}
    # Examples: retinanet_2026-01-05_12-42-21, fasterrcnn_physics_2026-01-05_12-42-21
    
    # Model name
    MODEL_NAME="${LOCALIZATION_MODEL}"
    
    # Physics suffix
    PHYSICS_SUFFIX=""
    if [ "$USE_PHYSICS_LOSS" == "1" ]; then
        PHYSICS_SUFFIX="_physics"
    fi
    
    # Combine
    EXPERIMENT="${MODEL_NAME}${PHYSICS_SUFFIX}"
fi

# ============================================================================
# Setup experiment directory
# ============================================================================

# Create new timestamped directory for each experiment run
# Format: outputs/spineclue_experiments/{model}_{physics}_{timestamp}/
SAVE_DIR="$PROJECT_ROOT/outputs/spineclue_experiments/${EXPERIMENT}_${TIMESTAMP}"
mkdir -p "$SAVE_DIR"

# Log file (save in same directory as model)
LOG_FILE="$SAVE_DIR/training.log"

# ============================================================================
# Logging setup
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
}

# ============================================================================
# Print configuration
# ============================================================================

log_section "SpineCLUE Training Configuration"
log "Experiment: $EXPERIMENT"
log "Timestamp: $TIMESTAMP"
log "Save directory: $SAVE_DIR"
log "Log file: $LOG_FILE"
if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
    log "Checkpoint (resuming): $CHECKPOINT_PATH"
else
    log "Checkpoint: Starting from scratch (no checkpoint loaded)"
fi
log ""
log "Training Stage: $TRAIN_STAGE"
log "Batch size: $BATCH_SIZE"
log "Learning rate: $LEARNING_RATE"
log "Epochs: $EPOCHS"
log "Number of GPUs: $NUM_GPUS"
log "CUDA devices: $CUDA_VISIBLE_DEVICES"
log ""
log "Localization Model: $LOCALIZATION_MODEL"
log "  Use pretrained: $USE_PRETRAINED"
log "  Score threshold: $SCORE_THRESH"
log "  NMS threshold: $NMS_THRESH"
log ""
log "Physics-guided clustering: $USE_PHYSICS_CLUSTERING"
log "Physics-guided loss: $USE_PHYSICS_LOSS"
if [ "$USE_PHYSICS_LOSS" == "1" ]; then
    log "  Physics loss weight: $PHYSICS_LOSS_WEIGHT"
fi
if [ "$USE_PHYSICS_CLUSTERING" == "1" ]; then
    log "  Physics laws path: $PHYSICS_LAWS_PATH"
    log "  Physics weight: $PHYSICS_WEIGHT"
    log "  Physics threshold: $PHYSICS_THRESHOLD"
fi
log ""
log "Uncertainty estimation: $USE_UNCERTAINTY"
log "Message fusion: $USE_MESSAGE_FUSION"
log ""
log "Data directory: $PROCESSED_DIR"
if [ -n "$CSV_PATH" ]; then
    log "CSV path: $CSV_PATH (custom)"
else
    log "CSV path: <default> ($PROCESSED_DIR/preprocessed_data_subject.csv)"
fi
log "Sample fraction: $SAMPLE_FRACTION"
if [ -n "$CHECKPOINT_PATH" ]; then
    log "Checkpoint path: $CHECKPOINT_PATH (resuming training)"
fi
log ""
log "Localization:"
log "  Batch size: $LOCALIZATION_BATCH_SIZE"
log "  Epochs: $LOCALIZATION_EPOCHS"
log "  Learning rate: $LOCALIZATION_LR"

# ============================================================================
# Check dependencies
# ============================================================================

log_section "Checking Dependencies"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found"
    exit 1
fi
log "Python: $(python3 --version)"

# Check PyTorch
if ! python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    log_error "PyTorch not found"
    exit 1
fi
log "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# Check CUDA
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    log "CUDA: Available"
    CUDA_DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    log "CUDA devices: $CUDA_DEVICE_COUNT"
else
    log "CUDA: Not available (using CPU)"
fi

# Check torchvision (for RetinaNet)
if python3 -c "import torchvision; from torchvision.models.detection import retinanet_resnet50_fpn_v2" 2>/dev/null; then
    log "torchvision: Installed"
else
    log_error "torchvision not installed or version too old!"
    log_error "RetinaNet requires torchvision >= 0.13"
    if [ "$TRAIN_STAGE" == "localization" ] || [ "$TRAIN_STAGE" == "all" ]; then
        log_error "Exiting..."
        exit 1
    fi
fi

# Check data directory
if [ ! -d "$PROCESSED_DIR" ]; then
    log_error "Processed data directory not found: $PROCESSED_DIR"
    exit 1
fi
log "Data directory: $PROCESSED_DIR (exists)"

# Check physics laws if using physics clustering
if [ "$USE_PHYSICS_CLUSTERING" == "1" ]; then
    if [ ! -f "$PHYSICS_LAWS_PATH" ]; then
        log_error "Physics laws file not found: $PHYSICS_LAWS_PATH"
        exit 1
    fi
    log "Physics laws: $PHYSICS_LAWS_PATH (exists)"
fi

# ============================================================================
# Training
# ============================================================================

log_section "Starting Training"

# Set environment variables
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Build command arguments
ARGS=(
    --save_path "$SAVE_DIR"
    --processed_dir "$PROCESSED_DIR"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --epochs "$EPOCHS"
    --train_stage "$TRAIN_STAGE"
    --num_workers "$NUM_WORKERS"
    --use_physics_clustering "$USE_PHYSICS_CLUSTERING"
    --use_physics_loss "$USE_PHYSICS_LOSS"
    --physics_loss_weight "$PHYSICS_LOSS_WEIGHT"
    --physics_laws_path "$PHYSICS_LAWS_PATH"
    --physics_weight "$PHYSICS_WEIGHT"
    --physics_threshold "$PHYSICS_THRESHOLD"
    --use_uncertainty "$USE_UNCERTAINTY"
    --use_message_fusion "$USE_MESSAGE_FUSION"
    --mc_dropout_samples "$MC_DROPOUT_SAMPLES"
    --message_fusion_hops "$MESSAGE_FUSION_HOPS"
    --message_fusion_weight "$MESSAGE_FUSION_WEIGHT"
    --contrastive_weight "$CONTRASTIVE_WEIGHT"
    --contrastive_temperature "$CONTRASTIVE_TEMPERATURE"
    --sequence_weight "$SEQUENCE_WEIGHT"
)

if [ -n "$CSV_PATH" ]; then
    ARGS+=(--csv_path "$CSV_PATH")
fi

# Sample fraction (for quick testing)
if [ -n "$SAMPLE_FRACTION" ]; then
    ARGS+=(--sample_fraction "$SAMPLE_FRACTION")
fi

# Checkpoint path (for resuming training or test mode)
if [ -n "$CHECKPOINT_PATH" ]; then
    ARGS+=(--checkpoint_path "$CHECKPOINT_PATH")
fi

# Test mode
if [ -n "$TEST_MODE" ]; then
    ARGS+=(--test_mode "$TEST_MODE")
fi

# Stage-specific arguments
if [ "$TRAIN_STAGE" == "localization" ] || [ "$TRAIN_STAGE" == "all" ]; then
    ARGS+=(
        --localization_model "$LOCALIZATION_MODEL"
        --localization_epochs "$LOCALIZATION_EPOCHS"
        --localization_lr "$LOCALIZATION_LR"
        --localization_batch_size "$LOCALIZATION_BATCH_SIZE"
        --use_pretrained "$USE_PRETRAINED"
        --score_thresh "$SCORE_THRESH"
        --nms_thresh "$NMS_THRESH"
    )
    if [ -n "$LOCALIZATION_WEIGHTS" ]; then
        ARGS+=(--localization_weights "$LOCALIZATION_WEIGHTS")
    fi
fi

if [ "$TRAIN_STAGE" == "segmentation" ] || [ "$TRAIN_STAGE" == "all" ]; then
    ARGS+=(
        --segmentation_epochs "$SEGMENTATION_EPOCHS"
        --segmentation_lr "$SEGMENTATION_LR"
        --segmentation_batch_size "$SEGMENTATION_BATCH_SIZE"
        --use_fasterrcnn_pipeline "$USE_FASTERRCNN_PIPELINE"
        --fasterrcnn_localization_weights "$FASTERRCNN_LOCALIZATION_WEIGHTS"
    )
fi

if [ "$TRAIN_STAGE" == "identification" ] || [ "$TRAIN_STAGE" == "all" ]; then
    ARGS+=(
        --identification_epochs "$IDENTIFICATION_EPOCHS"
        --identification_lr "$IDENTIFICATION_LR"
        --identification_batch_size "$IDENTIFICATION_BATCH_SIZE"
    )
fi

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    log "Using multi-GPU training (torchrun, $NUM_GPUS GPUs)"
    log "Master address: $MASTER_ADDR:$MASTER_PORT"
    
    torchrun \
        --nnode 1 \
        --nproc_per_node "$NUM_GPUS" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        workspace/trainers/spineclue_trainer.py \
        "${ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
else
    log "Using single-GPU training"
    
    python3 workspace/trainers/spineclue_trainer.py \
        "${ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
fi

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Final summary
# ============================================================================

log_section "Training Complete"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log "Training completed successfully!"
    log "Results saved to: $SAVE_DIR"
    log "Log file: $LOG_FILE"
else
    log_error "Training failed with exit code: $TRAIN_EXIT_CODE"
    log_error "Check log file for details: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

