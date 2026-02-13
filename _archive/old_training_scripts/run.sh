#!/bin/bash

# MNIST Training with Apptainer
# Usage: ./run_mnist_training.sh [OPTIONS]

set -e  # Exit on error

# Use absolute paths for all directories
PROJECT_ROOT="/gscratch/scrubbed/june0604/vindr"
WORKSPACE_DIR="$PROJECT_ROOT/workspace"
OUTPUT_DIR="$PROJECT_ROOT/outputs"
# VerSe Multi-task Training Configuration
PROCESSED_DIR="$PROJECT_ROOT/VerSe/processed"

# Export PROJECT_ROOT for Python scripts to use
export PROJECT_ROOT
TASK="verse"  # VerSe segmentation
USE_CALIBRATION=0  # Not used for segmentation

# Default training parameters
BATCH_SIZE=1
LEARNING_RATE=1e-4
EPOCHS=100
OPTIMIZER="adamW"
SCHEDULER="cosinewarmlr"
LOSS="segmentation"  # Segmentation loss (Dice + Focal)
MODEL="spine_mednext"  # Backbone for segmentation model
MODE="3d"  # 3D mode for spine segmentation
NUM_CLASSES=28  # 28 vertebrae
WANDB=0
TEST_MODE=0  # 0: train, 1: test

# VerSe segmentation parameters
DICE_WEIGHT=1.0
FOCAL_WEIGHT=1.0
ORDER_WEIGHT=0.4  # Spatial Order Loss weight (0 to disable)
USE_BFLOAT16=1  # Use bfloat16 mixed precision
USE_VRM=1  # Use Vertebra Relation Module in SpineMedNeXt
MODEL_SIZE="base"  # Model size (small/base/large)

# Model Weight
MODEL_WEIGHT=None

## MedNeXt
MODEL_WEIGHT="$PROJECT_ROOT/outputs/VerSe_MEDNEXT_SEGMENTATION_11-26_10-09/best_model_dice.pth"

# SpineMedNeXt
# MODEL_WEIGHT="$PROJECT_ROOT/outputs/VerSe_SPINE_MEDNEXT_SEGMENTATION_11-29_10-54/best_model_dice.pth"


# DDP parameters
NUM_GPUS=1
MASTER_ADDR="localhost"
MASTER_PORT="12357"

# Parse command line arguments
USE_GPU=true
DEBUG_MODE=false
NUM_WORKERS=4

# Help function
show_help() {
    echo "VerSe 3D Segmentation Training with DDP (Distributed Data Parallel)"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              show this help message and exit"
    echo "  -b, --batch-size N      batch size per GPU (default: 1)"
    echo "  -l, --learning-rate F   learning rate (default: 1e-4)"
    echo "  -e, --epochs N          epochs (default: 50)"
    echo "  -m, --model NAME        model backbone (resnet18/resnet34, default: resnet18)"
    echo "  -n, --num_classes N     number of classes (28 vertebrae, default: 28)"
    echo "  -o, --optimizer NAME    optimizer (adam/adamw/sgd, default: adamW)"
    echo "  -s, --scheduler NAME    scheduler (step/cosine, default: cosine)"
    echo "  -loss, --loss NAME      loss (segmentation, default: segmentation)"
    echo "  -w, --workers N         data loading workers (default: 4)"
    echo "  -g, --num-gpus N        number of GPUs to use (default: 8)"
    echo "  -d, --debug             enable debug mode"
    echo "  --master-addr ADDR      master address for DDP (default: localhost)"
    echo "  --master-port PORT      master port for DDP (default: 12357)"
    echo "  --wandb                 whether to use wandb (default: 0)"
    echo "  --model-weight PATH     model weight path (default: None)"
    echo "  --task NAME             task to run (verse, default: verse)"
    echo "  --processed-dir PATH    VerSe processed data directory"
    echo "  --dice-weight F         weight for Dice loss (default: 1.0)"
    echo "  --focal-weight F        weight for Focal loss (default: 1.0)"
    echo "  --use-bfloat16          use bfloat16 mixed precision (default: enabled)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # 8 GPUs with default settings"
    echo "  $0 -g 4 -b 2 -e 100                  # 4 GPUs, batch size 2 per GPU, 100 epochs"
    echo "  $0 -g 8 -l 5e-4 --wandb              # 8 GPUs, wandb logging"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num_classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        -o|--optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        -s|--scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        -loss|--loss)
            LOSS="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -g|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_MODE=true
            shift
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --wandb)
            WANDB="$2"
            shift 2
            ;;
        --model-weight)
            MODEL_WEIGHT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --use_calibration)
            USE_CALIBRATION="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --dice-weight)
            DICE_WEIGHT="$2"
            shift 2
            ;;
        --focal-weight)
            FOCAL_WEIGHT="$2"
            shift 2
            ;;
        --use-bfloat16)
            USE_BFLOAT16=1
            shift
            ;;
        --order-weight)
            ORDER_WEIGHT="$2"
            shift 2
            ;;
        --use-vrm)
            USE_VRM="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# # Check if SIF file exists
# if [[ ! -f "$SIF_FILE" ]]; then
#     echo "Error: SIF file '$SIF_FILE' not found!"
#     echo "Please make sure the container image exists in the project directory."
#     exit 1
# fi

# Check actual available GPUs
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
# For NUM_GPUS=1: 0
# For NUM_GPUS=2: 0,1
# For NUM_GPUS=3: 0,1,2
# etc.
GPU_IDS=""
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ $i -eq 0 ]]; then
        GPU_IDS="$i"
    else
        GPU_IDS="$GPU_IDS,$i"
    fi
done

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DIR"

# GPU flag setting
GPU_FLAG=""
if [[ "$USE_GPU" == true ]]; then
    GPU_FLAG="--nv"
    echo "GPU mode enabled (--nv flag added)"
else
    echo "CPU mode (use --gpu flag to enable GPU)"
fi

# Debug flag setting
DEBUG_FLAG=""
if [[ "$DEBUG_MODE" == true ]]; then
    DEBUG_FLAG="--debug"
    echo "Debug mode enabled"
fi

# Find available port
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
# Use the configured MASTER_PORT or find an available one
MASTER_PORT=$(find_available_port "$MASTER_PORT")

# Print training parameters
echo "=========================================="
echo "VerSe 3D Segmentation Training Configuration"
echo "=========================================="
echo "Workspace: $WORKSPACE_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "Processed Data: $PROCESSED_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Model (Backbone): $MODEL"
echo "Num Classes: $NUM_CLASSES (28 vertebrae)"
echo "Optimizer: $OPTIMIZER"
echo "Scheduler: $SCHEDULER"
echo "Loss: $LOSS (Dice + Focal)"
echo "Mode: 3D Segmentation (fixed)"
echo "Loss Weights: Dice=$DICE_WEIGHT, Focal=$FOCAL_WEIGHT"
echo "Mixed Precision: bfloat16=$USE_BFLOAT16"
echo "Workers: $NUM_WORKERS"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Debug: $DEBUG_MODE"
echo "Wandb: $WANDB"
echo "Model weight: $MODEL_WEIGHT"
echo "Test mode: $TEST_MODE"
echo "Task: $TASK"
echo "=========================================="


# Run training
echo "Starting VerSe 3D Segmentation training..."
echo ""

# Set GPU IDs as environment variable for torchrun
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

BFLOAT16_FLAG=""
if [ "$USE_BFLOAT16" -eq 1 ]; then
    BFLOAT16_FLAG="--use_bfloat16"
fi

torchrun \
    --nnode 1 \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$WORKSPACE_DIR/main.py" \
        --world_size "$NUM_GPUS" \
        --processed_dir "$PROCESSED_DIR" \
        --save_path "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --epochs "$EPOCHS" \
        --model "$MODEL" \
        --mode "$MODE" \
        --model_size "$MODEL_SIZE" \
        --num_classes "$NUM_CLASSES" \
        --optimizer "$OPTIMIZER" \
        --scheduler "$SCHEDULER" \
        --loss "$LOSS" \
        --num_workers "$NUM_WORKERS" \
        --wandb "$WANDB" \
        --model_weight "$MODEL_WEIGHT" \
        --test_mode "$TEST_MODE" \
        --task "$TASK" \
        --dice_weight "$DICE_WEIGHT" \
        --focal_weight "$FOCAL_WEIGHT" \
        --order_loss_weight "$ORDER_WEIGHT" \
        --use_vrm "$USE_VRM" \
        $BFLOAT16_FLAG \
        $DEBUG_FLAG

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"

