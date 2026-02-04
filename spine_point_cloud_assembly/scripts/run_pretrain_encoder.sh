#!/bin/bash
#
# Phase 2: Encoder Pretraining Script
# Supports both single-GPU and multi-GPU (DDP) training
#
# Usage:
#   Single GPU: bash scripts/run_pretrain_encoder.sh [OPTIONS]
#   Multi GPU:  torchrun --nproc_per_node=2 scripts/pretrain_encoder.py [OPTIONS]
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
POINT_CLOUD_DIR="$PROJECT_ROOT/outputs/point_clouds"
OUTPUT_DIR="$PROJECT_ROOT/outputs/embeddings"
BATCH_SIZE=8
NUM_EPOCHS=256
LEARNING_RATE=1e-4
HIDDEN_DIM=256
NUM_LAYERS=4
OUTPUT_DIM=512
MAX_POINTS=2048
NUM_WORKERS=4
USE_ROTATION=1
USE_CONTRASTIVE=1
USE_MASKED=1
NUM_GPUS=2

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --point_cloud_dir)
            POINT_CLOUD_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
export CUDA_VISIBLE_DEVICES="0,1"

# Create output directory for logging
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "============================================================"
echo "Starting Encoder Pretraining"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using multi-GPU training (DDP, $NUM_GPUS GPUs)"
    echo "Master port: $MASTER_PORT"
    
    torchrun \
        --nnode=1 \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="localhost" \
        --master_port="$MASTER_PORT" \
        scripts/pretrain_encoder.py \
        --point_cloud_dir "$POINT_CLOUD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --output_dim "$OUTPUT_DIM" \
        --max_points "$MAX_POINTS" \
        --num_workers "$NUM_WORKERS" \
        --use_rotation \
        --use_contrastive \
        --use_masked \
        ${RESUME:+--resume "$RESUME"}
else
    echo "Using single-GPU training"
    
    python scripts/pretrain_encoder.py \
        --point_cloud_dir "$POINT_CLOUD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --output_dim "$OUTPUT_DIM" \
        --max_points "$MAX_POINTS" \
        --num_workers "$NUM_WORKERS" \
        --use_rotation \
        --use_contrastive \
        --use_masked \
        --device cuda \
        ${RESUME:+--resume "$RESUME"}
fi

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo "Log file saved to: $LOG_FILE"
echo "Check the log file for detailed training information."
echo "============================================================"

