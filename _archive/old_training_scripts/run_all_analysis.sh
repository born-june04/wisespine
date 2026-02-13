#!/bin/bash
# Run all analysis scripts in sequence

# Don't use set -e, we want to handle GPU errors manually

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Setup conda environment (for SSH/remote execution)
if [ -f "/gscratch/ubicomp/june/miniconda3/etc/profile.d/conda.sh" ]; then
    export LD_LIBRARY_PATH="/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH"
    source /gscratch/ubicomp/june/miniconda3/etc/profile.d/conda.sh
    conda activate py311
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate py311
else
    echo "Warning: conda not found, trying to use system Python"
fi

# Configuration
MODEL_PATH="/gscratch/scrubbed/june0604/vindr/outputs/VerSe_COARSE_FINE_12-01_08-42/stage1/best_model.pth"
DATA_DIR="/gscratch/scrubbed/june0604/vindr/VerSe/processed"
BASE_OUTPUT_DIR="/gscratch/scrubbed/june0604/vindr/outputs/analysis"
NUM_SAMPLES=5
MODEL_SIZE="tiny"
USE_VRM=1

# Create output directory and log file
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="$BASE_OUTPUT_DIR/analysis_log.txt"

# GPU Configuration - Auto-detect working GPU
echo "Checking available GPUs..."
ACTUAL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [[ "$ACTUAL_GPUS" -eq 0 ]]; then
    echo "Error: No GPUs available. Please run on GPU server"
    exit 1
fi

# Test GPUs and find working one
WORKING_GPU=-1
for gpu_id in 1 2 3; do
    if [[ $gpu_id -ge $ACTUAL_GPUS ]]; then
        continue
    fi
    echo "Testing GPU $gpu_id..."
    if CUDA_VISIBLE_DEVICES=$gpu_id python -c "import torch; torch.zeros(1).cuda(); print('OK')" 2>/dev/null; then
        WORKING_GPU=$gpu_id
        echo "✓ GPU $gpu_id is working"
        break
    else
        echo "✗ GPU $gpu_id failed (ECC error or unavailable)"
    fi
done

if [[ $WORKING_GPU -eq -1 ]]; then
    echo "Error: No working GPU found"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$WORKING_GPU
DEVICE="cuda"
echo "Using GPU $WORKING_GPU"
echo ""

LOG_FILE="$BASE_OUTPUT_DIR/analysis_log.txt"
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Running All Analysis Scripts"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $BASE_OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Log: $LOG_FILE"
echo "=========================================="
echo ""

# Redirect output to log file and also display
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# 1. Network Stages Analysis
echo "=========================================="
echo "Step 1/4: Network Stages Analysis"
echo "=========================================="
OUTPUT_DIR_1="$BASE_OUTPUT_DIR/network_stages"
mkdir -p "$OUTPUT_DIR_1"

cd "$PROJECT_DIR" && python workspace/utils/visualize_network_stages.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_1" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --model_size "$MODEL_SIZE" \
    --use_vrm "$USE_VRM" 2>&1 | tee -a "$LOG_FILE" || {
    echo "Error in network stages analysis, trying next GPU..."
    # Try next GPU
    for next_gpu in 1 2 3 0; do
        if [[ $next_gpu -eq $WORKING_GPU ]] || [[ $next_gpu -ge $ACTUAL_GPUS ]]; then
            continue
        fi
        echo "Retrying with GPU $next_gpu..."
        export CUDA_VISIBLE_DEVICES=$next_gpu
        cd "$PROJECT_DIR" && python workspace/utils/visualize_network_stages.py \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR_1" \
            --num_samples "$NUM_SAMPLES" \
            --device "$DEVICE" \
            --model_size "$MODEL_SIZE" \
            --use_vrm "$USE_VRM" && break
    done
}

echo "✓ Network stages analysis complete: $OUTPUT_DIR_1"
echo ""

# 2. Target Heatmap Visualization
echo "=========================================="
echo "Step 2/4: Target Heatmap Visualization"
echo "=========================================="
OUTPUT_DIR_2="$BASE_OUTPUT_DIR/target_heatmaps"
mkdir -p "$OUTPUT_DIR_2"

cd "$PROJECT_DIR" && python workspace/utils/visualize_target_heatmaps.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_2" \
    --num_samples "$NUM_SAMPLES" \
    --sigma 3.0 2>&1 | tee -a "$LOG_FILE"

echo "✓ Target heatmap visualization complete: $OUTPUT_DIR_2"
echo ""

# 3. Loss Distribution Analysis
echo "=========================================="
echo "Step 3/4: Loss Distribution Analysis"
echo "=========================================="
OUTPUT_DIR_3="$BASE_OUTPUT_DIR/loss_distribution"
mkdir -p "$OUTPUT_DIR_3"

cd "$PROJECT_DIR" && python workspace/utils/analyze_loss_distribution.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_3" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --model_size "$MODEL_SIZE" \
    --use_vrm "$USE_VRM" 2>&1 | tee -a "$LOG_FILE" || {
    echo "Error in loss distribution analysis, trying next GPU..."
    for next_gpu in 1 2 3 0; do
        if [[ $next_gpu -eq $WORKING_GPU ]] || [[ $next_gpu -ge $ACTUAL_GPUS ]]; then
            continue
        fi
        echo "Retrying with GPU $next_gpu..."
        export CUDA_VISIBLE_DEVICES=$next_gpu
        cd "$PROJECT_DIR" && python workspace/utils/analyze_loss_distribution.py \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR_3" \
            --num_samples "$NUM_SAMPLES" \
            --device "$DEVICE" \
            --model_size "$MODEL_SIZE" \
            --use_vrm "$USE_VRM" 2>&1 | tee -a "$LOG_FILE" && break
    done
}

echo "✓ Loss distribution analysis complete: $OUTPUT_DIR_3"
echo ""

# 4. Heatmap-to-Vertebra Mapping Analysis
echo "=========================================="
echo "Step 4/4: Heatmap-to-Vertebra Mapping Analysis"
echo "=========================================="
OUTPUT_DIR_4="$BASE_OUTPUT_DIR/heatmap_vertebra_mapping"
mkdir -p "$OUTPUT_DIR_4"

cd "$PROJECT_DIR" && python workspace/utils/analyze_heatmap_vertebra_mapping.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR_4" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --model_size "$MODEL_SIZE" \
    --use_vrm "$USE_VRM" 2>&1 | tee -a "$LOG_FILE" || {
    echo "Error in heatmap-vertebra mapping analysis, trying next GPU..."
    for next_gpu in 1 2 3 0; do
        if [[ $next_gpu -eq $WORKING_GPU ]] || [[ $next_gpu -ge $ACTUAL_GPUS ]]; then
            continue
        fi
        echo "Retrying with GPU $next_gpu..."
        export CUDA_VISIBLE_DEVICES=$next_gpu
        cd "$PROJECT_DIR" && python workspace/utils/analyze_heatmap_vertebra_mapping.py \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR_4" \
            --num_samples "$NUM_SAMPLES" \
            --device "$DEVICE" \
            --model_size "$MODEL_SIZE" \
            --use_vrm "$USE_VRM" 2>&1 | tee -a "$LOG_FILE" && break
    done
}

echo "✓ Heatmap-to-vertebra mapping analysis complete: $OUTPUT_DIR_4"
echo ""

echo "=========================================="
echo "All Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  1. Network Stages: $OUTPUT_DIR_1"
echo "  2. Target Heatmaps: $OUTPUT_DIR_2"
echo "  3. Loss Distribution: $OUTPUT_DIR_3"
echo "  4. Heatmap-Vertebra Mapping: $OUTPUT_DIR_4"
echo ""
echo "Base directory: $BASE_OUTPUT_DIR"
echo "=========================================="

