#!/bin/bash
#
# Test Inference with Visualization
# Test the improved heatmap model with ROI visualization
#
# Usage: Run on GPU server (ssh g3120)
#   bash scripts/test_inference_visualization.sh

# Don't use set -e here, we want to handle errors manually for GPU retry logic

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

# GPU Configuration (similar to run.sh)
NUM_GPUS=1  # Use single GPU for inference

# Check actual available GPUs
echo "Checking available GPUs..."
ACTUAL_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [[ "$ACTUAL_GPUS" -eq 0 ]]; then
    echo "Error: No GPUs available. Please run on GPU server (ssh g3120)"
    exit 1
fi

# Check if requested number of GPUs is available
if [[ "$NUM_GPUS" -gt "$ACTUAL_GPUS" ]]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $ACTUAL_GPUS are available."
    echo "Adjusting to use $ACTUAL_GPUS GPU(s) instead."
    NUM_GPUS=$ACTUAL_GPUS
fi

# If CUDA_VISIBLE_DEVICES is not set, try to find a working GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Testing GPUs to find one without ECC errors..."
    GPU_FOUND=false
    for gpu_id in 0 1 2 3; do
        if [[ $gpu_id -ge $ACTUAL_GPUS ]]; then
            break
        fi
        echo "  Testing GPU $gpu_id..."
        # Quick test: try to create a small tensor on this GPU
        if python -c "import torch; torch.cuda.set_device($gpu_id); x = torch.zeros(1).cuda(); torch.cuda.synchronize(); print('OK')" 2>/dev/null; then
            GPU_IDS="$gpu_id"
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            GPU_FOUND=true
            echo "  GPU $gpu_id is working!"
            break
        else
            echo "  GPU $gpu_id has issues, trying next..."
        fi
    done
    
    if [ "$GPU_FOUND" = false ]; then
        echo "Error: No working GPU found. All GPUs seem to have issues."
        echo "Please try manually: CUDA_VISIBLE_DEVICES=1 bash scripts/test_inference_visualization.sh"
        exit 1
    fi
else
    # Use user-specified GPU
    GPU_IDS="$CUDA_VISIBLE_DEVICES"
    echo "Using user-specified GPU: $GPU_IDS"
fi

# Configuration
EXPERIMENT_DIR="outputs/VerSe_COARSE_FINE_12-03_22-36"  # SMALL_VRM1
MODEL_SIZE="small"
USE_VRM=1
INPUT_PATH="VerSe/processed/dataset-verse19training/sub-verse409_split-verse266/ct_volume.npy"
OUTPUT_DIR="outputs/inference_results/test_visualization"
DEVICE="cuda"

echo "============================================================"
echo "Testing Inference with ROI Visualization"
echo "============================================================"
echo "Experiment: $EXPERIMENT_DIR"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python workspace/inference/full_volume_segmentation.py \
    --input_path "$INPUT_PATH" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_size "$MODEL_SIZE" \
    --use_vrm $USE_VRM \
    --device "$DEVICE"
    
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "Error occurred (exit code: $EXIT_CODE)"
    echo "============================================================"
    
    # Check if it's a CUDA ECC error
    if grep -q "ECC error" "$OUTPUT_DIR/inference.log" 2>/dev/null || [ "$EXIT_CODE" -eq 1 ]; then
        echo "CUDA ECC error detected. Trying next available GPU..."
        echo ""
        
        CURRENT_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
        NEXT_GPU=$((CURRENT_GPU + 1))
        
        if [ $NEXT_GPU -lt $ACTUAL_GPUS ]; then
            echo "Retrying with GPU $NEXT_GPU..."
            export CUDA_VISIBLE_DEVICES="$NEXT_GPU"
            python workspace/inference/full_volume_segmentation.py \
                --input_path "$INPUT_PATH" \
                --experiment_dir "$EXPERIMENT_DIR" \
                --output_dir "$OUTPUT_DIR" \
                --model_size "$MODEL_SIZE" \
                --use_vrm $USE_VRM \
                --device "$DEVICE"
        else
            echo "No more GPUs to try. Please check GPU status:"
            echo "  nvidia-smi"
            echo ""
            echo "Or try manually:"
            echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/test_inference_visualization.sh"
            echo "  CUDA_VISIBLE_DEVICES=2 bash scripts/test_inference_visualization.sh"
            exit 1
        fi
    else
        echo "Non-CUDA error. Check logs: $OUTPUT_DIR/inference.log"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "Inference Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - roi_visualization_*.png: ROI visualizations"
echo "  - heatmap_visualization_*.png: Heatmap visualizations"
echo "  - roi_debug_*.json: ROI debugging information"
echo "  - inference_metadata.json: Full pipeline metadata"
echo ""

