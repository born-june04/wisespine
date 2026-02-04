#!/bin/bash
#
# 학습 데이터 시각화 스크립트
# Stage 1과 Stage 2 학습에 실제로 들어가는 데이터를 시각화합니다.
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

cd "$PROJECT_ROOT"

# Default parameters
SPLIT="val"
NUM_SAMPLES_STAGE1=3
NUM_SAMPLES_STAGE2=10
OUTPUT_DIR="$PROJECT_ROOT/outputs/training_data_visualizations"
STAGE="both"  # stage1, stage2, or both

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_samples_stage1)
            NUM_SAMPLES_STAGE1="$2"
            shift 2
            ;;
        --num_samples_stage2)
            NUM_SAMPLES_STAGE2="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --split SPLIT              Dataset split (train, val, test) [default: val]"
            echo "  --num_samples_stage1 N     Number of Stage 1 samples [default: 3]"
            echo "  --num_samples_stage2 N     Number of Stage 2 ROI samples [default: 10]"
            echo "  --output_dir DIR           Output directory [default: outputs/training_data_visualizations]"
            echo "  --stage STAGE              Which stage to visualize (stage1, stage2, both) [default: both]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Training Data Visualization"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Split: $SPLIT"
echo "Stage: $STAGE"
echo "Number of Stage 1 samples: $NUM_SAMPLES_STAGE1"
echo "Number of Stage 2 ROI samples: $NUM_SAMPLES_STAGE2"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Run visualization
python workspace/utils/visualize_training_data.py \
    --processed_dir "$PROJECT_ROOT/VerSe/processed" \
    --split "$SPLIT" \
    --num_samples_stage1 "$NUM_SAMPLES_STAGE1" \
    --num_samples_stage2 "$NUM_SAMPLES_STAGE2" \
    --output_dir "$OUTPUT_DIR" \
    --stage "$STAGE"

echo "============================================================"
echo "Visualization Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
