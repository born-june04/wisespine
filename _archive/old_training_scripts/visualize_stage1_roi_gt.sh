#!/bin/bash
#
# Stage1 ROI GT 시각화 스크립트
# Stage1 GT centroid를 사용해서 ROI를 추출하고 시각화하여
# 척추가 잘 들어가 있는지, GT가 맞는지 확인합니다.
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

cd "$PROJECT_ROOT"

# Default parameters
SPLIT="val"
NUM_SAMPLES=5
NUM_VERTEBRAE_PER_SAMPLE=3
OUTPUT_DIR="$PROJECT_ROOT/outputs/roi_gt_visualizations"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num_vertebrae)
            NUM_VERTEBRAE_PER_SAMPLE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --split SPLIT              Dataset split (train, val, test) [default: val]"
            echo "  --num_samples N            Number of samples to visualize [default: 5]"
            echo "  --num_vertebrae N          Number of vertebrae per sample [default: 3]"
            echo "  --output_dir DIR           Output directory [default: outputs/roi_gt_visualizations]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Stage1 ROI GT Visualization"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Split: $SPLIT"
echo "Number of samples: $NUM_SAMPLES"
echo "Number of vertebrae per sample: $NUM_VERTEBRAE_PER_SAMPLE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Run visualization
python workspace/utils/visualize_stage1_roi_gt.py \
    --processed_dir "$PROJECT_ROOT/VerSe/processed" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --num_vertebrae_per_sample "$NUM_VERTEBRAE_PER_SAMPLE" \
    --output_dir "$OUTPUT_DIR"

echo "============================================================"
echo "Visualization Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
