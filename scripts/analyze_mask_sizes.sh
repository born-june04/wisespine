#!/bin/bash
#
# 척추 마스크 크기 분석 스크립트
# 각 척추별 마스크 크기 통계를 수집하고 시각화합니다.
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

cd "$PROJECT_ROOT"

# Default parameters
NUM_PROCESSES=16
OUTPUT_DIR="$PROJECT_ROOT/outputs/mask_size_analysis"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_processes)
            NUM_PROCESSES="$2"
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
            echo "  --num_processes N    Number of parallel processes [default: 16]"
            echo "  --output_dir DIR     Output directory [default: outputs/mask_size_analysis]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Vertebra Mask Size Analysis"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Number of processes: $NUM_PROCESSES"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Run analysis (using original VerSe data, not processed)
python workspace/utils/analyze_mask_sizes.py \
    --verse_dir "$PROJECT_ROOT/VerSe" \
    --output_dir "$OUTPUT_DIR" \
    --num_processes "$NUM_PROCESSES"

echo "============================================================"
echo "Analysis Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
