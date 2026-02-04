#!/bin/bash
#
# Dice와 IoU 계산 테스트 스크립트
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PROJECT_ROOT

# Fix LD_LIBRARY_PATH for conda environment
export LD_LIBRARY_PATH=/gscratch/ubicomp/june/miniconda3/envs/py311/lib:$LD_LIBRARY_PATH

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Dice and IoU Calculation Test"
echo "============================================================"

# Run tests
python workspace/utils/test_dice_iou.py --test_all

echo "============================================================"
echo "Test Complete!"
echo "============================================================"
