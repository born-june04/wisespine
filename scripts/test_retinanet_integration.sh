#!/bin/bash
# Quick test of run_spineclue.sh with RetinaNet

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Testing run_spineclue.sh with RetinaNet..."
echo ""

bash scripts/run_spineclue.sh \
    --train_stage localization \
    --sample_fraction 0.01 \
    --epochs 1 \
    --batch_size 2 \
    --num_gpus 1 \
    --cuda_devices "0" \
    --experiment retinanet_test 2>&1 | tail -100
