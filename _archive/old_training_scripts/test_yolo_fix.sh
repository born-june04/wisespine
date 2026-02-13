#!/bin/bash
#
# Test YOLO Loss Fix
#

set -e

cd /gscratch/scrubbed/june0604/vindr

echo "============================================================"
echo "Testing YOLO Loss Fix"
echo "============================================================"
echo "Date: $(date)"
echo ""

# Stop any running training first
pkill -f spineclue_trainer.py || true
sleep 2

# Run with fixed code
bash scripts/run_spineclue.sh \
    --train_stage localization \
    --processed_dir VerSe/processed \
    --sample_fraction 0.01 \
    --localization_epochs 1 \
    --localization_batch_size 2 \
    --localization_lr 1e-4 \
    --use_pretrained_yolo 1 \
    --yolo_pretrained_model yolov8s \
    --num_gpus 1 \
    --cuda_devices 0 \
    --experiment yolo_loss_fixed 2>&1 | tee yolo_loss_fixed.log

echo ""
echo "============================================================"
echo "Checking results..."
echo "============================================================"

# Check for success indicators
if grep -q "\[DEBUG LOSS\].*Got loss from" yolo_loss_fixed.log; then
    echo "✓ SUCCESS: Ultralytics loss is working!"
    echo ""
    echo "Loss values:"
    grep "\[DEBUG LOSS\].*Got loss" yolo_loss_fixed.log | head -5
else
    echo "✗ FAILED: Still using fallback loss"
    echo ""
    echo "Errors:"
    grep -E "WARNING|AttributeError|Traceback" yolo_loss_fixed.log | head -10
fi

