#!/bin/bash
#
# Quick test with fixed device placement
#

set -e

cd /gscratch/scrubbed/june0604/vindr

echo "============================================================"
echo "Testing YOLO Loss with Device Fix"
echo "============================================================"
echo "Date: $(date)"
echo ""

# Kill any existing training
pkill -f spineclue_trainer.py || true
sleep 2

# Run quick test (1% data, 1 epoch, batch=2)
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
    --experiment yolo_device_fixed 2>&1 | tee yolo_device_fixed.log

echo ""
echo "============================================================"
echo "Checking Results"
echo "============================================================"

# Check for errors
echo "Errors:"
grep -E "AttributeError|RuntimeError|Traceback" yolo_device_fixed.log | head -5 || echo "  None found!"

echo ""
echo "Debug messages:"
grep "\[DEBUG LOSS\]" yolo_device_fixed.log | head -10 || echo "  None found"

echo ""
echo "Loss progression:"
grep "loss=" yolo_device_fixed.log | tail -10 | sed 's/.*loss=\([0-9.]*\).*/\1/'

echo ""
echo "Done!"

