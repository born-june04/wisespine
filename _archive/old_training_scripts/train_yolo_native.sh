#!/bin/bash
#
# Use Ultralytics YOLO's native training API
# This is the CORRECT way to train YOLO
#

set -e

cd /gscratch/scrubbed/june0604/vindr

echo "============================================================"
echo "SpineCLUE YOLO Native Training (Ultralytics API)"
echo "============================================================"
echo "Date: $(date)"
echo ""

# Activate environment
source /gscratch/ubicomp/june/miniconda3/etc/profile.d/conda.sh
conda activate py311

# Setup environment
export PYTHONPATH=/gscratch/scrubbed/june0604/vindr:$PYTHONPATH
export ULTRALYTICS_HOME=/gscratch/scrubbed/june0604/vindr/outputs/ultralytics_config
mkdir -p "$ULTRALYTICS_HOME"

# Run native YOLO training
python3 << 'EOF'
from ultralytics import YOLO
import torch

# Load pretrained model
model = YOLO('yolov8s.pt')  # Use YOLOv8-S (smaller, faster)

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Train using Ultralytics native API
# This will automatically handle:
# - Data loading
# - Loss computation
# - Optimization
# - Validation
results = model.train(
    data='VerSe/processed/yolo_dataset.yaml',  # We need to create this
    epochs=2,
    batch=8,
    imgsz=640,
    device=0,
    project='outputs/spineclue_experiments',
    name='yolo_native_pilot',
    exist_ok=True,
    verbose=True,
)

print("\nTraining complete!")
print(f"Results saved to: {results.save_dir}")
EOF

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo "End time: $(date)"

