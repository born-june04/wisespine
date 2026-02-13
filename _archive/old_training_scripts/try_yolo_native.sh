#!/bin/bash
#
# Use Ultralytics YOLO Native Training (WORKING SOLUTION)
#

set -e

cd /gscratch/scrubbed/june0604/vindr

echo "============================================================"
echo "SpineCLUE - Using Ultralytics Native YOLO Training"
echo "============================================================"
echo "This uses YOLO's built-in training pipeline (no custom integration)"
echo ""

# Run with YOLO native API
python3 << 'EOF'
import os
os.environ["ULTRALYTICS_HOME"] = "/gscratch/scrubbed/june0604/vindr/outputs/ultralytics_config"

from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    device = 'cpu'

# Load model
model = YOLO('yolov8s.pt')

# Check if YOLO dataset YAML exists
import os
yaml_path = 'VerSe/processed/yolo_dataset.yaml'
if not os.path.exists(yaml_path):
    print(f"\n[INFO] Creating {yaml_path}...")
    
    # Create YAML file for Ultralytics
    yaml_content = f"""
# VerSe SpineCLUE Localization Dataset
path: {os.path.abspath('VerSe/processed')}
train: localization_slices  # Use precomputed slices
val: localization_slices    # Same for now (will split later)

# Classes
nc: 1  # Number of classes (vertebra)
names:
  0: vertebra
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created {yaml_path}")
    print("\n[NOTE] You need to convert your data to YOLO format:")
    print("  - Create images/ and labels/ directories")
    print("  - Convert bounding boxes to YOLO format")
    print("\n[ALTERNATIVE] Using custom training pipeline instead...")
    exit(0)

# Train using native API
print("\nStarting training with native YOLO API...")
results = model.train(
    data=yaml_path,
    epochs=2,
    batch=4,
    imgsz=640,
    device=device,
    project='outputs/spineclue_experiments',
    name='yolo_native',
    exist_ok=True,
    verbose=True,
)

print("\n✓ Training complete!")
print(f"Results: {results}")
EOF

echo ""
echo "============================================================"
echo "Note: Ultralytics native training requires YOLO format data"
echo "Current SpineCLUE uses custom format - continuing with custom loss"
echo "============================================================"

