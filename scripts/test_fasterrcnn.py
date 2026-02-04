#!/usr/bin/env python3
"""
Test Faster R-CNN implementation
"""
import os
import sys
from pathlib import Path

# Set cache directories to avoid home directory quota issues
project_root = Path(__file__).parent.parent
os.environ['TORCH_HOME'] = str(project_root / '.cache' / 'torch')
os.environ['HF_HOME'] = str(project_root / '.cache' / 'huggingface')
os.environ['XDG_CACHE_HOME'] = str(project_root / '.cache')

# Create cache directories
(project_root / '.cache' / 'torch').mkdir(parents=True, exist_ok=True)
(project_root / '.cache' / 'huggingface').mkdir(parents=True, exist_ok=True)

# Add workspace to path
sys.path.insert(0, str(project_root / 'workspace'))

import torch
from networks.spineclue.localization_fasterrcnn import create_fasterrcnn_localization

def test_fasterrcnn():
    """Test Faster R-CNN model"""
    print("=" * 80)
    print("Testing Faster R-CNN Localization Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating Faster R-CNN model...")
    model = create_fasterrcnn_localization(
        num_classes=1,
        pretrained=True,
        img_size=640,
        score_thresh=0.3,
        nms_thresh=0.4,
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   ✓ Model on device: {device}")
    
    # Test inference
    print("\n2. Testing inference mode...")
    model.eval()
    batch_size = 4
    images = torch.randn(batch_size, 1, 640, 640).to(device)
    
    with torch.no_grad():
        detections = model(images)
    
    print(f"   ✓ Processed {len(detections)} images")
    for i, det in enumerate(detections):
        print(f"     Image {i}: {len(det)} detections")
        if len(det) > 0:
            print(f"       First bbox: x={det[0]['x']:.1f}, y={det[0]['y']:.1f}, "
                  f"w={det[0]['w']:.1f}, h={det[0]['h']:.1f}, conf={det[0]['confidence']:.3f}")
    
    # Test training mode
    print("\n3. Testing training mode...")
    model.train()
    
    targets = [
        [
            {'x': 320.0, 'y': 320.0, 'w': 40.0, 'h': 40.0, 'confidence': 1.0},
            {'x': 400.0, 'y': 200.0, 'w': 50.0, 'h': 60.0, 'confidence': 1.0},
        ],
        [
            {'x': 200.0, 'y': 400.0, 'w': 45.0, 'h': 55.0, 'confidence': 1.0},
        ],
        [
            {'x': 300.0, 'y': 300.0, 'w': 50.0, 'h': 50.0, 'confidence': 1.0},
            {'x': 500.0, 'y': 500.0, 'w': 40.0, 'h': 40.0, 'confidence': 1.0},
            {'x': 100.0, 'y': 100.0, 'w': 35.0, 'h': 35.0, 'confidence': 1.0},
        ],
        [],  # Empty target
    ]
    
    loss_dict = model(images, targets)
    
    print(f"   ✓ Loss computed successfully")
    print(f"   Loss components:")
    total_loss = 0
    for k, v in loss_dict.items():
        print(f"     {k}: {v.item():.4f}")
        total_loss += v.item()
    print(f"   Total loss: {total_loss:.4f}")
    
    # Test threshold adjustment
    print("\n4. Testing threshold adjustment...")
    model.set_thresholds(score_thresh=0.5, nms_thresh=0.3)
    print(f"   ✓ Thresholds updated")
    print(f"     Score threshold: {model.score_thresh}")
    print(f"     NMS threshold: {model.nms_thresh}")
    
    # Model statistics
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    
    print("\nFaster R-CNN vs RetinaNet:")
    print("  Faster R-CNN (Two-stage):")
    print("    + More accurate (especially for small objects)")
    print("    + Better with class imbalance")
    print("    + More robust to false positives")
    print("    - Slower inference (~2x)")
    print("    - More memory intensive")
    print()
    print("  RetinaNet (Single-stage):")
    print("    + Faster inference")
    print("    + Lower memory usage")
    print("    + Good for real-time applications")
    print("    - Slightly lower accuracy")
    print("    - May produce more false positives")

if __name__ == '__main__':
    test_fasterrcnn()

