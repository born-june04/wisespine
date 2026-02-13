#!/usr/bin/env python3
"""
Test augmentation functionality
"""
import os
import sys
from pathlib import Path

# Set cache directories to avoid home directory quota issues
project_root = Path(__file__).parent.parent
os.environ['TORCH_HOME'] = str(project_root / '.cache' / 'torch')
os.environ['HF_HOME'] = str(project_root / '.cache' / 'huggingface')
os.environ['XDG_CACHE_HOME'] = str(project_root / '.cache')

# Add workspace to path
sys.path.insert(0, str(project_root / 'workspace'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataset.spineclue_dataset import SpineCLUELocalizationDataset

def test_augmentation():
    """Test augmentation on a single sample"""
    print("=" * 80)
    print("Testing CT-aware Augmentation for Localization")
    print("=" * 80)
    
    processed_dir = "/gscratch/scrubbed/june0604/vindr/VerSe/processed"
    csv_path = None  # Use default
    
    # Create dataset with augmentation
    print("\n1. Creating dataset with augmentation enabled...")
    dataset = SpineCLUELocalizationDataset(
        processed_dir=processed_dir,
        csv_path=csv_path,
        split='train',
        num_slices=200,
        slice_size=(640, 640),
        planes=['axial'],
        augment=True,  # Enable augmentation
        sample_fraction=0.01,  # Use 1% for quick test
    )
    
    print(f"   ✓ Dataset created: {len(dataset)} samples")
    print(f"   ✓ Augmentation enabled: {dataset.augment}")
    
    # Get a sample with bboxes
    print("\n2. Finding sample with bboxes...")
    sample_idx = None
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        if len(sample['bboxes']) > 0:
            sample_idx = i
            break
    
    if sample_idx is None:
        print("   ✗ No samples with bboxes found!")
        return
    
    print(f"   ✓ Found sample {sample_idx} with {len(sample['bboxes'])} bboxes")
    
    # Get original and augmented versions
    print("\n3. Generating augmented samples...")
    np.random.seed(42)
    original = dataset[sample_idx]
    
    # Generate 5 augmented versions
    augmented_samples = []
    for i in range(5):
        np.random.seed(i)  # Different seed for each
        aug_sample = dataset[sample_idx]
        augmented_samples.append(aug_sample)
    
    print(f"   ✓ Generated 5 augmented versions")
    
    # Visualize
    print("\n4. Visualizing results...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CT-Aware Augmentation Examples', fontsize=16)
    
    # Original
    ax = axes[0, 0]
    slice_img = original['slice'][0].numpy()  # (640, 640)
    ax.imshow(slice_img, cmap='gray', vmin=-1, vmax=1)
    ax.set_title(f'Original\n{len(original["bboxes"])} bboxes')
    ax.axis('off')
    
    # Draw bboxes
    for bbox in original['bboxes']:
        rect = plt.Rectangle(
            (bbox['x'] - bbox['w']/2, bbox['y'] - bbox['h']/2),
            bbox['w'], bbox['h'],
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
    
    # Augmented versions
    for idx, (aug_sample, ax) in enumerate(zip(augmented_samples, axes.flat[1:])):
        slice_img = aug_sample['slice'][0].numpy()
        ax.imshow(slice_img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'Augmented {idx+1}\n{len(aug_sample["bboxes"])} bboxes')
        ax.axis('off')
        
        # Draw bboxes
        for bbox in aug_sample['bboxes']:
            rect = plt.Rectangle(
                (bbox['x'] - bbox['w']/2, bbox['y'] - bbox['h']/2),
                bbox['w'], bbox['h'],
                fill=False, edgecolor='red', linewidth=2
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    output_path = '/gscratch/scrubbed/june0604/vindr/outputs/augmentation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {output_path}")
    
    # Statistics
    print("\n5. Augmentation statistics:")
    print(f"   - Original shape: {original['slice'].shape}")
    print(f"   - Original bboxes: {len(original['bboxes'])}")
    print(f"   - Slice value range: [{slice_img.min():.3f}, {slice_img.max():.3f}]")
    
    # Check bbox consistency
    bbox_counts = [len(s['bboxes']) for s in augmented_samples]
    print(f"   - Augmented bbox counts: {bbox_counts}")
    print(f"   - All counts match: {all(c == len(original['bboxes']) for c in bbox_counts)}")
    
    print("\n" + "=" * 80)
    print("Augmentation test completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    test_augmentation()

