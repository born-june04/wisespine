#!/usr/bin/env python3
"""
Compare GT mask vs TotalSegmentator predictions on warped CT.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import glob

print("="*70)
print("COMPARING: GT vs TS on Warped CT")
print("="*70)

# Load GT mask
gt_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_nii.get_fdata()

# Load TS predictions (need to combine individual files)
ts_dir = "outputs/ts_warped"
ts_files = sorted(glob.glob(f"{ts_dir}/vertebrae_*.nii.gz"))

print(f"\nFound {len(ts_files)} TS vertebrae files")

if len(ts_files) == 0:
    print("❌ No TS predictions found! Checking directory...")
    all_files = glob.glob(f"{ts_dir}/*.nii.gz")
    print(f"Available files: {[os.path.basename(f) for f in all_files[:10]]}")
    
    # Try to find the combined file
    if os.path.exists(f"{ts_dir}/vertebrae_body.nii.gz"):
        print("✓ Found vertebrae_body.nii.gz")
        ts_mask_nii = nib.load(f"{ts_dir}/vertebrae_body.nii.gz")
        ts_mask = ts_mask_nii.get_fdata()
    else:
        print("❌ Could not find TS predictions")
        exit(1)
else:
    # Combine individual vertebrae files
    print("Combining TS predictions...")
    ts_mask = np.zeros_like(gt_mask)
    
    for ts_file in ts_files:
        # Extract vertebra name (e.g., vertebrae_L1.nii.gz -> L1)
        vert_name = os.path.basename(ts_file).replace("vertebrae_", "").replace(".nii.gz", "")
        
        # Load this vertebra
        vert_nii = nib.load(ts_file)
        vert_data = vert_nii.get_fdata()
        
        # Map to label (simplified - using hash)
        label = hash(vert_name) % 100 + 1
        
        ts_mask[vert_data > 0] = label
    
    print(f"✓ Combined {len(ts_files)} vertebrae")

print(f"\nGT mask:")
print(f"  Shape: {gt_mask.shape}")
print(f"  Labels: {sorted(np.unique(gt_mask)[1:].astype(int))}")

print(f"\nTS mask:")
print(f"  Shape: {ts_mask.shape}")
print(f"  Unique values: {len(np.unique(ts_mask))-1}")

# Find L1 region from GT
l1_mask = (gt_mask == 20)
l1_coords = np.where(l1_mask)

if len(l1_coords[0]) == 0:
    print("Warning: L1 not found in GT, using center of all vertebrae")
    all_vert_mask = gt_mask > 0
    l1_coords = np.where(all_vert_mask)

margin = 50
l1_bbox = [
    (max(0, l1_coords[0].min() - margin), min(gt_mask.shape[0], l1_coords[0].max() + margin)),
    (max(0, l1_coords[1].min() - margin), min(gt_mask.shape[1], l1_coords[1].max() + margin)),
    (max(0, l1_coords[2].min() - margin), min(gt_mask.shape[2], l1_coords[2].max() + margin)),
]

l1_center = [int(np.mean(c)) for c in l1_coords]

print(f"\nL1 center (from GT): {l1_center}")
print(f"Bbox: X=[{l1_bbox[0][0]}, {l1_bbox[0][1]}], Y=[{l1_bbox[1][0]}, {l1_bbox[1][1]}], Z=[{l1_bbox[2][0]}, {l1_bbox[2][1]}]")

# Create colormap for masks
colors = plt.cm.tab20(np.linspace(0, 1, 20))
cmap_mask = ListedColormap(colors)

# Create 2x3 comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('L1 Region: GT Mask vs TotalSegmentator on Warped CT', fontsize=16, fontweight='bold')

# Extract zoomed regions
gt_sag = gt_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
gt_ax = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
gt_cor = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

ts_sag = ts_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
ts_ax = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
ts_cor = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

# Row 1: GT
axes[0, 0].imshow(gt_sag.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[0, 0].set_title('GT Mask - Sagittal', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_ax.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[0, 1].set_title('GT Mask - Axial', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(gt_cor.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[0, 2].set_title('GT Mask - Coronal', fontsize=12)
axes[0, 2].axis('off')

# Row 2: TS on Warped
axes[1, 0].imshow(ts_sag.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[1, 0].set_title('TS on Warped CT - Sagittal', fontsize=12, color='red')
axes[1, 0].axis('off')

axes[1, 1].imshow(ts_ax.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[1, 1].set_title('TS on Warped CT - Axial', fontsize=12, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(ts_cor.T, cmap=cmap_mask, origin='lower', interpolation='nearest')
axes[1, 2].set_title('TS on Warped CT - Coronal', fontsize=12, color='red')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ts_comparison_warped.png', dpi=200, bbox_inches='tight')

print("\n✓ Saved: outputs/ts_comparison_warped.png")

# Calculate Dice score
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    return 2.0 * intersection / (np.sum(mask1 > 0) + np.sum(mask2 > 0) + 1e-6)

# Calculate overall Dice
gt_binary = gt_mask > 0
ts_binary = ts_mask > 0
dice = dice_coefficient(gt_binary, ts_binary)

print(f"\n📊 METRICS:")
print(f"  Overall Dice (binary): {dice:.4f}")
print(f"  GT vertebrae count: {len(np.unique(gt_mask))-1}")
print(f"  TS vertebrae count: {len(np.unique(ts_mask))-1}")

print("\n" + "="*70)
print("RESULT:")
print(f"  📊 outputs/ts_comparison_warped.png")
print(f"  🎯 Dice score: {dice:.4f}")
print("\n👀 Check if TS successfully segmented the warped vertebrae!")
print("="*70)

