#!/usr/bin/env python3
"""
Compare GT vs TS predictions on fractured CT.

Shows how TotalSegmentator handles fractured vertebra.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
import os

print("="*70)
print("FRACTURED CT: GT vs TS Comparison")
print("="*70)

# Load data
print("\nLoading data...")
fractured_ct_nii = nib.load("outputs/phase3_physics_fracture/ct_renderings/fractured_ct_manual.nii.gz")
fractured_ct = fractured_ct_nii.get_fdata()

gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_mask_nii.get_fdata()

# Load TS predictions on fractured CT
ts_files = sorted(glob.glob("outputs/phase3_physics_fracture/ts_predictions/ts_fractured/vertebrae_*.nii.gz"))
ts_mask = np.zeros_like(gt_mask)

print(f"  Found {len(ts_files)} TS vertebrae files")

for ts_file in ts_files:
    vert_name = os.path.basename(ts_file).replace("vertebrae_", "").replace(".nii.gz", "")
    vert_data = nib.load(ts_file).get_fdata()
    label = hash(vert_name) % 100 + 1
    ts_mask[vert_data > 0] = label

print(f"  GT mask: {len(np.unique(gt_mask))-1} vertebrae")
print(f"  TS mask: {len(np.unique(ts_mask))-1} vertebrae")

# Find L1 region
l1_mask = (gt_mask == 20)
l1_coords = np.where(l1_mask)
l1_center = [int(np.mean(c)) for c in l1_coords]

margin = 50
l1_bbox = [
    (max(0, l1_coords[0].min() - margin), min(gt_mask.shape[0], l1_coords[0].max() + margin)),
    (max(0, l1_coords[1].min() - margin), min(gt_mask.shape[1], l1_coords[1].max() + margin)),
    (max(0, l1_coords[2].min() - margin), min(gt_mask.shape[2], l1_coords[2].max() + margin)),
]

print(f"\nL1 center: {l1_center}")

# Calculate Dice
gt_binary = gt_mask > 0
ts_binary = ts_mask > 0
intersection = np.logical_and(gt_binary, ts_binary).sum()
dice = 2.0 * intersection / (gt_binary.sum() + ts_binary.sum() + 1e-6)

print(f"\n📊 Metrics:")
print(f"  Dice Score: {dice:.4f}")

# Create visualization
print("\n--- Creating visualization ---")

vmin, vmax = -200, 1500
colors = plt.cm.tab20(np.linspace(0, 1, 20))
colors[:, 3] = 0.6
cmap_overlay = ListedColormap(colors)

# 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle(f'Fractured L1: GT vs TS Prediction (Dice={dice:.3f})', 
             fontsize=16, fontweight='bold')

# Extract slices
frac_sag = fractured_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
frac_ax = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
frac_cor = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

gt_sag = gt_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
gt_ax = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
gt_cor = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

ts_sag = ts_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
ts_ax = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
ts_cor = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

# Row 1: Fractured CT only
axes[0, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title('Fractured CT - Sagittal', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title('Fractured CT - Axial', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title('Fractured CT - Coronal', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Row 2: Fractured CT + GT mask overlay
axes[1, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].imshow(np.ma.masked_where(gt_sag.T == 0, gt_sag.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=25)
axes[1, 0].set_title('GT Mask Overlay', fontsize=12, color='green', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].imshow(np.ma.masked_where(gt_ax.T == 0, gt_ax.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=25)
axes[1, 1].set_title('GT Mask Overlay', fontsize=12, color='green', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].imshow(np.ma.masked_where(gt_cor.T == 0, gt_cor.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=25)
axes[1, 2].set_title('GT Mask Overlay', fontsize=12, color='green', fontweight='bold')
axes[1, 2].axis('off')

# Row 3: Fractured CT + TS prediction overlay
axes[2, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[2, 0].imshow(np.ma.masked_where(ts_sag.T == 0, ts_sag.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=100)
axes[2, 0].set_title('TS Prediction Overlay', fontsize=12, color='red', fontweight='bold')
axes[2, 0].axis('off')

axes[2, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[2, 1].imshow(np.ma.masked_where(ts_ax.T == 0, ts_ax.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=100)
axes[2, 1].set_title('TS Prediction Overlay', fontsize=12, color='red', fontweight='bold')
axes[2, 1].axis('off')

axes[2, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[2, 2].imshow(np.ma.masked_where(ts_cor.T == 0, ts_cor.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.6, vmin=0, vmax=100)
axes[2, 2].set_title('TS Prediction Overlay', fontsize=12, color='red', fontweight='bold')
axes[2, 2].axis('off')

plt.tight_layout()

output_path = "outputs/phase3_physics_fracture/visualizations/fractured_comparison_final.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')

print(f"✓ Saved: {output_path}")

# Also create difference visualization
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Segmentation Difference: GT vs TS on Fractured CT', 
              fontsize=16, fontweight='bold')

# Create difference masks
diff_sag = np.zeros((*gt_sag.T.shape, 3))
diff_ax = np.zeros((*gt_ax.T.shape, 3))
diff_cor = np.zeros((*gt_cor.T.shape, 3))

# GT only (Green)
diff_sag[(gt_sag.T > 0) & (ts_sag.T == 0)] = [0, 1, 0]
diff_ax[(gt_ax.T > 0) & (ts_ax.T == 0)] = [0, 1, 0]
diff_cor[(gt_cor.T > 0) & (ts_cor.T == 0)] = [0, 1, 0]

# TS only (Red)
diff_sag[(gt_sag.T == 0) & (ts_sag.T > 0)] = [1, 0, 0]
diff_ax[(gt_ax.T == 0) & (ts_ax.T > 0)] = [1, 0, 0]
diff_cor[(gt_cor.T == 0) & (ts_cor.T > 0)] = [1, 0, 0]

# Both (Yellow)
diff_sag[(gt_sag.T > 0) & (ts_sag.T > 0)] = [1, 1, 0]
diff_ax[(gt_ax.T > 0) & (ts_ax.T > 0)] = [1, 1, 0]
diff_cor[(gt_cor.T > 0) & (ts_cor.T > 0)] = [1, 1, 0]

axes2[0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[0].imshow(diff_sag, origin='lower', alpha=0.6)
axes2[0].set_title('Sagittal (Green=GT, Red=TS, Yellow=Both)', fontsize=11)
axes2[0].axis('off')

axes2[1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[1].imshow(diff_ax, origin='lower', alpha=0.6)
axes2[1].set_title('Axial', fontsize=11)
axes2[1].axis('off')

axes2[2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[2].imshow(diff_cor, origin='lower', alpha=0.6)
axes2[2].set_title('Coronal', fontsize=11)
axes2[2].axis('off')

plt.tight_layout()

output_diff = "outputs/phase3_physics_fracture/visualizations/fractured_difference_final.png"
plt.savefig(output_diff, dpi=200, bbox_inches='tight')

print(f"✓ Saved: {output_diff}")

print("\n" + "="*70)
print("🎉 FRACTURED CT COMPARISON COMPLETE!")
print("="*70)
print(f"\n📊 Results:")
print(f"  • Dice Score: {dice:.4f}")
print(f"  • GT vertebrae: {len(np.unique(gt_mask))-1}")
print(f"  • TS detected: {len(np.unique(ts_mask))-1}")
print(f"\n📁 Visualizations:")
print(f"  • {output_path}")
print(f"  • {output_diff}")
print(f"\n💡 Key Insight:")
print(f"  TS performance on fractured vertebra shows how")
print(f"  segmentation fails with realistic abnormalities!")
print("="*70)

