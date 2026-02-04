#!/usr/bin/env python3
"""
Enhanced visualization: CT with mask overlay.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
import os

print("="*70)
print("ENHANCED VISUALIZATION: CT + Mask Overlay")
print("="*70)

# Load data
original_ct_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original_ct = original_ct_nii.get_fdata()

warped_ct_nii = nib.load("outputs/rendered_ct_warped.nii.gz")
warped_ct = warped_ct_nii.get_fdata()

gt_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_nii.get_fdata()

# Load TS predictions
ts_files = sorted(glob.glob("outputs/ts_warped/vertebrae_*.nii.gz"))
ts_mask = np.zeros_like(gt_mask)

for ts_file in ts_files:
    vert_name = os.path.basename(ts_file).replace("vertebrae_", "").replace(".nii.gz", "")
    vert_nii = nib.load(ts_file)
    vert_data = vert_nii.get_fdata()
    label = hash(vert_name) % 100 + 1
    ts_mask[vert_data > 0] = label

# Find L1 region
l1_mask = (gt_mask == 20)
l1_coords = np.where(l1_mask)
margin = 50

l1_bbox = [
    (max(0, l1_coords[0].min() - margin), min(gt_mask.shape[0], l1_coords[0].max() + margin)),
    (max(0, l1_coords[1].min() - margin), min(gt_mask.shape[1], l1_coords[1].max() + margin)),
    (max(0, l1_coords[2].min() - margin), min(gt_mask.shape[2], l1_coords[2].max() + margin)),
]

l1_center = [int(np.mean(c)) for c in l1_coords]

print(f"L1 center: {l1_center}")

# Create colormap for overlay (with transparency)
colors = plt.cm.tab20(np.linspace(0, 1, 20))
colors[:, 3] = 0.5  # Set alpha to 0.5 for transparency
cmap_overlay = ListedColormap(colors)

# Create 2x3 visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('CT + Mask Overlay: Original (GT) vs Warped (TS Prediction)', 
             fontsize=16, fontweight='bold')

vmin, vmax = -200, 1500

# Extract slices
orig_ct_sag = original_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
orig_ct_ax = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
orig_ct_cor = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

warp_ct_sag = warped_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
warp_ct_ax = warped_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
warp_ct_cor = warped_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

gt_sag = gt_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
gt_ax = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
gt_cor = gt_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

ts_sag = ts_mask[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
ts_ax = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
ts_cor = ts_mask[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

# Row 1: Original CT + GT mask
axes[0, 0].imshow(orig_ct_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].imshow(np.ma.masked_where(gt_sag.T == 0, gt_sag.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=25)
axes[0, 0].set_title('Original CT + GT - Sagittal', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(orig_ct_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].imshow(np.ma.masked_where(gt_ax.T == 0, gt_ax.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=25)
axes[0, 1].set_title('Original CT + GT - Axial', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(orig_ct_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].imshow(np.ma.masked_where(gt_cor.T == 0, gt_cor.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=25)
axes[0, 2].set_title('Original CT + GT - Coronal', fontsize=12)
axes[0, 2].axis('off')

# Row 2: Warped CT + TS prediction
axes[1, 0].imshow(warp_ct_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].imshow(np.ma.masked_where(ts_sag.T == 0, ts_sag.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=100)
axes[1, 0].set_title('Warped CT + TS Prediction - Sagittal', fontsize=12, color='red')
axes[1, 0].axis('off')

axes[1, 1].imshow(warp_ct_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].imshow(np.ma.masked_where(ts_ax.T == 0, ts_ax.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=100)
axes[1, 1].set_title('Warped CT + TS Prediction - Axial', fontsize=12, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(warp_ct_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].imshow(np.ma.masked_where(ts_cor.T == 0, ts_cor.T), 
                  cmap=cmap_overlay, origin='lower', alpha=0.5, vmin=0, vmax=100)
axes[1, 2].set_title('Warped CT + TS Prediction - Coronal', fontsize=12, color='red')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ts_overlay_comparison.png', dpi=200, bbox_inches='tight')

print("\n✓ Saved: outputs/ts_overlay_comparison.png")

# Also create difference visualization
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Segmentation Difference: GT vs TS on Warped CT', fontsize=16, fontweight='bold')

# Create difference masks (GT - TS)
# Green: GT only, Red: TS only, Yellow: Both
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

axes2[0].imshow(warp_ct_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[0].imshow(diff_sag, origin='lower', alpha=0.6)
axes2[0].set_title('Sagittal\n(Green=GT only, Red=TS only, Yellow=Both)', fontsize=11)
axes2[0].axis('off')

axes2[1].imshow(warp_ct_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[1].imshow(diff_ax, origin='lower', alpha=0.6)
axes2[1].set_title('Axial\n(Green=GT only, Red=TS only, Yellow=Both)', fontsize=11)
axes2[1].axis('off')

axes2[2].imshow(warp_ct_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[2].imshow(diff_cor, origin='lower', alpha=0.6)
axes2[2].set_title('Coronal\n(Green=GT only, Red=TS only, Yellow=Both)', fontsize=11)
axes2[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ts_difference_overlay.png', dpi=200, bbox_inches='tight')

print("✓ Saved: outputs/ts_difference_overlay.png")

print("\n" + "="*70)
print("RESULTS:")
print("  📊 outputs/ts_overlay_comparison.png    - CT + Mask overlay")
print("  📊 outputs/ts_difference_overlay.png    - Difference visualization")
print("\n👀 Now you can clearly see:")
print("  - How well TS predicted on deformed CT")
print("  - Where GT and TS differ (Green/Red/Yellow)")
print("="*70)

