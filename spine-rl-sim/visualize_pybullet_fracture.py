#!/usr/bin/env python3
"""
Visualize PyBullet-generated fractured CT.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("="*70)
print("PyBullet Fractured CT Visualization")
print("="*70)

# Load data
original_ct_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original_ct = original_ct_nii.get_fdata()

fractured_ct_nii = nib.load("outputs/phase3_physics_fracture/ct_renderings/pybullet_fractured.nii.gz")
fractured_ct = fractured_ct_nii.get_fdata()

gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_mask_nii.get_fdata()

fractured_mask_nii = nib.load("outputs/phase3_physics_fracture/ct_renderings/pybullet_fractured_mask.nii.gz")
fractured_mask = fractured_mask_nii.get_fdata()

print(f"Original CT: {original_ct.shape}")
print(f"Fractured CT: {fractured_ct.shape}")

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

print(f"L1 center: {l1_center}")

vmin, vmax = -200, 1500

# 2x3 comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('PyBullet Fracture: Original vs Fractured CT (L1 Region)', 
             fontsize=16, fontweight='bold')

# Extract slices
orig_sag = original_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
orig_ax = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
orig_cor = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

frac_sag = fractured_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
frac_ax = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
frac_cor = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

# Row 1: Original
axes[0, 0].imshow(orig_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title('Original - Sagittal', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(orig_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title('Original - Axial', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(orig_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title('Original - Coronal', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Row 2: Fractured
axes[1, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].set_title('PyBullet Fractured - Sagittal', fontsize=12, color='red', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].set_title('PyBullet Fractured - Axial', fontsize=12, color='red', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].set_title('PyBullet Fractured - Coronal', fontsize=12, color='red', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()

output_path = "outputs/phase3_physics_fracture/visualizations/pybullet_fracture_ct.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')

print(f"\n✓ Saved: {output_path}")

# Difference visualization
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('CT Difference: Original vs PyBullet Fractured', fontsize=16, fontweight='bold')

diff_sag = frac_sag - orig_sag
diff_ax = frac_ax - orig_ax
diff_cor = frac_cor - orig_cor

axes2[0].imshow(diff_sag.T, cmap='RdBu_r', vmin=-100, vmax=100, origin='lower')
axes2[0].set_title('Difference - Sagittal', fontsize=12)
axes2[0].axis('off')

axes2[1].imshow(diff_ax.T, cmap='RdBu_r', vmin=-100, vmax=100, origin='lower')
axes2[1].set_title('Difference - Axial', fontsize=12)
axes2[1].axis('off')

axes2[2].imshow(diff_cor.T, cmap='RdBu_r', vmin=-100, vmax=100, origin='lower')
axes2[2].set_title('Difference - Coronal', fontsize=12)
axes2[2].axis('off')

plt.tight_layout()

diff_path = "outputs/phase3_physics_fracture/visualizations/pybullet_fracture_diff.png"
plt.savefig(diff_path, dpi=200, bbox_inches='tight')

print(f"✓ Saved: {diff_path}")

print("\n" + "="*70)
print("✓ Visualization complete!")
print("="*70)

