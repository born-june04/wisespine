#!/usr/bin/env python3
"""
Visualize CT deformation with ZOOM on deformed region.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print("="*70)
print("CREATING ZOOMED COMPARISON")
print("="*70)

# Load CTs
original_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original = original_nii.get_fdata()

warped_nii = nib.load("outputs/rendered_ct_warped.nii.gz")
warped = warped_nii.get_fdata()

# Load GT mask to find L1
gt_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_nii.get_fdata()

# Find L1 region (label 20)
l1_mask = (gt_mask == 20)
l1_coords = np.where(l1_mask)

if len(l1_coords[0]) == 0:
    print("Error: L1 not found!")
    exit(1)

# Get L1 bounding box with margin
margin = 50  # voxels
l1_bbox = [
    (max(0, l1_coords[0].min() - margin), min(original.shape[0], l1_coords[0].max() + margin)),
    (max(0, l1_coords[1].min() - margin), min(original.shape[1], l1_coords[1].max() + margin)),
    (max(0, l1_coords[2].min() - margin), min(original.shape[2], l1_coords[2].max() + margin)),
]

l1_center = [int(np.mean(c)) for c in l1_coords]

print(f"\nL1 location:")
print(f"  Center: {l1_center}")
print(f"  Bbox: X=[{l1_bbox[0][0]}, {l1_bbox[0][1]}], "
      f"Y=[{l1_bbox[1][0]}, {l1_bbox[1][1]}], "
      f"Z=[{l1_bbox[2][0]}, {l1_bbox[2][1]}]")

# Create visualization with 3 rows:
# Row 1: Full view (original)
# Row 2: Full view (warped) with bbox
# Row 3: Zoomed view (warped)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('CT Deformation: Full View → Zoomed on L1 Region', fontsize=16, fontweight='bold')

vmin, vmax = -200, 1500

# === Row 1: Original (full) ===
ax = axes[0, 0]
slice_sag = original[l1_center[0], :, :]
ax.imshow(slice_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, original.shape[1], 0, original.shape[2]])
ax.set_title('Original - Sagittal (full)', fontsize=11)
ax.axis('on')
ax.set_xlabel('Y')
ax.set_ylabel('Z')

ax = axes[0, 1]
slice_ax = original[:, :, l1_center[2]]
ax.imshow(slice_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, original.shape[0], 0, original.shape[1]])
ax.set_title('Original - Axial (full)', fontsize=11)
ax.axis('on')
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax = axes[0, 2]
slice_cor = original[:, l1_center[1], :]
ax.imshow(slice_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, original.shape[0], 0, original.shape[2]])
ax.set_title('Original - Coronal (full)', fontsize=11)
ax.axis('on')
ax.set_xlabel('X')
ax.set_ylabel('Z')

# === Row 2: Warped (full) with bbox ===
ax = axes[1, 0]
slice_sag = warped[l1_center[0], :, :]
ax.imshow(slice_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, warped.shape[1], 0, warped.shape[2]])
# Draw bbox
rect = Rectangle((l1_bbox[1][0], l1_bbox[2][0]), 
                 l1_bbox[1][1]-l1_bbox[1][0], 
                 l1_bbox[2][1]-l1_bbox[2][0],
                 linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title('Warped - Sagittal (bbox shown)', fontsize=11, color='red')
ax.axis('on')
ax.set_xlabel('Y')
ax.set_ylabel('Z')

ax = axes[1, 1]
slice_ax = warped[:, :, l1_center[2]]
ax.imshow(slice_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, warped.shape[0], 0, warped.shape[1]])
rect = Rectangle((l1_bbox[0][0], l1_bbox[1][0]), 
                 l1_bbox[0][1]-l1_bbox[0][0], 
                 l1_bbox[1][1]-l1_bbox[1][0],
                 linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title('Warped - Axial (bbox shown)', fontsize=11, color='red')
ax.axis('on')
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax = axes[1, 2]
slice_cor = warped[:, l1_center[1], :]
ax.imshow(slice_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', extent=[0, warped.shape[0], 0, warped.shape[2]])
rect = Rectangle((l1_bbox[0][0], l1_bbox[2][0]), 
                 l1_bbox[0][1]-l1_bbox[0][0], 
                 l1_bbox[2][1]-l1_bbox[2][0],
                 linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title('Warped - Coronal (bbox shown)', fontsize=11, color='red')
ax.axis('on')
ax.set_xlabel('X')
ax.set_ylabel('Z')

# === Row 3: Zoomed on L1 ===
ax = axes[2, 0]
slice_sag_zoom = warped[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
ax.imshow(slice_sag_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('ZOOMED - Sagittal (L1 region)', fontsize=11, fontweight='bold', color='darkred')
ax.axis('off')

ax = axes[2, 1]
slice_ax_zoom = warped[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
ax.imshow(slice_ax_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('ZOOMED - Axial (L1 region)', fontsize=11, fontweight='bold', color='darkred')
ax.axis('off')

ax = axes[2, 2]
slice_cor_zoom = warped[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
ax.imshow(slice_cor_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('ZOOMED - Coronal (L1 region)', fontsize=11, fontweight='bold', color='darkred')
ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/ct_warped_zoomed.png', dpi=200, bbox_inches='tight')

print("\n✓ Saved: outputs/ct_warped_zoomed.png")

# Also create side-by-side original vs warped zoom
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('L1 Region: Original vs Warped (ZOOMED)', fontsize=16, fontweight='bold')

# Original zoomed
ax = axes2[0, 0]
slice_sag_zoom_orig = original[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
ax.imshow(slice_sag_zoom_orig.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Original - Sagittal', fontsize=12)
ax.axis('off')

ax = axes2[0, 1]
slice_ax_zoom_orig = original[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
ax.imshow(slice_ax_zoom_orig.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Original - Axial', fontsize=12)
ax.axis('off')

ax = axes2[0, 2]
slice_cor_zoom_orig = original[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
ax.imshow(slice_cor_zoom_orig.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Original - Coronal', fontsize=12)
ax.axis('off')

# Warped zoomed
ax = axes2[1, 0]
ax.imshow(slice_sag_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Warped - Sagittal (displaced)', fontsize=12, color='red')
ax.axis('off')

ax = axes2[1, 1]
ax.imshow(slice_ax_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Warped - Axial (displaced)', fontsize=12, color='red')
ax.axis('off')

ax = axes2[1, 2]
ax.imshow(slice_cor_zoom.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
ax.set_title('Warped - Coronal (displaced)', fontsize=12, color='red')
ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/ct_warped_sidebyside.png', dpi=200, bbox_inches='tight')

print("✓ Saved: outputs/ct_warped_sidebyside.png")

print("\n" + "="*70)
print("RESULTS:")
print("  📊 outputs/ct_warped_zoomed.png      - Full → Zoomed progression")
print("  📊 outputs/ct_warped_sidebyside.png  - Original vs Warped (zoomed)")
print("\n👀 Now you can clearly see L1 deformation!")
print("="*70)

