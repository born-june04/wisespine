#!/usr/bin/env python3
"""Debug: Compare original CT with rendered CT to find coordinate issues."""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

print("="*70)
print("DEBUG: Analyzing CT coordinates")
print("="*70)

# Load original CT
original_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original = original_nii.get_fdata()

print(f"\nOriginal CT:")
print(f"  Shape: {original.shape}")
print(f"  HU range: [{original.min():.0f}, {original.max():.0f}]")
print(f"  Affine:\n{original_nii.affine}")

# Find where bone is in original
bone_original = original > 200  # Bone threshold
if bone_original.sum() > 0:
    coords = np.where(bone_original)
    print(f"\nBone location in original CT:")
    print(f"  X range: [{coords[0].min()}, {coords[0].max()}] / {original.shape[0]}")
    print(f"  Y range: [{coords[1].min()}, {coords[1].max()}] / {original.shape[1]}")
    print(f"  Z range: [{coords[2].min()}, {coords[2].max()}] / {original.shape[2]}")
    print(f"  Centroid: ({np.mean(coords[0]):.0f}, {np.mean(coords[1]):.0f}, {np.mean(coords[2]):.0f})")

# Load rendered CT
rendered_nii = nib.load("outputs/rendered_ct_initial.nii.gz")
rendered = rendered_nii.get_fdata()

print(f"\nRendered CT:")
print(f"  Shape: {rendered.shape}")
print(f"  HU range: [{rendered.min():.0f}, {rendered.max():.0f}]")

# Find where bone is in rendered
bone_rendered = rendered > 500
if bone_rendered.sum() > 0:
    coords = np.where(bone_rendered)
    print(f"\nBone location in rendered CT:")
    print(f"  X range: [{coords[0].min()}, {coords[0].max()}] / {rendered.shape[0]}")
    print(f"  Y range: [{coords[1].min()}, {coords[1].max()}] / {rendered.shape[1]}")
    print(f"  Z range: [{coords[2].min()}, {coords[2].max()}] / {rendered.shape[2]}")
    print(f"  Centroid: ({np.mean(coords[0]):.0f}, {np.mean(coords[1]):.0f}, {np.mean(coords[2]):.0f})")
else:
    print(f"\n⚠️ NO BONE FOUND in rendered CT!")

# Visualize both at correct location
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('DEBUG: Original vs Rendered CT (at bone centroids)', fontsize=14, fontweight='bold')

# Get centroids for slicing
if bone_original.sum() > 0:
    coords_orig = np.where(bone_original)
    center_orig = [int(np.mean(c)) for c in coords_orig]
else:
    center_orig = [s//2 for s in original.shape]

if bone_rendered.sum() > 0:
    coords_rend = np.where(bone_rendered)
    center_rend = [int(np.mean(c)) for c in coords_rend]
else:
    center_rend = [s//2 for s in rendered.shape]

vmin, vmax = -200, 1500

# Original CT
axes[0, 0].imshow(original[center_orig[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title(f'Original - Sagittal (x={center_orig[0]})', fontsize=10)
axes[0, 0].axis('off')

axes[0, 1].imshow(original[:, :, center_orig[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title(f'Original - Axial (z={center_orig[2]})', fontsize=10)
axes[0, 1].axis('off')

axes[0, 2].imshow(original[:, center_orig[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title(f'Original - Coronal (y={center_orig[1]})', fontsize=10)
axes[0, 2].axis('off')

# Rendered CT
axes[1, 0].imshow(rendered[center_rend[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].set_title(f'Rendered - Sagittal (x={center_rend[0]})', fontsize=10, color='red')
axes[1, 0].axis('off')

axes[1, 1].imshow(rendered[:, :, center_rend[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].set_title(f'Rendered - Axial (z={center_rend[2]})', fontsize=10, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(rendered[:, center_rend[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].set_title(f'Rendered - Coronal (y={center_rend[1]})', fontsize=10, color='red')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/debug_ct_coordinates.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved debug visualization: outputs/debug_ct_coordinates.png")

print("\n" + "="*70)
print("PROBLEM IDENTIFIED:")
print("  The rendered vertebrae are in wrong location!")
print("  Need to fix coordinate transformation in ct_renderer.py")
print("="*70)

