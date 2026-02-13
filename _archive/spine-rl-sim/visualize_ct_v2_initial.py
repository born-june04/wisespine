#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Load CTs
original_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original = original_nii.get_fdata()

rendered_nii = nib.load("outputs/rendered_ct_v2_initial.nii.gz")
rendered = rendered_nii.get_fdata()

# Find bone center
bone_mask = original > 200
coords = np.where(bone_mask)
center = [int(np.mean(c)) for c in coords]

print(f"Bone center at voxel: {center}")
print(f"Original CT range: [{original.min():.0f}, {original.max():.0f}]")
print(f"Rendered CT range: [{rendered.min():.0f}, {rendered.max():.0f}]")
print(f"Rendered bone voxels: {(rendered > 500).sum()}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('CT Rendering V2: Original vs Rendered (Initial State)', fontsize=16, fontweight='bold')

vmin, vmax = -200, 1500

# Original
axes[0, 0].imshow(original[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title('Original CT - Sagittal', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(original[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title('Original CT - Axial', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(original[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title('Original CT - Coronal', fontsize=12)
axes[0, 2].axis('off')

# Rendered
axes[1, 0].imshow(rendered[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].set_title('Rendered CT - Sagittal', fontsize=12, color='red')
axes[1, 0].axis('off')

axes[1, 1].imshow(rendered[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].set_title('Rendered CT - Axial', fontsize=12, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(rendered[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].set_title('Rendered CT - Coronal', fontsize=12, color='red')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ct_v2_initial.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: outputs/ct_v2_initial.png")
print("\n📍 Location: /gscratch/scrubbed/june0604/vindr/outputs/ct_v2_initial.png")

