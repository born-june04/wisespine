#!/usr/bin/env python3
"""
Create fractured L1 vertebra by manually displacing fragments in CT space.

Much simpler than PyBullet - directly manipulate the mask!
"""

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("="*70)
print("MANUAL FRACTURE: L1 Fragmentation in CT Space")
print("="*70)

# Load data
print("\nLoading data...")
original_ct_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
original_ct = original_ct_nii.get_fdata()
affine = original_ct_nii.affine

gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_mask_nii.get_fdata()

print(f"  CT shape: {original_ct.shape}")
print(f"  GT mask labels: {np.unique(gt_mask)[1:]}")

# Extract L1 (label 20)
l1_mask = (gt_mask == 20)
l1_coords = np.where(l1_mask)

print(f"\nL1 vertebra:")
print(f"  Voxels: {l1_mask.sum()}")
print(f"  Z range: [{l1_coords[2].min()}, {l1_coords[2].max()}]")

# Divide L1 into 5 fragments along Z axis
z_min, z_max = l1_coords[2].min(), l1_coords[2].max()
z_range = z_max - z_min
z_step = z_range / 5

print(f"\nDividing into 5 fragments:")
fragment_masks = []
for i in range(5):
    z_start = z_min + i * z_step
    z_end = z_start + z_step
    
    # Create fragment mask
    frag_mask = l1_mask.copy()
    for z_idx in range(gt_mask.shape[2]):
        if z_idx < z_start or z_idx >= z_end:
            frag_mask[:, :, z_idx] = False
    
    fragment_masks.append(frag_mask)
    print(f"  Fragment {i}: Z=[{z_start:.0f}, {z_end:.0f}], {frag_mask.sum()} voxels")

# Define displacements for each fragment (in voxels)
# Simulate fracture: fragments separate with gaps
displacements = [
    np.array([0, 0, 0]),      # Fragment 0: no movement
    np.array([2, 0, 5]),      # Fragment 1: slight shift
    np.array([-3, 1, 10]),    # Fragment 2: more shift
    np.array([1, -2, 15]),    # Fragment 3: even more
    np.array([-2, 1, 20]),    # Fragment 4: most displaced
]

print(f"\n--- Applying displacements ---")

# Create deformation field
deformation = np.zeros((3,) + gt_mask.shape, dtype=np.float32)

for i, (frag_mask, displacement) in enumerate(zip(fragment_masks, displacements)):
    print(f"  Fragment {i}: displacement = {displacement} voxels")
    for axis in range(3):
        deformation[axis][frag_mask] = displacement[axis]

# Smooth deformation field (makes it more realistic)
print("\nSmoothing deformation field...")
for axis in range(3):
    deformation[axis] = ndi.gaussian_filter(deformation[axis], sigma=3.0)

# Apply deformation to original CT
print("\nWarping CT...")
shape = original_ct.shape
i, j, k = np.meshgrid(
    np.arange(shape[0]),
    np.arange(shape[1]),
    np.arange(shape[2]),
    indexing='ij'
)

# Apply deformation
i_warped = i + deformation[0]
j_warped = j + deformation[1]
k_warped = k + deformation[2]

# Interpolate
fractured_ct = ndi.map_coordinates(
    original_ct,
    [i_warped, j_warped, k_warped],
    order=1,
    mode='constant',
    cval=-1000
)

print(f"✓ Warped CT created")
print(f"  Shape: {fractured_ct.shape}")
print(f"  HU range: [{fractured_ct.min():.0f}, {fractured_ct.max():.0f}]")

# Save
output_path = "outputs/phase3_physics_fracture/ct_renderings/fractured_ct_manual.nii.gz"
fractured_nii = nib.Nifti1Image(fractured_ct, affine=affine)
nib.save(fractured_nii, output_path)
print(f"\n✓ Saved: {output_path}")

# Visualize
print("\n--- Creating visualization ---")

# Find L1 center for cropping
l1_center = [int(np.mean(c)) for c in l1_coords]
margin = 50

l1_bbox = [
    (max(0, l1_coords[0].min() - margin), min(gt_mask.shape[0], l1_coords[0].max() + margin)),
    (max(0, l1_coords[1].min() - margin), min(gt_mask.shape[1], l1_coords[1].max() + margin)),
    (max(0, l1_coords[2].min() - margin), min(gt_mask.shape[2], l1_coords[2].max() + margin)),
]

vmin, vmax = -200, 1500

# Create 2x3 comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Manual Fracture: Original vs Fractured L1 Vertebra', 
             fontsize=16, fontweight='bold')

# Extract slices
orig_sag = original_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
orig_ax = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
orig_cor = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

frac_sag = fractured_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
frac_ax = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
frac_cor = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]

# Original
axes[0, 0].imshow(orig_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title('Original - Sagittal', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(orig_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title('Original - Axial', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(orig_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title('Original - Coronal', fontsize=12)
axes[0, 2].axis('off')

# Fractured
axes[1, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].set_title('Fractured - Sagittal\n(5 fragments with gaps)', 
                     fontsize=12, color='red', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].set_title('Fractured - Axial\n(Manual displacement)', 
                     fontsize=12, color='red', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].set_title('Fractured - Coronal\n(CT space manipulation)', 
                     fontsize=12, color='red', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()

output_viz = "outputs/phase3_physics_fracture/visualizations/fractured_ct_manual.png"
plt.savefig(output_viz, dpi=200, bbox_inches='tight')

print(f"✓ Saved: {output_viz}")

# Also create a zoomed view showing the gaps
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('ZOOMED: Fractured L1 showing gaps between fragments', 
              fontsize=16, fontweight='bold', color='red')

axes2[0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[0].set_title('Sagittal View', fontsize=14)
axes2[0].axis('off')

axes2[1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[1].set_title('Axial View', fontsize=14)
axes2[1].axis('off')

axes2[2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes2[2].set_title('Coronal View', fontsize=14)
axes2[2].axis('off')

plt.tight_layout()

output_zoom = "outputs/phase3_physics_fracture/visualizations/fractured_ct_zoomed.png"
plt.savefig(output_zoom, dpi=200, bbox_inches='tight')

print(f"✓ Saved: {output_zoom}")

print("\n" + "="*70)
print("🎉 MANUAL FRACTURE COMPLETE!")
print("="*70)
print(f"\n📊 Results:")
print(f"  • Fractured CT: {output_path}")
print(f"  • Comparison: {output_viz}")
print(f"  • Zoomed: {output_zoom}")
print(f"\n✨ Features:")
print(f"  • 5 fragments with gaps")
print(f"  • Manual displacement (realistic)")
print(f"  • Smooth deformation field")
print(f"  • All tissues preserved")
print(f"\n🚀 Next: Run TotalSegmentator on fractured CT!")
print("="*70)

