#!/usr/bin/env python3
"""Create gentle deformation (no physics instability)."""

import numpy as np
import mujoco
import sys
sys.path.append('spine-rl-sim/modules')
from ct_renderer_v2 import render_ct_from_deformed_mask

print("="*70)
print("CREATING GENTLE DEFORMATION")
print("="*70)

gt_mask = "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
original_ct = "VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz"
mujoco_xml = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"

# Load MuJoCo
model = mujoco.MjModel.from_xml_path(mujoco_xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Get L1 body ID
l1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L1")

# Save initial position
initial_pos = data.xpos[l1_id].copy()
print(f"\nL1 initial position: {initial_pos} (meters)")

# Apply GENTLE force for SHORT time
force = np.array([0, 0, 10.0])  # Reduced from 50 to 10
print(f"Applying gentle force: {force} N for 10 steps...")

for i in range(10):  # Reduced from 100 to 10
    data.xfrc_applied[l1_id, :3] = force
    mujoco.mj_step(model, data)

final_pos = data.xpos[l1_id].copy()
displacement = (final_pos - initial_pos) * 1000  # to mm

print(f"L1 final position: {final_pos} (meters)")
print(f"Displacement: {displacement} mm")
print(f"Distance: {np.linalg.norm(displacement):.1f} mm")

if np.linalg.norm(displacement) > 200:
    print("\n⚠️  Still too much displacement! Using even gentler force...")
    
    # Reset and try even gentler
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    force = np.array([0, 0, 2.0])
    for i in range(5):
        data.xfrc_applied[l1_id, :3] = force
        mujoco.mj_step(model, data)
    
    final_pos = data.xpos[l1_id].copy()
    displacement = (final_pos - initial_pos) * 1000
    print(f"\nGentle displacement: {displacement} mm ({np.linalg.norm(displacement):.1f} mm)")

# Render
print("\nRendering deformed CT...")
ct_def, mask_def = render_ct_from_deformed_mask(
    gt_mask, original_ct, mujoco_xml, data,
    "outputs/rendered_ct_v2_gentle.nii.gz"
)

print("\n✓ Saved: outputs/rendered_ct_v2_gentle.nii.gz")
print(f"  Bone voxels: {(mask_def > 0).sum()}")

# Visualize
print("\nCreating visualization...")
import nibabel as nib
import matplotlib.pyplot as plt

original_nii = nib.load(original_ct)
original = original_nii.get_fdata()

rendered_nii = nib.load("outputs/rendered_ct_v2_gentle.nii.gz")
rendered = rendered_nii.get_fdata()

bone_mask = original > 200
coords = np.where(bone_mask)
center = [int(np.mean(c)) for c in coords]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('CT Rendering: Original vs Gentle Deformation', fontsize=16, fontweight='bold')

vmin, vmax = -200, 1500

axes[0, 0].imshow(original[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 0].set_title('Original - Sagittal')
axes[0, 0].axis('off')

axes[0, 1].imshow(original[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 1].set_title('Original - Axial')
axes[0, 1].axis('off')

axes[0, 2].imshow(original[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[0, 2].set_title('Original - Coronal')
axes[0, 2].axis('off')

axes[1, 0].imshow(rendered[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 0].set_title('Deformed - Sagittal', color='red')
axes[1, 0].axis('off')

axes[1, 1].imshow(rendered[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 1].set_title('Deformed - Axial', color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(rendered[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axes[1, 2].set_title('Deformed - Coronal', color='red')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ct_v2_gentle_comparison.png', dpi=150, bbox_inches='tight')

print("\n✓ Saved: outputs/ct_v2_gentle_comparison.png")
print("\n📍 Check: /gscratch/scrubbed/june0604/vindr/outputs/ct_v2_gentle_comparison.png")
print("="*70)

