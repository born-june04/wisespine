#!/usr/bin/env python3
"""
Render fractured vertebra to CT volume.

Takes PyBullet simulation state and renders it as a CT image.
"""

import numpy as np
import nibabel as nib
import pybullet as p
import pybullet_data
import glob
import os
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

print("="*70)
print("FRACTURED VERTEBRA → CT RENDERING")
print("="*70)

# Step 1: Run simulation to get fractured state
def get_fractured_state():
    """Run PyBullet and return fragment positions."""
    print("\n--- Running PyBullet simulation ---")
    
    fragment_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"
    urdf_paths = sorted(glob.glob(os.path.join(fragment_dir, "L1_frag_*.urdf")))
    
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", [0, 0, -1.3])
    
    # Load fragments
    fragment_ids = []
    base_pos = [0.005, 0.2, -1.165]
    z_offset = 0.0
    
    for urdf_path in urdf_paths:
        pos = [base_pos[0], base_pos[1], base_pos[2] + z_offset]
        frag_id = p.loadURDF(urdf_path, pos)
        fragment_ids.append(frag_id)
        z_offset += 0.011
    
    # Create constraints
    constraints = []
    for i in range(len(fragment_ids) - 1):
        constraint_id = p.createConstraint(
            fragment_ids[i], -1,
            fragment_ids[i+1], -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.005],
            [0, 0, 0.005]
        )
        constraints.append(constraint_id)
    
    # Initial state (assembled)
    for _ in range(10):
        p.stepSimulation()
    
    initial_states = []
    for frag_id in fragment_ids:
        pos, orn = p.getBasePositionAndOrientation(frag_id)
        initial_states.append({'pos': np.array(pos), 'orn': np.array(orn)})
    
    # Apply force and break
    force = [0, 0, 10000]  # Very strong force
    for step in range(300):
        if step < 100:
            p.applyExternalForce(fragment_ids[-1], -1, force, [0, 0, 0], p.LINK_FRAME)
        
        # Manually break constraints at certain steps to ensure fracture
        if step == 120 and len(constraints) > 0:
            p.removeConstraint(constraints[-1])
            print(f"  💥 Manually broke top constraint")
        if step == 140 and len(constraints) > 1:
            p.removeConstraint(constraints[-2])
            print(f"  💥 Manually broke middle constraint")
        
        p.stepSimulation()
    
    # Get fractured state
    fractured_states = []
    for frag_id in fragment_ids:
        pos, orn = p.getBasePositionAndOrientation(frag_id)
        fractured_states.append({'pos': np.array(pos), 'orn': np.array(orn)})
        print(f"  Fragment at: {pos}")
    
    p.disconnect()
    
    return initial_states, fractured_states


# Step 2: Render to CT
def render_fractured_to_ct(initial_states, fractured_states, output_path):
    """Render fractured vertebra state to CT volume."""
    print("\n--- Rendering to CT ---")
    
    # Load original CT and GT mask
    original_ct_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
    original_ct = original_ct_nii.get_fdata()
    affine = original_ct_nii.affine
    
    gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    gt_mask = gt_mask_nii.get_fdata()
    
    # Start with original CT
    fractured_ct = original_ct.copy()
    
    # L1 label is 20
    l1_mask = (gt_mask == 20)
    
    print(f"  L1 mask: {l1_mask.sum()} voxels")
    
    # Divide L1 mask into 5 fragments (by Z coordinate)
    l1_coords = np.where(l1_mask)
    z_coords = l1_coords[2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_step = (z_max - z_min) / 5
    
    fragment_masks = []
    for i in range(5):
        z_start = z_min + i * z_step
        z_end = z_start + z_step
        frag_mask = l1_mask & (gt_mask != 0)  # Start with L1
        frag_mask = frag_mask & (np.arange(gt_mask.shape[2])[None, None, :] >= z_start)
        frag_mask = frag_mask & (np.arange(gt_mask.shape[2])[None, None, :] < z_end)
        fragment_masks.append(frag_mask)
        print(f"  Fragment {i} mask: {frag_mask.sum()} voxels")
    
    # Calculate displacements
    affine_inv = np.linalg.inv(affine)
    scale_factor = 1000.0  # PyBullet is meters, CT is mm
    
    # First, "erase" original L1 from CT
    fractured_ct[l1_mask] = -1000  # Air
    
    # Then, place each fragment at new position
    for i, (init_state, frac_state, frag_mask) in enumerate(zip(initial_states, fractured_states, fragment_masks)):
        # Calculate displacement in PyBullet space (meters)
        displacement_m = frac_state['pos'] - init_state['pos']
        displacement_mm = displacement_m * scale_factor
        
        # Convert to voxel space
        displacement_voxel = affine_inv[:3, :3] @ displacement_mm
        
        print(f"  Fragment {i}: displacement = {np.linalg.norm(displacement_mm):.1f} mm")
        print(f"                = ({displacement_voxel[0]:.1f}, {displacement_voxel[1]:.1f}, {displacement_voxel[2]:.1f}) voxels")
        
        # Shift fragment mask
        shifted_mask = ndi.shift(frag_mask.astype(float), 
                                 shift=displacement_voxel,
                                 order=1,
                                 mode='constant',
                                 cval=0.0)
        
        shifted_mask = (shifted_mask > 0.5)
        
        # Get original CT values for this fragment
        original_fragment_values = original_ct[frag_mask]
        
        # Place shifted fragment (use mean HU value)
        if shifted_mask.sum() > 0 and len(original_fragment_values) > 0:
            mean_hu = np.mean(original_fragment_values)
            fractured_ct[shifted_mask] = mean_hu
            print(f"    Placed {shifted_mask.sum()} voxels with HU={mean_hu:.0f}")
    
    # Save
    fractured_nii = nib.Nifti1Image(fractured_ct, affine=affine)
    nib.save(fractured_nii, output_path)
    
    print(f"\n✓ Saved: {output_path}")
    
    return fractured_ct, original_ct, l1_mask


# Step 3: Visualize
def visualize_fractured_ct(original_ct, fractured_ct, l1_mask, output_path):
    """Create visualization comparing original and fractured CT."""
    print("\n--- Creating visualization ---")
    
    # Find L1 center
    l1_coords = np.where(l1_mask)
    l1_center = [int(np.mean(c)) for c in l1_coords]
    
    margin = 50
    l1_bbox = [
        (max(0, l1_coords[0].min() - margin), min(original_ct.shape[0], l1_coords[0].max() + margin)),
        (max(0, l1_coords[1].min() - margin), min(original_ct.shape[1], l1_coords[1].max() + margin)),
        (max(0, l1_coords[2].min() - margin), min(original_ct.shape[2], l1_coords[2].max() + margin)),
    ]
    
    vmin, vmax = -200, 1500
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fractured Vertebra CT Rendering', fontsize=16, fontweight='bold')
    
    # Original CT
    orig_sag = original_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
    orig_ax = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
    orig_cor = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
    
    axes[0, 0].imshow(orig_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title('Original CT - Sagittal', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(orig_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title('Original CT - Axial', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(orig_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 2].set_title('Original CT - Coronal', fontsize=12)
    axes[0, 2].axis('off')
    
    # Fractured CT
    frac_sag = fractured_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
    frac_ax = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
    frac_cor = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
    
    axes[1, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 0].set_title('Fractured CT - Sagittal', fontsize=12, color='red', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 1].set_title('Fractured CT - Axial', fontsize=12, color='red', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 2].set_title('Fractured CT - Coronal', fontsize=12, color='red', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    
    print(f"✓ Saved: {output_path}")


# Main execution
print("\n=== STEP 1: Get fractured state from PyBullet ===")
initial_states, fractured_states = get_fractured_state()

print("\n=== STEP 2: Render to CT ===")
output_ct_path = "outputs/phase3_physics_fracture/ct_renderings/fractured_ct.nii.gz"
fractured_ct, original_ct, l1_mask = render_fractured_to_ct(
    initial_states, 
    fractured_states, 
    output_ct_path
)

print("\n=== STEP 3: Visualize ===")
output_viz_path = "outputs/phase3_physics_fracture/visualizations/fractured_ct_comparison.png"
visualize_fractured_ct(original_ct, fractured_ct, l1_mask, output_viz_path)

print("\n" + "="*70)
print("🎉 FRACTURED CT RENDERING COMPLETE!")
print("="*70)
print(f"\n📊 Output files:")
print(f"  • CT: {output_ct_path}")
print(f"  • Visualization: {output_viz_path}")
print("\n✨ You can now see the fractured vertebra in CT!")
print("="*70)

