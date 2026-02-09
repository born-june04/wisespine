#!/usr/bin/env python3
"""
Simplified CT Renderer: Use GT mask directly, just apply MuJoCo deformations.

Much simpler and more accurate than voxelizing meshes!
"""

import numpy as np
import nibabel as nib
import mujoco
from scipy import ndimage as ndi
from typing import Tuple


def render_ct_from_deformed_mask(
    gt_mask_path: str,
    original_ct_path: str,
    mujoco_model_path: str,
    mujoco_data: mujoco.MjData,
    output_path: str,
):
    """
    Render CT by deforming GT mask according to MuJoCo positions.
    
    Strategy:
    1. Load GT mask (each vertebra = one label)
    2. For each vertebra:
       - Get displacement from MuJoCo vs initial position
       - Translate that label in the mask
    3. Convert mask back to CT (assign HU values)
    4. Save as NIfTI
    """
    print("Loading data...")
    
    # Load GT mask
    gt_nii = nib.load(gt_mask_path)
    gt_mask = gt_nii.get_fdata().astype(np.int16)
    affine = gt_nii.affine
    
    # Load original CT (for HU reference)
    ct_nii = nib.load(original_ct_path)
    ct_data = ct_nii.get_fdata()
    
    print(f"  GT mask shape: {gt_mask.shape}")
    print(f"  Labels: {np.unique(gt_mask)[1:]}")
    
    # Label mapping (vertebra name → GT label)
    label_map = {
        'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
        'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
        'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
        'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23,
    }
    
    # Load MuJoCo model to get initial positions
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    data_init = mujoco.MjData(model)
    mujoco.mj_forward(model, data_init)
    
    # Create deformed mask
    deformed_mask = np.zeros_like(gt_mask)
    
    print("\nApplying deformations...")
    
    affine_inv = np.linalg.inv(affine)
    
    for vert_name, label in label_map.items():
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, vert_name)
        except:
            continue
        
        # Get initial and current positions (in mm)
        pos_init_mm = data_init.xpos[body_id] * 1000
        pos_curr_mm = mujoco_data.xpos[body_id] * 1000
        
        # Displacement in physical space (mm)
        displacement_mm = pos_curr_mm - pos_init_mm
        
        # Convert displacement to voxel space
        # displacement_mm is [dx, dy, dz] in mm
        # We need to convert using affine matrix
        # Simplified: just use rotation part of affine
        displacement_voxel = (affine_inv[:3, :3] @ displacement_mm)
        
        # Get this vertebra's mask
        vert_mask = (gt_mask == label)
        
        if vert_mask.sum() == 0:
            continue
        
        # Translate the mask
        # scipy.ndimage.shift shifts by given offset
        shifted_mask = ndi.shift(vert_mask.astype(float), 
                                 shift=displacement_voxel,
                                 order=1,  # Linear interpolation
                                 mode='constant',
                                 cval=0.0)
        
        # Threshold back to binary
        shifted_mask = (shifted_mask > 0.5)
        
        # Add to deformed mask
        deformed_mask[shifted_mask] = label
        
        disp_norm = np.linalg.norm(displacement_mm)
        print(f"  {vert_name} (label {label}): displaced {disp_norm:.1f} mm "
              f"({displacement_voxel[0]:.1f}, {displacement_voxel[1]:.1f}, {displacement_voxel[2]:.1f} voxels)")
    
    # Convert mask to CT
    # Simple approach: bone where mask > 0
    rendered_ct = np.full_like(ct_data, -1000, dtype=np.float32)  # Air
    rendered_ct[deformed_mask > 0] = 1000  # Bone
    
    # Save
    nii = nib.Nifti1Image(rendered_ct, affine=affine)
    nib.save(nii, output_path)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Bone voxels: {(deformed_mask > 0).sum()}")
    
    return rendered_ct, deformed_mask


if __name__ == "__main__":
    print("="*70)
    print("SIMPLIFIED CT RENDERING (using GT mask)")
    print("="*70)
    
    gt_mask = "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
    original_ct = "VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz"
    mujoco_xml = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"
    
    # Test 1: No deformation
    print("\n--- Test 1: Initial state (no deformation) ---")
    model = mujoco.MjModel.from_xml_path(mujoco_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    ct_init, mask_init = render_ct_from_deformed_mask(
        gt_mask, original_ct, mujoco_xml, data,
        "outputs/rendered_ct_v2_initial.nii.gz"
    )
    
    # Test 2: With deformation
    print("\n--- Test 2: L1 displaced ---")
    l1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L1")
    force = np.array([0, 0, 50.0])
    
    for _ in range(100):
        data.xfrc_applied[l1_id, :3] = force
        mujoco.mj_step(model, data)
    
    ct_def, mask_def = render_ct_from_deformed_mask(
        gt_mask, original_ct, mujoco_xml, data,
        "outputs/rendered_ct_v2_deformed.nii.gz"
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("✓ outputs/rendered_ct_v2_initial.nii.gz")
    print("✓ outputs/rendered_ct_v2_deformed.nii.gz")
    print("\nNow visualizing...")
    print("="*70)

