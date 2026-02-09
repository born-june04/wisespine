#!/usr/bin/env python3
"""
Correct CT Rendering: Apply deformation field to original CT.

Instead of creating CT from scratch, we:
1. Load original CT (with all tissues)
2. Create deformation field from MuJoCo
3. Warp original CT using this field
→ Preserves all information, realistic!
"""

import numpy as np
import nibabel as nib
import mujoco
from scipy import ndimage as ndi
from scipy.interpolate import griddata


def create_deformation_field(
    gt_mask_path: str,
    mujoco_model_path: str,
    mujoco_data: mujoco.MjData,
    ct_shape: tuple,
    affine: np.ndarray,
):
    """
    Create 3D deformation field from MuJoCo displacements.
    
    Returns:
        deformation_field: [3, H, W, D] - displacement vector at each voxel
    """
    print("Creating deformation field...")
    
    # Load GT mask
    gt_nii = nib.load(gt_mask_path)
    gt_mask = gt_nii.get_fdata().astype(np.int16)
    
    # Label mapping
    label_map = {
        'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
        'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
        'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
        'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23,
    }
    
    # Load MuJoCo model for initial positions
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    data_init = mujoco.MjData(model)
    mujoco.mj_forward(model, data_init)
    
    # Initialize deformation field (displacement in voxel coordinates)
    deformation = np.zeros((3,) + ct_shape, dtype=np.float32)
    
    affine_inv = np.linalg.inv(affine)
    
    # For each vertebra, set its displacement
    for vert_name, label in label_map.items():
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, vert_name)
        except:
            continue
        
        # Get displacement in mm
        pos_init_mm = data_init.xpos[body_id] * 1000
        pos_curr_mm = mujoco_data.xpos[body_id] * 1000
        displacement_mm = pos_curr_mm - pos_init_mm
        
        # Convert to voxel displacement
        displacement_voxel = affine_inv[:3, :3] @ displacement_mm
        
        # Get vertebra mask
        vert_mask = (gt_mask == label)
        
        if vert_mask.sum() == 0:
            continue
        
        # Set deformation for this vertebra region
        for axis in range(3):
            deformation[axis][vert_mask] = displacement_voxel[axis]
        
        disp_norm = np.linalg.norm(displacement_mm)
        print(f"  {vert_name}: {disp_norm:.1f} mm displacement")
    
    # Smooth deformation field to affect surrounding tissue
    # This makes the deformation more realistic (nearby tissue moves too)
    print("\nSmoothing deformation field...")
    for axis in range(3):
        # Gaussian smoothing to spread deformation
        deformation[axis] = ndi.gaussian_filter(deformation[axis], sigma=5.0)
    
    return deformation


def apply_deformation(ct_data: np.ndarray, deformation: np.ndarray):
    """
    Warp CT using deformation field.
    
    Args:
        ct_data: Original CT [H, W, D]
        deformation: Deformation field [3, H, W, D]
    
    Returns:
        warped_ct: Deformed CT [H, W, D]
    """
    print("\nApplying deformation to CT...")
    
    # Create coordinate grid
    shape = ct_data.shape
    i, j, k = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    
    # Apply deformation: new_coords = old_coords + displacement
    i_warped = i + deformation[0]
    j_warped = j + deformation[1]
    k_warped = k + deformation[2]
    
    # Interpolate CT values at warped coordinates
    # Use map_coordinates for efficient interpolation
    warped_ct = ndi.map_coordinates(
        ct_data,
        [i_warped, j_warped, k_warped],
        order=1,  # Linear interpolation
        mode='constant',
        cval=-1000  # Air outside
    )
    
    print("✓ Deformation applied")
    
    return warped_ct


def render_deformed_ct(
    original_ct_path: str,
    gt_mask_path: str,
    mujoco_model_path: str,
    mujoco_data: mujoco.MjData,
    output_path: str,
):
    """
    Render deformed CT by warping original CT.
    
    This preserves all tissue information from original CT!
    """
    print("="*70)
    print("RENDERING DEFORMED CT (with warping)")
    print("="*70)
    
    # Load original CT
    ct_nii = nib.load(original_ct_path)
    ct_data = ct_nii.get_fdata()
    affine = ct_nii.affine
    
    print(f"\nOriginal CT:")
    print(f"  Shape: {ct_data.shape}")
    print(f"  HU range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    
    # Create deformation field
    deformation = create_deformation_field(
        gt_mask_path,
        mujoco_model_path,
        mujoco_data,
        ct_data.shape,
        affine
    )
    
    # Apply deformation
    warped_ct = apply_deformation(ct_data, deformation)
    
    print(f"\nWarped CT:")
    print(f"  Shape: {warped_ct.shape}")
    print(f"  HU range: [{warped_ct.min():.0f}, {warped_ct.max():.0f}]")
    
    # Save
    nii = nib.Nifti1Image(warped_ct, affine=affine)
    nib.save(nii, output_path)
    
    print(f"\n✓ Saved: {output_path}")
    print("="*70)
    
    return warped_ct


if __name__ == "__main__":
    original_ct = "VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz"
    gt_mask = "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
    mujoco_xml = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"
    
    # Load MuJoCo and apply gentle force
    model = mujoco.MjModel.from_xml_path(mujoco_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    l1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L1")
    
    # Gentle force
    force = np.array([0, 0, 2.0])
    for _ in range(5):
        data.xfrc_applied[l1_id, :3] = force
        mujoco.mj_step(model, data)
    
    # Render deformed CT
    warped_ct = render_deformed_ct(
        original_ct,
        gt_mask,
        mujoco_xml,
        data,
        "outputs/rendered_ct_warped.nii.gz"
    )
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    # Visualize
    import matplotlib.pyplot as plt
    
    original_nii = nib.load(original_ct)
    original = original_nii.get_fdata()
    
    bone_mask = original > 200
    coords = np.where(bone_mask)
    center = [int(np.mean(c)) for c in coords]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CT Warping: Original vs Deformed (ALL TISSUES PRESERVED)', 
                 fontsize=16, fontweight='bold')
    
    vmin, vmax = -200, 1500
    
    # Original
    axes[0, 0].imshow(original[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title('Original - Sagittal')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title('Original - Axial')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 2].set_title('Original - Coronal')
    axes[0, 2].axis('off')
    
    # Warped
    axes[1, 0].imshow(warped_ct[center[0], :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 0].set_title('Warped - Sagittal', color='red')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(warped_ct[:, :, center[2]].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 1].set_title('Warped - Axial', color='red')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(warped_ct[:, center[1], :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 2].set_title('Warped - Coronal', color='red')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/ct_warped_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ Saved: outputs/ct_warped_comparison.png")
    print("\n📍 Check: /gscratch/scrubbed/june0604/vindr/outputs/ct_warped_comparison.png")
    print("\n🎉 Now this looks like a REAL CT with deformation!")
    print("="*70)

