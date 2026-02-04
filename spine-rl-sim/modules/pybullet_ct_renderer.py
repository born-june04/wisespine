#!/usr/bin/env python3
"""
Render PyBullet fracture state to CT volume.

This converts PyBullet fragment positions into a fractured CT by:
1. Getting fragment displacements from PyBullet
2. Creating fractured mask by applying displacements
3. Warping original CT based on deformation field
"""

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from typing import List, Tuple


def pybullet_to_ct_displacement(
    pybullet_displacement: np.ndarray,
    pybullet_scale: float = 0.001,
    ct_spacing: np.ndarray = np.array([1.0, 1.0, 1.0])
) -> np.ndarray:
    """
    Convert PyBullet displacement (meters) to CT displacement (voxels).
    
    The key insight:
    - Mesh was loaded with scale=0.001, so 1 PyBullet unit = 1mm in physical space
    - PyBullet displacement is in PyBullet units (NOT meters!)
    - Just need to convert from PyBullet units to voxels
    
    Args:
        pybullet_displacement: [dx, dy, dz] in PyBullet units
        pybullet_scale: Scale factor used when loading mesh (default 0.001)
        ct_spacing: Voxel spacing in mm (from CT affine)
    
    Returns:
        ct_displacement: [dx, dy, dz] in voxels
    """
    # PyBullet displacement is already in physical mm due to scale=0.001
    # (1 PyBullet unit = 1mm when mesh loaded with scale 0.001)
    displacement_mm = pybullet_displacement * 1000.0  # Convert from meters to mm
    
    # Convert to voxels
    ct_displacement = displacement_mm / ct_spacing
    
    return ct_displacement


def create_fractured_mask_from_pybullet(
    gt_mask: np.ndarray,
    l1_label: int,
    fragment_displacements: List[Tuple[int, np.ndarray]],
    num_fragments: int = 5,
    pybullet_scale: float = 0.001,
    ct_spacing: np.ndarray = np.array([1.0, 1.0, 1.0])
) -> np.ndarray:
    """
    Create fractured mask from PyBullet displacements.
    
    Args:
        gt_mask: Ground truth mask
        l1_label: Label for L1 vertebra
        fragment_displacements: List of (fragment_idx, displacement_3d) from PyBullet
        num_fragments: Number of fragments
        pybullet_scale: Mesh scale in PyBullet
        ct_spacing: CT voxel spacing
    
    Returns:
        fractured_mask: Mask with displaced fragments
    """
    # Extract L1 mask
    l1_mask = (gt_mask == l1_label)
    
    # Get L1 bounding box
    coords = np.where(l1_mask)
    z_min, z_max = coords[2].min(), coords[2].max()
    
    # Divide L1 into fragments by Z-slices
    z_step = (z_max - z_min) / num_fragments
    fragment_masks = []
    
    for i in range(num_fragments):
        z_start = z_min + i * z_step
        z_end = z_min + (i + 1) * z_step
        
        frag_mask = l1_mask.copy()
        frag_mask[..., :int(z_start)] = 0
        frag_mask[..., int(z_end):] = 0
        
        fragment_masks.append(frag_mask)
    
    # Create fractured mask
    fractured_mask = gt_mask.copy()
    
    # Remove original L1
    fractured_mask[l1_mask] = 0
    
    # Add displaced fragments
    for frag_idx, pybullet_disp in fragment_displacements:
        if frag_idx >= len(fragment_masks):
            continue
        
        frag_mask = fragment_masks[frag_idx]
        
        # Convert PyBullet displacement to voxels
        disp_voxels = pybullet_to_ct_displacement(
            pybullet_disp,
            pybullet_scale,
            ct_spacing
        )
        
        print(f"  Fragment {frag_idx}: PyBullet={pybullet_disp}, voxels={disp_voxels}")
        
        # Apply displacement using shift
        frag_coords = np.array(np.where(frag_mask)).T
        
        if len(frag_coords) == 0:
            continue
        
        # Shift coordinates
        new_coords = frag_coords + disp_voxels.astype(int)
        
        # Clip to valid range
        new_coords[:, 0] = np.clip(new_coords[:, 0], 0, fractured_mask.shape[0]-1)
        new_coords[:, 1] = np.clip(new_coords[:, 1], 0, fractured_mask.shape[1]-1)
        new_coords[:, 2] = np.clip(new_coords[:, 2], 0, fractured_mask.shape[2]-1)
        
        # Set in fractured mask
        fractured_mask[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = l1_label
    
    return fractured_mask


def create_deformation_field_from_fragments(
    gt_mask: np.ndarray,
    l1_label: int,
    fragment_displacements: List[Tuple[int, np.ndarray]],
    num_fragments: int = 5,
    pybullet_scale: float = 0.001,
    ct_spacing: np.ndarray = np.array([1.0, 1.0, 1.0])
) -> np.ndarray:
    """
    Create 3D deformation field from fragment displacements.
    
    Returns:
        deformation_field: [H, W, D, 3] - displacement at each voxel
    """
    H, W, D = gt_mask.shape
    deformation = np.zeros((H, W, D, 3), dtype=np.float32)
    
    # Extract L1 mask
    l1_mask = (gt_mask == l1_label)
    coords = np.where(l1_mask)
    z_min, z_max = coords[2].min(), coords[2].max()
    
    # Divide into fragments
    z_step = (z_max - z_min) / num_fragments
    
    for frag_idx, pybullet_disp in fragment_displacements:
        if frag_idx >= num_fragments:
            continue
        
        z_start = z_min + frag_idx * z_step
        z_end = z_min + (frag_idx + 1) * z_step
        
        # Create fragment mask
        frag_mask = l1_mask.copy()
        frag_mask[..., :int(z_start)] = 0
        frag_mask[..., int(z_end):] = 0
        
        # Convert displacement to voxels
        disp_voxels = pybullet_to_ct_displacement(
            pybullet_disp,
            pybullet_scale,
            ct_spacing
        )
        
        # Apply constant displacement to this fragment
        deformation[frag_mask] = disp_voxels
    
    return deformation


def apply_deformation_to_ct(
    original_ct: np.ndarray,
    deformation_field: np.ndarray
) -> np.ndarray:
    """
    Apply deformation field to CT.
    
    Args:
        original_ct: Original CT volume [H, W, D]
        deformation_field: Displacement at each voxel [H, W, D, 3]
    
    Returns:
        deformed_ct: Warped CT volume
    """
    H, W, D = original_ct.shape
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.arange(W),
        np.arange(H),
        np.arange(D),
        indexing='xy'
    )
    
    # Apply deformation
    x_new = x + deformation_field[..., 1]  # Note: deformation is [dx, dy, dz] in i,j,k
    y_new = y + deformation_field[..., 0]
    z_new = z + deformation_field[..., 2]
    
    # Sample from original CT
    from scipy.ndimage import map_coordinates
    
    coords = np.array([y_new.ravel(), x_new.ravel(), z_new.ravel()])
    deformed_ct = map_coordinates(original_ct, coords, order=1, mode='nearest')
    deformed_ct = deformed_ct.reshape(H, W, D)
    
    return deformed_ct


def render_pybullet_to_ct(
    pybullet_env,
    output_ct_path: str = None,
    output_mask_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render current PyBullet state to CT volume.
    
    Args:
        pybullet_env: PyBulletFractureEnv instance
        output_ct_path: Optional path to save rendered CT
        output_mask_path: Optional path to save fractured mask
    
    Returns:
        (fractured_ct, fractured_mask)
    """
    print("Rendering PyBullet state to CT...")
    
    # Get fragment displacements
    displacements = pybullet_env.get_fragment_displacements()
    
    print(f"  Got {len(displacements)} fragment displacements")
    
    # Get CT spacing from affine
    affine = pybullet_env.gt_ct_nii.affine
    ct_spacing = np.abs(np.diag(affine[:3, :3]))
    
    print(f"  CT spacing: {ct_spacing}")
    
    # Create deformation field
    deformation_field = create_deformation_field_from_fragments(
        pybullet_env.gt_mask,
        pybullet_env.l1_label,
        displacements,
        pybullet_env.num_fragments,
        ct_spacing=ct_spacing
    )
    
    print("  Created deformation field")
    
    # Apply to original CT
    fractured_ct = apply_deformation_to_ct(
        pybullet_env.gt_ct,
        deformation_field
    )
    
    print("  Applied deformation to CT")
    
    # Create fractured mask
    fractured_mask = create_fractured_mask_from_pybullet(
        pybullet_env.gt_mask,
        pybullet_env.l1_label,
        displacements,
        pybullet_env.num_fragments,
        ct_spacing=ct_spacing
    )
    
    print("  Created fractured mask")
    
    # Save if paths provided
    if output_ct_path:
        fractured_ct_nii = nib.Nifti1Image(
            fractured_ct,
            pybullet_env.gt_ct_nii.affine,
            pybullet_env.gt_ct_nii.header
        )
        nib.save(fractured_ct_nii, output_ct_path)
        print(f"  ✓ Saved CT: {output_ct_path}")
    
    if output_mask_path:
        fractured_mask_nii = nib.Nifti1Image(
            fractured_mask.astype(np.int16),
            pybullet_env.gt_mask_nii.affine,
            pybullet_env.gt_mask_nii.header
        )
        nib.save(fractured_mask_nii, output_mask_path)
        print(f"  ✓ Saved mask: {output_mask_path}")
    
    return fractured_ct, fractured_mask


if __name__ == "__main__":
    """Test CT rendering from PyBullet."""
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from modules.pybullet_fracture_env import PyBulletFractureEnv
    
    print("="*70)
    print("PyBullet → CT Rendering Test")
    print("="*70)
    
    # Create environment with tuned parameters
    env = PyBulletFractureEnv(
        gui=False,
        max_force=0.0001,  # Tuned
        max_torque=0.00001
    )
    env.reset()
    
    # Apply realistic random forces
    print("\nApplying tuned forces to create realistic fracture...")
    for i in range(10):  # 10 steps
        action = env.action_space.sample()
        env.step(action)
    
    print(f"  Applied {10} steps of random forces")
    
    # Render to CT
    print("\n" + "="*70)
    fractured_ct, fractured_mask = render_pybullet_to_ct(
        env,
        output_ct_path="outputs/phase3_physics_fracture/ct_renderings/pybullet_fractured.nii.gz",
        output_mask_path="outputs/phase3_physics_fracture/ct_renderings/pybullet_fractured_mask.nii.gz"
    )
    
    print("\n" + "="*70)
    print("✓ Rendering complete!")
    print("="*70)
    print(f"\nGenerated:")
    print(f"  • CT: pybullet_fractured.nii.gz")
    print(f"  • Mask: pybullet_fractured_mask.nii.gz")
    print(f"\n🚀 Next: Run TotalSegmentator on pybullet_fractured.nii.gz")
    print("="*70)
    
    env.close()

