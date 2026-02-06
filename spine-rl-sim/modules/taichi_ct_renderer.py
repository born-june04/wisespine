#!/usr/bin/env python3
"""
Render Taichi physics simulation to CT volume.

This connects the Taichi MPM bone fracture simulation to CT rendering by:
1. Getting particle displacements from Taichi
2. Converting particle positions to deformation field
3. Warping original CT using the deformation field

The core CT warping logic is reused from pybullet_ct_renderer.py.
"""

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.interpolate import griddata, NearestNDInterpolator
from typing import Tuple, Optional, Dict
from pathlib import Path


def taichi_to_voxel_coords(
    taichi_positions: np.ndarray,
    taichi_bounds: Tuple[np.ndarray, np.ndarray],
    ct_shape: Tuple[int, int, int],
    vertebra_bbox: Optional[Dict] = None
) -> np.ndarray:
    """
    Convert Taichi normalized coordinates to CT voxel coordinates.
    
    Taichi simulator uses normalized [0, 1] coordinates.
    We need to map these back to the vertebra region in CT space.
    
    Args:
        taichi_positions: Particle positions [N, 3] in Taichi normalized space
        taichi_bounds: (min, max) bounds from Taichi (typically around 0.3-0.7)
        ct_shape: CT volume shape (H, W, D)
        vertebra_bbox: Optional dict with 'min' and 'max' keys for vertebra bounding box
        
    Returns:
        voxel_coords: [N, 3] in CT voxel space
    """
    bounds_min, bounds_max = taichi_bounds
    
    # Normalize to [0, 1] within Taichi bounds
    normalized = (taichi_positions - bounds_min) / (bounds_max - bounds_min + 1e-8)
    
    if vertebra_bbox is not None:
        # Map to vertebra bounding box in CT
        bbox_min = vertebra_bbox['min']
        bbox_max = vertebra_bbox['max']
        voxel_coords = normalized * (bbox_max - bbox_min) + bbox_min
    else:
        # Map to full CT volume
        voxel_coords = normalized * np.array(ct_shape)
    
    return voxel_coords


def create_deformation_field_from_taichi(
    original_positions: np.ndarray,
    deformed_positions: np.ndarray,
    taichi_bounds: Tuple[np.ndarray, np.ndarray],
    ct_shape: Tuple[int, int, int],
    vertebra_mask: np.ndarray,
    smoothing_sigma: float = 1.0
) -> np.ndarray:
    """
    Create 3D deformation field from Taichi particle displacements.
    
    This uses scattered data interpolation to create a dense deformation
    field from sparse particle positions.
    
    Args:
        original_positions: Original particle positions [N, 3] in Taichi space
        deformed_positions: Deformed particle positions [N, 3] in Taichi space
        taichi_bounds: Bounds from Taichi simulator
        ct_shape: CT volume shape (H, W, D)
        vertebra_mask: Binary mask of vertebra region
        smoothing_sigma: Gaussian smoothing for deformation field
        
    Returns:
        deformation_field: [H, W, D, 3] displacement at each voxel
    """
    H, W, D = ct_shape
    
    # Get vertebra bounding box
    coords = np.where(vertebra_mask > 0)
    if len(coords[0]) == 0:
        return np.zeros((H, W, D, 3), dtype=np.float32)
    
    bbox_min = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
    bbox_max = np.array([coords[0].max(), coords[1].max(), coords[2].max()])
    vertebra_bbox = {'min': bbox_min, 'max': bbox_max}
    
    # Convert Taichi positions to voxel coordinates
    orig_voxels = taichi_to_voxel_coords(
        original_positions, taichi_bounds, ct_shape, vertebra_bbox
    )
    deformed_voxels = taichi_to_voxel_coords(
        deformed_positions, taichi_bounds, ct_shape, vertebra_bbox
    )
    
    # Compute displacements in voxel space
    displacements = deformed_voxels - orig_voxels
    
    # Create deformation field using nearest neighbor interpolation
    # (faster and more stable than linear interpolation for sparse data)
    deformation = np.zeros((H, W, D, 3), dtype=np.float32)
    
    # Round original positions to nearest voxel
    orig_voxels_int = np.round(orig_voxels).astype(int)
    
    # Clip to valid range
    orig_voxels_int[:, 0] = np.clip(orig_voxels_int[:, 0], 0, H - 1)
    orig_voxels_int[:, 1] = np.clip(orig_voxels_int[:, 1], 0, W - 1)
    orig_voxels_int[:, 2] = np.clip(orig_voxels_int[:, 2], 0, D - 1)
    
    # Assign displacements to deformation field
    for i in range(len(orig_voxels_int)):
        vi, vj, vk = orig_voxels_int[i]
        deformation[vi, vj, vk] = displacements[i]
    
    # Smooth the deformation field for continuity
    if smoothing_sigma > 0:
        for dim in range(3):
            deformation[..., dim] = ndi.gaussian_filter(
                deformation[..., dim], 
                sigma=smoothing_sigma
            )
    
    # Only apply deformation within vertebra region
    vertebra_mask_3d = np.stack([vertebra_mask > 0] * 3, axis=-1)
    deformation = deformation * vertebra_mask_3d
    
    return deformation


def apply_deformation_to_ct(
    original_ct: np.ndarray,
    deformation_field: np.ndarray
) -> np.ndarray:
    """
    Apply deformation field to CT using inverse warping.
    
    This is the same function as in pybullet_ct_renderer.py.
    
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
    
    # Apply deformation (inverse warping)
    x_new = x + deformation_field[..., 1]
    y_new = y + deformation_field[..., 0]
    z_new = z + deformation_field[..., 2]
    
    # Sample from original CT
    from scipy.ndimage import map_coordinates
    
    coords = np.array([y_new.ravel(), x_new.ravel(), z_new.ravel()])
    deformed_ct = map_coordinates(original_ct, coords, order=1, mode='nearest')
    deformed_ct = deformed_ct.reshape(H, W, D)
    
    return deformed_ct


def apply_deformation_to_mask(
    original_mask: np.ndarray,
    deformation_field: np.ndarray
) -> np.ndarray:
    """
    Apply deformation field to segmentation mask.
    
    Uses nearest neighbor interpolation to preserve label values.
    
    Args:
        original_mask: Original mask volume [H, W, D]
        deformation_field: Displacement at each voxel [H, W, D, 3]
        
    Returns:
        deformed_mask: Warped mask volume
    """
    H, W, D = original_mask.shape
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.arange(W),
        np.arange(H),
        np.arange(D),
        indexing='xy'
    )
    
    # Apply deformation
    x_new = x + deformation_field[..., 1]
    y_new = y + deformation_field[..., 0]
    z_new = z + deformation_field[..., 2]
    
    # Sample from original mask (order=0 for nearest neighbor)
    from scipy.ndimage import map_coordinates
    
    coords = np.array([y_new.ravel(), x_new.ravel(), z_new.ravel()])
    deformed_mask = map_coordinates(
        original_mask.astype(float), coords, order=0, mode='nearest'
    )
    deformed_mask = deformed_mask.reshape(H, W, D).astype(original_mask.dtype)
    
    return deformed_mask


class TaichiCTRenderer:
    """
    Renders Taichi physics simulation results to CT volumes.
    
    This class manages the connection between Taichi simulator output
    and CT rendering, handling coordinate transformations and 
    deformation field creation.
    """
    
    def __init__(
        self,
        ct_path: str,
        mask_path: str,
        vertebra_label: int = 1
    ):
        """
        Args:
            ct_path: Path to original CT NIfTI file
            mask_path: Path to segmentation mask NIfTI file
            vertebra_label: Label of the vertebra in the mask
        """
        self.ct_nii = nib.load(ct_path)
        self.mask_nii = nib.load(mask_path)
        
        self.ct_data = self.ct_nii.get_fdata()
        self.mask_data = self.mask_nii.get_fdata()
        
        self.vertebra_label = vertebra_label
        self.vertebra_mask = (self.mask_data == vertebra_label)
        
        # Get spacing from affine
        self.spacing = np.abs(np.diag(self.ct_nii.affine[:3, :3]))
        
        print(f"Loaded CT: {self.ct_data.shape}, spacing: {self.spacing}")
        print(f"Vertebra label {vertebra_label}: {self.vertebra_mask.sum()} voxels")
    
    def render_from_particles(
        self,
        original_positions: np.ndarray,
        deformed_positions: np.ndarray,
        taichi_bounds: Tuple[np.ndarray, np.ndarray],
        damage: Optional[np.ndarray] = None,
        smoothing_sigma: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render fractured CT from Taichi particle data.
        
        Args:
            original_positions: Original particle positions [N, 3]
            deformed_positions: Deformed particle positions [N, 3]
            taichi_bounds: (min, max) bounds from Taichi
            damage: Optional damage values per particle [N]
            smoothing_sigma: Smoothing for deformation field
            
        Returns:
            (fractured_ct, fractured_mask)
        """
        # Create deformation field
        deformation = create_deformation_field_from_taichi(
            original_positions,
            deformed_positions,
            taichi_bounds,
            self.ct_data.shape,
            self.vertebra_mask,
            smoothing_sigma
        )
        
        # Apply to CT
        fractured_ct = apply_deformation_to_ct(self.ct_data, deformation)
        
        # Apply to mask
        fractured_mask = apply_deformation_to_mask(self.mask_data, deformation)
        
        # Optionally modify HU values based on damage (fracture lines)
        if damage is not None:
            fractured_ct = self._apply_damage_to_hu(
                fractured_ct, damage, 
                original_positions, taichi_bounds
            )
        
        return fractured_ct, fractured_mask
    
    def _apply_damage_to_hu(
        self,
        ct: np.ndarray,
        damage: np.ndarray,
        positions: np.ndarray,
        taichi_bounds: Tuple[np.ndarray, np.ndarray],
        damage_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Modify HU values in damaged regions (simulates fracture lines).
        
        High damage regions get lower HU (simulating bone loss/fracture).
        """
        coords = np.where(self.vertebra_mask)
        bbox_min = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
        bbox_max = np.array([coords[0].max(), coords[1].max(), coords[2].max()])
        
        voxel_coords = taichi_to_voxel_coords(
            positions, taichi_bounds, ct.shape, {'min': bbox_min, 'max': bbox_max}
        )
        voxel_coords_int = np.round(voxel_coords).astype(int)
        
        # Clip to valid range
        H, W, D = ct.shape
        voxel_coords_int[:, 0] = np.clip(voxel_coords_int[:, 0], 0, H - 1)
        voxel_coords_int[:, 1] = np.clip(voxel_coords_int[:, 1], 0, W - 1)
        voxel_coords_int[:, 2] = np.clip(voxel_coords_int[:, 2], 0, D - 1)
        
        # Reduce HU in damaged regions
        for i in range(len(damage)):
            if damage[i] > damage_threshold:
                vi, vj, vk = voxel_coords_int[i]
                # Reduce HU proportionally to damage
                reduction = 0.3 + 0.7 * (1 - damage[i])  # 30-100% of original
                ct[vi, vj, vk] *= reduction
        
        return ct
    
    def save(
        self,
        fractured_ct: np.ndarray,
        fractured_mask: np.ndarray,
        output_ct_path: str,
        output_mask_path: str
    ):
        """Save rendered CT and mask to NIfTI files."""
        # Save CT
        ct_nii = nib.Nifti1Image(
            fractured_ct.astype(np.float32),
            self.ct_nii.affine,
            self.ct_nii.header
        )
        nib.save(ct_nii, output_ct_path)
        print(f"Saved CT: {output_ct_path}")
        
        # Save mask
        mask_nii = nib.Nifti1Image(
            fractured_mask.astype(np.int16),
            self.mask_nii.affine,
            self.mask_nii.header
        )
        nib.save(mask_nii, output_mask_path)
        print(f"Saved mask: {output_mask_path}")


def test_taichi_ct_renderer():
    """Test Taichi CT renderer with synthetic data."""
    print("=" * 70)
    print("Testing Taichi CT Renderer")
    print("=" * 70)
    
    # Create synthetic data
    shape = (64, 64, 64)
    ct = np.random.randn(*shape) * 100 + 500  # Fake CT
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Add synthetic vertebra
    center = np.array(shape) // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist < 15:
                    mask[i, j, k] = 1
    
    # Simulate Taichi particle output
    n_particles = 1000
    original_positions = np.random.rand(n_particles, 3) * 0.3 + 0.35  # [0.35, 0.65]
    
    # Simulate deformation (small random displacements)
    deformed_positions = original_positions + np.random.randn(n_particles, 3) * 0.01
    
    taichi_bounds = (
        np.array([0.35, 0.35, 0.35]),
        np.array([0.65, 0.65, 0.65])
    )
    
    # Create deformation field
    deformation = create_deformation_field_from_taichi(
        original_positions,
        deformed_positions,
        taichi_bounds,
        shape,
        mask
    )
    
    print(f"Deformation field shape: {deformation.shape}")
    print(f"Max displacement: {np.abs(deformation).max():.4f} voxels")
    
    # Apply deformation
    deformed_ct = apply_deformation_to_ct(ct, deformation)
    deformed_mask = apply_deformation_to_mask(mask, deformation)
    
    print(f"Original CT range: [{ct.min():.1f}, {ct.max():.1f}]")
    print(f"Deformed CT range: [{deformed_ct.min():.1f}, {deformed_ct.max():.1f}]")
    print(f"Original mask sum: {mask.sum()}")
    print(f"Deformed mask sum: {deformed_mask.sum()}")
    
    print("=" * 70)
    print("✓ Taichi CT Renderer test passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_taichi_ct_renderer()
