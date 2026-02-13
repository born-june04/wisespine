"""
Simple Pedicle Screw Placement and Rasterization

Goal: Insert a pedicle screw into L1 vertebra and render it to CT space.
This is Phase 4 Step 1: Basic geometry before adding physics-based artifacts.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, center_of_mass
from skimage.morphology import ball
import matplotlib.pyplot as plt
from pathlib import Path

# Import project configuration
try:
    from config import (
        get_verse_ct_path, get_verse_seg_path, get_phase4_output_path,
        DEFAULT_SUBJECT, CT_SPACING
    )
    USE_CONFIG = True
except ImportError:
    print("⚠ Warning: config.py not found, using hardcoded paths")
    USE_CONFIG = False


def find_pedicle_centers(vertebra_mask):
    """
    Find left and right pedicle centers for a vertebra.
    
    Simplified approach:
    - Pedicles are lateral projections from vertebral body
    - Located in posterior-lateral regions
    - Usually symmetric (left/right)
    
    Returns: (left_center, right_center) in voxel coordinates
    """
    # Get bounding box
    coords = np.argwhere(vertebra_mask > 0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Vertebra center
    center = (min_coords + max_coords) / 2
    
    # Pedicles are:
    # - Posterior (lower Z in typical CT orientation)
    # - Lateral (left/right X)
    # - Mid-height (center Y)
    
    # Divide vertebra into posterior/anterior halves
    z_mid = center[2]
    posterior_mask = vertebra_mask.copy()
    posterior_mask[:, :, int(z_mid):] = 0  # Keep posterior half
    
    # Find lateral peaks (pedicles)
    # Project to X-Z plane
    projection = posterior_mask.sum(axis=1)  # Sum over Y
    
    # Find left and right peaks
    x_profile = projection.sum(axis=1)  # Sum over Z
    
    # Smooth to find peaks
    from scipy.ndimage import gaussian_filter1d
    x_profile_smooth = gaussian_filter1d(x_profile.astype(float), sigma=3)
    
    # Find peaks (left and right pedicles)
    peaks = []
    threshold = 0.3 * x_profile_smooth.max()
    
    for i in range(1, len(x_profile_smooth) - 1):
        if (x_profile_smooth[i] > threshold and
            x_profile_smooth[i] > x_profile_smooth[i-1] and
            x_profile_smooth[i] > x_profile_smooth[i+1]):
            peaks.append(i)
    
    # Take leftmost and rightmost peaks
    if len(peaks) >= 2:
        left_x = peaks[0]
        right_x = peaks[-1]
    else:
        # Fallback: use lateral edges
        left_x = int(min_coords[0] + 0.2 * (max_coords[0] - min_coords[0]))
        right_x = int(min_coords[0] + 0.8 * (max_coords[0] - min_coords[0]))
    
    # Y: mid-height
    y_center = int(center[1])
    
    # Z: posterior quarter
    z_pedicle = int(min_coords[2] + 0.3 * (max_coords[2] - min_coords[2]))
    
    left_center = np.array([left_x, y_center, z_pedicle])
    right_center = np.array([right_x, y_center, z_pedicle])
    
    return left_center, right_center


def rasterize_cylinder(start_point, direction, diameter, length, volume_shape):
    """
    Rasterize a cylinder (pedicle screw) into a 3D volume.
    
    Args:
        start_point: (x, y, z) entry point in voxel coordinates
        direction: (dx, dy, dz) unit vector for screw trajectory
        diameter: screw diameter in voxels
        length: screw length in voxels
        volume_shape: shape of output volume
    
    Returns:
        Binary mask of screw
    """
    screw_mask = np.zeros(volume_shape, dtype=bool)
    
    # Normalize direction
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    
    # Radius in voxels
    radius = diameter / 2.0
    
    # Sample points along screw axis
    num_samples = int(length * 2)  # Oversample for smooth rasterization
    
    for t in np.linspace(0, length, num_samples):
        # Point on screw axis
        center = start_point + t * direction
        
        # Rasterize cross-section (circle)
        # Check all voxels within radius
        x_c, y_c, z_c = center.astype(int)
        
        # Bounds check
        x_min = max(0, int(x_c - radius - 1))
        x_max = min(volume_shape[0], int(x_c + radius + 2))
        y_min = max(0, int(y_c - radius - 1))
        y_max = min(volume_shape[1], int(y_c + radius + 2))
        z_min = max(0, int(z_c - radius - 1))
        z_max = min(volume_shape[2], int(z_c + radius + 2))
        
        # Check voxels in neighborhood
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    # Distance from voxel to screw axis
                    voxel = np.array([x, y, z])
                    to_voxel = voxel - center
                    
                    # Project onto perpendicular plane
                    along_axis = np.dot(to_voxel, direction) * direction
                    perpendicular = to_voxel - along_axis
                    
                    # Distance to axis
                    dist = np.linalg.norm(perpendicular)
                    
                    if dist <= radius:
                        screw_mask[x, y, z] = True
    
    return screw_mask


def place_pedicle_screw(vertebra_mask, side='left', spacing=(1.0, 1.0, 1.0)):
    """
    Place a pedicle screw in a vertebra.
    
    Args:
        vertebra_mask: Binary mask of vertebra
        side: 'left' or 'right'
        spacing: Voxel spacing in mm (for size conversion)
    
    Returns:
        screw_mask: Binary mask of screw
        entry_point: Entry point in voxel coordinates
        trajectory: Trajectory vector (unit)
    """
    # Find pedicle centers
    left_center, right_center = find_pedicle_centers(vertebra_mask)
    
    # Choose side
    entry_point = left_center if side == 'left' else right_center
    
    # Typical pedicle screw parameters
    diameter_mm = 5.5  # mm
    length_mm = 45  # mm
    
    # Convert to voxels
    diameter_voxels = diameter_mm / spacing[0]  # Assume isotropic or use avg
    length_voxels = length_mm / spacing[0]
    
    # Trajectory: from pedicle into vertebral body
    # Typically: medial (toward center), slightly anterior, slightly caudal
    
    # Get vertebra center of mass
    com = np.array(center_of_mass(vertebra_mask))
    
    # Direction: toward center of mass
    trajectory = com - entry_point
    trajectory = trajectory / np.linalg.norm(trajectory)
    
    # Adjust: slightly anterior (positive Z) and slightly caudal (positive Y)
    # This is a simplified model; real screws follow pedicle anatomy
    trajectory[1] += 0.1  # Slight caudal
    trajectory[2] += 0.1  # Slight anterior
    trajectory = trajectory / np.linalg.norm(trajectory)  # Renormalize
    
    # Rasterize screw
    screw_mask = rasterize_cylinder(
        entry_point, trajectory, 
        diameter_voxels, length_voxels, 
        vertebra_mask.shape
    )
    
    return screw_mask, entry_point, trajectory


def visualize_screw_placement(ct_volume, vertebra_mask, screw_mask, output_path):
    """
    Visualize CT with vertebra and screw overlaid.
    """
    # Find center slice through screw
    screw_coords = np.argwhere(screw_mask > 0)
    if len(screw_coords) == 0:
        print("Warning: Screw mask is empty!")
        return
    
    center = screw_coords.mean(axis=0).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sagittal view
    slice_idx = center[0]
    axes[0, 0].imshow(ct_volume[slice_idx, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].contour(vertebra_mask[slice_idx, :, :].T, colors='blue', linewidths=1)
    axes[0, 0].contour(screw_mask[slice_idx, :, :].T, colors='red', linewidths=2)
    axes[0, 0].set_title(f'Sagittal (X={slice_idx})')
    axes[0, 0].axis('off')
    
    # Coronal view
    slice_idx = center[1]
    axes[0, 1].imshow(ct_volume[:, slice_idx, :].T, cmap='gray', origin='lower')
    axes[0, 1].contour(vertebra_mask[:, slice_idx, :].T, colors='blue', linewidths=1)
    axes[0, 1].contour(screw_mask[:, slice_idx, :].T, colors='red', linewidths=2)
    axes[0, 1].set_title(f'Coronal (Y={slice_idx})')
    axes[0, 1].axis('off')
    
    # Axial view
    slice_idx = center[2]
    axes[0, 2].imshow(ct_volume[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 2].contour(vertebra_mask[:, :, slice_idx].T, colors='blue', linewidths=1)
    axes[0, 2].contour(screw_mask[:, :, slice_idx].T, colors='red', linewidths=2)
    axes[0, 2].set_title(f'Axial (Z={slice_idx})')
    axes[0, 2].axis('off')
    
    # Zoomed views
    zoom_size = 50
    
    # Sagittal zoomed
    y_min = max(0, center[1] - zoom_size)
    y_max = min(ct_volume.shape[1], center[1] + zoom_size)
    z_min = max(0, center[2] - zoom_size)
    z_max = min(ct_volume.shape[2], center[2] + zoom_size)
    
    axes[1, 0].imshow(
        ct_volume[center[0], y_min:y_max, z_min:z_max].T, 
        cmap='gray', origin='lower'
    )
    axes[1, 0].contour(
        vertebra_mask[center[0], y_min:y_max, z_min:z_max].T, 
        colors='blue', linewidths=1
    )
    axes[1, 0].contour(
        screw_mask[center[0], y_min:y_max, z_min:z_max].T, 
        colors='red', linewidths=2
    )
    axes[1, 0].set_title('Sagittal (Zoomed)')
    axes[1, 0].axis('off')
    
    # Coronal zoomed
    x_min = max(0, center[0] - zoom_size)
    x_max = min(ct_volume.shape[0], center[0] + zoom_size)
    
    axes[1, 1].imshow(
        ct_volume[x_min:x_max, center[1], z_min:z_max].T, 
        cmap='gray', origin='lower'
    )
    axes[1, 1].contour(
        vertebra_mask[x_min:x_max, center[1], z_min:z_max].T, 
        colors='blue', linewidths=1
    )
    axes[1, 1].contour(
        screw_mask[x_min:x_max, center[1], z_min:z_max].T, 
        colors='red', linewidths=2
    )
    axes[1, 1].set_title('Coronal (Zoomed)')
    axes[1, 1].axis('off')
    
    # Axial zoomed
    axes[1, 2].imshow(
        ct_volume[x_min:x_max, y_min:y_max, center[2]].T, 
        cmap='gray', origin='lower'
    )
    axes[1, 2].contour(
        vertebra_mask[x_min:x_max, y_min:y_max, center[2]].T, 
        colors='blue', linewidths=1
    )
    axes[1, 2].contour(
        screw_mask[x_min:x_max, y_min:y_max, center[2]].T, 
        colors='red', linewidths=2
    )
    axes[1, 2].set_title('Axial (Zoomed)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")


def main():
    """
    Test pedicle screw placement on verse563 L1 vertebra.
    """
    # Paths (use config if available, otherwise fallback to hardcoded)
    if USE_CONFIG:
        ct_path = get_verse_ct_path(DEFAULT_SUBJECT)
        mask_path = get_verse_seg_path(DEFAULT_SUBJECT)
        output_dir = get_phase4_output_path()
    else:
        ct_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
        mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
        output_dir = Path("outputs/phase4_surgical_artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading CT and mask...")
    ct_nii = nib.load(str(ct_path))
    ct_volume = ct_nii.get_fdata()
    spacing = ct_nii.header.get_zooms()
    
    mask_nii = nib.load(str(mask_path))
    mask_volume = mask_nii.get_fdata()
    
    # Extract L1 vertebra (label 22)
    l1_mask = (mask_volume == 22)
    
    print(f"L1 vertebra size: {l1_mask.sum()} voxels")
    print(f"Voxel spacing: {spacing} mm")
    
    # Place pedicle screw (left side)
    print("\nPlacing pedicle screw (LEFT)...")
    screw_mask_left, entry_left, traj_left = place_pedicle_screw(
        l1_mask, side='left', spacing=spacing
    )
    print(f"  Entry point: {entry_left}")
    print(f"  Trajectory: {traj_left}")
    print(f"  Screw size: {screw_mask_left.sum()} voxels")
    
    # Place pedicle screw (right side)
    print("\nPlacing pedicle screw (RIGHT)...")
    screw_mask_right, entry_right, traj_right = place_pedicle_screw(
        l1_mask, side='right', spacing=spacing
    )
    print(f"  Entry point: {entry_right}")
    print(f"  Trajectory: {traj_right}")
    print(f"  Screw size: {screw_mask_right.sum()} voxels")
    
    # Combine screws
    screws_mask = screw_mask_left | screw_mask_right
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_screw_placement(
        ct_volume, l1_mask, screws_mask,
        output_dir / "screw_placement_test.png"
    )
    
    # Save screw mask (for next step: artifact generation)
    screws_nii = nib.Nifti1Image(screws_mask.astype(np.uint8), ct_nii.affine)
    screws_path = output_dir / "implant_models" / "L1_pedicle_screws_mask.nii.gz"
    screws_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(screws_nii, str(screws_path))
    print(f"\n✓ Screw mask saved: {screws_path}")
    
    print("\n" + "="*60)
    print("✓ Step 1 COMPLETE: Pedicle screw placement & rasterization!")
    print("="*60)
    print("\nNext: Add physics-based metal artifacts (streak, blooming)")


if __name__ == "__main__":
    main()

