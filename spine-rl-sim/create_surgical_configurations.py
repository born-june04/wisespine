"""
Enhanced Surgical Hardware: Pedicle Screws + Rods + Multi-level

Implements realistic surgical instrumentation:
1. Bilateral pedicle screws (left + right)
2. Connecting rod (posterior fixation)
3. Multi-level support (L1, L2, L3, etc.)

This creates clinically realistic surgical configurations for testing.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass
from pathlib import Path
import matplotlib.pyplot as plt


def find_pedicle_centers(vertebra_mask):
    """Find left and right pedicle centers."""
    coords = np.argwhere(vertebra_mask > 0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = (min_coords + max_coords) / 2
    
    # Pedicles are posterior-lateral
    z_pedicle = int(min_coords[2] + 0.25 * (max_coords[2] - min_coords[2]))
    y_center = int(center[1])
    
    # Left and right positions
    left_x = int(min_coords[0] + 0.25 * (max_coords[0] - min_coords[0]))
    right_x = int(min_coords[0] + 0.75 * (max_coords[0] - min_coords[0]))
    
    left_center = np.array([left_x, y_center, z_pedicle])
    right_center = np.array([right_x, y_center, z_pedicle])
    
    return left_center, right_center


def rasterize_cylinder(start_point, direction, diameter, length, volume_shape):
    """Rasterize a cylinder into 3D volume."""
    screw_mask = np.zeros(volume_shape, dtype=bool)
    
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    
    radius = diameter / 2.0
    num_samples = int(length * 2)
    
    for t in np.linspace(0, length, num_samples):
        center = start_point + t * direction
        x_c, y_c, z_c = center.astype(int)
        
        x_min = max(0, int(x_c - radius - 1))
        x_max = min(volume_shape[0], int(x_c + radius + 2))
        y_min = max(0, int(y_c - radius - 1))
        y_max = min(volume_shape[1], int(y_c + radius + 2))
        z_min = max(0, int(z_c - radius - 1))
        z_max = min(volume_shape[2], int(z_c + radius + 2))
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    voxel = np.array([x, y, z])
                    to_voxel = voxel - center
                    along_axis = np.dot(to_voxel, direction) * direction
                    perpendicular = to_voxel - along_axis
                    dist = np.linalg.norm(perpendicular)
                    
                    if dist <= radius:
                        screw_mask[x, y, z] = True
    
    return screw_mask


def place_bilateral_screws(vertebra_mask, spacing=(1.0, 1.0, 1.0)):
    """Place left and right pedicle screws."""
    left_center, right_center = find_pedicle_centers(vertebra_mask)
    
    # Screw parameters
    diameter_mm = 5.5
    length_mm = 45
    diameter_voxels = diameter_mm / spacing[0]
    length_voxels = length_mm / spacing[0]
    
    # Trajectories: toward center of mass
    com = np.array(center_of_mass(vertebra_mask))
    
    # Left screw
    traj_left = com - left_center
    traj_left[1] += 0.1  # Slight caudal
    traj_left[2] += 0.1  # Slight anterior
    traj_left = traj_left / np.linalg.norm(traj_left)
    
    screw_left = rasterize_cylinder(
        left_center, traj_left, diameter_voxels, length_voxels, vertebra_mask.shape
    )
    
    # Right screw
    traj_right = com - right_center
    traj_right[1] += 0.1
    traj_right[2] += 0.1
    traj_right = traj_right / np.linalg.norm(traj_right)
    
    screw_right = rasterize_cylinder(
        right_center, traj_right, diameter_voxels, length_voxels, vertebra_mask.shape
    )
    
    return screw_left, screw_right, left_center, right_center


def place_connecting_rod(left_entry, right_entry, vertebra_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Place a rod connecting left and right screws (posterior fixation).
    
    In real surgery, the rod sits posterior to the vertebra body,
    connecting the heads of the pedicle screws.
    """
    # Rod parameters
    rod_diameter_mm = 5.5  # Similar to screw diameter
    rod_diameter_voxels = rod_diameter_mm / spacing[0]
    
    # Rod runs from left entry to right entry (lateral direction)
    rod_start = left_entry.copy()
    rod_end = right_entry.copy()
    
    # Rod is posterior to entry points (shift backward)
    posterior_offset = -5  # voxels backward
    rod_start[2] += posterior_offset
    rod_end[2] += posterior_offset
    
    # Rod direction
    rod_direction = rod_end - rod_start
    rod_length = np.linalg.norm(rod_direction)
    rod_direction = rod_direction / rod_length
    
    # Rasterize rod
    rod_mask = rasterize_cylinder(
        rod_start, rod_direction, rod_diameter_voxels, rod_length, vertebra_mask.shape
    )
    
    return rod_mask


def create_instrumentation(vertebra_labels, gt_mask, spacing, include_rod=True):
    """
    Create full surgical instrumentation for specified vertebrae.
    
    Args:
        vertebra_labels: list of (label_id, name) tuples, e.g., [(22, 'L1'), (23, 'L2')]
        gt_mask: full ground truth mask
        spacing: voxel spacing
        include_rod: whether to add connecting rods
    
    Returns:
        hardware_mask: combined mask of all hardware
        description: text description of configuration
    """
    hardware_mask = np.zeros_like(gt_mask, dtype=bool)
    screw_entries = []  # For rod placement
    
    description = []
    
    for label_id, name in vertebra_labels:
        vertebra_mask = (gt_mask == label_id)
        
        if vertebra_mask.sum() == 0:
            print(f"Warning: {name} (label {label_id}) not found in mask!")
            continue
        
        print(f"Instrumenting {name}...")
        
        # Place bilateral screws
        screw_left, screw_right, left_entry, right_entry = place_bilateral_screws(
            vertebra_mask, spacing
        )
        
        hardware_mask |= screw_left
        hardware_mask |= screw_right
        
        screw_entries.append((left_entry, right_entry, name))
        
        description.append(f"{name}: bilateral pedicle screws")
        
        # Add connecting rod for this level
        if include_rod:
            rod_mask = place_connecting_rod(left_entry, right_entry, vertebra_mask, spacing)
            hardware_mask |= rod_mask
            description.append(f"{name}: connecting rod")
    
    # Multi-level rod (connects across vertebrae)
    if include_rod and len(screw_entries) > 1:
        print("Adding multi-level rods...")
        # Left side rod
        left_entries = [e[0] for e in screw_entries]
        for i in range(len(left_entries) - 1):
            rod_mask = place_connecting_rod(
                left_entries[i], left_entries[i+1], gt_mask, spacing
            )
            hardware_mask |= rod_mask
        
        # Right side rod
        right_entries = [e[1] for e in screw_entries]
        for i in range(len(right_entries) - 1):
            rod_mask = place_connecting_rod(
                right_entries[i], right_entries[i+1], gt_mask, spacing
            )
            hardware_mask |= rod_mask
        
        description.append(f"Multi-level rods connecting {screw_entries[0][2]} to {screw_entries[-1][2]}")
    
    description_text = "\n  - ".join(description)
    return hardware_mask, f"Configuration:\n  - {description_text}"


def visualize_instrumentation(ct_volume, hardware_mask, vertebra_labels, gt_mask, output_path, config_name):
    """Visualize surgical instrumentation."""
    # Get center through hardware
    hw_coords = np.argwhere(hardware_mask > 0)
    if len(hw_coords) == 0:
        print("Warning: No hardware to visualize!")
        return
    
    center = hw_coords.mean(axis=0).astype(int)
    
    # Combine vertebra masks
    combined_vertebra = np.zeros_like(gt_mask, dtype=bool)
    for label_id, _ in vertebra_labels:
        combined_vertebra |= (gt_mask == label_id)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    vmin, vmax = -200, 1500
    
    # Sagittal
    slice_idx = center[0]
    axes[0, 0].imshow(ct_volume[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].contour(combined_vertebra[slice_idx, :, :].T, colors='blue', linewidths=1, alpha=0.3)
    axes[0, 0].set_title('Original - Sagittal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(ct_volume[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 0].contour(combined_vertebra[slice_idx, :, :].T, colors='blue', linewidths=1, alpha=0.3)
    axes[1, 0].contour(hardware_mask[slice_idx, :, :].T, colors='red', linewidths=2)
    axes[1, 0].set_title(f'{config_name} - Sagittal', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Coronal
    slice_idx = center[1]
    axes[0, 1].imshow(ct_volume[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 1].contour(combined_vertebra[:, slice_idx, :].T, colors='blue', linewidths=1, alpha=0.3)
    axes[0, 1].set_title('Original - Coronal', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(ct_volume[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 1].contour(combined_vertebra[:, slice_idx, :].T, colors='blue', linewidths=1, alpha=0.3)
    axes[1, 1].contour(hardware_mask[:, slice_idx, :].T, colors='red', linewidths=2)
    axes[1, 1].set_title(f'{config_name} - Coronal', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Axial
    slice_idx = center[2]
    axes[0, 2].imshow(ct_volume[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 2].contour(combined_vertebra[:, :, slice_idx].T, colors='blue', linewidths=1, alpha=0.3)
    axes[0, 2].set_title('Original - Axial', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(ct_volume[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 2].contour(combined_vertebra[:, :, slice_idx].T, colors='blue', linewidths=1, alpha=0.3)
    axes[1, 2].contour(hardware_mask[:, :, slice_idx].T, colors='red', linewidths=2)
    axes[1, 2].set_title(f'{config_name} - Axial', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")


def main():
    """
    Test different instrumentation configurations:
    1. L1 only, screws only
    2. L1 only, screws + rod
    3. L1+L2+L3, screws + rods (multi-level)
    """
    # Load data
    ct_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
    mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    output_dir = Path("outputs/phase4_surgical_artifacts/configurations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    ct_nii = nib.load(str(ct_path))
    ct_volume = ct_nii.get_fdata()
    spacing = ct_nii.header.get_zooms()
    
    gt_mask = nib.load(str(mask_path)).get_fdata()
    
    print(f"CT shape: {ct_volume.shape}")
    print(f"Spacing: {spacing} mm")
    print()
    
    # Configuration 1: L1 only, screws only
    print("="*70)
    print("Configuration 1: L1 - Screws Only")
    print("="*70)
    hw1, desc1 = create_instrumentation(
        [(22, 'L1')], gt_mask, spacing, include_rod=False
    )
    print(f"Hardware voxels: {hw1.sum()}")
    print(desc1)
    print()
    
    # Save
    hw1_nii = nib.Nifti1Image(hw1.astype(np.uint8), ct_nii.affine)
    nib.save(hw1_nii, str(output_dir / "config1_L1_screws_only.nii.gz"))
    
    visualize_instrumentation(
        ct_volume, hw1, [(22, 'L1')], gt_mask,
        output_dir / "config1_L1_screws_only.png",
        "L1 Screws Only"
    )
    
    # Configuration 2: L1 only, screws + rod
    print("="*70)
    print("Configuration 2: L1 - Screws + Rod")
    print("="*70)
    hw2, desc2 = create_instrumentation(
        [(22, 'L1')], gt_mask, spacing, include_rod=True
    )
    print(f"Hardware voxels: {hw2.sum()}")
    print(desc2)
    print()
    
    hw2_nii = nib.Nifti1Image(hw2.astype(np.uint8), ct_nii.affine)
    nib.save(hw2_nii, str(output_dir / "config2_L1_screws_rod.nii.gz"))
    
    visualize_instrumentation(
        ct_volume, hw2, [(22, 'L1')], gt_mask,
        output_dir / "config2_L1_screws_rod.png",
        "L1 Screws + Rod"
    )
    
    # Configuration 3: Multi-level (L1+L2+L3), screws + rods
    print("="*70)
    print("Configuration 3: Multi-level (L1+L2+L3) - Screws + Rods")
    print("="*70)
    hw3, desc3 = create_instrumentation(
        [(22, 'L1'), (23, 'L2'), (24, 'L3')], gt_mask, spacing, include_rod=True
    )
    print(f"Hardware voxels: {hw3.sum()}")
    print(desc3)
    print()
    
    hw3_nii = nib.Nifti1Image(hw3.astype(np.uint8), ct_nii.affine)
    nib.save(hw3_nii, str(output_dir / "config3_multi_level.nii.gz"))
    
    visualize_instrumentation(
        ct_volume, hw3, [(22, 'L1'), (23, 'L2'), (24, 'L3')], gt_mask,
        output_dir / "config3_multi_level.png",
        "Multi-level (L1+L2+L3)"
    )
    
    print("\n" + "="*70)
    print("✓ All configurations created!")
    print("="*70)
    print("\nSummary:")
    print(f"  Config 1: {hw1.sum():,} voxels (L1 screws only)")
    print(f"  Config 2: {hw2.sum():,} voxels (L1 screws + rod)")
    print(f"  Config 3: {hw3.sum():,} voxels (L1+L2+L3 multi-level)")
    print(f"\nNext: Generate artifacts and measure Dice for each configuration")


if __name__ == "__main__":
    main()


