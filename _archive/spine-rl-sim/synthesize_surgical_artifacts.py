"""
Metal Artifact Synthesis for Surgical Hardware in CT

Implements physics-based artifact simulation:
1. Streak artifacts (photon starvation)
2. Blooming effect (partial volume)
3. HU value corruption (beam hardening)

This creates realistic surgical hardware artifacts for robust segmentation training.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, convolve
from skimage.morphology import binary_dilation, ball
from pathlib import Path
import matplotlib.pyplot as plt

# Import project configuration
try:
    from config import (
        get_verse_ct_path, get_phase4_output_path,
        DEFAULT_SUBJECT, STREAK_INTENSITY, BLOOMING_SIGMA, CORRUPTION_RADIUS
    )
    USE_CONFIG = True
except ImportError:
    print("⚠ Warning: config.py not found, using hardcoded paths")
    USE_CONFIG = False


def add_metal_hu_values(ct_volume, metal_mask, metal_hu=20000):
    """
    Set metal voxels to high HU values.
    
    Titanium screws: ~10,000-30,000 HU
    Stainless steel: ~20,000-40,000 HU
    """
    ct_with_metal = ct_volume.copy()
    ct_with_metal[metal_mask] = metal_hu
    return ct_with_metal


def add_streak_artifacts(ct_volume, metal_mask, severity=0.5):
    """
    Add streak artifacts radiating from metal.
    
    Physics: Photon starvation causes dark/bright streaks.
    Implementation: Radial convolution from metal regions.
    """
    # Find metal center of mass for each connected component
    from scipy.ndimage import label as ndlabel, center_of_mass
    
    labeled_metal, num_features = ndlabel(metal_mask)
    
    if num_features == 0:
        return ct_volume
    
    ct_with_streaks = ct_volume.copy()
    
    # For each metal object, add radial streaks
    for i in range(1, num_features + 1):
        component_mask = (labeled_metal == i)
        center = np.array(center_of_mass(component_mask))
        
        # Create streak pattern
        # Streaks are strongest in axial plane (perpendicular to CT scan direction)
        # Simulate by adding radial lines in Z-slices
        
        z_center = int(center[2])
        z_range = 20  # Streaks extend +/- 20 slices
        
        z_min = max(0, z_center - z_range)
        z_max = min(ct_volume.shape[2], z_center + z_range)
        
        for z in range(z_min, z_max):
            # Get 2D slice
            slice_2d = ct_with_streaks[:, :, z]
            
            # Create radial streak pattern
            Y, X = np.ogrid[:slice_2d.shape[0], :slice_2d.shape[1]]
            
            cx, cy = center[0], center[1]
            
            # Distance from center
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            
            # Angle from center
            angle = np.arctan2(Y - cy, X - cx)
            
            # Streak pattern: radial lines with alternating dark/bright
            # Use sinusoidal pattern in angle
            num_streaks = 12  # Number of radial streaks
            streak_pattern = np.sin(num_streaks * angle)
            
            # Decay with distance
            decay = np.exp(-dist / 100.0)
            
            # Intensity: stronger near metal, fades with distance
            z_decay = 1.0 - abs(z - z_center) / z_range
            
            streak_intensity = streak_pattern * decay * z_decay * severity * 200
            
            # Add to slice
            ct_with_streaks[:, :, z] = slice_2d + streak_intensity
    
    return ct_with_streaks


def add_blooming_effect(ct_volume, metal_mask, bloom_radius=3):
    """
    Add blooming effect: metal appears larger due to partial volume averaging.
    
    Implementation: Gaussian blur at metal edges with high HU values.
    """
    # Dilate metal mask to get blooming region
    from scipy.ndimage import binary_dilation
    
    bloomed_mask = binary_dilation(metal_mask, structure=ball(bloom_radius))
    bloom_edge = bloomed_mask & (~metal_mask)  # Only the expanded region
    
    if bloom_edge.sum() == 0:
        return ct_volume
    
    ct_with_bloom = ct_volume.copy()
    
    # Calculate distance from metal for falloff
    from scipy.ndimage import distance_transform_edt
    
    dist_from_metal = distance_transform_edt(~metal_mask)
    
    # Blooming intensity: high near metal, decays with distance
    bloom_intensity = np.zeros_like(ct_volume)
    bloom_intensity[bloom_edge] = 5000 / (1 + dist_from_metal[bloom_edge])
    
    # Add to CT
    ct_with_bloom += bloom_intensity
    
    return ct_with_bloom


def add_hu_corruption(ct_volume, metal_mask, corruption_radius=10):
    """
    Add HU value corruption in tissue surrounding metal.
    
    Physics: Beam hardening causes dark bands and bright regions.
    Implementation: Add spatially-varying offset near metal.
    """
    # Calculate distance from metal
    from scipy.ndimage import distance_transform_edt
    
    dist_from_metal = distance_transform_edt(~metal_mask)
    
    # Create corruption field
    corruption_field = np.zeros_like(ct_volume)
    
    # Region of influence
    influence_mask = dist_from_metal < corruption_radius
    
    if influence_mask.sum() == 0:
        return ct_volume
    
    # Corruption pattern: alternating positive/negative based on geometry
    # Simplified: radial pattern with sinusoidal variation
    
    # Get metal center
    from scipy.ndimage import center_of_mass
    metal_center = np.array(center_of_mass(metal_mask))
    
    # Create coordinate grids
    Z, Y, X = np.ogrid[:ct_volume.shape[0], :ct_volume.shape[1], :ct_volume.shape[2]]
    
    # Angle from metal center in XY plane
    angle = np.arctan2(Y - metal_center[1], X - metal_center[0])
    
    # Corruption: sinusoidal in angle, decays with distance
    corruption_pattern = np.sin(4 * angle)  # 4 lobes
    decay = np.exp(-dist_from_metal / 5.0)
    
    corruption_field = corruption_pattern * decay * 150  # Scale factor
    corruption_field[~influence_mask] = 0
    
    ct_with_corruption = ct_volume + corruption_field
    
    return ct_with_corruption


def synthesize_surgical_artifacts(ct_volume, metal_mask, severity='moderate'):
    """
    Full pipeline: synthesize all metal artifacts.
    
    Args:
        ct_volume: Original CT (HU values)
        metal_mask: Binary mask of surgical hardware
        severity: 'mild', 'moderate', or 'severe'
    
    Returns:
        CT with realistic metal artifacts
    """
    severity_params = {
        'mild': {'metal_hu': 15000, 'streak': 0.3, 'bloom': 2, 'corruption': 8},
        'moderate': {'metal_hu': 20000, 'streak': 0.5, 'bloom': 3, 'corruption': 10},
        'severe': {'metal_hu': 30000, 'streak': 0.8, 'bloom': 4, 'corruption': 15}
    }
    
    params = severity_params.get(severity, severity_params['moderate'])
    
    print(f"Synthesizing {severity} surgical artifacts...")
    
    # Step 1: Set metal HU
    print("  1. Adding metal HU values...")
    ct_artifact = add_metal_hu_values(ct_volume, metal_mask, params['metal_hu'])
    
    # Step 2: Add streak artifacts
    print("  2. Adding streak artifacts...")
    ct_artifact = add_streak_artifacts(ct_artifact, metal_mask, params['streak'])
    
    # Step 3: Add blooming
    print("  3. Adding blooming effect...")
    ct_artifact = add_blooming_effect(ct_artifact, metal_mask, params['bloom'])
    
    # Step 4: Add HU corruption
    print("  4. Adding HU corruption...")
    ct_artifact = add_hu_corruption(ct_artifact, metal_mask, params['corruption'])
    
    print("  ✓ Artifact synthesis complete!")
    
    return ct_artifact


def visualize_artifact_comparison(ct_original, ct_artifact, metal_mask, output_path):
    """
    Visualize original CT vs artifact CT.
    """
    # Find center slice through metal
    metal_coords = np.argwhere(metal_mask > 0)
    if len(metal_coords) == 0:
        print("Warning: Metal mask is empty!")
        return
    
    center = metal_coords.mean(axis=0).astype(int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Sagittal
    slice_idx = center[0]
    axes[0, 0].imshow(ct_original[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[0, 0].set_title('Original - Sagittal')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(ct_artifact[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[1, 0].contour(metal_mask[slice_idx, :, :].T, colors='red', linewidths=1)
    axes[1, 0].set_title('With Artifacts - Sagittal')
    axes[1, 0].axis('off')
    
    diff = ct_artifact[slice_idx, :, :] - ct_original[slice_idx, :, :]
    axes[2, 0].imshow(diff.T, cmap='seismic', origin='lower', vmin=-500, vmax=500)
    axes[2, 0].set_title('Difference - Sagittal')
    axes[2, 0].axis('off')
    
    # Coronal
    slice_idx = center[1]
    axes[0, 1].imshow(ct_original[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[0, 1].set_title('Original - Coronal')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(ct_artifact[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[1, 1].contour(metal_mask[:, slice_idx, :].T, colors='red', linewidths=1)
    axes[1, 1].set_title('With Artifacts - Coronal')
    axes[1, 1].axis('off')
    
    diff = ct_artifact[:, slice_idx, :] - ct_original[:, slice_idx, :]
    axes[2, 1].imshow(diff.T, cmap='seismic', origin='lower', vmin=-500, vmax=500)
    axes[2, 1].set_title('Difference - Coronal')
    axes[2, 1].axis('off')
    
    # Axial
    slice_idx = center[2]
    axes[0, 2].imshow(ct_original[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[0, 2].set_title('Original - Axial')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(ct_artifact[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=-200, vmax=1500)
    axes[1, 2].contour(metal_mask[:, :, slice_idx].T, colors='red', linewidths=1)
    axes[1, 2].set_title('With Artifacts - Axial')
    axes[1, 2].axis('off')
    
    diff = ct_artifact[:, :, slice_idx] - ct_original[:, :, slice_idx]
    axes[2, 2].imshow(diff.T, cmap='seismic', origin='lower', vmin=-500, vmax=500)
    axes[2, 2].set_title('Difference - Axial')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")


def main():
    """
    Generate CT with surgical artifacts and visualize.
    """
    # Paths (use config if available, otherwise fallback to hardcoded)
    if USE_CONFIG:
        ct_path = get_verse_ct_path(DEFAULT_SUBJECT)
        output_dir = get_phase4_output_path("artifact_synthesis")
        metal_path = get_phase4_output_path() / "implant_models" / "L1_pedicle_screws_mask.nii.gz"
    else:
        ct_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
        metal_path = Path("outputs/phase4_surgical_artifacts/implant_models/L1_pedicle_screws_mask.nii.gz")
        output_dir = Path("outputs/phase4_surgical_artifacts/artifact_synthesis")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    ct_nii = nib.load(str(ct_path))
    ct_original = ct_nii.get_fdata()
    
    metal_nii = nib.load(str(metal_path))
    metal_mask = metal_nii.get_fdata().astype(bool)
    
    print(f"CT shape: {ct_original.shape}")
    print(f"Metal voxels: {metal_mask.sum()}")
    
    # Synthesize artifacts
    ct_artifact = synthesize_surgical_artifacts(ct_original, metal_mask, severity='moderate')
    
    # Save artifact CT
    artifact_nii = nib.Nifti1Image(ct_artifact.astype(np.float32), ct_nii.affine)
    artifact_path = output_dir / "ct_with_pedicle_screws.nii.gz"
    nib.save(artifact_nii, str(artifact_path))
    print(f"\n✓ Artifact CT saved: {artifact_path}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_artifact_comparison(
        ct_original, ct_artifact, metal_mask,
        output_dir / "artifact_comparison.png"
    )
    
    print("\n" + "="*60)
    print("✓ Step 2 COMPLETE: Metal artifact synthesis!")
    print("="*60)
    print("\nNext: Run TotalSegmentator on artifact CT to measure Dice drop")


if __name__ == "__main__":
    main()

