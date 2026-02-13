"""
Improved visualization: Show ACTUAL artifact CT with screws visible as bright metal.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_screw_in_ct(ct_original, ct_artifact, metal_mask, vertebra_mask, output_path):
    """
    Show original CT vs artifact CT with visible metal screws.
    """
    # Find center through screws
    metal_coords = np.argwhere(metal_mask > 0)
    center = metal_coords.mean(axis=0).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Window settings for bone visualization
    vmin, vmax = -200, 1500
    
    # Sagittal
    slice_idx = center[0]
    axes[0, 0].imshow(ct_original[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].contour(vertebra_mask[slice_idx, :, :].T, colors='blue', linewidths=1, alpha=0.5)
    axes[0, 0].set_title('Original CT - Sagittal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(ct_artifact[slice_idx, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 0].contour(vertebra_mask[slice_idx, :, :].T, colors='blue', linewidths=1, alpha=0.5)
    axes[1, 0].contour(metal_mask[slice_idx, :, :].T, colors='red', linewidths=2)
    axes[1, 0].set_title('With Pedicle Screws - Sagittal', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Coronal
    slice_idx = center[1]
    axes[0, 1].imshow(ct_original[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 1].contour(vertebra_mask[:, slice_idx, :].T, colors='blue', linewidths=1, alpha=0.5)
    axes[0, 1].set_title('Original CT - Coronal', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(ct_artifact[:, slice_idx, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 1].contour(vertebra_mask[:, slice_idx, :].T, colors='blue', linewidths=1, alpha=0.5)
    axes[1, 1].contour(metal_mask[:, slice_idx, :].T, colors='red', linewidths=2)
    axes[1, 1].set_title('With Pedicle Screws - Coronal', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Axial
    slice_idx = center[2]
    axes[0, 2].imshow(ct_original[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 2].contour(vertebra_mask[:, :, slice_idx].T, colors='blue', linewidths=1, alpha=0.5)
    axes[0, 2].set_title('Original CT - Axial', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(ct_artifact[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 2].contour(vertebra_mask[:, :, slice_idx].T, colors='blue', linewidths=1, alpha=0.5)
    axes[1, 2].contour(metal_mask[:, :, slice_idx].T, colors='red', linewidths=2)
    axes[1, 2].set_title('With Pedicle Screws - Axial', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")


def main():
    # Load data
    ct_orig = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz").get_fdata()
    ct_artifact = nib.load("outputs/phase4_surgical_artifacts/artifact_synthesis/ct_with_pedicle_screws.nii.gz").get_fdata()
    metal_mask = nib.load("outputs/phase4_surgical_artifacts/implant_models/L1_pedicle_screws_mask.nii.gz").get_fdata().astype(bool)
    gt_mask = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz").get_fdata()
    vertebra_mask = (gt_mask == 22)  # L1
    
    output_path = Path("outputs/phase4_surgical_artifacts/screw_visualization_with_artifacts.png")
    
    print("Creating visualization with VISIBLE screws...")
    visualize_screw_in_ct(ct_orig, ct_artifact, metal_mask, vertebra_mask, output_path)
    
    # Also check HU values
    print(f"\nHU value check:")
    print(f"  Original CT at screw location: {ct_orig[metal_mask].mean():.1f} HU")
    print(f"  Artifact CT at screw location: {ct_artifact[metal_mask].mean():.1f} HU")
    print(f"  Expected metal HU: ~20,000 HU")
    
    if ct_artifact[metal_mask].mean() > 10000:
        print("  ✓ Screws are BRIGHT (high HU) - visible as white metal!")
    else:
        print("  ⚠️  Screws not bright enough!")


if __name__ == "__main__":
    main()

