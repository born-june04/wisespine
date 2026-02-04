#!/usr/bin/env python3
"""
Visualize original and deformed CT in 3 views (sagittal, axial, coronal).
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_ct_comparison(
    original_path: str,
    deformed_path: str,
    output_path: str = "outputs/ct_comparison.png",
):
    """
    Create 2x3 visualization: original vs deformed in sagittal/axial/coronal views.
    
    Args:
        original_path: Path to original/initial CT
        deformed_path: Path to deformed CT
        output_path: Where to save PNG
    """
    print(f"Loading CTs for visualization...")
    
    # Load CTs
    original_nii = nib.load(original_path)
    deformed_nii = nib.load(deformed_path)
    
    original = original_nii.get_fdata()
    deformed = deformed_nii.get_fdata()
    
    print(f"  Original shape: {original.shape}")
    print(f"  Deformed shape: {deformed.shape}")
    
    # Get center slices (where vertebrae are)
    # Find where bone is (HU > 500)
    bone_mask = deformed > 500
    
    # Get bounding box of bone
    coords = np.where(bone_mask)
    if len(coords[0]) == 0:
        print("Warning: No bone found, using center slices")
        center = [s // 2 for s in deformed.shape]
    else:
        center = [int(np.mean(c)) for c in coords]
    
    print(f"  Center slice indices: {center}")
    
    # Create figure with 2 rows (original, deformed) × 3 columns (sag, ax, cor)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CT Rendering Results: Original vs Deformed (L1 displaced)', 
                 fontsize=16, fontweight='bold')
    
    # Window level for bone visualization
    vmin, vmax = -200, 1500
    
    # Row 1: Original
    # Sagittal (YZ plane, slice along X)
    ax = axes[0, 0]
    slice_sag = original[center[0], :, :]
    ax.imshow(slice_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Original - Sagittal', fontsize=12, fontweight='bold')
    ax.set_xlabel('Y (anterior-posterior)')
    ax.set_ylabel('Z (superior-inferior)')
    ax.axis('off')
    
    # Axial (XY plane, slice along Z)
    ax = axes[0, 1]
    slice_ax = original[:, :, center[2]]
    ax.imshow(slice_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Original - Axial', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Y (anterior-posterior)')
    ax.axis('off')
    
    # Coronal (XZ plane, slice along Y)
    ax = axes[0, 2]
    slice_cor = original[:, center[1], :]
    ax.imshow(slice_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Original - Coronal', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Z (superior-inferior)')
    ax.axis('off')
    
    # Row 2: Deformed
    # Sagittal
    ax = axes[1, 0]
    slice_sag = deformed[center[0], :, :]
    ax.imshow(slice_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Deformed - Sagittal', fontsize=12, fontweight='bold', color='red')
    ax.set_xlabel('Y (anterior-posterior)')
    ax.set_ylabel('Z (superior-inferior)')
    ax.axis('off')
    
    # Axial
    ax = axes[1, 1]
    slice_ax = deformed[:, :, center[2]]
    ax.imshow(slice_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Deformed - Axial', fontsize=12, fontweight='bold', color='red')
    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Y (anterior-posterior)')
    ax.axis('off')
    
    # Coronal
    ax = axes[1, 2]
    slice_cor = deformed[:, center[1], :]
    ax.imshow(slice_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title('Deformed - Coronal', fontsize=12, fontweight='bold', color='red')
    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Z (superior-inferior)')
    ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(axes[0, 0].images[0], ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.05)
    cbar.set_label('HU (Hounsfield Units)', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    
    # Also save with difference highlighting
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Difference Map: Deformed - Original (Red = bone displacement)', 
                  fontsize=14, fontweight='bold')
    
    # Compute difference
    diff = deformed - original
    
    # Sagittal difference
    ax = axes2[0]
    diff_sag = diff[center[0], :, :]
    im = ax.imshow(diff_sag.T, cmap='seismic', vmin=-1000, vmax=1000, origin='lower')
    ax.set_title('Sagittal Difference', fontsize=12)
    ax.axis('off')
    
    # Axial difference
    ax = axes2[1]
    diff_ax = diff[:, :, center[2]]
    ax.imshow(diff_ax.T, cmap='seismic', vmin=-1000, vmax=1000, origin='lower')
    ax.set_title('Axial Difference', fontsize=12)
    ax.axis('off')
    
    # Coronal difference
    ax = axes2[2]
    diff_cor = diff[:, center[1], :]
    ax.imshow(diff_cor.T, cmap='seismic', vmin=-1000, vmax=1000, origin='lower')
    ax.set_title('Coronal Difference', fontsize=12)
    ax.axis('off')
    
    plt.colorbar(im, ax=axes2, orientation='horizontal', fraction=0.05, pad=0.05, 
                 label='HU Difference (Deformed - Original)')
    plt.tight_layout()
    
    diff_path = output_path.parent / "ct_difference.png"
    plt.savefig(diff_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved difference map: {diff_path}")
    
    return output_path, diff_path


if __name__ == "__main__":
    print("="*70)
    print("CT VISUALIZATION")
    print("="*70)
    
    original_ct = "outputs/rendered_ct_initial.nii.gz"
    deformed_ct = "outputs/rendered_ct_deformed.nii.gz"
    
    vis_path, diff_path = visualize_ct_comparison(
        original_path=original_ct,
        deformed_path=deformed_ct,
        output_path="outputs/ct_comparison.png",
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"📊 Main visualization: {vis_path}")
    print(f"📊 Difference map:     {diff_path}")
    print("\n👀 Check these PNG files to see:")
    print("   - Vertebrae rendering quality")
    print("   - L1 displacement (upward)")
    print("   - Deformation visibility")
    print("="*70)

