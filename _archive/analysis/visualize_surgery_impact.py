
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    pre_path = root / f"scoliosis_cobb{angle}.nii.gz"
    post_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    if not post_path.exists():
        print(f"Error: {post_path} not found.")
        return
        
    print(f"Loading Volumes...")
    nii_pre = nib.load(pre_path)
    pre_data = np.asanyarray(nii_pre.dataobj)
    
    nii_post = nib.load(post_path)
    post_data = np.asanyarray(nii_post.dataobj)
    
    nii_hw = nib.load(hw_path)
    hw_data = np.asanyarray(nii_hw.dataobj)
    
    # Find a good sagittal slice to visualize
    # Laminectomy happens near the MIDLINE (X-center of screws)
    # Hardware is lateral (Pedicles). Midline is between them.
    
    # Find active Z range
    z_indices = np.where(hw_data.any(axis=(0, 1)))[0]
    z_center = (z_indices.min() + z_indices.max()) // 2
    
    # Find X-center (Midline)
    x_indices = np.where(hw_data.any(axis=(1, 2)))[0]
    x_midline = (x_indices.min() + x_indices.max()) // 2
    
    print(f"Visualizing Midline Sagittal Slice at X={x_midline}, Z-center={z_center}")
    
    # Extract ROI around the surgery site
    pad_z = 100
    z_start = max(0, z_center - pad_z) 
    z_end = min(pre_data.shape[2], z_center + pad_z)
    
    # Y range (Posterior focus)
    # Usually low Y?
    # Let's verify by finding bone limits
    y_indices = np.where(pre_data[x_midline, :, z_center] > 100)[0]
    if len(y_indices) > 0:
        y_min = max(0, y_indices.min() - 20)
        y_max = min(pre_data.shape[1], y_indices.max() + 20)
    else:
        y_min, y_max = 0, pre_data.shape[1]
        
    # Slices
    # Pre-Op
    pre_slice = pre_data[x_midline, y_min:y_max, z_start:z_end].T
    # Post-Op
    post_slice = post_data[x_midline, y_min:y_max, z_start:z_end].T
    
    # Compute Diff for Heatmap
    diff = post_slice - pre_slice
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    vmin, vmax = -200, 1000
    
    axes[0].imshow(pre_slice, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("Pre-Op (Intact Lamina)", fontsize=14)
    axes[0].set_ylabel("Superior (Z)", fontsize=12)
    axes[0].set_xlabel("Posterior (Y)", fontsize=12)
    
    axes[1].imshow(post_slice, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("Post-Op (Laminectomy & Graft)", fontsize=14)
    axes[1].set_xlabel("Posterior (Y)", fontsize=12)
    
    # Comparison / Diff
    # Show what changed
    # Red = Removed (Bone -> Tissue/Air)
    # Green = Added (Tissue -> Graft)
    
    # Normalize diff for visualization
    # Negative diff (Bone removed) -> Red
    # Positive diff (Graft added) -> Green
    
    # We can use a custom colormap or simple overlay
    # Create RGB image from Post-Op
    # Overlay Red where diff < -100
    # Overlay Green where diff > 100
    
    rgb = np.stack([pre_slice]*3, axis=-1)
    # Normalize to 0-1 for plotting base
    rgb = (rgb - vmin) / (vmax - vmin)
    rgb = np.clip(rgb, 0, 1)
    
    mask_removed = (diff < -50) # Bone removed
    mask_added = (diff > 50) # Graft/Artifact added
    
    # Highlight
    # Red channel boost for removed (Show where it WAS)
    rgb[mask_removed, 0] = 1.0 
    rgb[mask_removed, 1] *= 0.2
    rgb[mask_removed, 2] *= 0.2
    
    # Green channel boost for added
    rgb[mask_added, 1] = 1.0
    rgb[mask_added, 0] *= 0.2
    rgb[mask_added, 2] *= 0.2
    
    axes[2].imshow(rgb, origin='lower')
    axes[2].set_title("Surgical Change Map\nRed: Resected Bone | Green: Graft/Hardware", fontsize=14)
    axes[2].set_xlabel("Posterior (Y)", fontsize=12)
    
    # Annotate
    if np.any(mask_removed):
        # Find center of removed region
        coords = np.argwhere(mask_removed)
        c = coords.mean(axis=0)
        axes[2].annotate("Resected Lamina", xy=(c[1], c[0]), xytext=(c[1], c[0]+30),
                        arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontsize=12, fontweight='bold')
                        
    if np.any(mask_added):
        coords = np.argwhere(mask_added)
        c = coords.mean(axis=0)
        axes[2].annotate("Bone Graft / Fill", xy=(c[1], c[0]), xytext=(c[1], c[0]-30),
                        arrowprops=dict(facecolor='lime', shrink=0.05), color='lime', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = root / "surgery_impact_comparison.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
