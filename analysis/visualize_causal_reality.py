
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    # Paths
    post_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    causal_path = root / f"scoliosis_cobb{angle}_causal.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    if not causal_path.exists():
        print(f"Error: {causal_path} not found.")
        return # Simulation probably still running
        
    print(f"Initializing Visualization (Memory Safe)...")
    
    # Load Proxies (Lazy)
    nii_post = nib.load(post_path)
    nii_causal = nib.load(causal_path)
    nii_hw = nib.load(hw_path)
    
    shape = nii_post.shape
    
    # 1. Find ROI using Hardware (Lazy check?)
    # We need to find where hardware is without loading full mask if possible.
    # But hardware mask is small (uint8). Loading it is fine (512^3 uint8 = 128MB).
    # It's the float32 CTs that are big.
    
    print("Loading Hardware Mask...")
    hw_data = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    
    # Find Center indices
    x_indices = np.where(hw_data.any(axis=(1, 2)))[0]
    if len(x_indices) > 0:
        x_mid = (x_indices.min() + x_indices.max()) // 2
    else:
        x_mid = shape[0] // 2
        
    z_indices = np.where(hw_data.any(axis=(0, 1)))[0]
    if len(z_indices) > 0:
        z_mid = (z_indices.min() + z_indices.max()) // 2
    else:
        z_mid = shape[2] // 2
        
    print(f"ROI Center: X={x_mid}, Z={z_mid}")
    
    # 2. Extract Slices (On-demand)
    
    # Sagittal Midline (X = x_mid)
    # Range Z: z_mid +/- 60
    z_start = max(0, z_mid - 60)
    z_end = min(shape[2], z_mid + 60)
    
    # Slice object
    s_sag = (slice(x_mid, x_mid+1), slice(None), slice(z_start, z_end))
    
    print("Reading Sagittal Slice...")
    # Read from disk
    sag_post = np.asanyarray(nii_post.dataobj[s_sag]).squeeze().T
    sag_causal = np.asanyarray(nii_causal.dataobj[s_sag]).squeeze().T
    
    # Axial Slice (Z = z_mid)
    s_ax = (slice(None), slice(None), slice(z_mid, z_mid+1))
    
    print("Reading Axial Slice...")
    ax_post = np.asanyarray(nii_post.dataobj[s_ax]).squeeze().T
    ax_causal = np.asanyarray(nii_causal.dataobj[s_ax]).squeeze().T
    
    # Halo Zoom (On Axial)
    # Find a screw in this axial slice
    slice_hw = hw_data[:, :, z_mid]
    contours = np.argwhere(slice_hw > 0)
    
    if len(contours) > 0:
        # Pick one
        c = contours[contours[:, 0].argmin()]
        cx, cy = c[0], c[1]
        
        # Crop 40x40
        s = 30
        x_s, x_e = max(0, cx-s), min(shape[0], cx+s)
        y_s, y_e = max(0, cy-s), min(shape[1], cy+s)
        
        # We can just crop from the loaded axial slice
        halo_post = ax_post.T[x_s:x_e, y_s:y_e].T # ax_post is transposed, need to be careful
        # ax_post was .T -> (Y, X). 
        # coords cx, cy are (X, Y).
        # So ax_post[y, x]. 
        # Let's re-crop from source to be sure of orientation, or just use array slicing on 2D
        
        # ax_post is (Y_dim, X_dim). 
        # cx is index in X_dim (dim 1 of ax_post)
        # cy is index in Y_dim (dim 0 of ax_post)
        
        halo_post = ax_post[y_s:y_e, x_s:x_e]
        halo_causal = ax_causal[y_s:y_e, x_s:x_e]
    else:
        halo_post = np.zeros((60, 60))
        halo_causal = np.zeros((60, 60))
        
    # Free HW memory
    del hw_data
    
    # Plotting
    print("Generating Figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    vmin, vmax = -200, 1500
    
    # Titles and Images
    # ... (Same plotting code as before)
    axes[0,0].imshow(sag_post, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0,0].set_title("Sagittal: Laminectomy Void (Air/Gap)", fontsize=12)
    
    axes[0,1].imshow(ax_post, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0,1].set_title("Axial: Normal Muscle Density", fontsize=12)
    
    axes[0,2].imshow(halo_post, cmap='gray', origin='lower', vmin=vmin, vmax=2500)
    axes[0,2].set_title("Detail: Tight Screw Integration", fontsize=12)
    
    axes[1,0].imshow(sag_causal, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1,0].set_title("Sagittal: Hematoma Fill (Fluid)", fontsize=12, fontweight='bold', color='crimson')
    axes[1,0].annotate("Fluid Collection", xy=(sag_causal.shape[1]//2, sag_causal.shape[0]//2), 
                      xytext=(sag_causal.shape[1]//2, sag_causal.shape[0]//2+20),
                      arrowprops=dict(facecolor='red', shrink=0.05), color='red')
    
    axes[1,1].imshow(ax_causal, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1,1].set_title("Axial: Muscle Edema (Swelling)", fontsize=12, fontweight='bold', color='orange')
    
    axes[1,2].imshow(halo_causal, cmap='gray', origin='lower', vmin=vmin, vmax=2500)
    axes[1,2].set_title("Detail: Periprosthetic Halo (Lucency)", fontsize=12, fontweight='bold', color='gold')
    
    plt.tight_layout()
    out_path = root / "causal_reality_visualization.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
