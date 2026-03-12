
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    
    # We want to show Cobb 60 as the most severe case
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    art_path = root / f"scoliosis_cobb{angle}_artifacts.nii.gz"
    
    if not art_path.exists():
        print("Artifacts missing (wait for generation)")
        return
        
    print(f"Loading data for Cobb {angle}...")
    nii_ct = nib.load(ct_path)
    ct = nii_ct.get_fdata()
    
    nii_hw = nib.load(hw_path)
    hw = nii_hw.get_fdata()
    
    nii_art = nib.load(art_path)
    art = nii_art.get_fdata()
    
    # Find slice with metal
    z_indices = np.where(hw.sum(axis=(0,1)) > 0)[0]
    if len(z_indices) > 0:
        z_metal = z_indices[len(z_indices)//2]
    else:
        z_metal = ct.shape[2] // 2
        print("No metal found?")
        
    print(f"Selected Slice Z={z_metal}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin, vmax = -200, 1500 # Bone window
    
    # 1. Baseline
    axes[0].imshow(ct[:, :, z_metal].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Baseline (Cobb {angle}°)", fontsize=14)
    
    # 2. Hardware Mask
    axes[1].imshow(hw[:, :, z_metal].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[1].set_title("Hardware Mask (Implantation)", fontsize=14)
    
    # 3. Artifact (Simple)
    axes[2].imshow(art[:, :, z_metal].T, cmap='gray', origin='lower', vmin=vmin, vmax=3000)
    axes[2].set_title("Surgical Artifact (Explicit + Bloom)", fontsize=14, fontweight='bold', color='crimson')
    
    # Zoom Inset
    # Find metal center
    coords = np.argwhere(hw[:, :, z_metal])
    if len(coords) > 0:
        c = coords.mean(axis=0).astype(int)
        s = 50
        # Draw box on plot 3
        rect = plt.Rectangle((c[0]-s, c[1]-s), 2*s, 2*s, linewidth=2, edgecolor='yellow', facecolor='none')
        # axes[2].add_patch(rect) # Axis T is transposed? Be careful with coords.
        
        # imshow is transposed (.T). So x is row, y is col in array -> y, x in plot.
        # c[0] is array x (plot y), c[1] is array y (plot x).
        # Rectangle (x, y) = (c[1]-s, c[0]-s)
        rect = plt.Rectangle((c[0]-s, c[1]-s), 2*s, 2*s, linewidth=2, edgecolor='yellow', facecolor='none')
        # Wait, orig=lower.
        # array[x, y].T -> [y, x].
        # imshow shows [y, x]. 
        # So array x is Vertical axis (Y in plot). Array y is Horizontal (X in plot).
        # c[0] = x = Y_plot. c[1] = y = X_plot.
        rect = plt.Rectangle((c[0]-s, c[1]-s), 2*s, 2*s, linewidth=2, edgecolor='yellow', facecolor='none')
        # No, Rectangle is (x,y)
        rect = plt.Rectangle((c[0]-s, c[1]-s), 2*s, 2*s, linewidth=2, edgecolor='yellow', facecolor='none')
        
    for ax in axes:
        ax.axis('off')
        
    out_png = root / "surgical_hardware_visualization.png"
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
