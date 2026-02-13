
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import gc

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    art_path = root / f"scoliosis_cobb{angle}_artifacts.nii.gz"
    
    # Check
    if not art_path.exists():
        print("Artifact path invalid")
        return

    # Find metal slice using header or iterative check
    # We can't load full volume.
    print("Finding metal slice...")
    nii_hw = nib.load(hw_path)
    shape = nii_hw.shape
    
    z_metal = shape[2] // 2
    for z in range(0, shape[2], 20):
        s = np.asanyarray(nii_hw.dataobj[:, :, z])
        if s.max() > 0:
            z_metal = z
            print(f"Found metal at {z}")
            break
            
    # Load just that slice
    print(f"Loading slice {z_metal}...")
    nii_ct = nib.load(ct_path)
    ctl = np.asanyarray(nii_ct.dataobj[:, :, z_metal]).T
    
    nii_hw = nib.load(hw_path)
    hwl = np.asanyarray(nii_hw.dataobj[:, :, z_metal]).T
    
    nii_art = nib.load(art_path)
    artl = np.asanyarray(nii_art.dataobj[:, :, z_metal]).T
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin, vmax = -200, 1500
    
    axes[0].imshow(ctl, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Baseline (Cobb {angle}°)", fontsize=14)
    
    axes[1].imshow(hwl, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[1].set_title("Hardware Mask", fontsize=14)
    
    axes[2].imshow(artl, cmap='gray', origin='lower', vmin=vmin, vmax=3000)
    axes[2].set_title("Surgical Implants (Hardware)", fontsize=14, fontweight='bold', color='crimson')
    
    # Zoom
    coords = np.argwhere(hwl)
    if len(coords) > 0:
        c = coords.mean(axis=0).astype(int)
        s = 50
        rect = plt.Rectangle((c[1]-s, c[0]-s), 2*s, 2*s, linewidth=2, edgecolor='yellow', facecolor='none')
        axes[2].add_patch(rect)
        
    plt.tight_layout()
    out_png = root / "surgical_hardware_visualization.png"
    plt.savefig(str(out_png), dpi=150)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
