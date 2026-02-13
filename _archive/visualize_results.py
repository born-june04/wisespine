import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_scoliosis():
    out_dir = Path("outputs/phase4_scoliosis")
    angles = [20, 40, 60]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # Load original for reference logic? No, just the 3 outputs.
    # User feedback was about the progression.
    
    for i, angle in enumerate(angles):
        nii_path = out_dir / f"scoliosis_cobb{angle}.nii.gz"
        print(f"Loading {nii_path}...")
        
        try:
            nii = nib.load(nii_path)
            shape = nii.shape
            mid_x, mid_y, mid_z = shape[0]//2, shape[1]//2, shape[2]//2
            
            # Load only the middle slices!
            sag = np.rot90(np.asarray(nii.dataobj[mid_x, :, :]))
            cor = np.rot90(np.asarray(nii.dataobj[:, mid_y, :]))
            axi = np.rot90(np.asarray(nii.dataobj[:, :, mid_z]))
            
            # Contrast
            vmin, vmax = -1000, 1000
            
            row = i
            axes[row, 0].imshow(sag, cmap='gray', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f"Cobb {angle}° (Refined) - Sagittal", fontsize=14, fontweight='bold', color='crimson')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(cor, cmap='gray', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f"Cobb {angle}° (Refined) - Coronal", fontsize=14, fontweight='bold', color='crimson')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(axi, cmap='gray', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f"Cobb {angle}° (Refined) - Axial", fontsize=14, fontweight='bold', color='crimson')
            axes[row, 2].axis('off')
            
            # del data
            
        except Exception as e:
            print(f"Error processing Cobb {angle}: {e}")
            
    out_path = out_dir / "scoliosis_comparison_refined.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize_scoliosis()
