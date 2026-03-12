
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    
    # We want to show Cobb 60 as the most severe case
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    art_path = root / f"scoliosis_cobb{angle}_artifacts.nii.gz"
    tum_path = root / f"scoliosis_cobb{angle}_tumors.nii.gz"
    
    if not ct_path.exists():
        print("Baseline CT missing")
        return
        
    print(f"Loading data for Cobb {angle}...")
    nii_ct = nib.load(ct_path)
    ct = nii_ct.get_fdata()
    
    art = None
    if art_path.exists():
        art = nib.load(art_path).get_fdata()
        
    tum = None
    if tum_path.exists():
        tum = nib.load(tum_path).get_fdata()
        
    # Find interesting slices
    # Artifacts: Need slice with metal
    z_art = ct.shape[2] // 2
    if art is not None:
        # Diff
        diff = np.abs(art - ct)
        z_indices = np.where(diff.sum(axis=(0,1)) > 1000)[0]
        if len(z_indices) > 0:
            z_art = z_indices[len(z_indices)//2] # Pick middle one
            
    # Tumors: Need slice with tumor
    z_tum = ct.shape[2] // 2
    if tum is not None:
        diff = np.abs(tum - ct)
        z_indices = np.where(diff.sum(axis=(0,1)) > 100)[0]
        if len(z_indices) > 0:
            # Pick one that looks like lytic (low density) and one blastic (high)
            # Just pick the one with max change
            z_tum = z_indices[np.argmax([diff[:,:,z].sum() for z in z_indices])]
            
    print(f"Slices selected: Art={z_art}, Tum={z_tum}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    vmin, vmax = -200, 1500
    
    # Row 1: Full Slice
    # Baseline (at Art slice)
    axes[0, 0].imshow(ct[:, :, z_art].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Baseline (Cobb {angle}°)", fontsize=14, fontweight='bold')
    
    # Artifacts
    if art is not None:
        axes[0, 1].imshow(art[:, :, z_art].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("Surgical Artifacts (Radon)", fontsize=14, fontweight='bold', color='crimson')
    else:
        axes[0, 1].text(0.5, 0.5, "Preprocessing...", ha='center')
        
    # Tumors
    if tum is not None:
        axes[0, 2].imshow(tum[:, :, z_tum].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title("Tumor Simulation (Lytic/Blastic)", fontsize=14, fontweight='bold', color='crimson')
    else:
        axes[0, 2].text(0.5, 0.5, "Preprocessing...", ha='center')
        
    # Row 2: Zoomed Regions
    # Artifact Zoom
    if art is not None:
        # Find metal center
        diff = np.abs(art[:,:,z_art] - ct[:,:,z_art])
        coords = np.argwhere(diff > 500)
        if len(coords) > 0:
            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            mid = (min_c + max_c) // 2
            s = 60 # 120x120 window
            zoom_slice = art[mid[0]-s:mid[0]+s, mid[1]-s:mid[1]+s, z_art]
            axes[1, 1].imshow(zoom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 1].set_title("Zoom: Beam Hardening & Streaks", fontsize=12)
            
            # Baseline reference
            zoom_base = ct[mid[0]-s:mid[0]+s, mid[1]-s:mid[1]+s, z_art]
            axes[1, 0].imshow(zoom_base.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 0].set_title("Zoom: Baseline Anatomy", fontsize=12)
    
    # Tumor Zoom
    if tum is not None:
        # Find tumor center
        diff = np.abs(tum[:,:,z_tum] - ct[:,:,z_tum])
        coords = np.argwhere(diff > 50)
        if len(coords) > 0:
            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            mid = (min_c + max_c) // 2
            s = 50
            zoom_tum = tum[mid[0]-s:mid[0]+s, mid[1]-s:mid[1]+s, z_tum]
            axes[1, 2].imshow(zoom_tum.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axes[1, 2].set_title("Zoom: Pathological Texture", fontsize=12)
            
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    out_png = root / "comprehensive_pathology_comparison.png"
    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
