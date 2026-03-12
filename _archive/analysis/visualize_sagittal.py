
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import center_of_mass, find_objects

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    print(f"Loading hardware for Cobb {angle}...")
    nii_hw = nib.load(hw_path)
    # Load fully if possible (hardware mask is sparse but volume is same)
    # 512x512x1000 uint8 ~ 250MB. Safe.
    hw_data = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    
    print("Finding screws...")
    # Label connected components to find individual screws
    from scipy.ndimage import label
    lbl_hw, n_feats = label(hw_data > 0)
    print(f"Found {n_feats} hardware components.")
    
    if n_feats == 0:
        print("No hardware found.")
        return
        
    # Find a representative screw (e.g. near the apex / middle of Z)
    slices = find_objects(lbl_hw)
    
    # Filter for "Screw-like" objects (not rods)
    # Rods are very long in Z. Screws are short in Z but long in Y (AP).
    screws = []
    for i, sl in enumerate(slices):
        if sl is None: continue
        # Dims
        dz = sl[2].stop - sl[2].start
        dy = sl[1].stop - sl[1].start
        dx = sl[0].stop - sl[0].start
        
        # Heuristic: Screws are roughly 45mm long (AP/Y) -> ~30-40 voxels?
        # Rods are >100 voxels in Z.
        if dy > dz and dy > dx: # Oriented in Y (AP)
            screws.append(i+1)
            
    print(f"Identified {len(screws)} potential screws.")
    if not screws:
        print("No clear screws found, visualizing largest component (Rod?)")
        target_label = 1 # Fallback
    else:
        # Pick one in the middle
        target_label = screws[len(screws)//2]
        
    print(f"Visualizing Screw Label {target_label} (Sagittal)...")
    
    # Get bounding box of this screw
    sl = slices[target_label-1]
    
    # Expand ROI for context
    pad = 30
    z_start = max(0, sl[2].start - pad)
    z_end = min(hw_data.shape[2], sl[2].stop + pad)
    y_start = max(0, sl[1].start - pad)
    y_end = min(hw_data.shape[1], sl[1].stop + pad)
    # For sagittal, we slice X.
    # We want the slice passing through the center of the screw in X.
    x_center = (sl[0].start + sl[0].stop) // 2
    
    x_start = max(0, x_center - 1)
    x_end = min(hw_data.shape[0], x_center + 1) # Single slice or slab
    
    # Load CT Crop
    print("Loading CT ROI...")
    nii_ct = nib.load(ct_path)
    # Slicing: X, Y, Z
    # We want Sagittal: Y-Z plane at fixed X.
    ct_crop = np.asanyarray(nii_ct.dataobj[x_center-2:x_center+3, y_start:y_end, z_start:z_end])
    hw_crop = hw_data[x_center-2:x_center+3, y_start:y_end, z_start:z_end]
    
    # MIP (Max Intensity Projection) typically creates a "Slab" effect
    # Projects the brightest voxels (bone/metal) to the viewing plane.
    # Sagittal View: Project along X axis (0)
    ct_mip = ct_crop.max(axis=0).T # Transpose to get Z up, Y right (or similar)
    hw_mip = hw_crop.max(axis=0).T
    
    # Orientation check:
    # Usually:
    # Y axis: Anterior-Posterior
    # Z axis: Superior-Inferior
    # Transposing makes Y horizontal, Z vertical. Correct for sagittal.
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ct_mip, cmap='gray', origin='lower', vmin=-200, vmax=2000)
    
    # Overlay Hardware
    # Create RGBA overlay
    overlay = np.zeros(hw_mip.shape + (4,))
    overlay[..., 0] = 1.0 # Red
    overlay[..., 3] = hw_mip * 0.6 # Alpha where mask is
    
    ax.imshow(overlay, origin='lower')
    
    ax.set_title(f"Sagittal View (MIP 5-slice slab)\nBiomechanical Alignment Check\nCenter X={x_center}", fontsize=15)
    ax.set_xlabel("Posterior <---> Anterior (Y)", fontsize=12)
    ax.set_ylabel("Inferior <---> Superior (Z)", fontsize=12)
    
    # Annotation
    # Draw arrow showing trajectory
    if hw_mip.sum() > 0:
        # Find axis via PCA on 2D
        coords = np.argwhere(hw_mip)
        if len(coords) > 5:
            c = coords.mean(axis=0)
            # Annotate
            ax.annotate("Pedicle Isthmus\n(Intramedullary)", 
                       xy=(c[1], c[0]), 
                       xytext=(c[1]+20, c[0]+20),
                       arrowprops=dict(facecolor='yellow', shrink=0.05),
                       color='yellow', fontsize=12, fontweight='bold')
                       
    plt.tight_layout()
    out_path = root / "surgical_hardware_sagittal.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
