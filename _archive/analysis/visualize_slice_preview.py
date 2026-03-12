
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.spatial.distance import cdist

# Fix paths
sys.path.append(str(Path(__file__).parent / "spine-rl-sim"))
from modules.artifact_physics import simulate_metal_artifact_radon
from modules.tumor_synthesis import generate_lytic_lesion, generate_blastic_lesion

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    
    if not ct_path.exists():
        print("CT missing")
        return
        
    print(f"Loading Cobb {angle} headers...")
    nii_ct = nib.load(ct_path)
    nii_hw = nib.load(hw_path)
    
    shape = nii_ct.shape
    print(f"Shape: {shape}")
    
    # 1. Find Interesting Slices (Scan hardware mask)
    print("Scanning for metal (stride 10)...")
    z_metal = -1
    for z in range(0, shape[2], 10):
        # Load small slice
        s = np.asanyarray(nii_hw.dataobj[:, :, z]).astype(np.uint8)
        if s.max() > 0:
            z_metal = z
            print(f"Found metal at Z={z}")
            break
            
    if z_metal == -1:
        print("No metal found, defaulting to middle")
        z_metal = shape[2] // 2
        
    # 2. Simulate Metal Artifact (2D)
    print(f"Simulating Metal Artifact at Z={z_metal}...")
    slice_ct = np.asanyarray(nii_ct.dataobj[:, :, z_metal]).astype(np.float32)
    slice_hw = np.asanyarray(nii_hw.dataobj[:, :, z_metal]).astype(bool)
    
    # If no metal on this exact slice (e.g. if we defaulted), make a fake screw for viz
    if slice_hw.sum() == 0:
        print("Injecting fake screw for visualization...")
        slice_hw[240:270, 200:230] = 1
        
    art_img = simulate_metal_artifact_radon(
        slice_ct, slice_hw,
        num_angles=180,
        beam_hardening_strength=0.2,
        scatter_strength=0.05
    )
    
    # 3. Simulate Tumors (Slab)
    # Pick a different slice for tumors (e.g., +100 slices away)
    z_tumor = min(z_metal + 100, shape[2] - 5)
    print(f"Simulating Tumors at Z={z_tumor}...")
    
    # Load 5-slice slab
    slab = np.asanyarray(nii_ct.dataobj[:, :, z_tumor-2:z_tumor+3]).astype(np.float32)
    
    # Approximate bone mask (Threshold)
    bone_mask = (slab > 200)
    
    # Find bone center
    coords = np.argwhere(bone_mask)
    if len(coords) > 0:
        # Pick random bone spot
        # Center of slab is Z=2
        # Restrict to middle slice
        mid_coords = coords[coords[:, 2] == 2]
        if len(mid_coords) == 0:
             mid_coords = coords
        
        # Lytic
        c_lytic = mid_coords[len(mid_coords)//3]
        slab_lytic, _ = generate_lytic_lesion(
            slab, bone_mask, c_lytic,
            radius_mm=10.0, irregularity=0.5
        )
        
        # Blastic (far away)
        dists = cdist([c_lytic], mid_coords)[0]
        far_idx = np.argmax(dists)
        c_blastic = mid_coords[far_idx]
        
        slab_combined, _ = generate_blastic_lesion(
            slab_lytic, bone_mask, c_blastic,
            radius_mm=10.0, density_increase=900.0
        )
        
        tum_img = slab_combined[:, :, 2] # Extract middle
    else:
        tum_img = slab[:, :, 2]
        
    # 4. Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    vmin, vmax = -200, 1500
    
    # Row 1: Baselines & Full
    # Original
    axes[0, 0].imshow(slice_ct.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original CT", fontsize=14)
    
    # Artifact
    axes[0, 1].imshow(art_img.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Radon Artifact Simulation", fontsize=14, fontweight='bold', color='crimson')
    
    # Tumor
    axes[0, 2].imshow(tum_img.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Tumor Simulation", fontsize=14, fontweight='bold', color='crimson')
    
    # Row 2: Zooms
    # Artifact Zoom (Center of image roughly)
    # Find metal center
    ct_center = np.argwhere(slice_hw)
    if len(ct_center) > 0:
        c = ct_center.mean(axis=0).astype(int)
        s = 60
        zoom_art = art_img[c[0]-s:c[0]+s, c[1]-s:c[1]+s]
        axes[1, 1].imshow(zoom_art.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("Zoom: Streak & Blooming", fontsize=12)
        
    # Tumor Zoom
    c = c_lytic[:2] # X, Y
    s = 40
    zoom_tum = tum_img[c[0]-s:c[0]+s, c[1]-s:c[1]+s]
    axes[1, 2].imshow(zoom_tum.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("Zoom: Lytic Lesion", fontsize=12)
    
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    out_png = root / "scoliosis_pathology_preview.png"
    plt.savefig(str(out_png), dpi=150)
    print(f"Saved {out_png}")

if __name__ == "__main__":
    main()
