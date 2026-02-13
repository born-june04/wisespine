
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import time
import gc

# Import physics module
# Import physics module
# Handle spine-rl-sim directory (dashes problem)
sys.path.append(str(Path(__file__).parent / "spine-rl-sim"))
from modules.artifact_physics import simulate_metal_artifact_radon

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine")
    out_dir = root / "outputs/phase4_scoliosis"
    
    angles = [20, 40, 60]
    
    for angle in angles:
        print(f"\nProcessing Cobb {angle} Artifacts...")
        ct_path = out_dir / f"scoliosis_cobb{angle}.nii.gz"
        mask_path = out_dir / f"scoliosis_cobb{angle}_hardware.nii.gz"
        
        if not ct_path.exists() or not mask_path.exists():
            print(f"Skipping {angle} (missing input)")
            continue
            
        print("Loading data...")
        nii_ct = nib.load(ct_path)
        nii_mask = nib.load(mask_path)
        
        # Load data (iterative usually better for RAM, but let's try full load for speed if manageable)
        # 1626 slices... 1.2GB float32. Two volumes = 2.4GB. 
        # Python overhead... maybe 6GB. Safe on 120GB node.
        
        ct_data = np.asanyarray(nii_ct.dataobj).astype(np.float32)
        mask_data = np.asanyarray(nii_mask.dataobj).astype(bool)
        
        # Output volume
        out_vol = ct_data.copy()
        
        # Find slices with metal
        z_indices = np.where(mask_data.sum(axis=(0,1)) > 0)[0]
        print(f"Found {len(z_indices)} slices with metal.")
        
        if len(z_indices) == 0:
            print("No metal found?")
            continue
            
        # Process metal slices
        t0 = time.time()
        for i, z in enumerate(z_indices):
            if i % 10 == 0:
                print(f"  Slice {z} ({i+1}/{len(z_indices)})", end='\r')
                
            slice_ct = ct_data[:, :, z]
            slice_mask = mask_data[:, :, z]
            
            # Apply Radon Physics
            # Downsample for speed? Radon is O(N^3). 512x512 is slow.
            # But we need high res.
            # Let's crop to spine?
            # Or just run it.
            
            artifact_slice = simulate_metal_artifact_radon(
                slice_ct, slice_mask, 
                num_angles=180, # Trade-off speed/quality
                beam_hardening_strength=0.2,
                scatter_strength=0.05
            )
            
            out_vol[:, :, z] = artifact_slice
            
        print(f"  Processed in {time.time()-t0:.1f}s")
        
        # Save
        out_path = out_dir / f"scoliosis_cobb{angle}_artifacts.nii.gz"
        nib.save(nib.Nifti1Image(out_vol, nii_ct.affine), out_path)
        print(f"Saved {out_path}")
        
        del out_vol, ct_data, mask_data
        gc.collect()

if __name__ == "__main__":
    main()
