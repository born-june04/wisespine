
import numpy as np
import nibabel as nib
from pathlib import Path
import scipy.ndimage as ndi
import sys

def run_artifact_synthesis(ct_path_in, hw_path_in, out_path_in):
    ct_path = Path(ct_path_in)
    mask_path = Path(hw_path_in)
    out_path = Path(out_path_in)
    
    if not ct_path.exists() or not mask_path.exists():
        print(f"Skipping artifact synthesis (missing input: {ct_path} or {mask_path})")
        return
        
    print(f"Loading data for Artifacts: {ct_path.name}...")
    nii_ct = nib.load(ct_path)
    nii_mask = nib.load(mask_path)
    
    ct_data = np.asanyarray(nii_ct.dataobj).astype(np.float32)
    mask_data = np.asanyarray(nii_mask.dataobj).astype(bool) 
    
    # 1. Implant Hardware
    print("Implanting hardware (3000 HU)...")
    ct_data[mask_data] = 3000.0
    
    # 2. Simple Blooming (Scatter)
    z_indices = np.where(mask_data.sum(axis=(0,1)) > 0)[0]
    print(f"Applying blooming to {len(z_indices)} slices...")
    
    for z in z_indices:
        slice_mask = mask_data[:, :, z].astype(np.float32)
        bloom = ndi.gaussian_filter(slice_mask, sigma=1.5)
        # Add bloom to CT (glare)
        ct_data[:, :, z] += bloom * 500.0
        
    # Clip
    ct_data = np.clip(ct_data, -1000, 3095) 
    
    print(f"Saving {out_path}...")
    nib.save(nib.Nifti1Image(ct_data, nii_ct.affine), out_path)
    print("Done.")

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angles = [20, 40, 60]
    
    for angle in angles:
        print(f"\nProcessing Cobb {angle} Hardware (Simplified)...")
        ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
        mask_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
        out_path = root / f"scoliosis_cobb{angle}_artifacts.nii.gz"
        
        run_artifact_synthesis(ct_path, mask_path, out_path)

if __name__ == "__main__":
    main()
