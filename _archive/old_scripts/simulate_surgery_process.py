
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path
import sys

def run_surgery_simulation(ct_path_in, hw_path_in, out_path_in):
    ct_path = Path(ct_path_in)
    hw_path = Path(hw_path_in)
    out_path = Path(out_path_in)
    
    if not ct_path.exists():
        print(f"Error: {ct_path} not found.")
        return
        
    print(f"Loading CT: {ct_path}...")
    nii_ct = nib.load(ct_path)
    ct_data = np.asanyarray(nii_ct.dataobj).astype(np.float32)
    
    if not hw_path.exists():
        print(f"Error: {hw_path} not found.")
        return
        
    print(f"Loading Hardware Mask: {hw_path}...")
    nii_hw = nib.load(hw_path)
    hw_data = np.asanyarray(nii_hw.dataobj).astype(np.uint8)
    
    # 2. Identify Surgical Bounds
    hw_coords = np.argwhere(hw_data > 0)
    if len(hw_coords) == 0:
        print("No hardware found.")
        # Just save copy
        nib.save(nii_ct, out_path)
        return
        
    print(f"Hardware bounds: {hw_coords.min(axis=0)} to {hw_coords.max(axis=0)}")
    
    # 3. Create Surgical Bed (Dilation)
    print("Creating Surgical Bed (Dilation)...")
    struct = generate_binary_structure(3, 1) # Connectivity 1
    
    min_c = np.maximum(0, hw_coords.min(axis=0) - 20)
    max_c = np.minimum(ct_data.shape, hw_coords.max(axis=0) + 20)
    
    slices = (slice(min_c[0], max_c[0]), slice(min_c[1], max_c[1]), slice(min_c[2], max_c[2]))
    
    hw_crop = hw_data[slices]
    ct_crop = ct_data[slices]
    
    print("  Dilating...")
    bed_mask = binary_dilation(hw_crop > 0, structure=struct, iterations=5)
    
    # Exclude metal itself
    bed_mask = bed_mask & (hw_crop == 0)
    
    # 4. Define Laminectomy Zone (Bone Removal)
    x_indices = np.where(hw_crop > 0)[0]
    x_center = (x_indices.min() + x_indices.max()) // 2
    
    lamina_mask = np.zeros_like(bed_mask)
    if x_center-15 >= 0 and x_center+15 <= bed_mask.shape[0]:
         lamina_mask[x_center-15:x_center+15, :, :] = 1
    
    resection_zone = bed_mask & lamina_mask & (ct_crop > 150)
    
    print(f"  Resecting {np.count_nonzero(resection_zone)} voxels...")
    
    if np.count_nonzero(resection_zone) > 0:
        noise = np.random.normal(40, 15, np.count_nonzero(resection_zone))
        ct_crop[resection_zone] = noise
        
    # 5. Define Bone Graft (Bone Addition)
    graft_zone = bed_mask & (ct_crop < 100)
    
    print("  Adding Bone Graft...")
    chips_prob = (np.random.rand(np.count_nonzero(graft_zone)) < 0.10)
    graft_vals = ct_crop[graft_zone]
    graft_vals[chips_prob] = np.random.normal(600, 100, np.count_nonzero(chips_prob))
    ct_crop[graft_zone] = graft_vals
    
    # 6. Air Pockets (Vacuum)
    print("  Adding Air Pockets...")
    air_zone = bed_mask & (~resection_zone) 
    air_prob = (np.random.rand(np.count_nonzero(air_zone)) < 0.005) 
    
    air_vals = ct_crop[air_zone]
    air_vals[air_prob] = -950
    ct_crop[air_zone] = air_vals
    
    # 7. Write back and Save
    ct_data[slices] = ct_crop
    
    print(f"Saving {out_path}...")
    nib.save(nib.Nifti1Image(ct_data, nii_ct.affine), out_path)
    print("Done.")

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz" # Maybe input has artifacts now?
    # Usually surgery is after artifacts.
    # But for demo, it might take tumor output?
    # Let's assume input is scoliosis_cobb{angle}.nii.gz for now as per original script.
    
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    out_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    
    run_surgery_simulation(ct_path, hw_path, out_path)

if __name__ == "__main__":
    main()
