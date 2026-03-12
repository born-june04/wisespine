
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path
import sys
import gc

def run_causal_simulation(post_op_path, pre_op_path, hw_path, out_path):
    post_op_path = Path(post_op_path)
    pre_op_path = Path(pre_op_path)
    hw_path = Path(hw_path)
    out_path = Path(out_path)
    
    if not post_op_path.exists(): 
        print(f"Post-op not found: {post_op_path}")
        return
    
    print("Initializing Memory-Safe Causal Simulation...")
    
    nii_post = nib.load(post_op_path)
    affine = nii_post.affine
    shape = nii_post.shape
    
    print("Loading Post-Op volume (int16)...")
    # Load into memory? Or chunk?
    # We load full output into memory as int16 (800MB)
    vol_data = np.zeros(shape, dtype=np.int16)
    
    chunk_size = 100
    n_slices = shape[2]
    
    print(f"Processing in chunks of {chunk_size} slices...")
    
    nii_pre = nib.load(pre_op_path)
    nii_hw = nib.load(hw_path)
    
    for z_start in range(0, n_slices, chunk_size):
        z_end = min(z_start + chunk_size, n_slices)
        # print(f"  Chunk {z_start}-{z_end}...")
        
        s_z = slice(z_start, z_end)
        
        # Load chunks
        chunk_post = np.asanyarray(nii_post.dataobj[:, :, s_z]).astype(np.int16)
        chunk_pre = np.asanyarray(nii_pre.dataobj[:, :, s_z]).astype(np.int16)
        chunk_hw = np.asanyarray(nii_hw.dataobj[:, :, s_z]).astype(np.uint8)
        
        # 1. Hematoma (Resected Zone)
        resected = (chunk_pre > 150) & (chunk_post < 100)
        if np.any(resected):
            noise = np.random.normal(50, 5, np.count_nonzero(resected)).astype(np.int16)
            chunk_post[resected] = noise
            
        # 2. Muscle Edema
        hw_locs = np.argwhere(chunk_hw > 0)
        if len(hw_locs) > 0:
            x_min, x_max = hw_locs[:, 0].min(), hw_locs[:, 0].max()
            
            muscle_mask = (chunk_post > 30) & (chunk_post < 100)
            edema_mask = np.zeros_like(chunk_post, dtype=bool)
            
            edema_mask[max(0, x_min-40):x_min, :, :] = True
            edema_mask[x_max:min(shape[0], x_max+40), :, :] = True
            
            edema_mask = edema_mask & muscle_mask
            if np.any(edema_mask):
                chunk_post[edema_mask] -= 20
        
        # 3. Halo
        if np.any(chunk_hw):
            struct = generate_binary_structure(3, 1)
            dilated = binary_dilation(chunk_hw > 0, structure=struct, iterations=1)
            halo = dilated & (chunk_hw == 0)
            
            halo_target = halo & (chunk_post > 150)
            if np.any(halo_target):
                chunk_post[halo_target] = 60
                
        vol_data[:, :, s_z] = chunk_post
        
        del chunk_post, chunk_pre, chunk_hw
        
    print(f"Saving Full Volume: {out_path}...")
    nib.save(nib.Nifti1Image(vol_data, affine), out_path)
    print("Done.")

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    angle = 60
    
    post_op_path = root / f"scoliosis_cobb{angle}_postop.nii.gz"
    pre_op_path = root / f"scoliosis_cobb{angle}.nii.gz"
    hw_path = root / f"scoliosis_cobb{angle}_hardware.nii.gz"
    out_path = root / f"scoliosis_cobb{angle}_causal.nii.gz"
    
    run_causal_simulation(post_op_path, pre_op_path, hw_path, out_path)

if __name__ == "__main__":
    main()
