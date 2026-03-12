
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys

# Replicate v7 Logic
def generate_scoliosis_curve_v7(centers, cobb_angle):
    """Generate v7-compatible curve (axial_gain=0.3)."""
    sorted_labels = sorted(centers.keys(), key=lambda l: centers[l][2])
    z_coords = np.array([centers[l][2] for l in sorted_labels])
    
    z_min, z_max = z_coords.min(), z_coords.max()
    z_height = z_max - z_min
    z_norm = (z_coords - z_min) / (z_height + 1e-6)
    
    # 1. Lateral Deviation (Sine wave)
    # Cobb angle is roughly the max derivative change
    # Amplitude A -> max derivative is A * pi / L ?
    # Angle ~ arctan(dy/dz). 
    # This approximation is sufficient for matching.
    amplitude = (cobb_angle / 180.0 * np.pi) * (z_height * 0.15)
    deviation = amplitude * np.sin(np.pi * z_norm)
    
    # 2. Variable Axial Rotation (Coupling)
    # v7 used 0.3
    axial_gain = 0.3
    axial_rot = (cobb_angle * axial_gain) * np.sin(np.pi * z_norm)
    
    new_centers = {}
    rotations = {}
    
    for i, label in enumerate(sorted_labels):
        old = centers[label]
        z = z_norm[i]
        
        # Lateral translation
        dx = deviation[i]
        
        # Coronal tilt
        deriv = amplitude * (np.pi / z_height) * np.cos(np.pi * z)
        tilt = np.degrees(np.arctan(deriv))
        
        r_tilt = R.from_euler('y', -tilt, degrees=True)
        r_axial = R.from_euler('z', axial_rot[i], degrees=True)
        rotations[label] = r_axial * r_tilt
        
        # In v7, we didn't apply 'dx' to centers? 
        # Wait, check simulate_scoliosis.py logic.
        # "new_centers[label] = old + np.array([dx, 0, 0])"
        new_centers[label] = old + np.array([dx, 0, 0])
        
    return new_centers, rotations

def compute_centers_v7(mask_path):
    print("Loading mask metadata...")
    nii = nib.load(mask_path)
    shape = nii.shape
    
    # Slice-wise center computation
    # Accumulate moments: sum_x, sum_y, sum_z, count
    accum = {} # label -> [sx, sy, sz, count]
    
    print(f"Scanning {shape[2]} slices...")
    step = 10 # scan every 10th slice? No, need precision. 
    # Use block loading
    chunk_size = 100
    
    for z_start in range(0, shape[2], chunk_size):
        z_end = min(z_start + chunk_size, shape[2])
        # Load small chunk
        block = np.asarray(nii.dataobj[:, :, z_start:z_end], dtype=np.uint8)
        
        # Find labels in block
        unique = np.unique(block)
        unique = unique[unique > 0]
        
        for l in unique:
            coords = np.argwhere(block == l)
            # coords are (x, y, relative_z)
            # Global Z = z_start + relative_z
            
            # Vectorized sum
            s_x = coords[:, 0].sum()
            s_y = coords[:, 1].sum()
            s_z = (coords[:, 2] + z_start).sum()
            cnt = len(coords)
            
            if l not in accum:
                accum[l] = np.array([0.0, 0.0, 0.0, 0.0])
                
            accum[l] += np.array([s_x, s_y, s_z, cnt])
            
    centers = {}
    for l, vals in accum.items():
        if vals[3] > 0:
            centers[int(l)] = vals[:3] / vals[3]
            
    return centers

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine")
    mask_path = root / "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
    out_dir = root / "outputs/phase4_scoliosis"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Compute Centers (Original)
    # In v7 script we cropped the CT and adjusted centers.
    # We must replicate the CROP logic!
    # "z_min_spine = 521, z_max_spine = 1338"
    # "pad = 100"
    # "z_crop_start = 421"
    
    centers_global = compute_centers_v7(mask_path)
    
    all_z = [c[2] for c in centers_global.values()]
    z_min_spine = min(all_z)
    z_max_spine = max(all_z)
    pad = 100
    z_crop_start = max(0, int(z_min_spine - pad))
    
    print(f"Crop start Z: {z_crop_start}")
    
    # Adjust centers
    centers_cropped = {}
    for l, c in centers_global.items():
        centers_cropped[l] = c.copy()
        centers_cropped[l][2] -= z_crop_start
        
    angles = [20, 40, 60]
    pose_export = {}
    
    for angle in angles:
        print(f"Computing Cobb {angle}...")
        new_centers, rotations = generate_scoliosis_curve_v7(centers_cropped, angle)
        
        pose_export[str(angle)] = {
            "centers": {k: v.tolist() for k, v in new_centers.items()},
            "rotations": {k: rotations[k].as_euler('xyz', degrees=True).tolist() for k in rotations}
        }
        
    out_json = out_dir / "scoliosis_pose.json"
    with open(out_json, "w") as f:
        json.dump(pose_export, f, indent=2)
        
    print(f"Saved {out_json}")

if __name__ == "__main__":
    main()
