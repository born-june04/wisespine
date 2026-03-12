
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, center_of_mass
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
import sys
import gc

def rasterize_screw(shape, entry, direction, length_mm, radius_mm, spacing):
    """
    Rasterize a cylinder (pedicle screw) into the volume.
    All geometry is computed in mm-space to handle anisotropic voxels correctly.
    
    Args:
        shape: volume shape in voxels
        entry: entry point in voxel coordinates
        direction: insertion direction (unit vector in voxel space)
        length_mm: screw length in mm (typically 40-50mm)
        radius_mm: screw radius in mm (typically 2.75-3.25mm)
        spacing: voxel spacing (sx, sy, sz) in mm
    """
    mask = np.zeros(shape, dtype=bool)
    sp = np.array(spacing[:3], dtype=np.float64)
    
    # Convert direction from voxel-space to mm-space, then normalize
    d_mm = direction * sp  # scale by spacing
    d_mm = d_mm / (np.linalg.norm(d_mm) + 1e-9)
    
    # Convert entry to mm-space
    entry_mm = entry * sp
    
    # Bounding box in voxel space (generous)
    # Max possible extent in each dimension
    max_extent_vox = np.ceil(length_mm / sp + 2 * radius_mm / sp + 3).astype(int)
    
    center_mm = entry_mm + d_mm * length_mm / 2.0
    center_vox = center_mm / sp
    
    min_idx = np.maximum(0, np.floor(center_vox - max_extent_vox / 2).astype(int))
    max_idx = np.minimum(np.array(shape), np.ceil(center_vox + max_extent_vox / 2).astype(int))
    
    ranges = [np.arange(min_idx[i], max_idx[i]) for i in range(3)]
    if any(len(r) == 0 for r in ranges):
        return mask
    
    # Meshgrid in voxel coordinates
    I, J, K = np.meshgrid(*ranges, indexing='ij')
    
    # Convert grid to mm-space
    Imm = I * sp[0]
    Jmm = J * sp[1]
    Kmm = K * sp[2]
    
    # Vector from entry to each grid point (in mm)
    vec = np.stack([Imm - entry_mm[0], Jmm - entry_mm[1], Kmm - entry_mm[2]], axis=-1)
    
    # Project onto screw axis (in mm)
    proj = np.dot(vec, d_mm)
    
    # Radial distance squared (in mm²)
    vec_sq = np.sum(vec**2, axis=-1)
    dist_sq = vec_sq - proj**2
    
    # Cylinder mask: within length and radius (in mm)
    cyl_mask = (proj >= 0) & (proj <= length_mm) & (dist_sq <= radius_mm**2)
    
    mask[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = cyl_mask
    
    return mask

def simulate_physical_insertion(vertebra_mask, side='left'):
    """
    Physically optimal insertion using Distance Transform Ridge (Medial Axis).
    """
    if vertebra_mask.sum() == 0:
        return None, None
        
    dist_field = distance_transform_edt(vertebra_mask)
    com = center_of_mass(vertebra_mask)
    
    # Crop to Posterior half
    y_com = int(com[1])
    posterior_crop = dist_field.copy()
    posterior_crop[:, y_com:, :] = 0 
    
    # Crop to Side
    x_com = int(com[0])
    side_crop = posterior_crop.copy()
    if side == 'left':
         side_crop[x_com:, :, :] = 0 
    else:
         side_crop[:x_com, :, :] = 0 
         
    if side_crop.max() == 0:
        return None, None
        
    max_idx = np.unravel_index(np.argmax(side_crop), side_crop.shape)
    peak_val = side_crop[max_idx]
    
    # Threshold to find the "tube"
    tube_mask = (side_crop > (peak_val * 0.85))
    coords = np.argwhere(tube_mask)
    if len(coords) < 10:
        return None, None
        
    # PCA to find axis
    mean_c = coords.mean(axis=0)
    centered = coords - mean_c
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 2] 
    
    # Orientation: We want Posterior -> Anterior (+Y)
    if axis[1] < 0:
        axis = -axis
        
    center_pt = np.array(max_idx, dtype=float)
    
    # Ray cast backwards (-axis) until distance < 1 (Surface)
    t = 0
    trace_pt = center_pt
    while True:
        idx = np.round(trace_pt).astype(int)
        if (idx < 0).any() or (idx >= dist_field.shape).any():
            break
        d = dist_field[tuple(idx)]
        if d < 1.0: 
            break
        trace_pt -= axis * 0.5 
        t += 0.5
        if t > 50: 
            break
            
    entry_point = trace_pt
    return entry_point, axis

def run_hardware_placement(ct_path_in, mask_path_in, pose_path_in, out_dir, angle=60):
    ct_path = Path(ct_path_in)
    mask_path = Path(mask_path_in) # Original mask for bounds
    pose_path = Path(pose_path_in)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pose_path.exists():
        print(f"Pose not found: {pose_path}")
        return
        
    with open(pose_path) as f:
        full_pose_data = json.load(f)
        
    angle_str = str(angle)
    if angle_str not in full_pose_data:
        print(f"Angle {angle} not in pose data")
        return
        
    pose_data = full_pose_data[angle_str]
    centers_dict = pose_data["centers"]
    rotations_dict = pose_data["rotations"]
    
    print(f"Processing Cobb {angle} Physics Placement...")
    
    # Target volume shape (from output CT)
    if ct_path.exists():
        img = nib.load(ct_path)
        target_shape = img.header.get_data_shape()
        target_affine = img.header.get_best_affine()
    else:
        print("Warning: Output CT not found, guessing shape")
        target_shape = (583, 512, 1017)
        target_affine = np.eye(4)
    
    print(f"Target Shape: {target_shape}")
    final_hardware_mask = np.zeros(target_shape, dtype=np.uint8)
    
    print("Loading original mask headers...")
    nii_mask = nib.load(mask_path)
    mask_shape = nii_mask.header.get_data_shape()
    
    # Memory-Safe Bounding Box Detection
    print("Scanning mask for bounding boxes (Slice-by-Slice)...")
    label_bounds = {}
    n_slices = mask_shape[2]
    block_size = 50
    
    for z_start in range(0, n_slices, block_size):
        z_end = min(z_start + block_size, n_slices)
        block = nii_mask.dataobj[:, :, z_start:z_end].astype(np.uint8)
        
        present_labels = np.unique(block)
        present_labels = present_labels[present_labels > 0]
        
        for lbl in present_labels:
            lbl = int(lbl)
            coords = np.argwhere(block == lbl)
            if len(coords) == 0: continue
            coords[:, 2] += z_start
            
            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            
            if lbl not in label_bounds:
                label_bounds[lbl] = [min_c[0], min_c[1], min_c[2], max_c[0], max_c[1], max_c[2]]
            else:
                curr = label_bounds[lbl]
                label_bounds[lbl] = [
                    min(curr[0], min_c[0]), min(curr[1], min_c[1]), min(curr[2], min_c[2]),
                    max(curr[3], max_c[0]), max(curr[4], max_c[1]), max(curr[5], max_c[2])
                ]
        del block
        if z_start % 200 == 0:
             gc.collect()
            
    print(f"Found {len(label_bounds)} labels.")
    spacing = nii_mask.header.get_zooms()
    labels = [int(k) for k in centers_dict.keys()]
    
    count = 0
    for label in labels:
        label_str = str(label)
        if label_str not in rotations_dict: continue
        if label not in label_bounds: continue
            
        bounds = label_bounds[label]
        slices = (
            slice(bounds[0], bounds[3]+1),
            slice(bounds[1], bounds[4]+1),
            slice(bounds[2], bounds[5]+1)
        )
        vert_crop = np.asanyarray(nii_mask.dataobj[slices]).astype(np.uint8)
        vert_crop = (vert_crop == label)
        if vert_crop.sum() == 0: continue
            
        offset = np.array([bounds[0], bounds[1], bounds[2]])
        com_local = center_of_mass(vert_crop)
        com_orig = com_local + offset
        
        entry_l, ax_l = simulate_physical_insertion(vert_crop, 'left')
        entry_r, ax_r = simulate_physical_insertion(vert_crop, 'right')
        
        if entry_l is None and entry_r is None:
            continue
            
        euler = rotations_dict[label_str]
        center_new = np.array(centers_dict[label_str])
        rot = R.from_euler('xyz', euler, degrees=True)
        
        def transform_point_from_crop(pt_local_crop):
            pt_global_orig = pt_local_crop + offset
            pt_rel = pt_global_orig - com_orig
            pt_rot = rot.apply(pt_rel)
            pt_final = pt_rot + center_new
            return pt_final

        def transform_vec(vec):
            return rot.apply(vec)
            
        # Clinical pedicle screw dimensions (in mm)
        screw_length_mm = 45.0   # 40-50mm typical
        screw_radius_mm = 3.0    # 5.5-6.5mm diameter typical
        
        if entry_l is not None:
             start_new = transform_point_from_crop(entry_l)
             vec_new = transform_vec(ax_l)
             s_mask = rasterize_screw(target_shape, start_new, vec_new, screw_length_mm, screw_radius_mm, spacing)
             final_hardware_mask[s_mask] = 1
             
        if entry_r is not None:
             start_new = transform_point_from_crop(entry_r)
             vec_new = transform_vec(ax_r)
             s_mask = rasterize_screw(target_shape, start_new, vec_new, screw_length_mm, screw_radius_mm, spacing)
             final_hardware_mask[s_mask] = 1
             
        count += 1
        if count % 5 == 0: gc.collect()
        
    print(f"Placed hardware in {count} vertebrae.")
    
    out_hw = out_dir / f"scoliosis_cobb{angle}_hardware.nii.gz"
    print(f"Saving {out_hw}...")
    nib.save(nib.Nifti1Image(final_hardware_mask, target_affine), out_hw)
    print("Done.")

def main():
    root = Path("/gscratch/scrubbed/june0604/wisespine/outputs/phase4_scoliosis")
    pose_path = root / "scoliosis_pose.json"
    angle = 60
    ct_path = root / f"scoliosis_cobb{angle}.nii.gz"
    mask_path = Path("/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    
    run_hardware_placement(ct_path, mask_path, pose_path, root, angle)

if __name__ == "__main__":
    main()
