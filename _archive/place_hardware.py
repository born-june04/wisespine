
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os

def find_pedicle_offsets(mask_nii, centers):
    """Find pedicle offsets relative to center in original mask."""
    print("Finding pedicle offsets...")
    data = np.asarray(mask_nii.dataobj).astype(np.uint8)
    
    offsets = {} # label -> (left_offset, right_offset)
    
    for l, center in centers.items():
        coords = np.argwhere(data == l)
        if len(coords) == 0:
            continue
            
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        
        # Pedicle heuristic (posterior-lateral)
        # Z (posterio-anterior in VerSe? usually Z is inferior-superior in axial slices? No)
        # VerSe orientation: RAS or LIA?
        # Usually axial slices scan from top to bottom (Z).
        # Pedicles are posterior. Anterior is vertebral body.
        # But we don't know exact orientation without crawling headers.
        # Let's assume standard patient coordinates.
        # Center of mass is roughly body center.
        # Pedicles are "behind" and "lateral".
        
        # Simplified: Use relative bounding box logic like in create_surgical_configurations.py
        # It worked before.
        
        # Left/Right is X axis (typically dim 0 or 1 depending on orientation).
        # Assuming RAS: +x = Right, +y = Anterior, +z = Superior.
        # If shape is (583, 512, 1626). 
        # Sagittal slice shows spine.
        
        # Let's reuse logic:
        # P_L = min_x + 0.25 width
        # P_R = min_x + 0.75 width
        # P_Y = center_y (approx)
        # P_Z = min_z + 0.25 depth? (posterior?)
        
        # We'll use the bounding box logic relative to the computed Center.
        w = max_c[0] - min_c[0]
        h = max_c[1] - min_c[1]
        d = max_c[2] - min_c[2]
        
        # Assume X is L-R.
        left_pos = np.array([min_c[0] + 0.25 * w, center[1], min_c[2] + 0.3 * d])
        right_pos = np.array([min_c[0] + 0.75 * w, center[1], min_c[2] + 0.3 * d])
        
        # Offsets
        offsets[l] = (left_pos - center, right_pos - center)
        
    return offsets

def rasterize_cylinder_segment(volume, start, end, radius, val):
    """Simple cylinder rasterization."""
    # Vector
    vec = end - start
    length = np.linalg.norm(vec)
    direction = vec / (length + 1e-6)
    
    steps = int(length * 2)
    for t in np.linspace(0, length, steps):
        p = start + t * direction
        
        # Sphere at p with radius
        # Bounding box
        r_int = int(np.ceil(radius))
        x0 = int(p[0]) - r_int
        x1 = int(p[0]) + r_int + 1
        y0 = int(p[1]) - r_int
        y1 = int(p[1]) + r_int + 1
        z0 = int(p[2]) - r_int
        z1 = int(p[2]) + r_int + 1
        
        # Clip
        x0, x1 = max(0, x0), min(volume.shape[0], x1)
        y0, y1 = max(0, y0), min(volume.shape[1], y1)
        z0, z1 = max(0, z0), min(volume.shape[2], z1)
        
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
            
        # Brute force inside box (slow but works for sparse hardware)
        for xx in range(x0, x1):
            for yy in range(y0, y1):
                for zz in range(z0, z1):
                    if (xx-p[0])**2 + (yy-p[1])**2 + (zz-p[2])**2 <= radius**2:
                        volume[xx, yy, zz] = val

def place_hardware():
    root = Path("/gscratch/scrubbed/june0604/wisespine")
    out_dir = root / "outputs/phase4_scoliosis"
    
    # 1. Load Pose
    with open(out_dir / "scoliosis_pose.json", "r") as f:
        pose_data = json.load(f)
        
    # 2. Compute Original Offsets
    # We need centers from pose '0' or '20'? No, original centers.
    # We can invoke compute_centers logic again? Or save it in export_pose.py?
    # export_pose.py logic: cropped centers.
    
    # Wait, pose.json contains the *generated* centers for Cobb 20/40/60.
    # It does NOT contain original centroids or original mask path?
    # I know the mask path.
    mask_path = root / "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
    
    # We need a quick re-computation of original centers to get offsets.
    import nibabel as nib
    nii_mask = nib.load(mask_path)
    
    # Compute centers (memory efficient logic again!)
    # Actually, let's assume we can get them from the same logic.
    # But wait, the `scoliosis_pose.json` has `centers` for each angle.
    # The `new_centers` were computed from `centers_cropped`.
    # `centers_cropped` came from `centers_global` - `z_crop_start`.
    
    # So I need to compute `centers_cropped` (original) first to get offsets.
    # Then I can apply offsets to `new_centers`.
    # But `offsets` need to be relative to `centers_cropped`.
    
    # Let's copy the center computation logic.
    print("Computing original centers/offsets...")
    
    # Chunked load
    shape = nii_mask.shape
    accum = {} 
    chunk = 100
    for z in range(0, shape[2], chunk):
        z2 = min(z+chunk, shape[2])
        d = np.asarray(nii_mask.dataobj[:, :, z:z2], dtype=np.uint8)
        u = np.unique(d)
        for l in u[u>0]:
            c = np.argwhere(d==l)
            sx = c[:,0].sum()
            sy = c[:,1].sum()
            sz = (c[:,2]+z).sum()
            cnt = len(c)
            if l not in accum: accum[l] = np.zeros(4)
            accum[l] += [sx, sy, sz, cnt]
            
    centers_orig = {int(l): v[:3]/v[3] for l,v in accum.items()}
    
    # Crop adjustment
    all_z = [c[2] for c in centers_orig.values()]
    pad = 100
    z_crop_start = max(0, int(min(all_z) - pad))
    
    centers_cropped = {l: c - np.array([0, 0, z_crop_start]) for l,c in centers_orig.items()}
    
    # Calculate offsets relative to ADJUSTED centers (Wait, offsets are vector diffs, translation invariant)
    # But we need offsets from the MASK.
    # The mask indices are Global.
    # So Offset = Pedicle_Global - Center_Global.
    # This vector should be rotated by the VERTEBRA ROTATION.
    # In original spine (Cobb 0), rotation is Identity? (Approx).
    # So Offset is in local frame.
    
    # Let's compute offsets.
    # Need to find bounding box of each label to estimate pedicle.
    # Re-scan mask? Expensive.
    # Maybe use `accum` logic to find min/max?
    # `accum` only stores mean.
    # Finding min/max requires another pass.
    # Or keep track of min/max in the first pass!
    
    # I'll modify the loop above to track min/max.
    
    min_max = {} # l -> [minx, miny, minz, maxx, maxy, maxz]
    
    accum = {}
    for z in range(0, shape[2], chunk):
        z2 = min(z+chunk, shape[2])
        d = np.asarray(nii_mask.dataobj[:, :, z:z2], dtype=np.uint8)
        u = np.unique(d)
        for l in u[u>0]:
            c = np.argwhere(d==l)
            # Global Z
            c_glob = c.copy()
            c_glob[:, 2] += z
            
            # Mean
            sx = c[:,0].sum()
            sy = c[:,1].sum()
            sz = c_glob[:,2].sum()
            cnt = len(c)
            if l not in accum: accum[l] = np.zeros(4)
            accum[l] += [sx, sy, sz, cnt]
            
            # Min/Max
            curr_min = c_glob.min(axis=0)
            curr_max = c_glob.max(axis=0)
            
            if l not in min_max:
                min_max[l] = np.concatenate([curr_min, curr_max])
            else:
                old = min_max[l]
                new_min = np.minimum(old[:3], curr_min)
                new_max = np.maximum(old[3:], curr_max)
                min_max[l] = np.concatenate([new_min, new_max])
                
    centers_orig = {int(l): v[:3]/v[3] for l,v in accum.items()}
    
    offsets = {}
    for l, mm in min_max.items():
        c = centers_orig[l]
        w = mm[3] - mm[0]
        h = mm[4] - mm[1]
        d = mm[5] - mm[2]
        
        # Pedicles (based on bounding box)
        # Left
        p_l = np.array([mm[0] + 0.25*w, c[1], mm[2] + 0.3*d])
        # Right
        p_r = np.array([mm[0] + 0.75*w, c[1], mm[2] + 0.3*d])
        
        offsets[l] = (p_l - c, p_r - c)
        
    print("Offsets computed.")
    
    # 3. Process each Angle
    for angle in ["20", "40", "60"]:
        print(f"Angle {angle}")
        pose = pose_data[angle]
        
        # Load NIfTI (v7) to get shape
        nii_ct = nib.load(out_dir / f"scoliosis_cobb{angle}.nii.gz")
        shape = nii_ct.shape
        hardware_vol = np.zeros(shape, dtype=np.uint8) # 1=Metal
        
        # Place Screws (T4-L4? Or all?)
        # Let's do T5-L3 (typical long fusion)
        # Labels? Need to map labels to levels?
        # We don't have label->level map.
        # Just use indices 5 to 17 (roughly).
        sorted_labels = sorted([int(k) for k in pose["centers"].keys()])
        # subset = sorted_labels[4:-2] # Skip ends
        subset = sorted_labels
        
        for l in subset:
            if l not in offsets: continue
            
            c_new = np.array(pose["centers"][str(l)])
            rot_euler = pose["rotations"][str(l)] # degrees
            r = R.from_euler('xyz', rot_euler, degrees=True)
            
            off_l, off_r = offsets[l]
            
            # Apply rotation to offset
            p_l_new = c_new + r.apply(off_l)
            p_r_new = c_new + r.apply(off_r)
            
            # Screw direction: Towards center (approx)
            # Or just "Anterior" in local frame?
            # Anterior is +Y? Let's assume +Y.
            dir_local = np.array([0, 1.0, 0]) # Anterior
            dir_new = r.apply(dir_local)
            
            # Rasterize Cylinder
            # Radius 3mm (6mm diam)
            rasterize_cylinder_segment(hardware_vol, p_l_new, p_l_new + dir_new * 40, 3.0, 1)
            rasterize_cylinder_segment(hardware_vol, p_r_new, p_r_new + dir_new * 40, 3.0, 1)
            
        # Add Rods?
        # Connect screw heads
        # Spline? Linear segments for now.
        
        # Save Mask
        out_name = out_dir / f"scoliosis_cobb{angle}_hardware.nii.gz"
        nib.save(nib.Nifti1Image(hardware_vol, nii_ct.affine), out_name)
        print(f"Saved {out_name}")

if __name__ == "__main__":
    place_hardware()
